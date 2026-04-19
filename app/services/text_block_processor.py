from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.schemas import Block, LayoutBlock, OCRResult, PageContent, PageTextResponse, ProcessedBlock
from app.services.layout_analysis_surya import image_to_data_url, load_image, load_image_bytes
from app.services.page_content_presenter import build_page_content as build_presented_page_content

logger = logging.getLogger(__name__)

TEXT_BLOCK_TYPES = {"text", "title", "header", "footer", "page_number"}
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
GARBAGE_RE = re.compile(r"[�■□▪▫◆◇◻◼]+")
HYPHEN_BREAK_RE = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")
LINE_BREAK_RE = re.compile(r"\s*\n\s*")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
SPACE_AFTER_OPEN_RE = re.compile(r"([(\[{])\s+")
SPACE_BEFORE_CLOSE_RE = re.compile(r"\s+([)\]}])")


class TextBlockProcessingError(RuntimeError):
    pass


class PaddleOCRNotAvailableError(TextBlockProcessingError):
    pass


class PageTextNotFoundError(TextBlockProcessingError):
    pass


_PAGE_CONTENT_CACHE: OrderedDict[str, PageContent] = OrderedDict()
_MAX_PAGE_CONTENT_CACHE_ITEMS = 32
_PADDLE_UNAVAILABLE_MESSAGE: str | None = None


def _cache_page_content(page_content: PageContent) -> None:
    _PAGE_CONTENT_CACHE[page_content.page_content_id] = page_content
    _PAGE_CONTENT_CACHE.move_to_end(page_content.page_content_id)

    while len(_PAGE_CONTENT_CACHE) > _MAX_PAGE_CONTENT_CACHE_ITEMS:
        _PAGE_CONTENT_CACHE.popitem(last=False)


def cache_page_content(page_content: PageContent) -> None:
    _cache_page_content(page_content)


def get_cached_page_text(page_content_id: str) -> PageTextResponse:
    page_content = _PAGE_CONTENT_CACHE.get(page_content_id)
    if page_content is None:
        raise PageTextNotFoundError(
            "Собранный текст страницы не найден или уже очищен из памяти."
        )
    return PageTextResponse(
        page_content_id=page_content.page_content_id,
        page_text=page_content.page_text,
    )


def get_cached_page_content(page_content_id: str) -> PageContent:
    page_content = _PAGE_CONTENT_CACHE.get(page_content_id)
    if page_content is None:
        raise PageTextNotFoundError(
            "PageContent страницы не найден или уже очищен из памяти."
        )
    return page_content


def build_results_path(page_content_id: str) -> Path:
    result_dir = Path(tempfile.gettempdir()) / "text-block-results"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / f"{page_content_id}.json"


def save_results(page_content: PageContent, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(page_content.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _import_paddleocr() -> Any:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise PaddleOCRNotAvailableError(
            "PaddleOCR не установлен. Установите `paddleocr` и совместимый backend Paddle."
        ) from exc
    return PaddleOCR


@lru_cache(maxsize=1)
def get_paddleocr_engine() -> Any:
    global _PADDLE_UNAVAILABLE_MESSAGE

    if _PADDLE_UNAVAILABLE_MESSAGE is not None:
        raise PaddleOCRNotAvailableError(_PADDLE_UNAVAILABLE_MESSAGE)

    paddle_ocr_cls = _import_paddleocr()
    logger.info("Initializing PaddleOCR engine for text block processing")

    # PaddleOCR changed constructor parameters between major versions.
    # Try the newer API first, then fall back to the classic one.
    try:
        return paddle_ocr_cls(
            lang=settings.text_block_ocr_lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except TypeError:
        try:
            return paddle_ocr_cls(
                lang=settings.text_block_ocr_lang,
                use_angle_cls=False,
                use_gpu=False,
                show_log=False,
            )
        except Exception as exc:  # pragma: no cover - runtime dependent
            logger.exception("Failed to initialize PaddleOCR engine")
            _PADDLE_UNAVAILABLE_MESSAGE = (
                f"PaddleOCR не удалось инициализировать: {exc}"
            )
            raise PaddleOCRNotAvailableError(_PADDLE_UNAVAILABLE_MESSAGE) from exc
    except Exception as exc:  # pragma: no cover - runtime dependent
        logger.exception("Failed to initialize PaddleOCR engine")
        _PADDLE_UNAVAILABLE_MESSAGE = (
            f"PaddleOCR не удалось инициализировать: {exc}"
        )
        raise PaddleOCRNotAvailableError(_PADDLE_UNAVAILABLE_MESSAGE) from exc


def get_value(source: Any, *names: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        for name in names:
            if name in source:
                return source[name]
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return default


def load_page_image(image: np.ndarray | bytes | str | Path) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, (str, Path)):
        return load_image(image)
    return load_image_bytes(image)


def default_route_for_type(block_type: str) -> str:
    if block_type in TEXT_BLOCK_TYPES:
        return "text_pipeline"
    if block_type == "formula":
        return "formula_pipeline"
    if block_type == "table":
        return "table_pipeline"
    return "image_pipeline"


def build_routed_blocks_from_layout(
    blocks: list[LayoutBlock | dict[str, Any]],
    *,
    block_id_prefix: str | None = None,
) -> list[Block]:
    routed_blocks: list[Block] = []

    for index, raw_block in enumerate(blocks, start=1):
        layout_block = (
            raw_block if isinstance(raw_block, LayoutBlock) else LayoutBlock.model_validate(raw_block)
        )
        block_id = (
            f"{block_id_prefix}_block_{index:03d}"
            if block_id_prefix
            else f"block_{index:03d}"
        )
        routed_blocks.append(
            Block(
                block_id=block_id,
                type=layout_block.type,
                bbox=layout_block.bbox,
                reading_order=layout_block.reading_order,
                route_to=default_route_for_type(layout_block.type),
                seed_text=layout_block.text,
                seed_latex=layout_block.latex,
                seed_confidence=layout_block.confidence,
            )
        )

    return routed_blocks


def sort_blocks(blocks: list[Block | dict[str, Any]]) -> list[Block]:
    normalized: list[Block] = []

    for index, raw_block in enumerate(blocks, start=1):
        block = Block.model_validate(raw_block)
        block_id = block.block_id or f"block_{index:03d}"
        route_to = block.route_to or default_route_for_type(block.type)
        normalized.append(
            block.model_copy(
                update={
                    "block_id": block_id,
                    "route_to": route_to,
                }
            )
        )

    return sorted(
        normalized,
        key=lambda block: (block.reading_order, block.bbox[1], block.bbox[0], block.block_id),
    )


def clamp_bbox(image: np.ndarray, bbox: list[int]) -> list[int]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1 + 1, min(width, int(x2)))
    y2 = max(y1 + 1, min(height, int(y2)))
    return [x1, y1, x2, y2]


def crop_block(image: np.ndarray, bbox: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = clamp_bbox(image, bbox)
    return image[y1:y2, x1:x2].copy()


def _ocr_result_from_texts(texts: list[str], scores: list[float]) -> OCRResult:
    cleaned_texts = [text.strip() for text in texts if str(text).strip()]
    cleaned_scores = [float(score) for score in scores[: len(cleaned_texts)] if score is not None]
    confidence = (
        sum(cleaned_scores) / len(cleaned_scores)
        if cleaned_scores
        else 0.0
    )
    return OCRResult(text="\n".join(cleaned_texts), confidence=max(0.0, min(1.0, confidence)))


def _parse_predict_output(prediction: Any) -> OCRResult | None:
    rec_texts = get_value(prediction, "rec_texts", default=None)
    rec_scores = get_value(prediction, "rec_scores", default=None)
    if rec_texts is None:
        return None

    texts = [str(text) for text in (rec_texts or [])]
    scores = [float(score or 0.0) for score in (rec_scores or [])]
    return _ocr_result_from_texts(texts, scores)


def _parse_legacy_output(prediction: Any) -> OCRResult | None:
    if not isinstance(prediction, list):
        return None

    raw_lines = prediction
    if raw_lines and isinstance(raw_lines[0], list):
        first_item = raw_lines[0]
        if first_item and isinstance(first_item[0], (list, tuple)):
            raw_lines = first_item

    texts: list[str] = []
    scores: list[float] = []
    for line in raw_lines:
        if (
            isinstance(line, (list, tuple))
            and len(line) >= 2
            and isinstance(line[1], (list, tuple))
            and len(line[1]) >= 2
        ):
            texts.append(str(line[1][0] or ""))
            scores.append(float(line[1][1] or 0.0))

    if not texts:
        return None
    return _ocr_result_from_texts(texts, scores)


def recognize_text_block(crop: np.ndarray) -> OCRResult:
    ocr_engine = get_paddleocr_engine()

    try:
        if hasattr(ocr_engine, "predict"):
            predictions = ocr_engine.predict(crop)
            if predictions:
                parsed = _parse_predict_output(predictions[0])
                if parsed is not None:
                    return parsed

        if hasattr(ocr_engine, "ocr"):
            predictions = ocr_engine.ocr(crop, cls=False)
            parsed = _parse_legacy_output(predictions)
            if parsed is not None:
                return parsed
    except PaddleOCRNotAvailableError:
        raise
    except Exception as exc:  # pragma: no cover - runtime dependent
        logger.exception("PaddleOCR failed on a text block")
        raise TextBlockProcessingError(f"Ошибка OCR текстового блока: {exc}") from exc

    return OCRResult(text="", confidence=0.0)


def fallback_text_result(block: Block) -> OCRResult:
    return OCRResult(
        text=(block.seed_text or "").strip(),
        confidence=float(block.seed_confidence or 0.0),
    )


def postprocess_text(text: str) -> str:
    cleaned = ZERO_WIDTH_RE.sub("", text or "")
    cleaned = CONTROL_RE.sub(" ", cleaned)
    cleaned = GARBAGE_RE.sub(" ", cleaned)
    cleaned = HYPHEN_BREAK_RE.sub("", cleaned)
    cleaned = LINE_BREAK_RE.sub(" ", cleaned)
    cleaned = MULTISPACE_RE.sub(" ", cleaned)
    cleaned = SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = SPACE_AFTER_OPEN_RE.sub(r"\1", cleaned)
    cleaned = SPACE_BEFORE_CLOSE_RE.sub(r"\1", cleaned)
    return cleaned.strip()


def should_mark_for_review(content: str | None, confidence: float | None) -> bool:
    text = (content or "").strip()
    if not text:
        return True
    if confidence is None or confidence < settings.text_block_review_confidence:
        return True
    return len(text) < settings.text_block_min_length


def build_page_content(
    blocks: list[ProcessedBlock],
    *,
    page_number: int | None = None,
    page_content_id: str | None = None,
) -> PageContent:
    return build_presented_page_content(
        blocks,
        page_number=page_number,
        page_content_id=page_content_id,
    )


def process_text_blocks(
    image: np.ndarray | bytes | str | Path,
    blocks: list[Block | dict[str, Any]],
    *,
    page_number: int | None = None,
) -> PageContent:
    image_rgb = load_page_image(image)
    ordered_blocks = sort_blocks(blocks)
    processed_blocks: list[ProcessedBlock] = []
    paddle_unavailable = False

    for block in ordered_blocks:
        if block.route_to == "text_pipeline":
            crop = crop_block(image_rgb, block.bbox)
            ocr_backend = "paddleocr"
            try:
                ocr_result = recognize_text_block(crop)
            except (PaddleOCRNotAvailableError, TextBlockProcessingError):
                logger.warning(
                    "PaddleOCR failed or is unavailable, using Surya seed text fallback for block %s",
                    block.block_id,
                )
                paddle_unavailable = True
                ocr_result = fallback_text_result(block)
                ocr_backend = "surya_seed"
            processed_text = postprocess_text(ocr_result.text)
            processed_blocks.append(
                ProcessedBlock(
                    block_id=block.block_id or "block",
                    type=block.type,
                    reading_order=block.reading_order,
                    bbox=clamp_bbox(image_rgb, block.bbox),
                    route_to=block.route_to or "text_pipeline",
                    content=processed_text or None,
                    confidence=ocr_result.confidence,
                    needs_review=should_mark_for_review(
                        processed_text, ocr_result.confidence
                    )
                    or ocr_backend != "paddleocr",
                    crop_data_url=image_to_data_url(crop),
                    ocr_result=ocr_result,
                    ocr_backend=ocr_backend,
                )
            )
            continue

        processed_blocks.append(
            ProcessedBlock(
                block_id=block.block_id or "block",
                type=block.type,
                reading_order=block.reading_order,
                bbox=clamp_bbox(image_rgb, block.bbox),
                route_to=block.route_to or default_route_for_type(block.type),
                content=None,
                confidence=None,
                needs_review=False,
                crop_data_url=None,
                latex=block.seed_latex,
                ocr_result=None,
                ocr_backend="none",
            )
        )

    page_content = build_page_content(processed_blocks, page_number=page_number)
    result_path = build_results_path(page_content.page_content_id)
    page_content = page_content.model_copy(update={"result_json_path": str(result_path)})
    save_results(page_content, result_path)
    _cache_page_content(page_content)
    logger.info(
        "Processed %s routed blocks; %s need review",
        len(page_content.blocks),
        sum(1 for block in page_content.blocks if block.needs_review),
    )
    if paddle_unavailable:
        logger.warning(
            "Text block OCR completed with Surya fallback because PaddleOCR backend was unavailable"
        )
    return page_content
