from __future__ import annotations

import json
import logging
import re
import tempfile
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.schemas import (
    Block,
    FormulaBlockResult,
    FormulaOCRResult,
    PageContent,
    ProcessedBlock,
)
from app.services.layout_analysis_surya import (
    SuryaNotAvailableError,
    extract_latex_text,
    get_surya_predictors,
    get_value,
    image_to_data_url,
    load_image,
    load_image_bytes,
    numpy_to_pil,
)
from app.services.text_block_processor import (
    build_page_content,
    cache_page_content,
    default_route_for_type,
)

logger = logging.getLogger(__name__)

FORMULA_BLOCK_TYPES = {"formula"}
SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")

_FORMULA_CROP_CACHE: OrderedDict[str, Path] = OrderedDict()
_MAX_FORMULA_CROP_CACHE_ITEMS = 64
_PIX2TEX_UNAVAILABLE_MESSAGE: str | None = None


class FormulaBlockProcessingError(RuntimeError):
    pass


class Pix2TexNotAvailableError(FormulaBlockProcessingError):
    pass


class FormulaCropNotFoundError(FormulaBlockProcessingError):
    pass


def _cache_formula_crop(block_id: str, crop_path: Path) -> None:
    _FORMULA_CROP_CACHE[block_id] = crop_path
    _FORMULA_CROP_CACHE.move_to_end(block_id)

    while len(_FORMULA_CROP_CACHE) > _MAX_FORMULA_CROP_CACHE_ITEMS:
        _FORMULA_CROP_CACHE.popitem(last=False)


def build_crop_path(block_id: str) -> Path:
    crop_dir = Path(tempfile.gettempdir()) / "formula-block-crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    safe_block_id = SAFE_ID_RE.sub("_", block_id).strip("._") or "formula_block"
    return crop_dir / f"{safe_block_id}.png"


def build_results_path(page_content_id: str) -> Path:
    result_dir = Path(tempfile.gettempdir()) / "formula-block-results"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / f"{page_content_id}.json"


def save_results(page_content: PageContent, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(page_content.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_formula_crop_bytes(block_id: str) -> bytes:
    crop_path = _FORMULA_CROP_CACHE.get(block_id)
    if crop_path is None or not crop_path.exists():
        raise FormulaCropNotFoundError(
            f"Crop формулы для блока {block_id} не найден или уже очищен."
        )
    return crop_path.read_bytes()


def load_page_image(image: np.ndarray | bytes | str | Path) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, (str, Path)):
        return load_image(image)
    return load_image_bytes(image)


def clamp_bbox(image: np.ndarray, bbox: list[int]) -> list[int]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1 + 1, min(width, int(x2)))
    y2 = max(y1 + 1, min(height, int(y2)))
    return [x1, y1, x2, y2]


def normalize_latex(latex: str | None) -> str | None:
    if not latex:
        return None
    cleaned = str(latex).replace("\r", "\n").strip()
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned or None


def normalize_processed_blocks(
    blocks: list[Block | ProcessedBlock | dict[str, Any]],
) -> list[ProcessedBlock]:
    normalized: list[ProcessedBlock] = []

    for index, raw_block in enumerate(blocks, start=1):
        if isinstance(raw_block, ProcessedBlock):
            block = raw_block
        elif isinstance(raw_block, Block):
            source = raw_block
            block = ProcessedBlock(
                block_id=source.block_id or f"block_{index:03d}",
                type=source.type,
                reading_order=source.reading_order,
                bbox=source.bbox,
                route_to=source.route_to or default_route_for_type(source.type),
                content=None,
                confidence=None,
                needs_review=False,
                crop_data_url=None,
                crop_path=None,
                latex=source.seed_latex,
                formula_result=None,
                formula_backend=None,
                ocr_result=None,
                ocr_backend="none",
            )
        else:
            raw_dict = raw_block.model_dump() if hasattr(raw_block, "model_dump") else raw_block
            try:
                block = ProcessedBlock.model_validate(raw_dict)
            except Exception:
                source = Block.model_validate(raw_dict)
                block = ProcessedBlock(
                    block_id=source.block_id or f"block_{index:03d}",
                    type=source.type,
                    reading_order=source.reading_order,
                    bbox=source.bbox,
                    route_to=source.route_to or default_route_for_type(source.type),
                    content=None,
                    confidence=None,
                    needs_review=False,
                    crop_data_url=None,
                    crop_path=None,
                    latex=source.seed_latex,
                    formula_result=None,
                    formula_backend=None,
                    ocr_result=None,
                    ocr_backend="none",
                )

        normalized.append(
            block.model_copy(
                update={
                    "block_id": block.block_id or f"block_{index:03d}",
                    "route_to": block.route_to or default_route_for_type(block.type),
                }
            )
        )

    return sorted(
        normalized,
        key=lambda block: (block.reading_order, block.bbox[1], block.bbox[0], block.block_id),
    )


def extract_formula_blocks(
    blocks: list[Block | ProcessedBlock | dict[str, Any]],
) -> list[ProcessedBlock]:
    ordered_blocks = normalize_processed_blocks(blocks)
    return [
        block
        for block in ordered_blocks
        if block.route_to == "formula_pipeline" or block.type in FORMULA_BLOCK_TYPES
    ]


def crop_formula_block(image: np.ndarray, bbox: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = clamp_bbox(
        image,
        [
            bbox[0] - settings.formula_block_crop_padding,
            bbox[1] - settings.formula_block_crop_padding,
            bbox[2] + settings.formula_block_crop_padding,
            bbox[3] + settings.formula_block_crop_padding,
        ],
    )
    return image[y1:y2, x1:x2].copy()


def extract_formula_confidence(prediction: Any) -> float | None:
    direct_confidence = get_value(
        prediction,
        "confidence",
        "score",
        "avg_score",
        "avg_prob",
        default=None,
    )
    if direct_confidence is not None:
        return max(0.0, min(1.0, float(direct_confidence)))

    raw_lines = get_value(prediction, "text_lines", "lines", default=[]) or []
    line_scores: list[float] = []
    for line in raw_lines:
        score = get_value(line, "confidence", "score", default=None)
        if score is not None:
            line_scores.append(float(score))

    if not line_scores:
        return None
    return max(0.0, min(1.0, sum(line_scores) / len(line_scores)))


def recognize_formula_with_surya(crop: np.ndarray) -> FormulaOCRResult:
    predictors = get_surya_predictors()
    crop_height, crop_width = crop.shape[:2]

    try:
        prediction = predictors["recognition_predictor"](
            [numpy_to_pil(crop)],
            task_names=[predictors["task_names"].block_without_boxes],
            bboxes=[[[0, 0, crop_width, crop_height]]],
        )[0]
    except SuryaNotAvailableError:
        raise
    except Exception as exc:  # pragma: no cover - model runtime dependent
        raise FormulaBlockProcessingError(
            f"Surya не смог распознать формулу: {exc}"
        ) from exc

    return FormulaOCRResult(
        latex=normalize_latex(extract_latex_text(prediction)),
        confidence=extract_formula_confidence(prediction),
    )


def _import_pix2tex() -> Any:
    try:
        from pix2tex.cli import LatexOCR
    except ImportError as exc:
        raise Pix2TexNotAvailableError(
            "pix2tex не установлен. Установите `pix2tex` для fallback-распознавания формул."
        ) from exc
    return LatexOCR


@lru_cache(maxsize=1)
def get_pix2tex_engine() -> Any:
    global _PIX2TEX_UNAVAILABLE_MESSAGE

    if _PIX2TEX_UNAVAILABLE_MESSAGE is not None:
        raise Pix2TexNotAvailableError(_PIX2TEX_UNAVAILABLE_MESSAGE)

    latex_ocr_cls = _import_pix2tex()
    logger.info("Initializing pix2tex engine for formula block processing")

    try:
        return latex_ocr_cls()
    except Exception as exc:  # pragma: no cover - runtime dependent
        logger.exception("Failed to initialize pix2tex engine")
        _PIX2TEX_UNAVAILABLE_MESSAGE = f"pix2tex не удалось инициализировать: {exc}"
        raise Pix2TexNotAvailableError(_PIX2TEX_UNAVAILABLE_MESSAGE) from exc


def recognize_formula_with_pix2tex(crop: np.ndarray) -> FormulaOCRResult:
    engine = get_pix2tex_engine()

    try:
        latex = engine(numpy_to_pil(crop))
    except Pix2TexNotAvailableError:
        raise
    except Exception as exc:  # pragma: no cover - runtime dependent
        logger.exception("pix2tex failed on a formula block")
        raise FormulaBlockProcessingError(
            f"pix2tex не смог распознать формулу: {exc}"
        ) from exc

    return FormulaOCRResult(latex=normalize_latex(str(latex or "")), confidence=None)


def save_formula_crop(block_id: str, crop: np.ndarray) -> Path:
    crop_path = build_crop_path(block_id)
    numpy_to_pil(crop).save(crop_path, format="PNG")
    _cache_formula_crop(block_id, crop_path)
    return crop_path


def recognize_formula_block(crop: np.ndarray) -> tuple[FormulaOCRResult | None, str]:
    try:
        surya_result = recognize_formula_with_surya(crop)
        if surya_result.latex and (
            surya_result.confidence is None
            or surya_result.confidence >= settings.formula_block_surya_confidence_threshold
        ):
            return surya_result, "surya"
        logger.warning(
            "Surya formula OCR requested pix2tex fallback because result was empty or below confidence threshold"
        )
    except (SuryaNotAvailableError, FormulaBlockProcessingError):
        logger.exception("Surya formula OCR failed, trying pix2tex fallback")

    try:
        pix2tex_result = recognize_formula_with_pix2tex(crop)
        if pix2tex_result.latex:
            return pix2tex_result, "pix2tex"
    except (Pix2TexNotAvailableError, FormulaBlockProcessingError):
        logger.exception("pix2tex fallback failed on a formula block")

    return None, "none"


def update_page_content_with_formulas(
    blocks: list[Block | ProcessedBlock | dict[str, Any]],
    formula_results: list[FormulaBlockResult],
    *,
    page_number: int | None = None,
    page_content_id: str | None = None,
) -> PageContent:
    ordered_blocks = normalize_processed_blocks(blocks)
    formula_map = {result.block_id: result for result in formula_results}
    updated_blocks: list[ProcessedBlock] = []

    for block in ordered_blocks:
        formula_result = formula_map.get(block.block_id)
        if formula_result is None:
            updated_blocks.append(block)
            continue

        updated_blocks.append(
            block.model_copy(
                update={
                    "latex": formula_result.latex,
                    "confidence": (
                        formula_result.confidence
                        if formula_result.confidence is not None
                        else block.confidence
                    ),
                    "needs_review": block.needs_review or formula_result.needs_review,
                    "crop_data_url": formula_result.crop_data_url or block.crop_data_url,
                    "crop_path": formula_result.crop_path or block.crop_path,
                    "formula_result": formula_result.formula_result,
                    "formula_backend": formula_result.formula_backend,
                }
            )
        )

    page_content = build_page_content(
        updated_blocks,
        page_number=page_number,
        page_content_id=page_content_id,
    )
    result_path = build_results_path(page_content.page_content_id)
    page_content = page_content.model_copy(update={"result_json_path": str(result_path)})
    save_results(page_content, result_path)
    cache_page_content(page_content)
    return page_content


def process_formula_blocks(
    image: np.ndarray | bytes | str | Path,
    blocks: list[Block | ProcessedBlock | dict[str, Any]],
    *,
    page_number: int | None = None,
    page_content_id: str | None = None,
) -> PageContent:
    image_rgb = load_page_image(image)
    formula_blocks = extract_formula_blocks(blocks)
    formula_results: list[FormulaBlockResult] = []

    for block in formula_blocks:
        crop = crop_formula_block(image_rgb, block.bbox)
        crop_path = save_formula_crop(block.block_id, crop)
        if block.latex:
            formula_ocr_result = FormulaOCRResult(
                latex=normalize_latex(block.latex),
                confidence=block.confidence,
            )
            backend = "surya"
        else:
            formula_ocr_result, backend = recognize_formula_block(crop)
        formula_results.append(
            FormulaBlockResult(
                block_id=block.block_id,
                reading_order=block.reading_order,
                bbox=clamp_bbox(image_rgb, block.bbox),
                latex=formula_ocr_result.latex if formula_ocr_result else None,
                confidence=formula_ocr_result.confidence if formula_ocr_result else None,
                needs_review=formula_ocr_result is None,
                crop_path=str(crop_path),
                crop_data_url=image_to_data_url(crop),
                formula_result=formula_ocr_result,
                formula_backend=backend,
            )
        )
        logger.info(
            "Processed formula block %s with backend=%s",
            block.block_id,
            backend,
        )

    page_content = update_page_content_with_formulas(
        blocks,
        formula_results,
        page_number=page_number,
        page_content_id=page_content_id,
    )
    logger.info(
        "Processed %s formula blocks; %s need review",
        len(formula_results),
        sum(1 for result in formula_results if result.needs_review),
    )
    return page_content
