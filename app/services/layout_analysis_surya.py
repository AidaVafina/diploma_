from __future__ import annotations

import base64
import json
import logging
import re
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageFile

from app.schemas import LayoutAnalysisResponse, LayoutBlock

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

TEXTUAL_TYPES = {"text", "title", "header", "footer", "page_number", "table"}
MERGEABLE_TYPES = {"text", "title", "header", "footer"}

TYPE_COLORS: dict[str, tuple[int, int, int]] = {
    "text": (32, 114, 229),
    "title": (85, 98, 255),
    "formula": (220, 53, 69),
    "table": (34, 139, 34),
    "image": (245, 179, 0),
    "header": (145, 99, 255),
    "footer": (255, 140, 66),
    "page_number": (112, 112, 112),
}

SURYA_LABEL_MAP = {
    "Text": "text",
    "Caption": "text",
    "Footnote": "footer",
    "Formula": "formula",
    "Equation": "formula",
    "Text-inline-math": "formula",
    "List-item": "text",
    "Page-header": "header",
    "Page-footer": "footer",
    "Picture": "image",
    "Figure": "image",
    "Section-header": "title",
    "Title": "title",
    "Table": "table",
    "Table-of-contents": "text",
    "Form": "table",
    "Handwriting": "text",
}

PAGE_NUMBER_RE = re.compile(r"^[\[\(\-–—]?\s*\d{1,4}\s*[\]\)\-–—]?$")


class LayoutAnalysisError(RuntimeError):
    pass


class SuryaNotAvailableError(LayoutAnalysisError):
    pass


class VisualizationNotFoundError(LayoutAnalysisError):
    pass


@dataclass
class AnalysisArtifacts:
    blocks: list[LayoutBlock]
    visualization_png: bytes


_ANALYSIS_CACHE: OrderedDict[str, AnalysisArtifacts] = OrderedDict()
_MAX_ANALYSIS_CACHE_ITEMS = 16


def _cache_analysis(analysis_id: str, artifacts: AnalysisArtifacts) -> None:
    _ANALYSIS_CACHE[analysis_id] = artifacts
    _ANALYSIS_CACHE.move_to_end(analysis_id)

    while len(_ANALYSIS_CACHE) > _MAX_ANALYSIS_CACHE_ITEMS:
        _ANALYSIS_CACHE.popitem(last=False)


def get_cached_visualization(analysis_id: str) -> bytes:
    artifacts = _ANALYSIS_CACHE.get(analysis_id)
    if artifacts is None:
        raise VisualizationNotFoundError("Визуализация не найдена или уже очищена из памяти.")
    return artifacts.visualization_png


def load_image(path: str | Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise LayoutAnalysisError(f"Не удалось загрузить изображение по пути: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_image_bytes(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise LayoutAnalysisError("Пустое изображение.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            return np.array(image.convert("RGB"))
    except Exception as exc:  # pragma: no cover - defensive branch
        raise LayoutAnalysisError("Не удалось декодировать изображение.") from exc


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    return Image.fromarray(image.astype(np.uint8)).convert("RGB")


def image_to_png_bytes(image: np.ndarray) -> bytes:
    pil_image = numpy_to_pil(image)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_to_data_url(image: np.ndarray) -> str:
    encoded = base64.b64encode(image_to_png_bytes(image)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _import_surya() -> dict[str, Any]:
    try:
        from surya.common.surya.schema import TaskNames
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.layout import LayoutPredictor
        from surya.recognition import RecognitionPredictor
        from surya.settings import settings as surya_settings
    except ImportError as exc:
        raise SuryaNotAvailableError(
            "Surya или Torch не установлены. Установите `surya-ocr` и `torch`."
        ) from exc

    return {
        "TaskNames": TaskNames,
        "DetectionPredictor": DetectionPredictor,
        "FoundationPredictor": FoundationPredictor,
        "LayoutPredictor": LayoutPredictor,
        "RecognitionPredictor": RecognitionPredictor,
        "surya_settings": surya_settings,
    }


def get_installed_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def describe_surya_runtime_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__

    if "pad_token_id" in message:
        transformers_version = get_installed_version("transformers")
        version_hint = (
            f" Установлена версия transformers={transformers_version}."
            if transformers_version
            else ""
        )
        return (
            "Surya не смог инициализировать модель layout. "
            "Похоже на несовместимость между Surya и установленной версией "
            f"`transformers`.{version_hint} Обычно помогает зафиксировать "
            "`transformers<5` и переустановить зависимости."
        )

    return f"Surya не смог инициализировать модели: {message}"


@lru_cache(maxsize=1)
def get_surya_predictors() -> dict[str, Any]:
    imports = _import_surya()
    foundation_predictor_cls = imports["FoundationPredictor"]
    layout_predictor_cls = imports["LayoutPredictor"]
    detection_predictor_cls = imports["DetectionPredictor"]
    recognition_predictor_cls = imports["RecognitionPredictor"]
    task_names = imports["TaskNames"]
    surya_settings = imports["surya_settings"]

    logger.info("Initializing Surya predictors")
    try:
        layout_predictor = layout_predictor_cls(
            foundation_predictor_cls(checkpoint=surya_settings.LAYOUT_MODEL_CHECKPOINT)
        )
        recognition_predictor = recognition_predictor_cls(foundation_predictor_cls())
        detection_predictor = detection_predictor_cls()
    except Exception as exc:  # pragma: no cover - model runtime dependent
        logger.exception("Failed to initialize Surya predictors")
        raise SuryaNotAvailableError(describe_surya_runtime_error(exc)) from exc

    return {
        "layout_predictor": layout_predictor,
        "recognition_predictor": recognition_predictor,
        "detection_predictor": detection_predictor,
        "task_names": task_names,
    }


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {}


def get_value(source: Any, *names: str, default: Any = None) -> Any:
    source_dict = _as_dict(source)
    for name in names:
        if isinstance(source, dict) and name in source:
            return source[name]
        if hasattr(source, name):
            return getattr(source, name)
        if name in source_dict:
            return source_dict[name]
    return default


def normalize_bbox(raw_bbox: Any) -> list[int]:
    if raw_bbox is None:
        return [0, 0, 0, 0]

    if isinstance(raw_bbox, np.ndarray):
        raw_bbox = raw_bbox.tolist()

    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        x1, y1, x2, y2 = raw_bbox
        return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

    if isinstance(raw_bbox, (list, tuple)) and raw_bbox and isinstance(raw_bbox[0], (list, tuple)):
        xs = [point[0] for point in raw_bbox]
        ys = [point[1] for point in raw_bbox]
        return [
            int(round(min(xs))),
            int(round(min(ys))),
            int(round(max(xs))),
            int(round(max(ys))),
        ]

    return [0, 0, 0, 0]


def clamp_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    clamped = [
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    ]
    if clamped[2] <= clamped[0]:
        clamped[2] = min(width, clamped[0] + 1)
    if clamped[3] <= clamped[1]:
        clamped[3] = min(height, clamped[1] + 1)
    return clamped


def bbox_area(bbox: list[int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def intersection_area(box_a: list[int], box_b: list[int]) -> int:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def horizontal_overlap_ratio(box_a: list[int], box_b: list[int]) -> float:
    overlap = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    min_width = max(1, min(box_a[2] - box_a[0], box_b[2] - box_b[0]))
    return overlap / min_width


def vertical_gap(box_a: list[int], box_b: list[int]) -> int:
    if box_b[1] >= box_a[3]:
        return box_b[1] - box_a[3]
    if box_a[1] >= box_b[3]:
        return box_a[1] - box_b[3]
    return 0


def center_inside(box: list[int], container: list[int]) -> bool:
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    return (
        container[0] <= center_x <= container[2]
        and container[1] <= center_y <= container[3]
    )


def normalize_label(label: str | None) -> str:
    if not label:
        return "text"

    if hasattr(label, "value"):
        label = label.value

    label_text = str(label)
    if label_text in SURYA_LABEL_MAP:
        return SURYA_LABEL_MAP[label_text]

    enum_suffix = label_text.split(".")[-1].replace("_", "-")
    return SURYA_LABEL_MAP.get(enum_suffix, "text")


def extract_text_lines(ocr_prediction: Any, width: int, height: int) -> list[dict[str, Any]]:
    raw_lines = get_value(ocr_prediction, "text_lines", "lines", default=[]) or []
    lines: list[dict[str, Any]] = []

    for line in raw_lines:
        text = str(get_value(line, "text", default="") or "").strip()
        bbox = clamp_bbox(
            normalize_bbox(get_value(line, "bbox", "polygon", default=None)),
            width,
            height,
        )
        if not text or bbox_area(bbox) == 0:
            continue
        lines.append(
            {
                "text": text,
                "bbox": bbox,
                "confidence": float(get_value(line, "confidence", default=0.0) or 0.0),
            }
        )

    return lines


def extract_latex_text(prediction: Any) -> str | None:
    raw_lines = get_value(prediction, "text_lines", "lines", default=[]) or []
    if raw_lines:
        texts = [str(get_value(line, "text", default="") or "").strip() for line in raw_lines]
        joined = "\n".join(part for part in texts if part)
        return joined or None

    text = str(get_value(prediction, "text", default="") or "").strip()
    return text or None


def crop_image(image: np.ndarray, bbox: list[int], padding: int = 6) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    padded_bbox = clamp_bbox(
        [x1 - padding, y1 - padding, x2 + padding, y2 + padding],
        width,
        height,
    )
    return image[padded_bbox[1] : padded_bbox[3], padded_bbox[0] : padded_bbox[2]]


def analyze_layout(image: np.ndarray | str | Path) -> dict[str, Any]:
    if isinstance(image, (str, Path)):
        image_rgb = load_image(image)
    else:
        image_rgb = image

    predictors = get_surya_predictors()
    pil_image = numpy_to_pil(image_rgb)

    logger.info("Running Surya layout analysis")
    layout_prediction = predictors["layout_predictor"]([pil_image])[0]

    logger.info("Running Surya OCR")
    ocr_prediction = predictors["recognition_predictor"](
        [pil_image],
        task_names=[predictors["task_names"].ocr_with_boxes],
        det_predictor=predictors["detection_predictor"],
    )[0]

    return {
        "image_rgb": image_rgb,
        "layout_prediction": layout_prediction,
        "ocr_prediction": ocr_prediction,
        "recognition_predictor": predictors["recognition_predictor"],
        "task_names": predictors["task_names"],
    }


def normalize_blocks(raw_blocks: dict[str, Any]) -> list[dict[str, Any]]:
    image_rgb: np.ndarray = raw_blocks["image_rgb"]
    height, width = image_rgb.shape[:2]
    layout_prediction = raw_blocks["layout_prediction"]
    recognition_predictor = raw_blocks["recognition_predictor"]
    task_names = raw_blocks["task_names"]

    layout_boxes = get_value(layout_prediction, "bboxes", default=[]) or []
    ocr_lines = extract_text_lines(raw_blocks["ocr_prediction"], width, height)

    normalized: list[dict[str, Any]] = []

    for index, raw_box in enumerate(layout_boxes, start=1):
        bbox = clamp_bbox(
            normalize_bbox(get_value(raw_box, "bbox", "polygon", default=None)),
            width,
            height,
        )
        block_type = normalize_label(str(get_value(raw_box, "label", default="Text")))
        reading_order = int(get_value(raw_box, "position", default=index) or index)
        confidence = float(get_value(raw_box, "confidence", default=0.0) or 0.0)

        related_lines = [
            line
            for line in ocr_lines
            if center_inside(line["bbox"], bbox)
            or intersection_area(line["bbox"], bbox) >= 0.5 * bbox_area(line["bbox"])
        ]
        related_lines.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        extracted_text = "\n".join(line["text"] for line in related_lines) or None
        latex_text: str | None = None

        if block_type == "formula":
            try:
                formula_crop = crop_image(image_rgb, bbox)
                crop_height, crop_width = formula_crop.shape[:2]
                latex_prediction = recognition_predictor(
                    [numpy_to_pil(formula_crop)],
                    task_names=[task_names.block_without_boxes],
                    bboxes=[[[0, 0, crop_width, crop_height]]],
                )[0]
                latex_text = extract_latex_text(latex_prediction)
            except Exception:  # pragma: no cover - model runtime dependent
                logger.exception("Failed to run Surya LaTeX OCR for bbox %s", bbox)

        normalized.append(
            {
                "type": block_type,
                "bbox": bbox,
                "confidence": max(0.0, min(1.0, confidence)),
                "reading_order": max(1, reading_order),
                "text": extracted_text,
                "latex": latex_text,
            }
        )

    return normalized


def should_merge_text_blocks(block_a: dict[str, Any], block_b: dict[str, Any], page_height: int) -> bool:
    if block_a["type"] != block_b["type"]:
        return False
    if block_a["type"] not in MERGEABLE_TYPES:
        return False

    box_a = block_a["bbox"]
    box_b = block_b["bbox"]
    max_gap = max(18, int(page_height * 0.012))

    if horizontal_overlap_ratio(box_a, box_b) < 0.55:
        return False

    if vertical_gap(box_a, box_b) > max_gap:
        return False

    return True


def merge_blocks(block_a: dict[str, Any], block_b: dict[str, Any]) -> dict[str, Any]:
    merged_text = "\n".join(
        part for part in [block_a.get("text"), block_b.get("text")] if part
    ) or None
    merged_latex = "\n".join(
        part for part in [block_a.get("latex"), block_b.get("latex")] if part
    ) or None

    return {
        "type": block_a["type"],
        "bbox": [
            min(block_a["bbox"][0], block_b["bbox"][0]),
            min(block_a["bbox"][1], block_b["bbox"][1]),
            max(block_a["bbox"][2], block_b["bbox"][2]),
            max(block_a["bbox"][3], block_b["bbox"][3]),
        ],
        "confidence": max(block_a["confidence"], block_b["confidence"]),
        "reading_order": min(block_a["reading_order"], block_b["reading_order"]),
        "text": merged_text,
        "latex": merged_latex,
    }


def classify_page_number(block: dict[str, Any], page_height: int) -> dict[str, Any]:
    text = (block.get("text") or "").strip()
    if not text or not PAGE_NUMBER_RE.fullmatch(text):
        return block

    y1, y2 = block["bbox"][1], block["bbox"][3]
    near_edge = y2 < int(page_height * 0.14) or y1 > int(page_height * 0.86)
    if near_edge:
        updated = dict(block)
        updated["type"] = "page_number"
        return updated
    return block


def postprocess_blocks(blocks: list[dict[str, Any]], page_shape: tuple[int, int]) -> list[dict[str, Any]]:
    height, width = page_shape
    page_area = height * width
    filtered: list[dict[str, Any]] = []

    for block in blocks:
        area = bbox_area(block["bbox"])
        min_area = int(page_area * 0.00045)
        if block["type"] in {"formula", "page_number"}:
            min_area = int(page_area * 0.00018)

        if area < min_area:
            logger.info("Dropping small block %s with area %s", block["type"], area)
            continue

        block = classify_page_number(block, height)
        filtered.append(block)

    sorted_blocks = sort_reading_order(filtered, renumber=False)
    merged: list[dict[str, Any]] = []

    for block in sorted_blocks:
        if merged and should_merge_text_blocks(merged[-1], block, height):
            merged[-1] = merge_blocks(merged[-1], block)
        else:
            merged.append(block)

    return merged


def sort_reading_order(
    blocks: list[dict[str, Any]],
    *,
    renumber: bool = True,
) -> list[dict[str, Any]]:
    sorted_blocks = sorted(
        blocks,
        key=lambda block: (
            block.get("reading_order", 10**9),
            block["bbox"][1],
            block["bbox"][0],
        ),
    )

    if renumber:
        for index, block in enumerate(sorted_blocks, start=1):
            block["reading_order"] = index

    return sorted_blocks


def draw_layout_boxes(image: np.ndarray, blocks: list[dict[str, Any] | LayoutBlock]) -> np.ndarray:
    canvas = image.copy()

    for block in blocks:
        block_dict = block.model_dump() if isinstance(block, LayoutBlock) else block
        color = TYPE_COLORS.get(block_dict["type"], (32, 114, 229))
        x1, y1, x2, y2 = block_dict["bbox"]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{block_dict['reading_order']}. {block_dict['type']}"
        anchor_y = max(22, y1 - 8)
        cv2.rectangle(
            canvas,
            (x1, anchor_y - 18),
            (min(canvas.shape[1] - 1, x1 + max(90, len(label) * 8)), anchor_y + 4),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 4, anchor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


def save_results(blocks: list[dict[str, Any] | LayoutBlock], path: str | Path) -> None:
    serializable = [
        block.model_dump() if isinstance(block, LayoutBlock) else block for block in blocks
    ]
    Path(path).write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_results_path(analysis_id: str) -> Path:
    result_dir = Path(tempfile.gettempdir()) / "surya-layout-results"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / f"{analysis_id}.json"


def analyze_page_layout(
    image: np.ndarray | str | Path,
) -> LayoutAnalysisResponse:
    raw_result = analyze_layout(image)
    normalized_blocks = normalize_blocks(raw_result)

    if not normalized_blocks:
        raise LayoutAnalysisError("Surya не вернул блоков макета для этой страницы.")

    page_shape = raw_result["image_rgb"].shape[:2]
    processed_blocks = postprocess_blocks(normalized_blocks, page_shape)
    if not processed_blocks:
        raise LayoutAnalysisError(
            "После постобработки не осталось валидных блоков макета."
        )
    ordered_blocks = sort_reading_order(processed_blocks)
    block_models = [LayoutBlock(**block) for block in ordered_blocks]

    visualization = draw_layout_boxes(raw_result["image_rgb"], block_models)
    analysis_id = uuid4().hex
    result_json_path = build_results_path(analysis_id)
    visualization_data_url = image_to_data_url(visualization)
    save_results(block_models, result_json_path)
    _cache_analysis(
        analysis_id,
        AnalysisArtifacts(blocks=block_models, visualization_png=image_to_png_bytes(visualization)),
    )

    visualization_url = f"/visualization?analysis_id={analysis_id}"
    logger.info("Completed layout analysis with %s blocks", len(block_models))
    return LayoutAnalysisResponse(
        analysis_id=analysis_id,
        visualization_url=visualization_url,
        visualization_data_url=visualization_data_url,
        result_json_path=str(result_json_path),
        blocks=block_models,
    )
