from __future__ import annotations

import base64
import json
import os
import sys
import textwrap
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from app.core.config import settings
from app.services.layout_analysis_surya import analyze_page_layout
from app.services.text_block_processor import build_routed_blocks_from_layout, process_text_blocks

ARTIFACTS_DIR = ROOT / "artifacts" / "section_4_3"
LAYOUT_DIR = ROOT / "layout_artifacts"


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(40, bold=True)
FONT_SUBTITLE = load_font(26, bold=True)
FONT_BODY = load_font(22)
FONT_SMALL = load_font(20)
FONT_TINY = load_font(18)


def review_reason(block: dict) -> str:
    reasons: list[str] = []
    content = (block.get("content") or "").strip()
    confidence = block.get("confidence")
    backend = block.get("ocr_backend")

    if not content:
        reasons.append("пустой текст")
    if confidence is None or float(confidence) < settings.text_block_review_confidence:
        reasons.append(f"confidence < {settings.text_block_review_confidence:.2f}")
    if len(content) < settings.text_block_min_length:
        reasons.append(f"длина < {settings.text_block_min_length}")
    if backend != "paddleocr":
        reasons.append("резервный backend")

    return "; ".join(reasons) if reasons else "проверка не требуется"


def decode_data_url(data_url: str | None) -> Image.Image:
    if not data_url:
        return Image.new("RGB", (420, 100), "white")
    _, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def wrap_text(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [text]


def truncate_text(text: str, limit: int) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def draw_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font, fill: str) -> None:
    lines = text.split("\n")
    line_heights: list[int] = []
    line_widths: list[int] = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    total_height = sum(line_heights) + max(0, len(lines) - 1) * 8
    y = box[1] + ((box[3] - box[1]) - total_height) / 2
    for line, width, height in zip(lines, line_widths, line_heights):
        x = box[0] + ((box[2] - box[0]) - width) / 2
        draw.text((x, y), line, font=font, fill=fill)
        y += height + 8


def build_page_artifact(image_name: str, page_number: int) -> dict:
    image_path = LAYOUT_DIR / image_name
    layout = analyze_page_layout(image_path)
    routed_blocks = build_routed_blocks_from_layout(
        layout.blocks,
        block_id_prefix=image_name.replace(".png", ""),
    )
    page_content = process_text_blocks(image_path, routed_blocks, page_number=page_number)

    page_dir = ARTIFACTS_DIR / image_name.replace(".png", "")
    page_dir.mkdir(parents=True, exist_ok=True)

    layout_json_path = page_dir / "layout_analysis.json"
    text_json_path = page_dir / "page_content.json"
    layout_json_path.write_text(
        json.dumps(layout.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    text_json_path.write_text(
        json.dumps(page_content.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    text_blocks = [block for block in page_content.model_dump(mode="json")["blocks"] if block["route_to"] == "text_pipeline"]
    average_confidence = round(
        sum(float(block.get("confidence") or 0.0) for block in text_blocks) / max(1, len(text_blocks)),
        3,
    )
    for block in text_blocks:
        block["review_reason"] = review_reason(block)

    return {
        "image_name": image_name,
        "page_number": page_number,
        "layout_json_path": str(layout_json_path),
        "page_content_json_path": str(text_json_path),
        "layout_block_count": len(layout.blocks),
        "text_block_count": len(text_blocks),
        "needs_review_count": page_content.needs_review_count,
        "average_confidence": average_confidence,
        "blocks": text_blocks,
    }


def create_pipeline_scheme(output_path: Path) -> None:
    canvas = Image.new("RGB", (1900, 980), "white")
    draw = ImageDraw.Draw(canvas)

    title = "Последовательность работы модуля OCR текстовых блоков"
    draw.text((120, 48), title, font=FONT_TITLE, fill="#233044")
    draw.text(
        (120, 106),
        "Ключевые этапы после layout-анализа и перед сборкой PageContent",
        font=FONT_SMALL,
        fill="#5B6B80",
    )

    boxes = [
        ("Вход:\npage_image + routed_blocks", (120, 190, 470, 318), "#EAF4FF", "#4A90E2"),
        ("sort_blocks:\nreading_order, y, x, block_id", (520, 190, 880, 318), "#F1EEFF", "#8E6BFF"),
        ("crop_block:\nвырезание text_pipeline-областей", (930, 190, 1310, 318), "#EFFCF4", "#30A46C"),
        ("PaddleOCR:\npredict() -> ocr()", (1360, 190, 1730, 318), "#FFF4E8", "#E58A1F"),
        ("fallback_text_result:\nseed_text + seed_confidence", (1360, 382, 1730, 510), "#FFF0F0", "#D9485F"),
        ("postprocess_text:\nочистка OCR-артефактов", (930, 574, 1310, 702), "#FFF7E6", "#D97706"),
        ("should_mark_for_review:\nпорог confidence, длина, backend", (520, 574, 880, 702), "#EEF7FF", "#3182CE"),
        ("build_page_content:\npage_text, readable_text, blocks", (120, 574, 470, 702), "#F4F7FB", "#64748B"),
        ("save_results + cache:\n/tmp/text-block-results + memory cache", (520, 790, 980, 900), "#F8FAFC", "#4B5563"),
    ]

    for text, (x1, y1, x2, y2), fill, outline in boxes:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=24, fill=fill, outline=outline, width=4)
        draw_centered(draw, (x1 + 18, y1 + 12, x2 - 18, y2 - 12), text, FONT_BODY, "#1F2937")

    def arrow(points: list[tuple[int, int]], color: str = "#566579") -> None:
        for start, end in zip(points, points[1:]):
            draw.line((start[0], start[1], end[0], end[1]), fill=color, width=5)
        x2, y2 = points[-1]
        draw.polygon([(x2, y2), (x2 - 18, y2 - 10), (x2 - 18, y2 + 10)], fill=color)

    arrow([(470, 254), (520, 254)])
    arrow([(880, 254), (930, 254)])
    arrow([(1310, 254), (1360, 254)])
    arrow([(1545, 318), (1545, 382)])
    arrow([(1360, 446), (1310, 446), (1310, 638)])
    arrow([(930, 638), (880, 638)])
    arrow([(520, 638), (470, 638)])
    arrow([(300, 702), (300, 845), (520, 845)])

    legend_y = 820
    legend_items = [
        ("Основной OCR-путь", "#E58A1F"),
        ("Резервный сценарий при ошибке OCR", "#D9485F"),
        ("Постобработка и правила контроля", "#3182CE"),
    ]
    x = 1150
    for label, color in legend_items:
        draw.rounded_rectangle((x, legend_y, x + 18, legend_y + 18), radius=4, fill=color)
        draw.text((x + 30, legend_y - 1), label, font=FONT_TINY, fill="#334155")
        legend_y += 34

    draw.text(
        (120, 930),
        "Нетекстовые блоки (formula/table/image) не проходят повторный OCR и сохраняются в общем ordered-списке с исходным route_to.",
        font=FONT_TINY,
        fill="#475569",
    )
    canvas.save(output_path)


def create_crop_examples(output_path: Path, page_artifact: dict) -> None:
    blocks = [block for block in page_artifact["blocks"] if (block.get("content") or "").strip()][:4]
    canvas = Image.new("RGB", (1900, 1500), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((90, 44), "Примеры обработанных текстовых блоков", font=FONT_TITLE, fill="#233044")
    subtitle = (
        "Демонстрационный прогон по странице tome1_p02_orig.png: "
        f"{page_artifact['text_block_count']} routed-блоков, "
        f"средний confidence {page_artifact['average_confidence']:.3f}, "
        "backend = surya_seed"
    )
    draw.text((90, 102), subtitle, font=FONT_SMALL, fill="#5B6B80")

    card_w = 810
    card_h = 580
    lefts = [90, 1000]
    tops = [170, 790]

    for index, block in enumerate(blocks):
        card_x = lefts[index % 2]
        card_y = tops[index // 2]
        draw.rounded_rectangle(
            (card_x, card_y, card_x + card_w, card_y + card_h),
            radius=28,
            fill="#F8FAFC",
            outline="#CBD5E1",
            width=3,
        )

        badge_fill = "#DCEEFF" if not block["needs_review"] else "#FFE3E3"
        badge_outline = "#3182CE" if not block["needs_review"] else "#D9485F"
        draw.rounded_rectangle(
            (card_x + 24, card_y + 22, card_x + 198, card_y + 64),
            radius=16,
            fill=badge_fill,
            outline=badge_outline,
            width=2,
        )
        badge_text = f"Блок {block['reading_order']} | {block['type']}"
        draw.text((card_x + 40, card_y + 31), badge_text, font=FONT_TINY, fill="#1E293B")

        crop = decode_data_url(block.get("crop_data_url"))
        crop.thumbnail((760, 190))
        crop_x = card_x + 25 + (760 - crop.width) // 2
        crop_y = card_y + 88 + (190 - crop.height) // 2
        canvas.paste(crop, (crop_x, crop_y))
        draw.rectangle(
            (card_x + 25, card_y + 88, card_x + 785, card_y + 278),
            outline="#CBD5E1",
            width=2,
        )

        meta_lines = [
            f"route_to = {block['route_to']}, ocr_backend = {block['ocr_backend']}",
            f"confidence = {float(block.get('confidence') or 0.0):.3f}, needs_review = {block['needs_review']}",
        ]
        text_y = card_y + 300
        for line in meta_lines:
            draw.text((card_x + 30, text_y), line, font=FONT_TINY, fill="#475569")
            text_y += 30

        draw.text((card_x + 30, text_y + 8), "Собранный текст блока:", font=FONT_TINY, fill="#1F2937")
        text_y += 44
        preview_text = truncate_text(block.get("content") or "∅", 120)
        for line in wrap_text(preview_text, 54)[:3]:
            draw.text((card_x + 30, text_y), line, font=FONT_BODY, fill="#111827")
            text_y += 33

        text_y += 12
        draw.text((card_x + 30, text_y), "Причина проверки:", font=FONT_TINY, fill="#1F2937")
        text_y += 34
        for line in wrap_text(block.get("review_reason") or "", 60)[:2]:
            draw.text((card_x + 30, text_y), line, font=FONT_SMALL, fill="#B42318")
            text_y += 28

    footer = (
        "Примечание: в текущем окружении Paddle backend отсутствует, поэтому модуль сохранил seed_text из layout-этапа "
        "и принудительно пометил блоки для ручной проверки."
    )
    draw.text((90, 1422), footer, font=FONT_TINY, fill="#475569")
    canvas.save(output_path)


def build_manifest(page_one: dict, page_two: dict) -> dict:
    page_one_blocks = page_one["blocks"]
    page_two_blocks = page_two["blocks"]

    selected_rows = []
    examples = [
        page_one_blocks[0],
        page_one_blocks[1],
        page_two_blocks[0],
        page_two_blocks[1],
        page_two_blocks[2],
        page_two_blocks[3],
    ]
    for block in examples:
        selected_rows.append(
            {
                "block_id": block["block_id"],
                "content": truncate_text((block.get("content") or "∅").strip() or "∅", 170),
                "confidence": round(float(block.get("confidence") or 0.0), 3),
                "backend": block.get("ocr_backend") or "none",
                "review_reason": block.get("review_reason") or "",
            }
        )

    return {
        "pages": {
            "tome1_p01_orig": page_one,
            "tome1_p02_orig": page_two,
        },
        "selected_table_rows": selected_rows,
        "review_threshold": settings.text_block_review_confidence,
        "min_length": settings.text_block_min_length,
        "formula_placeholder": settings.text_block_formula_placeholder,
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    page_one = build_page_artifact("tome1_p01_orig.png", page_number=1)
    page_two = build_page_artifact("tome1_p02_orig.png", page_number=2)

    pipeline_path = ARTIFACTS_DIR / "ocr_pipeline_scheme.png"
    examples_path = ARTIFACTS_DIR / "ocr_block_examples.png"
    manifest_path = ARTIFACTS_DIR / "manifest.json"

    create_pipeline_scheme(pipeline_path)
    create_crop_examples(examples_path, page_two)

    manifest = build_manifest(page_one, page_two)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Artifacts created in {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
