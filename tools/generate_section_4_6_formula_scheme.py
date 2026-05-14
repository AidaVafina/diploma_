from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "section_4_6"
PNG_PATH = OUT_DIR / "formula_module_block_scheme.png"

W = 1920
H = 1080
BG = (255, 255, 255, 255)

TEXT = "#1E293B"
MUTED = "#5B657A"
ARROW = "#64748B"

MAIN_FILL = "#F4F8FF"
MAIN_BORDER = "#2563EB"
DECISION_FILL = "#FFF7E8"
DECISION_BORDER = "#D97706"
SUCCESS_FILL = "#EEFDF4"
SUCCESS_BORDER = "#16A34A"
FALLBACK_FILL = "#FFF1F2"
FALLBACK_BORDER = "#E11D48"
SYSTEM_FILL = "#F8FAFC"
SYSTEM_BORDER = "#64748B"
NOTE_FILL = "#F7F7FF"
NOTE_BORDER = "#7C3AED"


def load_font(size: int, *, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(48, bold=True)
FONT_SUBTITLE = load_font(22)
FONT_BOX_TITLE = load_font(23, bold=True)
FONT_BOX_BODY = load_font(18)
FONT_DECISION = load_font(22, bold=True)
FONT_NOTE = load_font(18)


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            width, _ = text_size(draw, candidate, font)
            if width <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_multiline_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    body: str,
    *,
    title_font,
    body_font,
    fill: str = TEXT,
) -> None:
    title_lines = wrap_text(draw, title, title_font, box[2] - box[0] - 48)
    body_lines = wrap_text(draw, body, body_font, box[2] - box[0] - 48) if body else []

    title_h = text_size(draw, "Ag", title_font)[1]
    body_h = text_size(draw, "Ag", body_font)[1]
    gap = 8

    total_height = 0
    if title_lines:
        total_height += len(title_lines) * title_h + max(0, len(title_lines) - 1) * gap
    if body_lines:
        total_height += 16
        total_height += len(body_lines) * body_h + max(0, len(body_lines) - 1) * gap

    y = box[1] + ((box[3] - box[1]) - total_height) / 2

    for line in title_lines:
        width, _ = text_size(draw, line, title_font)
        x = box[0] + ((box[2] - box[0]) - width) / 2
        draw.text((x, y), line, font=title_font, fill=fill)
        y += title_h + gap

    if title_lines and body_lines:
        y += 8

    for line in body_lines:
        width, _ = text_size(draw, line, body_font)
        x = box[0] + ((box[2] - box[0]) - width) / 2
        draw.text((x, y), line, font=body_font, fill=fill)
        y += body_h + gap


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    title: str,
    body: str = "",
    fill: str,
    outline: str,
    radius: int = 28,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=5)
    draw_multiline_centered(
        draw,
        (box[0] + 14, box[1] + 12, box[2] - 14, box[3] - 12),
        title,
        body,
        title_font=FONT_BOX_TITLE,
        body_font=FONT_BOX_BODY,
    )


def draw_terminator(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    text: str,
    fill: str,
    outline: str,
) -> None:
    radius = (box[3] - box[1]) // 2
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=6)
    draw_multiline_centered(
        draw,
        box,
        text,
        "",
        title_font=FONT_BOX_TITLE,
        body_font=FONT_BOX_BODY,
    )


def draw_decision(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    size: tuple[int, int],
    *,
    text: str,
    fill: str,
    outline: str,
) -> None:
    cx, cy = center
    w, h = size
    points = [
        (cx, cy - h // 2),
        (cx + w // 2, cy),
        (cx, cy + h // 2),
        (cx - w // 2, cy),
    ]
    draw.polygon(points, fill=fill, outline=outline)
    draw.line(points + [points[0]], fill=outline, width=5)

    lines = wrap_text(draw, text, FONT_DECISION, w - 54)
    line_h = text_size(draw, "Ag", FONT_DECISION)[1]
    total_h = len(lines) * line_h + max(0, len(lines) - 1) * 8
    y = cy - total_h / 2
    for line in lines:
        width, _ = text_size(draw, line, FONT_DECISION)
        x = cx - width / 2
        draw.text((x, y), line, font=FONT_DECISION, fill=TEXT)
        y += line_h + 8


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    color: str = ARROW,
    width: int = 5,
) -> None:
    for start, end in zip(points, points[1:]):
        draw.line((start[0], start[1], end[0], end[1]), fill=color, width=width)

    x1, y1 = points[-2]
    x2, y2 = points[-1]
    size = 18
    spread = 11
    if abs(x2 - x1) >= abs(y2 - y1):
        sign = 1 if x2 >= x1 else -1
        head = [(x2, y2), (x2 - sign * size, y2 - spread), (x2 - sign * size, y2 + spread)]
    else:
        sign = 1 if y2 >= y1 else -1
        head = [(x2, y2), (x2 - spread, y2 - sign * size), (x2 + spread, y2 - sign * size)]
    draw.polygon(head, fill=color)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *, fill: str = MUTED) -> None:
    draw.text(xy, text, font=FONT_NOTE, fill=fill)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(image)

    draw.text(
        (90, 54),
        "Блок-схема модуля извлечения математических формул",
        font=FONT_TITLE,
        fill=TEXT,
    )
    draw.text(
        (90, 112),
        "Сокращённая версия для слайда: только ключевые шаги и fallback-ветка Surya / pix2tex.",
        font=FONT_SUBTITLE,
        fill=MUTED,
    )
    draw.line((90, 154, 1830, 154), fill="#D7DFEA", width=2)

    draw_box(
        draw,
        (1500, 175, 1830, 248),
        title="Контекст",
        body="После OCR текста в PDF-конвейере.",
        fill=NOTE_FILL,
        outline=NOTE_BORDER,
        radius=18,
    )

    draw_box(
        draw,
        (90, 235, 350, 360),
        title="1. Formula-блоки",
        body="sort + select\nformula_pipeline",
        fill=MAIN_FILL,
        outline=MAIN_BORDER,
    )
    draw_box(
        draw,
        (415, 235, 675, 360),
        title="2. Crop + cache",
        body="padding, PNG crop,\n/tmp cache",
        fill=MAIN_FILL,
        outline=MAIN_BORDER,
    )
    draw_decision(
        draw,
        (810, 298),
        (210, 140),
        text="Есть\nseed_latex?",
        fill=DECISION_FILL,
        outline=DECISION_BORDER,
    )
    draw_box(
        draw,
        (955, 185, 1265, 325),
        title="3a. Готовый LaTeX",
        body="normalize + confidence\nbackend = surya",
        fill=SUCCESS_FILL,
        outline=SUCCESS_BORDER,
    )
    draw_box(
        draw,
        (955, 405, 1265, 545),
        title="3b. Surya OCR",
        body="recognition_predictor\n+ confidence",
        fill=MAIN_FILL,
        outline=MAIN_BORDER,
    )
    draw_decision(
        draw,
        (1420, 475),
        (240, 150),
        text="latex и\nconfidence OK?",
        fill=DECISION_FILL,
        outline=DECISION_BORDER,
    )
    draw_box(
        draw,
        (1590, 390, 1830, 560),
        title="4. pix2tex",
        body="fallback;\nесли пусто -> review",
        fill=FALLBACK_FILL,
        outline=FALLBACK_BORDER,
    )
    draw_box(
        draw,
        (1360, 700, 1775, 845),
        title="5. FormulaBlockResult",
        body="latex, backend,\ncrop_path, needs_review",
        fill=SUCCESS_FILL,
        outline=SUCCESS_BORDER,
    )
    draw_box(
        draw,
        (875, 700, 1285, 845),
        title="6. Обновить PageContent",
        body="merge по block_id\n+ save_results",
        fill=MAIN_FILL,
        outline=MAIN_BORDER,
    )
    draw_box(
        draw,
        (390, 700, 800, 845),
        title="7. Выход",
        body="PageContent + JSON\n+ cache_page_content",
        fill=SYSTEM_FILL,
        outline=SYSTEM_BORDER,
    )

    draw_arrow(draw, [(350, 298), (415, 298)])
    draw_arrow(draw, [(675, 298), (705, 298)])
    draw_arrow(draw, [(915, 298), (955, 298), (955, 255)])
    draw_arrow(draw, [(810, 368), (810, 475), (955, 475)])
    draw_arrow(draw, [(1265, 475), (1300, 475)])
    draw_arrow(draw, [(1540, 475), (1590, 475)])
    draw_arrow(draw, [(1420, 550), (1420, 625), (1567, 625), (1567, 700)])
    draw_arrow(draw, [(1710, 560), (1710, 625), (1567, 625), (1567, 700)])
    draw_arrow(draw, [(1110, 325), (1110, 625), (1567, 625), (1567, 700)])
    draw_arrow(draw, [(1360, 772), (1285, 772)])
    draw_arrow(draw, [(875, 772), (800, 772)])

    draw_label(draw, (928, 248), "да")
    draw_label(draw, (835, 390), "нет")
    draw_label(draw, (1450, 585), "да")
    draw_label(draw, (1598, 442), "нет / слабый confidence")

    footer = (
        "Если Surya не проходит порог FORMULA_BLOCK_SURYA_CONFIDENCE_THRESHOLD "
        "или pix2tex возвращает пустой результат, блок помечается needs_review."
    )
    draw.text((90, 980), footer, font=FONT_NOTE, fill=MUTED)

    image.save(PNG_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
