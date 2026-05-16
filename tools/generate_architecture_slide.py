from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "architecture"
PNG_PATH = OUT_DIR / "high_level_architecture_slide.png"

W = 1920
H = 1080

BG = "#FFFFFF"
TEXT = "#111827"
MUTED = "#667085"
LINE = "#5A78A5"
LINE_SOFT = "#D8E2F1"
BOX_FILL = "#FFFFFF"
BOX_TITLE_FILL = "#F4F7FC"
BOX_BORDER = "#6A84B1"
INNER_FILL = "#FCFDFF"
INPUT_FILL = "#FBFCFE"
OUTPUT_FILL = "#FCFDFF"
DASH_FILL = "#FBFDFF"
DASH_BORDER = "#90A4C6"


@lru_cache(maxsize=None)
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


FONT_TITLE = load_font(54, bold=True)
FONT_META = load_font(15)


def measure(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
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
            width, _ = measure(draw, candidate, font)
            if width <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def fit_wrapped_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    max_width: int,
    max_height: int,
    base_size: int,
    bold: bool = False,
    min_size: int = 12,
    gap: int = 4,
) -> tuple[ImageFont.FreeTypeFont, list[str], int]:
    for size in range(base_size, min_size - 1, -1):
        font = load_font(size, bold=bold)
        lines = wrap_text(draw, text, font, max_width)
        heights = [measure(draw, line or "Ag", font)[1] for line in lines]
        total_h = sum(heights) + max(0, len(lines) - 1) * gap
        if total_h <= max_height:
            return font, lines, total_h

    font = load_font(min_size, bold=bold)
    lines = wrap_text(draw, text, font, max_width)
    heights = [measure(draw, line or "Ag", font)[1] for line in lines]
    total_h = sum(heights) + max(0, len(lines) - 1) * gap
    return font, lines, total_h


def draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    lines: list[tuple[str, object, str]],
    *,
    gap: int = 8,
) -> None:
    heights: list[int] = []
    for text, font, _fill in lines:
        _, h = measure(draw, text, font)
        heights.append(h)
    total_h = sum(heights) + max(0, len(lines) - 1) * gap
    y = box[1] + ((box[3] - box[1]) - total_h) / 2
    for (text, font, fill), h in zip(lines, heights):
        w, _ = measure(draw, text, font)
        x = box[0] + ((box[2] - box[0]) - w) / 2
        draw.text((x, y), text, font=font, fill=fill)
        y += h + gap


def draw_fitted_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    fill: str,
    base_size: int,
    bold: bool = False,
    min_size: int = 12,
    gap: int = 4,
) -> None:
    font, lines, _ = fit_wrapped_lines(
        draw,
        text,
        max_width=max(20, box[2] - box[0]),
        max_height=max(20, box[3] - box[1]),
        base_size=base_size,
        bold=bold,
        min_size=min_size,
        gap=gap,
    )
    draw_centered_lines(
        draw,
        box,
        [(line, font, fill) for line in lines],
        gap=gap,
    )


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str,
    radius: int = 18,
    width: int = 3,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_column(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    steps: list[str],
) -> None:
    x1, y1, x2, y2 = box
    rounded_box(draw, box, fill=BOX_FILL, outline=BOX_BORDER, radius=18, width=3)

    title_h = 108
    title_box = (x1, y1, x2, y1 + title_h)
    rounded_box(draw, title_box, fill=BOX_TITLE_FILL, outline=BOX_BORDER, radius=18, width=2)

    draw_fitted_centered_text(
        draw,
        (x1 + 18, y1 + 12, x2 - 18, y1 + 60),
        title,
        fill=TEXT,
        base_size=22,
        bold=True,
        min_size=17,
        gap=3,
    )
    draw_fitted_centered_text(
        draw,
        (x1 + 18, y1 + 60, x2 - 18, y1 + 98),
        subtitle,
        fill=MUTED,
        base_size=17,
        bold=False,
        min_size=13,
        gap=2,
    )

    inner_x1 = x1 + 18
    inner_x2 = x2 - 18
    current_y = y1 + title_h + 18
    inner_h = 102
    gap = 16
    for step in steps:
        inner_box = (inner_x1, current_y, inner_x2, current_y + inner_h)
        rounded_box(draw, inner_box, fill=INNER_FILL, outline=BOX_BORDER, radius=14, width=2)
        draw_fitted_centered_text(
            draw,
            (inner_x1 + 12, current_y + 10, inner_x2 - 12, current_y + inner_h - 10),
            step,
            fill=TEXT,
            base_size=18,
            bold=False,
            min_size=14,
            gap=4,
        )
        current_y += inner_h + gap

        if current_y + inner_h <= y2 - 8:
            cx = (x1 + x2) // 2
            draw.line((cx, current_y - gap + 2, cx, current_y - 2), fill=LINE, width=3)
            draw.polygon(
                [(cx, current_y - 2), (cx - 7, current_y - 14), (cx + 7, current_y - 14)],
                fill=LINE,
            )


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], *, color: str = LINE, width: int = 3) -> None:
    draw.line((start[0], start[1], end[0], end[1]), fill=color, width=width)
    if abs(end[0] - start[0]) >= abs(end[1] - start[1]):
        sign = 1 if end[0] >= start[0] else -1
        draw.polygon(
            [(end[0], end[1]), (end[0] - sign * 16, end[1] - 9), (end[0] - sign * 16, end[1] + 9)],
            fill=color,
        )
    else:
        sign = 1 if end[1] >= start[1] else -1
        draw.polygon(
            [(end[0], end[1]), (end[0] - 9, end[1] - sign * 16), (end[0] + 9, end[1] - sign * 16)],
            fill=color,
        )


def draw_dashed_rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str,
    radius: int = 16,
    dash: int = 12,
) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=None)

    def dash_h(y: int, left: int, right: int) -> None:
        x = left
        while x < right:
            draw.line((x, y, min(x + dash, right), y), fill=outline, width=3)
            x += dash * 2

    def dash_v(x: int, top: int, bottom: int) -> None:
        y = top
        while y < bottom:
            draw.line((x, y, x, min(y + dash, bottom)), fill=outline, width=3)
            y += dash * 2

    dash_h(y1, x1 + radius, x2 - radius)
    dash_h(y2, x1 + radius, x2 - radius)
    dash_v(x1, y1 + radius, y2 - radius)
    dash_v(x2, y1 + radius, y2 - radius)


def draw_dashed_polyline(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], *, color: str = DASH_BORDER, width: int = 3, dash: int = 12) -> None:
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if y1 == y2:
            x = min(x1, x2)
            end = max(x1, x2)
            while x < end:
                x_end = min(x + dash, end)
                draw.line((x, y1, x_end, y2), fill=color, width=width)
                x += dash * 2
        elif x1 == x2:
            y = min(y1, y2)
            end = max(y1, y2)
            while y < end:
                y_end = min(y + dash, end)
                draw.line((x1, y, x2, y_end), fill=color, width=width)
                y += dash * 2
        else:
            draw.line((x1, y1, x2, y2), fill=color, width=width)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)

    draw.text((80, 56), "Верхнеуровневая архитектура программного решения", font=FONT_TITLE, fill=TEXT)
    draw.line((80, 148, 1840, 148), fill=LINE_SOFT, width=2)

    input_box = (70, 320, 245, 730)

    rounded_box(draw, input_box, fill=INPUT_FILL, outline=BOX_BORDER, radius=18, width=3)
    draw_fitted_centered_text(
        draw,
        (input_box[0] + 12, input_box[1] + 16, input_box[2] - 12, input_box[3] - 16),
        "Вход:\nPDF-документ",
        fill=TEXT,
        base_size=19,
        bold=True,
        min_size=15,
        gap=10,
    )
    draw_fitted_centered_text(
        draw,
        (input_box[0] + 18, input_box[1] + 150, input_box[2] - 18, input_box[3] - 60),
        "(born-digital / scan)",
        fill=MUTED,
        base_size=17,
        bold=False,
        min_size=14,
        gap=2,
    )

    col1 = (300, 210, 575, 845)
    col2 = (612, 210, 887, 845)
    col3 = (924, 210, 1199, 845)
    col4 = (1236, 210, 1511, 845)
    pagecontent_box = (650, 890, 850, 970)
    out1 = (1610, 300, 1815, 380)
    out2 = (1610, 430, 1815, 510)
    out3 = (1610, 560, 1815, 640)
    out4 = (1610, 690, 1815, 790)
    review_box = (860, 915, 1290, 1000)

    draw_column(
        draw,
        col1,
        "Подготовка и оркестрация",
        "FastAPI, PyMuPDF",
        [
            "Загрузка PDF",
            "Извлечение текста\nи оценка достаточности",
            "Рендер страниц\nи OCR-подготовка",
            "Кэш артефактов\nи контроль этапов",
        ],
    )
    draw_column(
        draw,
        col2,
        "Layout-анализ, OCR\nи формулы",
        "Surya, PaddleOCR, pix2tex",
        [
            "Layout-анализ страницы",
            "Классификация блоков\nи reading order",
            "OCR текстовых блоков",
            "Распознавание формул в LaTeX\nи сборка PageContent",
        ],
    )
    draw_column(
        draw,
        col3,
        "Лингвистическая\nобработка",
        "NLP и нормализация",
        [
            "Постобработка OCR-текста",
            "Распознавание\nдореформенной орфографии",
            "Перевод в современную\nорфографию",
            "Исторический /\nнормализованный текст\nи needs_review",
        ],
    )
    draw_column(
        draw,
        col4,
        "Документная логика\nи экспорт",
        "Статьи, LaTeX, PDF, архив",
        [
            "Сегментация на статьи",
            "Извлечение метаданных",
            "Сборка LaTeX\nстраницы и статьи",
            "Экспорт читаемого PDF\nи архива коллекции",
        ],
    )

    rounded_box(draw, pagecontent_box, fill=OUTPUT_FILL, outline=BOX_BORDER, radius=14, width=3)
    draw_fitted_centered_text(
        draw,
        (pagecontent_box[0] + 12, pagecontent_box[1] + 8, pagecontent_box[2] - 12, pagecontent_box[3] - 8),
        "PageContent",
        fill=TEXT,
        base_size=18,
        bold=True,
        min_size=15,
        gap=2,
    )

    for box, label in [
        (out1, "LaTeX статьи"),
        (out2, "Читаемый PDF"),
        (out3, "Архив коллекции"),
        (out4, "Метаданные статьи"),
    ]:
        rounded_box(draw, box, fill=OUTPUT_FILL, outline=BOX_BORDER, radius=14, width=3)
        draw_fitted_centered_text(
            draw,
            (box[0] + 12, box[1] + 8, box[2] - 12, box[3] - 8),
            label,
            fill=TEXT,
            base_size=18,
            bold=True,
            min_size=14,
            gap=3,
        )

    draw_arrow(draw, (245, 525), (300, 525))
    draw_arrow(draw, (575, 525), (612, 525))
    draw_arrow(draw, (887, 525), (924, 525))
    draw_arrow(draw, (1199, 525), (1236, 525))

    draw_arrow(draw, ((col2[0] + col2[2]) // 2, col2[3]), ((pagecontent_box[0] + pagecontent_box[2]) // 2, pagecontent_box[1]))
    draw_arrow(draw, (1511, 340), (1610, 340))
    draw_arrow(draw, (1511, 470), (1610, 470))
    draw_arrow(draw, (1511, 600), (1610, 600))
    draw_arrow(draw, (1511, 730), (1610, 730))

    draw_dashed_rounded_box(draw, review_box, fill=DASH_FILL, outline=DASH_BORDER, radius=16, dash=12)
    draw_fitted_centered_text(
        draw,
        (review_box[0] + 16, review_box[1] + 10, review_box[2] - 16, review_box[3] - 10),
        "Ручная проверка сомнительных фрагментов",
        fill=MUTED,
        base_size=18,
        bold=True,
        min_size=14,
        gap=3,
    )
    draw_dashed_polyline(
        draw,
        [(1060, 845), (1060, 905)],
    )
    draw_arrow(draw, (1060, 905), (1060, 915), color=DASH_BORDER, width=3)
    draw_dashed_polyline(
        draw,
        [(1375, 845), (1375, 935), (1290, 935)],
    )
    draw_arrow(draw, (1290, 935), (1280, 935), color=DASH_BORDER, width=3)

    footer_y = 1020
    draw.text((110, footer_y), "Вафина А.И.", font=FONT_META, fill=MUTED)
    footer_text = (
        "Разработка системы автоматизации процесса формирования электронной коллекции научных\n"
        "ретродокументов"
    )
    draw_centered_lines(draw, (430, 1004, 1500, 1068), [(line, FONT_META, MUTED) for line in footer_text.split("\n")], gap=2)
    draw.text((1800, footer_y), "7/19", font=FONT_META, fill=MUTED)

    image.save(PNG_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
