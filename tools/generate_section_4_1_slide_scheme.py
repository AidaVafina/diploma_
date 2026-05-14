from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "section_4_1"
PNG_PATH = OUT_DIR / "scan_quality_module_slide.png"

WIDTH = 1920
HEIGHT = 1080

BG = "#FFFFFF"
TEXT = "#1F1F1F"
MUTED = "#6B7280"
ACCENT = "#1495E4"
ACCENT_DARK = "#1976D2"
ACCENT_SOFT = "#EAF5FF"
GREEN = "#EAF8F0"
GREEN_BORDER = "#2F9E64"
ORANGE = "#FFF4E8"
ORANGE_BORDER = "#D97706"
PURPLE = "#F3EEFF"
PURPLE_BORDER = "#8B5CF6"
GRAY_FILL = "#F8FAFC"
GRAY_BORDER = "#BFC8D6"
RED_SOFT = "#FFF1F1"
RED_BORDER = "#D9485F"
LINE = "#CBD5E1"


def load_font(size: int, *, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(56, bold=True)
FONT_SUBTITLE = load_font(26)
FONT_SECTION = load_font(24, bold=True)
FONT_BOX = load_font(23, bold=True)
FONT_BOX_SMALL = load_font(20)
FONT_NOTE = load_font(20)
FONT_META = load_font(18)


def draw_wrapped_center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, *, title_font, body_font, fill: str) -> None:
    lines = text.split("\n")
    rendered = []
    for idx, line in enumerate(lines):
        font = title_font if idx == 0 else body_font
        bbox = draw.textbbox((0, 0), line, font=font)
        rendered.append((line, font, bbox[2] - bbox[0], bbox[3] - bbox[1]))

    total_height = sum(item[3] for item in rendered) + max(0, len(rendered) - 1) * 8
    y = box[1] + ((box[3] - box[1]) - total_height) / 2
    for line, font, width, height in rendered:
        x = box[0] + ((box[2] - box[0]) - width) / 2
        draw.text((x, y), line, font=font, fill=fill)
        y += height + 8


def rounded_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], *, fill: str, outline: str, radius: int = 24, width: int = 4) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], *, color: str = ACCENT_DARK, width: int = 5) -> None:
    draw.line((start[0], start[1], end[0], end[1]), fill=color, width=width)
    if end[0] >= start[0]:
        draw.polygon(
            [
                (end[0], end[1]),
                (end[0] - 18, end[1] - 10),
                (end[0] - 18, end[1] + 10),
            ],
            fill=color,
        )
    else:
        draw.polygon(
            [
                (end[0], end[1]),
                (end[0] + 18, end[1] - 10),
                (end[0] + 18, end[1] + 10),
            ],
            fill=color,
        )


def dashed_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], *, fill: str, outline: str, radius: int = 18, dash: int = 14) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=None)
    x1, y1, x2, y2 = box
    points = [
        ((x1 + radius, y1), (x2 - radius, y1)),
        ((x2, y1 + radius), (x2, y2 - radius)),
        ((x2 - radius, y2), (x1 + radius, y2)),
        ((x1, y2 - radius), (x1, y1 + radius)),
    ]
    for (sx, sy), (ex, ey) in points:
        if sy == ey:
            step = dash * 2
            left, right = sorted((sx, ex))
            x = left
            while x < right:
                draw.line((x, sy, min(x + dash, right), ey), fill=outline, width=3)
                x += step
        else:
            step = dash * 2
            top, bottom = sorted((sy, ey))
            y = top
            while y < bottom:
                draw.line((sx, y, ex, min(y + dash, bottom)), fill=outline, width=3)
                y += step


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    draw.text((110, 74), "Модуль улучшения качества скан-документов", font=FONT_TITLE, fill=TEXT)
    draw.text(
        (110, 146),
        "Слайдовая блок-схема по главе 4.1 и текущей реализации в image_processor.py / pdf_processor.py",
        font=FONT_SUBTITLE,
        fill=MUTED,
    )

    # Header accent to match presentation
    draw.line((110, 198, 1810, 198), fill=LINE, width=2)
    draw.text((110, 214), "Текущая реализация", font=FONT_SECTION, fill=ACCENT_DARK)

    boxes = {
        "start": (110, 280, 400, 430),
        "render": (455, 280, 760, 430),
        "gray": (815, 280, 1115, 430),
        "denoise": (1170, 280, 1495, 430),
        "normalize": (1550, 280, 1810, 430),
        "threshold": (815, 525, 1115, 675),
        "result": (1170, 525, 1495, 675),
        "next": (1550, 525, 1810, 675),
        "note": (110, 774, 1810, 980),
    }

    rounded_box(draw, boxes["start"], fill=GRAY_FILL, outline=GRAY_BORDER)
    draw_wrapped_center(
        draw,
        boxes["start"],
        "Страница помечена\nдля OCR-подготовки\n(has_text = false)",
        title_font=FONT_BOX,
        body_font=FONT_BOX_SMALL,
        fill=TEXT,
    )

    rounded_box(draw, boxes["render"], fill=ACCENT_SOFT, outline=ACCENT)
    draw_wrapped_center(
        draw,
        boxes["render"],
        "Рендер PDF-страницы\nв pixmap\nPDF_RENDER_DPI = 200",
        title_font=FONT_BOX,
        body_font=FONT_BOX_SMALL,
        fill=TEXT,
    )

    rounded_box(draw, boxes["gray"], fill=PURPLE, outline=PURPLE_BORDER)
    draw_wrapped_center(
        draw,
        boxes["gray"],
        "RGB -> BGR ->\nGrayscale\ncv2.cvtColor",
        title_font=FONT_BOX,
        body_font=FONT_BOX_SMALL,
        fill=TEXT,
    )

    rounded_box(draw, boxes["denoise"], fill=GREEN, outline=GREEN_BORDER)
    draw_wrapped_center(
        draw,
        boxes["denoise"],
        "Шумоподавление\nfastNlMeansDenoising\n+ medianBlur(3)",
        title_font=FONT_BOX,
        body_font=FONT_BOX_SMALL,
        fill=TEXT,
    )

    rounded_box(draw, boxes["normalize"], fill=ORANGE, outline=ORANGE_BORDER)
    draw_wrapped_center(
        draw,
        boxes["normalize"],
        "Нормализация\nконтраста\ncv2.normalize",
        title_font=FONT_BOX,
        body_font=FONT_BOX_SMALL,
        fill=TEXT,
    )

    rounded_box(draw, boxes["threshold"], fill=ACCENT_SOFT, outline=ACCENT)
    draw_wrapped_center(
        draw,
        boxes["threshold"],
        "Адаптивная\nбинаризация\nblockSize=31, C=11",
        title_font=FONT_BOX,
        body_font=FONT_BOX_SMALL,
        fill=TEXT,
    )

    rounded_box(draw, boxes["result"], fill=GRAY_FILL, outline=GRAY_BORDER)
    draw_wrapped_center(
        draw,
        boxes["result"],
        "PIL.Image из\nобработанного\nbinary-массива",
        title_font=FONT_BOX,
        body_font=FONT_BOX_SMALL,
        fill=TEXT,
    )

    rounded_box(draw, boxes["next"], fill=GREEN, outline=GREEN_BORDER)
    draw_wrapped_center(
        draw,
        boxes["next"],
        "PNG data URL /\nOCR-preview\nдля следующих этапов",
        title_font=FONT_BOX,
        body_font=FONT_BOX_SMALL,
        fill=TEXT,
    )

    arrow(draw, (400, 355), (455, 355))
    arrow(draw, (760, 355), (815, 355))
    arrow(draw, (1115, 355), (1170, 355))
    arrow(draw, (1495, 355), (1550, 355))
    arrow(draw, (1680, 430), (1680, 525))
    arrow(draw, (1550, 600), (1495, 600))
    arrow(draw, (1170, 600), (1115, 600))

    # Section labels
    draw.text((825, 228), "Подготовка изображения", font=FONT_META, fill=MUTED)
    draw.text((1575, 473), "Передача результата", font=FONT_META, fill=MUTED)

    # Extension note based on chapter 4.1
    dashed_box(draw, boxes["note"], fill=RED_SOFT, outline=RED_BORDER)
    draw.text((140, 798), "Шаги, упомянутые в главе 4.1, но отсутствующие в текущем коде", font=FONT_SECTION, fill="#B42318")
    note_lines = [
        "deskew / коррекция наклона строк;",
        "Otsu-бинаризация как альтернативный вариант;",
        "морфологическая очистка артефактов и длинных линий;",
        "автовыбор лучшего OCR-варианта по confidence и PSM-конфигурации.",
    ]
    y = 846
    for line in note_lines:
        draw.text((165, y), f"• {line}", font=FONT_NOTE, fill=TEXT)
        y += 34

    draw.text(
        (110, 1020),
        "Рекомендуемая подпись на слайде: «Фактическая реализация модуля предобработки сканов в текущем прототипе»",
        font=FONT_META,
        fill=MUTED,
    )

    image.save(PNG_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
