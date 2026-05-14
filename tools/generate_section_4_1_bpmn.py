from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "section_4_1"
PNG_PATH = OUT_DIR / "scan_quality_module_bpmn.png"

W = 3200
H = 620
CENTER_Y = 250

BG = (255, 255, 255, 0)
TEXT = "#1F2937"
ARROW = "#3F4B65"
BOX_FILL = "#EEF3FC"
BOX_BORDER = "#6078B6"
START_FILL = "#E9F6EE"
START_BORDER = "#51A266"
END_FILL = "#FBECEC"
END_BORDER = "#C84A3D"


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


FONT_TITLE = load_font(42, bold=True)
FONT_BODY = load_font(38)
FONT_LABEL = load_font(34)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], *, color: str = ARROW, width: int = 6) -> None:
    draw.line((start[0], start[1], end[0], end[1]), fill=color, width=width)
    draw.polygon(
        [
            (end[0], end[1]),
            (end[0] - 22, end[1] - 12),
            (end[0] - 22, end[1] + 12),
        ],
        fill=color,
    )


def draw_centered_multiline(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], lines: list[str], fonts: list) -> None:
    rendered = []
    for line, font in zip(lines, fonts):
        bbox = draw.textbbox((0, 0), line, font=font)
        rendered.append((line, font, bbox[2] - bbox[0], bbox[3] - bbox[1]))

    total_h = sum(item[3] for item in rendered) + max(0, len(rendered) - 1) * 8
    y = box[1] + ((box[3] - box[1]) - total_h) / 2
    for line, font, width, height in rendered:
        x = box[0] + ((box[2] - box[0]) - width) / 2
        draw.text((x, y), line, font=font, fill=TEXT)
        y += height + 8


def draw_task(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], lines: list[str]) -> None:
    draw.rounded_rectangle(box, radius=34, fill=BOX_FILL, outline=BOX_BORDER, width=5)
    fonts = [FONT_TITLE] + [FONT_BODY for _ in lines[1:]]
    draw_centered_multiline(draw, box, lines, fonts)


def draw_start_event(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int = 44) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=START_BORDER, width=6, fill=START_FILL)
    bbox = draw.textbbox((0, 0), "старт", font=FONT_LABEL)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y + radius + 26), "старт", font=FONT_LABEL, fill=TEXT)


def draw_end_event(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int = 46) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=END_BORDER, width=6, fill=END_FILL)
    draw.ellipse((x - radius + 10, y - radius + 10, x + radius - 10, y + radius - 10), outline=END_BORDER, width=6, fill=None)
    bbox = draw.textbbox((0, 0), "конец", font=FONT_LABEL)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y + radius + 26), "конец", font=FONT_LABEL, fill=TEXT)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(image)

    start_x = 90
    event_y = CENTER_Y
    start_center = (start_x, event_y)
    draw_start_event(draw, start_center)

    tasks = [
        (260, 120, 620, 390, ["Рендер", "страницы", "pixmap", "200 DPI"]),
        (750, 120, 1145, 390, ["Преобразование", "цвета", "RGB → BGR", "grayscale"]),
        (1270, 120, 1770, 390, ["Шумоподавление", "fastNlMeansDenoising", "+ medianBlur"]),
        (1900, 120, 2260, 390, ["Нормализация", "контраста", "cv2.normalize"]),
        (2390, 120, 2890, 390, ["Адаптивная", "бинаризация", "cv2.adaptiveThreshold", "blockSize=31, C=11"]),
    ]

    previous_x = start_center[0] + 44
    for x1, y1, x2, y2, lines in tasks:
        arrow(draw, (previous_x, event_y), (x1, event_y))
        draw_task(draw, (x1, y1, x2, y2), lines)
        previous_x = x2

    end_center = (3090, event_y)
    arrow(draw, (previous_x, event_y), (end_center[0] - 46, event_y))
    draw_end_event(draw, end_center)

    image.save(PNG_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
