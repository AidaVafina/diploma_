from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "section_4_1"
PNG_PATH = OUT_DIR / "scan_quality_module_block_scheme.png"

W = 3200
H = 700
CENTER_Y = 280

BG = (255, 255, 255, 0)
TEXT = "#1F2937"
ARROW = "#404A5F"
BOX_FILL = "#EEF3FC"
BOX_BORDER = "#6078B6"
START_FILL = "#EAF8F0"
START_BORDER = "#4FA06A"
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


FONT_HEAD = load_font(40, bold=True)
FONT_BODY = load_font(38)
FONT_LABEL = load_font(34, bold=True)


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

    total_height = sum(item[3] for item in rendered) + max(0, len(rendered) - 1) * 8
    y = box[1] + ((box[3] - box[1]) - total_height) / 2
    for line, font, width, height in rendered:
        x = box[0] + ((box[2] - box[0]) - width) / 2
        draw.text((x, y), line, font=font, fill=TEXT)
        y += height + 8


def draw_process(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], lines: list[str]) -> None:
    draw.rounded_rectangle(box, radius=34, fill=BOX_FILL, outline=BOX_BORDER, width=5)
    fonts = [FONT_HEAD] + [FONT_BODY for _ in lines[1:]]
    draw_centered_multiline(draw, box, lines, fonts)


def draw_terminator(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str, *, fill: str, outline: str) -> None:
    draw.rounded_rectangle(box, radius=(box[3] - box[1]) // 2, fill=fill, outline=outline, width=6)
    bbox = draw.textbbox((0, 0), label, font=FONT_LABEL)
    x = box[0] + ((box[2] - box[0]) - (bbox[2] - bbox[0])) / 2
    y = box[1] + ((box[3] - box[1]) - (bbox[3] - bbox[1])) / 2
    draw.text((x, y), label, font=FONT_LABEL, fill=TEXT)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(image)

    start_box = (30, 218, 190, 342)
    end_box = (3000, 218, 3170, 342)
    draw_terminator(draw, start_box, "старт", fill=START_FILL, outline=START_BORDER)
    draw_terminator(draw, end_box, "конец", fill=END_FILL, outline=END_BORDER)

    tasks = [
        (260, 110, 620, 450, ["Рендер", "страницы", "pixmap", "200 DPI"]),
        (750, 110, 1145, 450, ["Преобразование", "цвета", "RGB -> BGR", "grayscale"]),
        (1270, 110, 1770, 450, ["Шумоподавление", "fastNlMeansDenoising", "+ medianBlur"]),
        (1900, 110, 2260, 450, ["Нормализация", "контраста", "cv2.normalize"]),
        (2390, 110, 2890, 450, ["Адаптивная", "бинаризация", "cv2.adaptiveThreshold", "blockSize=31, C=11"]),
    ]

    previous_x = start_box[2]
    for x1, y1, x2, y2, lines in tasks:
        arrow(draw, (previous_x, CENTER_Y), (x1, CENTER_Y))
        draw_process(draw, (x1, y1, x2, y2), lines)
        previous_x = x2

    arrow(draw, (previous_x, CENTER_Y), (end_box[0], CENTER_Y))

    image.save(PNG_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
