from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "section_4_2"

PNG_PATH = OUT_DIR / "layout_module_block_scheme.png"
SVG_PATH = OUT_DIR / "layout_module_block_scheme.svg"
PNG_SLIDE_PATH = OUT_DIR / "layout_module_block_scheme_slide.png"
SVG_SLIDE_PATH = OUT_DIR / "layout_module_block_scheme_slide.svg"

W = 3200
H = 1500
SLIDE_H = 1280

BG = (255, 255, 255, 0)
TEXT = "#1E293B"
MUTED = "#5B657A"
ARROW = "#64748B"
GRID = "#DCE4F0"

MAIN_FILL = "#F4F8FF"
MAIN_BORDER = "#4F46E5"
ROSE_FILL = "#FFF5F7"
ROSE_BORDER = "#E11D48"
GREEN_FILL = "#EEFDF4"
GREEN_BORDER = "#16A34A"

TITLE_FONT_SIZE = 58
SUBTITLE_FONT_SIZE = 30
BADGE_FONT_SIZE = 24
BOX_TITLE_FONT_SIZE = 32
BOX_BODY_FONT_SIZE = 25
NOTE_FONT_SIZE = 24

TITLE_OFFSET_Y = 160


@dataclass(frozen=True)
class Box:
    key: str
    x1: int
    y1: int
    x2: int
    y2: int
    title: str
    body: str
    fill: str
    outline: str
    radius: int = 34
    kind: str = "process"


@dataclass(frozen=True)
class Badge:
    x: int
    y: int
    text: str
    fill: str
    outline: str


@dataclass(frozen=True)
class ArrowPath:
    points: tuple[tuple[int, int], ...]
    dashed: bool = False
    label: str | None = None
    label_xy: tuple[int, int] | None = None


def load_font(size: int, *, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf" if not bold else "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(TITLE_FONT_SIZE, bold=True)
FONT_SUBTITLE = load_font(SUBTITLE_FONT_SIZE)
FONT_BADGE = load_font(BADGE_FONT_SIZE, bold=True)
FONT_BOX_TITLE = load_font(BOX_TITLE_FONT_SIZE, bold=True)
FONT_BOX_BODY = load_font(BOX_BODY_FONT_SIZE)
FONT_NOTE = load_font(NOTE_FONT_SIZE, bold=True)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    wrapped: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            wrapped.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            width, _ = text_size(draw, candidate, font)
            if width <= max_width:
                current = candidate
            else:
                wrapped.append(current)
                current = word
        wrapped.append(current)
    return wrapped


def box_layout(
    draw: ImageDraw.ImageDraw,
    box: Box,
) -> tuple[list[str], list[str], int]:
    inner_width = box.x2 - box.x1 - 72
    title_lines = wrap_text(draw, box.title, FONT_BOX_TITLE, inner_width)
    body_lines = wrap_text(draw, box.body, FONT_BOX_BODY, inner_width)
    line_gap = 12
    return title_lines, body_lines, line_gap


def line_height(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont) -> int:
    _, height = text_size(draw, "Ag", font)
    return height


def draw_multiline_centered(
    draw: ImageDraw.ImageDraw,
    box: Box,
    title_lines: list[str],
    body_lines: list[str],
    line_gap: int,
) -> None:
    title_h = line_height(draw, FONT_BOX_TITLE)
    body_h = line_height(draw, FONT_BOX_BODY)
    content_heights = 0
    if title_lines:
        content_heights += len(title_lines) * title_h
        content_heights += max(0, len(title_lines) - 1) * line_gap
    if body_lines:
        if title_lines:
            content_heights += 18
        content_heights += len(body_lines) * body_h
        content_heights += max(0, len(body_lines) - 1) * line_gap

    y = box.y1 + ((box.y2 - box.y1) - content_heights) / 2

    for line in title_lines:
        width, _ = text_size(draw, line, FONT_BOX_TITLE)
        x = box.x1 + ((box.x2 - box.x1) - width) / 2
        draw.text((x, y), line, font=FONT_BOX_TITLE, fill=TEXT)
        y += title_h + line_gap

    if title_lines and body_lines:
        y += 6

    for line in body_lines:
        if not line:
            y += body_h
            continue
        width, _ = text_size(draw, line, FONT_BOX_BODY)
        x = box.x1 + ((box.x2 - box.x1) - width) / 2
        draw.text((x, y), line, font=FONT_BOX_BODY, fill=TEXT)
        y += body_h + line_gap


def draw_box(draw: ImageDraw.ImageDraw, box: Box) -> None:
    width = 6 if box.kind != "note" else 5
    radius = box.radius if box.kind != "terminator" else (box.y2 - box.y1) // 2
    draw.rounded_rectangle(
        (box.x1, box.y1, box.x2, box.y2),
        radius=radius,
        fill=box.fill,
        outline=box.outline,
        width=width,
    )
    title_lines, body_lines, line_gap = box_layout(draw, box)
    draw_multiline_centered(draw, box, title_lines, body_lines, line_gap)


def draw_badge(draw: ImageDraw.ImageDraw, badge: Badge) -> None:
    text_w, text_h = text_size(draw, badge.text, FONT_BADGE)
    x2 = badge.x + text_w + 46
    y2 = badge.y + text_h + 26
    draw.rounded_rectangle(
        (badge.x, badge.y, x2, y2),
        radius=26,
        fill=badge.fill,
        outline=badge.outline,
        width=3,
    )
    draw.text((badge.x + 23, badge.y + 13), badge.text, font=FONT_BADGE, fill=badge.outline)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: str,
    width: int,
    dash: int = 18,
    gap: int = 12,
) -> None:
    x1, y1 = start
    x2, y2 = end
    if x1 == x2:
        step = dash + gap
        direction = 1 if y2 >= y1 else -1
        for y in range(y1, y2, direction * step):
            y_end = y + direction * min(dash, abs(y2 - y))
            draw.line((x1, y, x2, y_end), fill=fill, width=width)
        return
    if y1 == y2:
        step = dash + gap
        direction = 1 if x2 >= x1 else -1
        for x in range(x1, x2, direction * step):
            x_end = x + direction * min(dash, abs(x2 - x))
            draw.line((x, y1, x_end, y2), fill=fill, width=width)
        return
    draw.line((x1, y1, x2, y2), fill=fill, width=width)


def arrow_head_points(end: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = end
    dx, dy = direction
    size = 24
    spread = 14
    if abs(dx) >= abs(dy):
        sign = 1 if dx >= 0 else -1
        return [
            (x, y),
            (x - sign * size, y - spread),
            (x - sign * size, y + spread),
        ]
    sign = 1 if dy >= 0 else -1
    return [
        (x, y),
        (x - spread, y - sign * size),
        (x + spread, y - sign * size),
    ]


def draw_arrow(draw: ImageDraw.ImageDraw, path: ArrowPath, *, fill: str = ARROW, width: int = 6) -> None:
    points = path.points
    for start, end in zip(points, points[1:]):
        if path.dashed:
            draw_dashed_line(draw, start, end, fill=fill, width=width)
        else:
            draw.line((start[0], start[1], end[0], end[1]), fill=fill, width=width)
    final_start, final_end = points[-2], points[-1]
    direction = (final_end[0] - final_start[0], final_end[1] - final_start[1])
    draw.polygon(arrow_head_points(final_end, direction), fill=fill)
    if path.label and path.label_xy:
        draw.text(path.label_xy, path.label, font=FONT_BADGE, fill=MUTED)


def shift_box(box: Box, dy: int) -> Box:
    return Box(
        key=box.key,
        x1=box.x1,
        y1=box.y1 + dy,
        x2=box.x2,
        y2=box.y2 + dy,
        title=box.title,
        body=box.body,
        fill=box.fill,
        outline=box.outline,
        radius=box.radius,
        kind=box.kind,
    )


def shift_badge(badge: Badge, dy: int) -> Badge:
    return Badge(x=badge.x, y=badge.y + dy, text=badge.text, fill=badge.fill, outline=badge.outline)


def shift_arrow(path: ArrowPath, dy: int) -> ArrowPath:
    return ArrowPath(
        points=tuple((x, y + dy) for x, y in path.points),
        dashed=path.dashed,
        label=path.label,
        label_xy=(path.label_xy[0], path.label_xy[1] + dy) if path.label_xy else None,
    )


def base_boxes() -> list[Box]:
    return [
        Box(
            key="input",
            x1=120,
            y1=260,
            x2=520,
            y2=660,
            title="1. Вход модуля",
            body="RGB-изображение\nстраницы\nпосле рендера PDF",
            fill=MAIN_FILL,
            outline=MAIN_BORDER,
        ),
        Box(
            key="surya_raw",
            x1=640,
            y1=260,
            x2=1140,
            y2=660,
            title="2. Сырые данные Surya",
            body="LayoutPredictor выделяет\nобласти страницы\n\nRecognitionPredictor извлекает\nOCR-строки и их координаты",
            fill=MAIN_FILL,
            outline=MAIN_BORDER,
        ),
        Box(
            key="layout_block",
            x1=1260,
            y1=260,
            x2=1800,
            y2=660,
            title="3. Формирование LayoutBlock",
            body="нормализация типов блоков\nпривязка OCR-строк к bbox\nзаполнение text, confidence\nи reading_order",
            fill=ROSE_FILL,
            outline=ROSE_BORDER,
        ),
        Box(
            key="postprocess",
            x1=1920,
            y1=260,
            x2=2440,
            y2=660,
            title="4. Постобработка блоков",
            body="удаление мелких областей\nопределение page_number\nслияние соседних текстовых блоков\nсортировка и перенумерация",
            fill=MAIN_FILL,
            outline=MAIN_BORDER,
        ),
        Box(
            key="artifacts",
            x1=2560,
            y1=260,
            x2=3080,
            y2=660,
            title="5. Сохранение артефактов",
            body="строится PNG-визуализация\nрезультат сохраняется в JSON\nвизуализация кладётся в cache",
            fill=MAIN_FILL,
            outline=MAIN_BORDER,
        ),
        Box(
            key="response",
            x1=2560,
            y1=720,
            x2=3080,
            y2=910,
            title="6. Выход модуля",
            body="LayoutAnalysisResponse:\nanalysis_id, blocks,\nvisualization_url,\nresult_json_path",
            fill=GREEN_FILL,
            outline=GREEN_BORDER,
        ),
        Box(
            key="formula_note",
            x1=1310,
            y1=740,
            x2=1750,
            y2=920,
            title="3а. Для formula",
            body="вырезка области\nRecognitionPredictor\n(block_without_boxes)\nполучение LaTeX",
            fill=ROSE_FILL,
            outline=ROSE_BORDER,
            kind="note",
        ),
    ]


def base_badges() -> list[Badge]:
    return []


def base_arrows() -> list[ArrowPath]:
    return [
        ArrowPath(points=((520, 460), (640, 460))),
        ArrowPath(points=((1140, 460), (1260, 460))),
        ArrowPath(points=((1800, 460), (1920, 460))),
        ArrowPath(points=((2440, 460), (2560, 460))),
        ArrowPath(points=((2820, 660), (2820, 720))),
        ArrowPath(
            points=((1530, 660), (1530, 740)),
            dashed=True,
        ),
    ]


def draw_title(draw: ImageDraw.ImageDraw) -> None:
    title = "Модуль layout-анализа"
    subtitle = "Последовательность обработки страницы по разделу 4.2 и актуальному коду"
    draw.text((120, 56), title, font=FONT_TITLE, fill=TEXT)
    draw.text((120, 132), subtitle, font=FONT_SUBTITLE, fill=MUTED)
    draw.line((120, 190, W - 120, 190), fill=GRID, width=3)


def svg_text_block(
    draw: ImageDraw.ImageDraw,
    box: Box,
    title_lines: list[str],
    body_lines: list[str],
    line_gap: int,
) -> str:
    title_h = line_height(draw, FONT_BOX_TITLE)
    body_h = line_height(draw, FONT_BOX_BODY)
    content_heights = 0
    if title_lines:
        content_heights += len(title_lines) * title_h + max(0, len(title_lines) - 1) * line_gap
    if body_lines:
        if title_lines:
            content_heights += 18
        content_heights += len(body_lines) * body_h + max(0, len(body_lines) - 1) * line_gap
    y = box.y1 + ((box.y2 - box.y1) - content_heights) / 2

    parts: list[str] = []
    for line in title_lines:
        parts.append(
            f'<text x="{(box.x1 + box.x2) / 2:.1f}" y="{y + title_h:.1f}" '
            f'font-size="{BOX_TITLE_FONT_SIZE}" font-weight="700" text-anchor="middle" '
            f'fill="{TEXT}" font-family="Arial">{escape(line)}</text>'
        )
        y += title_h + line_gap
    if title_lines and body_lines:
        y += 6
    for line in body_lines:
        if not line:
            y += body_h
            continue
        parts.append(
            f'<text x="{(box.x1 + box.x2) / 2:.1f}" y="{y + body_h:.1f}" '
            f'font-size="{BOX_BODY_FONT_SIZE}" text-anchor="middle" fill="{TEXT}" '
            f'font-family="Arial">{escape(line)}</text>'
        )
        y += body_h + line_gap
    return "\n".join(parts)


def svg_box(draw: ImageDraw.ImageDraw, box: Box) -> str:
    rx = box.radius if box.kind != "terminator" else (box.y2 - box.y1) // 2
    title_lines, body_lines, line_gap = box_layout(draw, box)
    shape = (
        f'<rect x="{box.x1}" y="{box.y1}" width="{box.x2 - box.x1}" height="{box.y2 - box.y1}" '
        f'rx="{rx}" ry="{rx}" fill="{box.fill}" stroke="{box.outline}" '
        f'stroke-width="{6 if box.kind != "note" else 5}" />'
    )
    return f"{shape}\n{svg_text_block(draw, box, title_lines, body_lines, line_gap)}"


def svg_badge(draw: ImageDraw.ImageDraw, badge: Badge) -> str:
    text_w, text_h = text_size(draw, badge.text, FONT_BADGE)
    width = text_w + 46
    height = text_h + 26
    return (
        f'<rect x="{badge.x}" y="{badge.y}" width="{width}" height="{height}" rx="26" ry="26" '
        f'fill="{badge.fill}" stroke="{badge.outline}" stroke-width="3" />\n'
        f'<text x="{badge.x + 23}" y="{badge.y + 13 + text_h}" font-size="{BADGE_FONT_SIZE}" '
        f'font-weight="700" fill="{badge.outline}" font-family="Arial">{escape(badge.text)}</text>'
    )


def svg_arrow(path: ArrowPath) -> str:
    points = list(path.points)
    poly = " ".join(f"{x},{y}" for x, y in points)
    dash = ' stroke-dasharray="18 12"' if path.dashed else ""
    final_start, final_end = points[-2], points[-1]
    head = " ".join(f"{x},{y}" for x, y in arrow_head_points(final_end, (final_end[0] - final_start[0], final_end[1] - final_start[1])))
    label = ""
    if path.label and path.label_xy:
        label = (
            f'\n<text x="{path.label_xy[0]}" y="{path.label_xy[1]}" font-size="{BADGE_FONT_SIZE}" '
            f'fill="{MUTED}" font-family="Arial">{escape(path.label)}</text>'
        )
    return (
        f'<polyline points="{poly}" fill="none" stroke="{ARROW}" stroke-width="6" '
        f'stroke-linejoin="round" stroke-linecap="round"{dash} />\n'
        f'<polygon points="{head}" fill="{ARROW}" />{label}'
    )


def render_variant(*, with_title: bool) -> tuple[Image.Image, str]:
    height = H if with_title else SLIDE_H
    offset = TITLE_OFFSET_Y if with_title else 0
    image = Image.new("RGBA", (W, height), BG)
    draw = ImageDraw.Draw(image)

    boxes = base_boxes()
    badges = base_badges()
    arrows = base_arrows()
    if offset:
        boxes = [shift_box(box, offset) for box in boxes]
        badges = [shift_badge(badge, offset) for badge in badges]
        arrows = [shift_arrow(arrow, offset) for arrow in arrows]

    if with_title:
        draw_title(draw)

    for badge in badges:
        draw_badge(draw, badge)
    for box in boxes:
        draw_box(draw, box)
    for arrow in arrows:
        draw_arrow(draw, arrow)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}">',
    ]
    if with_title:
        svg_parts.append(
            f'<text x="120" y="120" font-size="{TITLE_FONT_SIZE}" font-weight="700" fill="{TEXT}" font-family="Arial">'
            f'{escape("Модуль layout-анализа")}</text>'
        )
        svg_parts.append(
            f'<text x="120" y="166" font-size="{SUBTITLE_FONT_SIZE}" fill="{MUTED}" font-family="Arial">'
            f'{escape("Последовательность обработки страницы по разделу 4.2 и актуальному коду")}</text>'
        )
        svg_parts.append(
            f'<line x1="120" y1="190" x2="{W - 120}" y2="190" stroke="{GRID}" stroke-width="3" />'
        )
    for badge in badges:
        svg_parts.append(svg_badge(draw, badge))
    for box in boxes:
        svg_parts.append(svg_box(draw, box))
    for arrow in arrows:
        svg_parts.append(svg_arrow(arrow))
    svg_parts.append("</svg>")
    return image, "\n".join(svg_parts)


def save_outputs(paths: Iterable[tuple[Path, Path, bool]]) -> None:
    for png_path, svg_path, with_title in paths:
        image, svg = render_variant(with_title=with_title)
        image.save(png_path)
        svg_path.write_text(svg, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_outputs(
        [
            (PNG_PATH, SVG_PATH, True),
            (PNG_SLIDE_PATH, SVG_SLIDE_PATH, False),
        ]
    )
    print(PNG_PATH)
    print(SVG_PATH)
    print(PNG_SLIDE_PATH)
    print(SVG_SLIDE_PATH)


if __name__ == "__main__":
    main()
