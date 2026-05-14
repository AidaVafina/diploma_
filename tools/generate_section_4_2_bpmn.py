from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "section_4_2"

PNG_PATH = OUT_DIR / "layout_module_bpmn.png"
SVG_PATH = OUT_DIR / "layout_module_bpmn.svg"
PNG_SLIDE_PATH = OUT_DIR / "layout_module_bpmn_slide.png"
SVG_SLIDE_PATH = OUT_DIR / "layout_module_bpmn_slide.svg"

W = 3400
H = 1500
SLIDE_H = 1300
TITLE_OFFSET_Y = 180

BG = (255, 255, 255, 0)
TEXT = "#1F2937"
MUTED = "#667085"
LINE = "#55637D"
GRID = "#DCE3EE"
POOL_FILL = "#F8FAFC"
POOL_BORDER = "#CBD5E1"
TASK_FILL = "#EEF3FC"
TASK_BORDER = "#6078B6"
GATEWAY_FILL = "#FFF8E8"
GATEWAY_BORDER = "#D18B00"
START_FILL = "#EAF8F0"
START_BORDER = "#4FA06A"
END_FILL = "#FBECEC"
END_BORDER = "#C84A3D"

TITLE_FONT_SIZE = 60
SUBTITLE_FONT_SIZE = 30
POOL_FONT_SIZE = 30
TASK_TITLE_FONT_SIZE = 28
TASK_BODY_FONT_SIZE = 24
LABEL_FONT_SIZE = 22
GATEWAY_FONT_SIZE = 38


@dataclass(frozen=True)
class Task:
    key: str
    x1: int
    y1: int
    x2: int
    y2: int
    title: str
    body: str


@dataclass(frozen=True)
class Gateway:
    key: str
    cx: int
    cy: int
    size: int
    kind: str


@dataclass(frozen=True)
class Event:
    key: str
    cx: int
    cy: int
    radius: int
    kind: str
    label: str


@dataclass(frozen=True)
class Flow:
    points: tuple[tuple[int, int], ...]
    label: str | None = None
    label_xy: tuple[int, int] | None = None


def load_font(size: int, *, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(TITLE_FONT_SIZE, bold=True)
FONT_SUBTITLE = load_font(SUBTITLE_FONT_SIZE)
FONT_POOL = load_font(POOL_FONT_SIZE, bold=True)
FONT_TASK_TITLE = load_font(TASK_TITLE_FONT_SIZE, bold=True)
FONT_TASK_BODY = load_font(TASK_BODY_FONT_SIZE)
FONT_LABEL = load_font(LABEL_FONT_SIZE, bold=True)
FONT_GATEWAY = load_font(GATEWAY_FONT_SIZE, bold=True)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def line_height(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont) -> int:
    _, h = text_size(draw, "Ag", font)
    return h


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


def draw_title(draw: ImageDraw.ImageDraw) -> None:
    draw.text((140, 60), "Модуль layout-анализа в формате BPMN", font=FONT_TITLE, fill=TEXT)
    draw.text(
        (140, 136),
        "Параллельная работа Surya-предикторов и условная ветка обработки formula",
        font=FONT_SUBTITLE,
        fill=MUTED,
    )
    draw.line((140, 198, W - 140, 198), fill=GRID, width=3)


def draw_pool(draw: ImageDraw.ImageDraw, *, offset_y: int) -> tuple[int, int, int, int]:
    pool = (140, 280 + offset_y, W - 140, 1180 + offset_y)
    header_w = 180
    draw.rounded_rectangle(pool, radius=34, fill=POOL_FILL, outline=POOL_BORDER, width=4)
    draw.line((pool[0] + header_w, pool[1], pool[0] + header_w, pool[3]), fill=POOL_BORDER, width=3)

    label = "Модуль\nlayout-\nанализа"
    lines = label.split("\n")
    lh = line_height(draw, FONT_POOL)
    total_h = len(lines) * lh + (len(lines) - 1) * 8
    y = pool[1] + ((pool[3] - pool[1]) - total_h) / 2
    center_x = pool[0] + header_w / 2
    for line in lines:
        w, _ = text_size(draw, line, FONT_POOL)
        draw.text((center_x - w / 2, y), line, font=FONT_POOL, fill=TEXT)
        y += lh + 8
    return pool


def draw_event(draw: ImageDraw.ImageDraw, event: Event) -> None:
    x, y, r = event.cx, event.cy, event.radius
    if event.kind == "start":
        draw.ellipse((x - r, y - r, x + r, y + r), outline=START_BORDER, width=6, fill=START_FILL)
    else:
        draw.ellipse((x - r, y - r, x + r, y + r), outline=END_BORDER, width=6, fill=END_FILL)
        draw.ellipse((x - r + 10, y - r + 10, x + r - 10, y + r - 10), outline=END_BORDER, width=6)

    w, _ = text_size(draw, event.label, FONT_LABEL)
    draw.text((x - w / 2, y + r + 20), event.label, font=FONT_LABEL, fill=TEXT)


def draw_task(draw: ImageDraw.ImageDraw, task: Task) -> None:
    draw.rounded_rectangle((task.x1, task.y1, task.x2, task.y2), radius=34, fill=TASK_FILL, outline=TASK_BORDER, width=5)
    inner_w = task.x2 - task.x1 - 56
    title_lines = wrap_text(draw, task.title, FONT_TASK_TITLE, inner_w)
    body_lines = wrap_text(draw, task.body, FONT_TASK_BODY, inner_w)
    title_h = line_height(draw, FONT_TASK_TITLE)
    body_h = line_height(draw, FONT_TASK_BODY)
    total_h = 0
    if title_lines:
        total_h += len(title_lines) * title_h + (len(title_lines) - 1) * 6
    if body_lines:
        if title_lines:
            total_h += 18
        total_h += len(body_lines) * body_h + (len(body_lines) - 1) * 6
    y = task.y1 + ((task.y2 - task.y1) - total_h) / 2
    for line in title_lines:
        w, _ = text_size(draw, line, FONT_TASK_TITLE)
        draw.text(((task.x1 + task.x2 - w) / 2, y), line, font=FONT_TASK_TITLE, fill=TEXT)
        y += title_h + 6
    if title_lines and body_lines:
        y += 12
    for line in body_lines:
        if not line:
            y += body_h
            continue
        w, _ = text_size(draw, line, FONT_TASK_BODY)
        draw.text(((task.x1 + task.x2 - w) / 2, y), line, font=FONT_TASK_BODY, fill=TEXT)
        y += body_h + 6


def draw_gateway(draw: ImageDraw.ImageDraw, gateway: Gateway) -> None:
    s = gateway.size
    cx, cy = gateway.cx, gateway.cy
    points = [(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)]
    draw.polygon(points, outline=GATEWAY_BORDER, fill=GATEWAY_FILL)
    draw.line(points + [points[0]], fill=GATEWAY_BORDER, width=5)
    if gateway.kind == "parallel":
        draw.text((cx - 13, cy - 22), "+", font=FONT_GATEWAY, fill=TEXT)
    else:
        draw.text((cx - 14, cy - 21), "X", font=FONT_GATEWAY, fill=TEXT)


def arrow_head(end: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = end
    dx, dy = direction
    size = 22
    spread = 13
    if abs(dx) >= abs(dy):
        sign = 1 if dx >= 0 else -1
        return [(x, y), (x - sign * size, y - spread), (x - sign * size, y + spread)]
    sign = 1 if dy >= 0 else -1
    return [(x, y), (x - spread, y - sign * size), (x + spread, y - sign * size)]


def draw_flow(draw: ImageDraw.ImageDraw, flow: Flow) -> None:
    pts = flow.points
    for start, end in zip(pts, pts[1:]):
        draw.line((start[0], start[1], end[0], end[1]), fill=LINE, width=6)
    last_start, last_end = pts[-2], pts[-1]
    draw.polygon(arrow_head(last_end, (last_end[0] - last_start[0], last_end[1] - last_start[1])), fill=LINE)
    if flow.label and flow.label_xy:
        draw.text(flow.label_xy, flow.label, font=FONT_LABEL, fill=MUTED)


def base_events() -> list[Event]:
    return [
        Event("start", 300, 730, 42, "start", "Старт"),
        Event("end", 3220, 730, 46, "end", "Конец"),
    ]


def base_tasks() -> list[Task]:
    return [
        Task(
            "input",
            420,
            580,
            730,
            880,
            "Получить вход страницы",
            "RGB-изображение\nпосле рендера PDF",
        ),
        Task(
            "layout",
            900,
            420,
            1250,
            620,
            "Определить layout-блоки",
            "Surya LayoutPredictor\nbbox, label, position",
        ),
        Task(
            "ocr",
            900,
            840,
            1250,
            1040,
            "Извлечь OCR-строки",
            "Surya RecognitionPredictor\nocr_with_boxes",
        ),
        Task(
            "blocks",
            1520,
            560,
            1910,
            900,
            "Сформировать LayoutBlock",
            "нормализовать тип блока\nсопоставить OCR-строки\nзаполнить text, confidence\nи reading_order",
        ),
        Task(
            "latex",
            2030,
            940,
            2390,
            1140,
            "Извлечь LaTeX",
            "вырезать formula-блок\nblock_without_boxes\nсохранить latex",
        ),
        Task(
            "post",
            2470,
            560,
            2850,
            900,
            "Постобработать блоки",
            "удалить мелкие области\nопределить page_number\nобъединить соседние text/title/header/footer\nперенумеровать reading_order",
        ),
        Task(
            "artifacts",
            2940,
            560,
            3180,
            900,
            "Сохранить артефакты",
            "PNG-визуализация\nJSON-результат\ncache и response",
        ),
    ]


def base_gateways() -> list[Gateway]:
    return [
        Gateway("parallel_split", 810, 730, 46, "parallel"),
        Gateway("parallel_join", 1360, 730, 46, "parallel"),
        Gateway("formula_split", 1985, 730, 46, "exclusive"),
        Gateway("formula_join", 2430, 730, 46, "exclusive"),
    ]


def base_flows() -> list[Flow]:
    return [
        Flow(points=((342, 730), (420, 730))),
        Flow(points=((730, 730), (810, 730))),
        Flow(points=((810, 730), (810, 520), (900, 520))),
        Flow(points=((810, 730), (810, 940), (900, 940))),
        Flow(points=((1250, 520), (1360, 520), (1360, 730))),
        Flow(points=((1250, 940), (1360, 940), (1360, 730))),
        Flow(points=((1360, 730), (1520, 730))),
        Flow(points=((1910, 730), (1985, 730))),
        Flow(points=((1985, 730), (2430, 730)), label="нет formula", label_xy=(2130, 686)),
        Flow(points=((1985, 730), (1985, 1040), (2030, 1040)), label="есть formula", label_xy=(1828, 886)),
        Flow(points=((2390, 1040), (2430, 1040), (2430, 730))),
        Flow(points=((2430, 730), (2470, 730))),
        Flow(points=((2850, 730), (2940, 730))),
        Flow(points=((3180, 730), (3174, 730))),
    ]


def shift_task(task: Task, dy: int) -> Task:
    return Task(task.key, task.x1, task.y1 + dy, task.x2, task.y2 + dy, task.title, task.body)


def shift_gateway(gateway: Gateway, dy: int) -> Gateway:
    return Gateway(gateway.key, gateway.cx, gateway.cy + dy, gateway.size, gateway.kind)


def shift_event(event: Event, dy: int) -> Event:
    return Event(event.key, event.cx, event.cy + dy, event.radius, event.kind, event.label)


def shift_flow(flow: Flow, dy: int) -> Flow:
    return Flow(
        points=tuple((x, y + dy) for x, y in flow.points),
        label=flow.label,
        label_xy=(flow.label_xy[0], flow.label_xy[1] + dy) if flow.label_xy else None,
    )


def svg_task(draw: ImageDraw.ImageDraw, task: Task) -> str:
    inner_w = task.x2 - task.x1 - 56
    title_lines = wrap_text(draw, task.title, FONT_TASK_TITLE, inner_w)
    body_lines = wrap_text(draw, task.body, FONT_TASK_BODY, inner_w)
    title_h = line_height(draw, FONT_TASK_TITLE)
    body_h = line_height(draw, FONT_TASK_BODY)
    total_h = 0
    if title_lines:
        total_h += len(title_lines) * title_h + (len(title_lines) - 1) * 6
    if body_lines:
        if title_lines:
            total_h += 18
        total_h += len(body_lines) * body_h + (len(body_lines) - 1) * 6
    y = task.y1 + ((task.y2 - task.y1) - total_h) / 2
    text_parts: list[str] = []
    for line in title_lines:
        text_parts.append(
            f'<text x="{(task.x1 + task.x2) / 2:.1f}" y="{y + title_h:.1f}" font-size="{TASK_TITLE_FONT_SIZE}" '
            f'font-weight="700" text-anchor="middle" fill="{TEXT}" font-family="Arial">{escape(line)}</text>'
        )
        y += title_h + 6
    if title_lines and body_lines:
        y += 12
    for line in body_lines:
        if not line:
            y += body_h
            continue
        text_parts.append(
            f'<text x="{(task.x1 + task.x2) / 2:.1f}" y="{y + body_h:.1f}" font-size="{TASK_BODY_FONT_SIZE}" '
            f'text-anchor="middle" fill="{TEXT}" font-family="Arial">{escape(line)}</text>'
        )
        y += body_h + 6
    return (
        f'<rect x="{task.x1}" y="{task.y1}" width="{task.x2 - task.x1}" height="{task.y2 - task.y1}" '
        f'rx="34" ry="34" fill="{TASK_FILL}" stroke="{TASK_BORDER}" stroke-width="5" />\n'
        + "\n".join(text_parts)
    )


def svg_gateway(gateway: Gateway) -> str:
    s = gateway.size
    cx, cy = gateway.cx, gateway.cy
    points = f"{cx},{cy - s} {cx + s},{cy} {cx},{cy + s} {cx - s},{cy}"
    symbol = "+" if gateway.kind == "parallel" else "X"
    return (
        f'<polygon points="{points}" fill="{GATEWAY_FILL}" stroke="{GATEWAY_BORDER}" stroke-width="5" />\n'
        f'<text x="{cx}" y="{cy + 14}" font-size="{GATEWAY_FONT_SIZE}" font-weight="700" text-anchor="middle" '
        f'fill="{TEXT}" font-family="Arial">{escape(symbol)}</text>'
    )


def svg_event(event: Event) -> str:
    outer = (
        f'<circle cx="{event.cx}" cy="{event.cy}" r="{event.radius}" '
        f'fill="{START_FILL if event.kind == "start" else END_FILL}" '
        f'stroke="{START_BORDER if event.kind == "start" else END_BORDER}" stroke-width="6" />'
    )
    inner = ""
    if event.kind == "end":
        inner = (
            f'\n<circle cx="{event.cx}" cy="{event.cy}" r="{event.radius - 10}" '
            f'fill="none" stroke="{END_BORDER}" stroke-width="6" />'
        )
    return (
        outer
        + inner
        + f'\n<text x="{event.cx}" y="{event.cy + event.radius + 48}" font-size="{LABEL_FONT_SIZE}" '
        f'font-weight="700" text-anchor="middle" fill="{TEXT}" font-family="Arial">{escape(event.label)}</text>'
    )


def svg_flow(flow: Flow) -> str:
    pts = " ".join(f"{x},{y}" for x, y in flow.points)
    last_start, last_end = flow.points[-2], flow.points[-1]
    head = " ".join(
        f"{x},{y}" for x, y in arrow_head(last_end, (last_end[0] - last_start[0], last_end[1] - last_start[1]))
    )
    label = ""
    if flow.label and flow.label_xy:
        label = (
            f'\n<text x="{flow.label_xy[0]}" y="{flow.label_xy[1]}" font-size="{LABEL_FONT_SIZE}" font-weight="700" '
            f'fill="{MUTED}" font-family="Arial">{escape(flow.label)}</text>'
        )
    return (
        f'<polyline points="{pts}" fill="none" stroke="{LINE}" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" />\n'
        f'<polygon points="{head}" fill="{LINE}" />{label}'
    )


def render_variant(*, with_title: bool) -> tuple[Image.Image, str]:
    offset = TITLE_OFFSET_Y if with_title else 0
    height = H if with_title else SLIDE_H
    image = Image.new("RGBA", (W, height), BG)
    draw = ImageDraw.Draw(image)

    if with_title:
        draw_title(draw)
    pool = draw_pool(draw, offset_y=offset)

    events = [shift_event(item, offset) for item in base_events()]
    tasks = [shift_task(item, offset) for item in base_tasks()]
    gateways = [shift_gateway(item, offset) for item in base_gateways()]
    flows = [shift_flow(item, offset) for item in base_flows()]

    for flow in flows:
        draw_flow(draw, flow)
    for gateway in gateways:
        draw_gateway(draw, gateway)
    for task in tasks:
        draw_task(draw, task)
    for event in events:
        draw_event(draw, event)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}">',
    ]
    if with_title:
        svg_parts.append(
            f'<text x="140" y="122" font-size="{TITLE_FONT_SIZE}" font-weight="700" fill="{TEXT}" font-family="Arial">'
            f'{escape("Модуль layout-анализа в формате BPMN")}</text>'
        )
        svg_parts.append(
            f'<text x="140" y="168" font-size="{SUBTITLE_FONT_SIZE}" fill="{MUTED}" font-family="Arial">'
            f'{escape("Параллельная работа Surya-предикторов и условная ветка обработки formula")}</text>'
        )
        svg_parts.append(f'<line x1="140" y1="198" x2="{W - 140}" y2="198" stroke="{GRID}" stroke-width="3" />')
    svg_parts.append(
        f'<rect x="{pool[0]}" y="{pool[1]}" width="{pool[2] - pool[0]}" height="{pool[3] - pool[1]}" '
        f'rx="34" ry="34" fill="{POOL_FILL}" stroke="{POOL_BORDER}" stroke-width="4" />'
    )
    svg_parts.append(
        f'<line x1="{pool[0] + 180}" y1="{pool[1]}" x2="{pool[0] + 180}" y2="{pool[3]}" stroke="{POOL_BORDER}" stroke-width="3" />'
    )
    pool_lines = ["Модуль", "layout-", "анализа"]
    lh = line_height(draw, FONT_POOL)
    total_h = len(pool_lines) * lh + (len(pool_lines) - 1) * 8
    y = pool[1] + ((pool[3] - pool[1]) - total_h) / 2
    cx = pool[0] + 90
    for line in pool_lines:
        svg_parts.append(
            f'<text x="{cx}" y="{y + lh:.1f}" font-size="{POOL_FONT_SIZE}" font-weight="700" text-anchor="middle" '
            f'fill="{TEXT}" font-family="Arial">{escape(line)}</text>'
        )
        y += lh + 8

    for flow in flows:
        svg_parts.append(svg_flow(flow))
    for gateway in gateways:
        svg_parts.append(svg_gateway(gateway))
    for task in tasks:
        svg_parts.append(svg_task(draw, task))
    for event in events:
        svg_parts.append(svg_event(event))
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
