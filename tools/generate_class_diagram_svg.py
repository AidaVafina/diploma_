from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "class_diagram"
SVG_PATH = OUT_DIR / "processing_system_class_diagram.svg"

W = 4300
H = 2650

BG = "#FFFFFF"
INK = "#1F2A44"
MUTED = "#5C6B87"
BORDER = "#5B6F95"
TITLE_FILL = "#F4F7FC"
SERVICE_FILL = "#FBFCFF"
LINE = "#4D668F"

TITLE_SIZE = 18
STEREOTYPE_SIZE = 14
BODY_SIZE = 14
TITLE_LINE_H = 24
BODY_LINE_H = 24


@dataclass(frozen=True)
class Node:
    key: str
    x: int
    y: int
    w: int
    title: str
    attrs: tuple[str, ...]
    ops: tuple[str, ...] = ()
    stereotype: str | None = None

    @property
    def title_lines(self) -> list[str]:
        lines = [self.title]
        if self.stereotype:
            lines.append(f"<<{self.stereotype}>>")
        return lines

    @property
    def title_h(self) -> int:
        return 18 + len(self.title_lines) * TITLE_LINE_H + 18

    @property
    def attr_h(self) -> int:
        return 16 + max(len(self.attrs), 1) * BODY_LINE_H + 16

    @property
    def ops_h(self) -> int:
        return 16 + max(len(self.ops), 1) * BODY_LINE_H + 16

    @property
    def h(self) -> int:
        return self.title_h + self.attr_h + self.ops_h

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def anchor(self, side: str, offset: int = 0) -> tuple[int, int]:
        if side == "top":
            return (self.x + self.w // 2 + offset, self.y)
        if side == "bottom":
            return (self.x + self.w // 2 + offset, self.y2)
        if side == "left":
            return (self.x, self.y + self.h // 2 + offset)
        if side == "right":
            return (self.x2, self.y + self.h // 2 + offset)
        raise ValueError(f"Unsupported side: {side}")


@dataclass(frozen=True)
class Edge:
    points: tuple[tuple[int, int], ...]
    dashed: bool = False
    start_arrow: bool = False
    end_arrow: bool = True
    label: tuple[str, ...] = ()
    label_pos: tuple[int, int] | None = None


def rect(x: int, y: int, w: int, h: int, *, fill: str, stroke: str, rx: int = 12, width: int = 2) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}" />'
    )


def text(x: int, y: int, content: str, *, size: int, weight: str = "400", fill: str = INK, anchor: str = "start", italic: bool = False) -> str:
    style = "font-style:italic;" if italic else ""
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" '
        f'style="font-family:Arial,Helvetica,sans-serif;font-size:{size}px;font-weight:{weight};{style}">'
        f"{escape(content)}"
        "</text>"
    )


def multiline_text(
    x: int,
    y: int,
    lines: list[str],
    *,
    size: int,
    line_h: int,
    weight: str = "400",
    fill: str = INK,
    anchor: str = "start",
    italic_index: set[int] | None = None,
) -> str:
    italic_index = italic_index or set()
    chunks = [
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" '
        f'style="font-family:Arial,Helvetica,sans-serif;font-size:{size}px;font-weight:{weight};">'
    ]
    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else line_h
        extra = ' font-style="italic"' if idx in italic_index else ""
        chunks.append(f'<tspan x="{x}" dy="{dy}"{extra}>{escape(line)}</tspan>')
    chunks.append("</text>")
    return "".join(chunks)


def draw_node(node: Node) -> str:
    outer_fill = SERVICE_FILL if node.stereotype else BG
    parts = [rect(node.x, node.y, node.w, node.h, fill=outer_fill, stroke=BORDER)]
    parts.append(rect(node.x, node.y, node.w, node.title_h, fill=TITLE_FILL, stroke=BORDER))

    y_attr = node.y + node.title_h
    y_ops = y_attr + node.attr_h
    parts.append(f'<line x1="{node.x}" y1="{y_attr}" x2="{node.x2}" y2="{y_attr}" stroke="{BORDER}" stroke-width="2" />')
    parts.append(f'<line x1="{node.x}" y1="{y_ops}" x2="{node.x2}" y2="{y_ops}" stroke="{BORDER}" stroke-width="2" />')

    title_lines = node.title_lines
    title_x = node.x + node.w / 2
    title_y = node.y + 28
    parts.append(
        multiline_text(
            int(title_x),
            int(title_y),
            title_lines,
            size=TITLE_SIZE,
            line_h=TITLE_LINE_H,
            weight="700",
            fill=INK,
            anchor="middle",
            italic_index={1} if node.stereotype else set(),
        )
    )

    attr_lines = list(node.attrs) if node.attrs else [""]
    attr_x = node.x + 18
    attr_y = y_attr + 30
    parts.append(
        multiline_text(
            attr_x,
            attr_y,
            attr_lines,
            size=BODY_SIZE,
            line_h=BODY_LINE_H,
            fill=INK,
        )
    )

    op_lines = list(node.ops) if node.ops else [""]
    op_x = node.x + 18
    op_y = y_ops + 30
    parts.append(
        multiline_text(
            op_x,
            op_y,
            op_lines,
            size=BODY_SIZE,
            line_h=BODY_LINE_H,
            fill=MUTED if node.ops else "#9AA7BE",
            italic_index={0} if not node.ops else set(),
        )
    )
    return "".join(parts)


def polyline(points: tuple[tuple[int, int], ...], *, dashed: bool, start_arrow: bool, end_arrow: bool) -> str:
    path_data = " ".join(f"{x},{y}" for x, y in points)
    dash = ' stroke-dasharray="12 10"' if dashed else ""
    marker_start = ' marker-start="url(#arrow)"' if start_arrow else ""
    marker_end = ' marker-end="url(#arrow)"' if end_arrow else ""
    return (
        f'<polyline fill="none" stroke="{LINE}" stroke-width="3"{dash}{marker_start}{marker_end} '
        f'points="{path_data}" />'
    )


def draw_edge(edge: Edge) -> str:
    parts = [polyline(edge.points, dashed=edge.dashed, start_arrow=edge.start_arrow, end_arrow=edge.end_arrow)]
    if edge.label and edge.label_pos:
        parts.append(
            multiline_text(
                edge.label_pos[0],
                edge.label_pos[1],
                list(edge.label),
                size=13,
                line_h=18,
                fill=MUTED,
                anchor="middle",
            )
        )
    return "".join(parts)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nodes = {
        "n1": Node("n1", 120, 50, 560, "1. DocumentProcessingResult", ("- pages", "- article_segmentation")),
        "n2": Node("n2", 120, 350, 550, "2. PageProcessingResult", ("- page_number", "- has_text", "- text", "- layout_analysis", "- text_block_content")),
        "n3": Node("n3", 120, 900, 550, "3. LayoutAnalysisResponse", ("- analysis_id", "- visualization_url", "- blocks")),
        "n4": Node("n4", 120, 1370, 550, "4. LayoutBlock", ("- type", "- bbox", "- confidence", "- reading_order", "- text", "- latex")),
        "n5": Node("n5", 120, 1970, 550, "5. Block", ("- block_id", "- type", "- route_to", "- seed_text", "- seed_latex")),
        "n6": Node("n6", 960, 290, 380, "6. ProcessedBlock", ("- block_id", "- type", "- route_to", "- content", "- latex", "- confidence", "- needs_review")),
        "n7": Node("n7", 1460, 310, 280, "7. OCRResult", ("- text", "- confidence")),
        "n8": Node("n8", 1860, 310, 330, "8. FormulaOCRResult", ("- latex", "- confidence")),
        "n9": Node("n9", 1640, 820, 380, "9. PagePresentation", ("- readable_text", "- tex_preview", "- latex_document")),
        "n10": Node("n10", 1580, 1230, 440, "10. PageContent", ("- page_content_id", "- page_number", "- page_text", "- historical_text", "- normalized_text", "- needs_review_count", "- presentation", "- blocks")),
        "n11": Node("n11", 890, 1160, 500, "11. TextPostprocessResponse", ("- cleaned_text", "- corrections", "- suspicious_tokens", "- needs_review")),
        "n12": Node("n12", 780, 1970, 430, "12. HistoricalSpellingMatch", ("- source_token", "- normalized_token", "- rule_id", "- confidence")),
        "n13": Node("n13", 1480, 1950, 560, "13. OrthographyNormalizationResult", ("- historical_text", "- normalized_text", "- matches", "- ambiguous_fragments", "- needs_review")),
        "n14": Node("n14", 2280, 90, 420, "14. ArticleBoundary", ("- page_number", "- is_article_start", "- score", "- needs_review")),
        "n15": Node("n15", 2810, 90, 430, "15. ArticlePreview", ("- article_id", "- start_page", "- end_page", "- title_preview", "- boundary_confidence")),
        "n16": Node("n16", 2420, 580, 520, "16. ArticleSegmentationResult", ("- total_pages", "- article_count", "- needs_review_count", "- boundaries", "- articles")),
        "n17": Node("n17", 3020, 1380, 340, "17. LatexMetadata", ("- title", "- author", "- documentclass")),
        "n18": Node("n18", 2340, 1190, 440, "18. ArticleContent", ("- article_id", "- page_numbers", "- title", "- author", "- article_text", "- historical_text", "- normalized_text", "- pages", "- blocks", "- metadata")),
        "n19": Node("n19", 2340, 2040, 380, "19. LatexDocumentResult", ("- latex_preview", "- latex_document", "- needs_review")),
        "n20": Node("n20", 2820, 2060, 340, "20. ReadablePdfSection", ("- heading", "- note", "- blocks")),
        "n21": Node("n21", 3240, 2080, 300, "21. ReadablePdfBlock", ("- kind", "- text", "- image_data_url")),
        "n22": Node("n22", 3490, 90, 620, "22. PDFProcessor", (), ("+ process_pdf_document()", "+ iter_pdf_processing_events()"), "service"),
        "n23": Node("n23", 3490, 640, 620, "23. TextPostprocessor", (), ("+ clean_text()", "+ correct_ocr_errors()"), "service"),
        "n24": Node("n24", 3470, 1120, 650, "24. PreReformOrthographyProcessor", (), ("+ detect_historical_spelling()", "+ translate_to_modern()", "+ preserve_original_form()", "+ build_dual_text_layer()"), "service"),
        "n25": Node("n25", 3490, 1680, 620, "25. ArticleSegmenter", (), ("+ segment_document_into_articles()",), "service"),
        "n26": Node("n26", 3470, 2120, 650, "26. LatexBuilder", (), ("+ build_latex_document()",), "service"),
        "n27": Node("n27", 3490, 2390, 620, "27. ReadablePdfExporter", (), ("+ export_readable_pdf()",), "service"),
    }

    n = nodes
    edges = [
        Edge((n["n1"].anchor("bottom"), n["n2"].anchor("top")), label=("содержит",), label_pos=(400, 315)),
        Edge((n["n2"].anchor("bottom"), n["n3"].anchor("top")), end_arrow=True),
        Edge((n["n3"].anchor("bottom"), n["n4"].anchor("top")), label=("содержит",), label_pos=(395, 1340)),
        Edge((n["n4"].anchor("bottom"), n["n5"].anchor("top")), label=("преобразуется в",), label_pos=(400, 1950)),
        Edge(((670, 700), (860, 700), n["n6"].anchor("left")), dashed=True, label=("использует",), label_pos=(800, 735)),
        Edge((n["n6"].anchor("right", -65), n["n7"].anchor("left")), dashed=True, start_arrow=True, end_arrow=True, label=("связан с",), label_pos=(1405, 485)),
        Edge((n["n7"].anchor("right"), n["n8"].anchor("left")), dashed=True, start_arrow=True, end_arrow=True, label=("связан с",), label_pos=(1800, 485)),
        Edge((n["n9"].anchor("bottom"), n["n10"].anchor("top")), label=("содержит",), label_pos=(1830, 1210)),
        Edge((n["n11"].anchor("right", -10), (1520, 1450), n["n10"].anchor("left", -10)), dashed=True, end_arrow=True),
        Edge((n["n11"].anchor("top"), (1140, 1080), n["n6"].anchor("bottom")), dashed=True, label=("используется", "для очистки текста", "страницы"), label_pos=(1380, 1000)),
        Edge(((1150, 1760), (1520, 1760), n["n10"].anchor("left", 60)), dashed=True, label=("результат передается", "в модуль орфографии"), label_pos=(1290, 1720)),
        Edge((n["n5"].anchor("right"), n["n12"].anchor("left")), dashed=True, label=("преобразуется в",), label_pos=(710, 2190)),
        Edge((n["n12"].anchor("right"), n["n13"].anchor("left")), dashed=True, start_arrow=True, end_arrow=True, label=("использует",), label_pos=(1340, 2200)),
        Edge((n["n13"].anchor("top", -120), n["n10"].anchor("bottom", -40)), dashed=True, label=("обогащает",), label_pos=(1730, 1910)),
        Edge((n["n13"].anchor("right"), n["n19"].anchor("left")), dashed=True, label=("обогащает",), label_pos=(2200, 2200)),
        Edge(((1210, 2225), (1210, 2500), (2040, 2500), (2040, 2425)), dashed=True, label=("двухслойное представление текста",), label_pos=(1625, 2550)),
        Edge((n["n16"].anchor("top", -120), n["n14"].anchor("bottom")), label=("содержит",), label_pos=(2460, 540)),
        Edge((n["n16"].anchor("top", 125), n["n15"].anchor("bottom")), label=("содержит",), label_pos=(3000, 540)),
        Edge((n["n16"].anchor("bottom"), n["n18"].anchor("top")), label=("формирует",), label_pos=(2670, 1160)),
        Edge((n["n18"].anchor("right"), n["n17"].anchor("left")), label=("использует",), label_pos=(2920, 1540)),
        Edge((n["n10"].anchor("right"), n["n18"].anchor("left")), dashed=True, start_arrow=True, end_arrow=True, label=("содержит",), label_pos=(2250, 1750)),
        Edge((n["n18"].anchor("bottom", -75), n["n19"].anchor("top")), label=("агрегирует",), label_pos=(2480, 2010)),
        Edge((n["n18"].anchor("bottom", 110), n["n20"].anchor("top")), label=("агрегирует",), label_pos=(2910, 2025)),
        Edge((n["n20"].anchor("right"), n["n21"].anchor("left")), label=("содержит",), label_pos=(3200, 2190)),
        Edge(((2700, 170), (760, 170), (760, 320), n["n1"].anchor("right", -40)), dashed=True, label=(), start_arrow=False, end_arrow=True),
        Edge((n["n22"].anchor("left"), (3360, 170), n["n15"].anchor("right", -20)), dashed=True),
        Edge((n["n23"].anchor("left"), (2940, 770), n["n16"].anchor("right", 20)), dashed=True),
        Edge(((3490, 760), (3320, 760), (3320, 1420), (2780, 1420)), dashed=True),
        Edge((n["n24"].anchor("left"), (3310, 1410), (3310, 1410), (3240, 1410)), dashed=True, end_arrow=True, label=("двухслойное", "представление", "текста"), label_pos=(3330, 1260)),
        Edge((n["n24"].anchor("left", 90), (3290, 1515), (3290, 1515), n["n18"].anchor("right", -20)), dashed=True),
        Edge((n["n24"].anchor("left", 165), (3260, 1700), (3260, 1700), n["n17"].anchor("right", 0)), dashed=True),
        Edge((n["n25"].anchor("left"), (3340, 1810), (3340, 1610), n["n17"].anchor("bottom")), dashed=True),
        Edge((n["n26"].anchor("left"), (3340, 2250), (3340, 1930), (3190, 1930), n["n17"].anchor("bottom", -10)), dashed=True, label=("связан с",), label_pos=(3320, 2130)),
        Edge((n["n27"].anchor("left"), (3360, 2520), (3360, 2215), n["n21"].anchor("right")), dashed=True),
    ]

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">',
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{LINE}" />',
        "</marker>",
        "</defs>",
        rect(0, 0, W, H, fill=BG, stroke=BG, rx=0, width=0),
    ]

    for edge in edges:
        parts.append(draw_edge(edge))
    for key in [
        "n1", "n2", "n3", "n4", "n5",
        "n6", "n7", "n8", "n9", "n10", "n11", "n12", "n13",
        "n14", "n15", "n16", "n18", "n17", "n19", "n20", "n21",
        "n22", "n23", "n24", "n25", "n26", "n27",
    ]:
        parts.append(draw_node(n[key]))

    parts.append("</svg>")
    SVG_PATH.write_text("".join(parts), encoding="utf-8")
    print(SVG_PATH)


if __name__ == "__main__":
    main()
