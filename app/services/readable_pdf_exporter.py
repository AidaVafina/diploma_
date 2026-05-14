from __future__ import annotations

import html
import os
import re
import tempfile
from pathlib import Path

import fitz

from app.schemas import ReadablePdfBlock, ReadablePdfExportRequest, ReadablePdfSection

A4_RECT = fitz.paper_rect("a4")
PAGE_MARGIN = 42
CONTENT_RECT = fitz.Rect(
    PAGE_MARGIN,
    PAGE_MARGIN,
    A4_RECT.width - PAGE_MARGIN,
    A4_RECT.height - PAGE_MARGIN,
)
FILENAME_SANITIZE_RE = re.compile(r'[\\/:*?"<>|]+')
MULTISPACE_RE = re.compile(r"\s+")
DEFAULT_FILENAME = "readable-text.pdf"
PDF_CSS = """
body {
  font-family: sans-serif;
  font-size: 11pt;
  line-height: 1.55;
  color: #1f2d33;
}
h1, h2, h3 {
  margin: 0 0 10pt 0;
  color: #0d6a73;
}
h1 {
  font-size: 22pt;
}
h2 {
  font-size: 15pt;
}
h3 {
  font-size: 13pt;
}
p {
  margin: 0 0 10pt 0;
}
.cover {
  margin-top: 48pt;
}
.cover-subtitle,
.section-note,
.author {
  color: #5a6b73;
}
.cover-subtitle {
  font-size: 12pt;
}
.section-note {
  font-size: 10pt;
}
.author {
  font-style: italic;
  margin-bottom: 14pt;
}
.formula {
  margin: 4pt 0 8pt 0;
  text-align: center;
}
.formula__image {
  display: block;
  margin: 0 auto;
  max-width: 100%;
  height: auto;
}
.placeholder {
  margin: 10pt 0 14pt 0;
  padding: 8pt 10pt;
  border: 1px solid #d4dcdf;
  background: #f6f8f9;
  color: #5a6b73;
  font-weight: bold;
}
"""


class ReadablePdfExportError(RuntimeError):
    pass


def resolve_readable_pdf_filename(
    raw_filename: str | None,
    fallback_title: str = "readable-text",
) -> str:
    filename = (raw_filename or fallback_title or "").strip()
    filename = filename.replace("\x00", "")
    filename = FILENAME_SANITIZE_RE.sub("_", filename)
    filename = MULTISPACE_RE.sub(" ", filename).strip(" ._")
    if not filename:
        filename = Path(DEFAULT_FILENAME).stem
    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"
    return filename


def _escape_html(text: str) -> str:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    return html.escape(normalized, quote=True).replace("\n", "<br />")


def _has_visible_section_content(section: ReadablePdfSection) -> bool:
    return any(
        (block.text or "").strip() or (block.image_data_url or "").strip()
        for block in section.blocks
    )


def _build_block_html(block: ReadablePdfBlock) -> str:
    text = (block.text or "").strip()
    image_data_url = (block.image_data_url or "").strip()
    if not text and not image_data_url:
        return ""

    if block.kind == "title":
        return f"<h3>{_escape_html(text)}</h3>"
    if block.kind == "author":
        return f"<p class='author'>{_escape_html(text)}</p>"
    if block.kind == "formula":
        if image_data_url:
            width_attr = (
                f" width='{int(block.image_width)}'"
                if block.image_width is not None
                else ""
            )
            height_attr = (
                f" height='{int(block.image_height)}'"
                if block.image_height is not None
                else ""
            )
            return (
                "<div class='formula'>"
                f"<img class='formula__image' src='{html.escape(image_data_url, quote=True)}'"
                f"{width_attr}{height_attr} alt='formula' />"
                "</div>"
            )
        return f"<div class='formula'>{_escape_html(text)}</div>"
    if block.kind == "placeholder":
        return f"<div class='placeholder'>{_escape_html(text)}</div>"
    return f"<p>{_escape_html(text)}</p>"


def _build_section_html(section: ReadablePdfSection) -> str:
    parts = ["<html><body><section>"]
    block_html = "".join(_build_block_html(block) for block in section.blocks)
    if not block_html:
        block_html = "<div class='placeholder'>Текст недоступен.</div>"

    parts.append(block_html)
    parts.extend(["</section>", "</body></html>"])
    return "".join(parts)


def _render_story(writer: fitz.DocumentWriter, html_text: str) -> None:
    story = fitz.Story(html=html_text, user_css=PDF_CSS)
    more = 1
    while more:
        device = writer.begin_page(A4_RECT)
        more, _ = story.place(CONTENT_RECT)
        story.draw(device)
        writer.end_page()


def export_readable_pdf(payload: ReadablePdfExportRequest) -> bytes:
    sections = [section for section in payload.sections if _has_visible_section_content(section)]
    if not sections:
        raise ReadablePdfExportError("Нет текста для экспорта в PDF.")

    tmp_path: str | None = None
    writer: fitz.DocumentWriter | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_path = tmp_file.name

        writer = fitz.DocumentWriter(tmp_path)
        for section in sections:
            _render_story(writer, _build_section_html(section))
        writer.close()
        writer = None
        return Path(tmp_path).read_bytes()
    except Exception as exc:  # pragma: no cover - defensive branch
        raise ReadablePdfExportError("Не удалось сформировать PDF с читаемым текстом.") from exc
    finally:
        if writer is not None:
            writer.close()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
