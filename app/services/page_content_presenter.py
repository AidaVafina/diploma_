from __future__ import annotations

import re
from typing import Iterable
from uuid import uuid4

from app.core.config import settings
from app.schemas import PageContent, PagePresentation, ProcessedBlock

TEXTUAL_BLOCK_TYPES = {"text", "title", "header", "footer", "page_number"}
TEX_ESCAPE_RE = re.compile(r"([\\&%$#_{}])")
MATH_TAG_OPEN_RE = re.compile(r"<math\b[^>]*>", re.IGNORECASE)
MATH_TAG_CLOSE_RE = re.compile(r"</math>", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
MULTIBREAK_RE = re.compile(r"\n{3,}")


def sort_processed_blocks(blocks: Iterable[ProcessedBlock]) -> list[ProcessedBlock]:
    return sorted(
        list(blocks),
        key=lambda block: (block.reading_order, block.bbox[1], block.bbox[0], block.block_id),
    )


def _normalize_preview_text(text: str) -> str:
    cleaned = text.replace("\r", "\n").strip()
    cleaned = MULTISPACE_RE.sub(" ", cleaned)
    cleaned = MULTIBREAK_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def _strip_math_wrappers(latex: str | None) -> str:
    if not latex:
        return ""
    cleaned = MATH_TAG_OPEN_RE.sub("", latex)
    cleaned = MATH_TAG_CLOSE_RE.sub("", cleaned)
    cleaned = cleaned.replace("<mrow>", "").replace("</mrow>", "")
    cleaned = cleaned.replace("<mi>", "").replace("</mi>", "")
    cleaned = cleaned.replace("<mn>", "").replace("</mn>", "")
    cleaned = cleaned.replace("<mo>", "").replace("</mo>", "")
    cleaned = cleaned.replace("<msub>", "").replace("</msub>", "")
    cleaned = cleaned.replace("<msup>", "").replace("</msup>", "")
    cleaned = cleaned.replace("<msubsup>", "").replace("</msubsup>", "")
    cleaned = cleaned.replace("<mfrac>", "").replace("</mfrac>", "")
    return _normalize_preview_text(cleaned)


def _escape_tex_text(text: str) -> str:
    escaped = TEX_ESCAPE_RE.sub(r"\\\1", text)
    escaped = escaped.replace("~", r"\textasciitilde{}")
    escaped = escaped.replace("^", r"\textasciicircum{}")
    return escaped


def build_page_text(blocks: Iterable[ProcessedBlock]) -> str:
    page_parts: list[str] = []

    for block in sort_processed_blocks(blocks):
        if block.route_to == "text_pipeline" and block.content:
            page_parts.append(block.content)
        elif block.type == "formula":
            page_parts.append(settings.text_block_formula_placeholder)

    return "\n".join(part for part in page_parts if part).strip()


def build_readable_text(blocks: Iterable[ProcessedBlock]) -> str:
    parts: list[str] = []

    for block in sort_processed_blocks(blocks):
        if block.type in TEXTUAL_BLOCK_TYPES and block.content:
            parts.append(block.content)
        elif block.type == "formula":
            parts.append(settings.text_block_formula_placeholder)
        elif block.type == "table":
            parts.append("[TABLE]")
        elif block.type == "image":
            parts.append("[IMAGE]")

    return _normalize_preview_text("\n\n".join(part for part in parts if part))


def build_tex_preview(blocks: Iterable[ProcessedBlock]) -> str:
    parts: list[str] = []

    for block in sort_processed_blocks(blocks):
        if block.type in TEXTUAL_BLOCK_TYPES and block.content:
            parts.append(_escape_tex_text(block.content))
            continue

        if block.type == "formula":
            raw_latex = block.latex
            if not raw_latex and block.formula_result is not None:
                raw_latex = block.formula_result.latex
            latex_body = _strip_math_wrappers(raw_latex)
            if latex_body:
                parts.append("\\[\n" + latex_body + "\n\\]")
            else:
                parts.append(f"% TODO: formula {block.block_id} requires review")
            continue

        if block.type == "table":
            parts.append(f"% [TABLE {block.block_id}]")
            continue

        if block.type == "image":
            parts.append(f"% [IMAGE {block.block_id}]")

    return _normalize_preview_text("\n\n".join(part for part in parts if part))


def build_page_content(
    blocks: list[ProcessedBlock],
    *,
    page_number: int | None = None,
    page_content_id: str | None = None,
) -> PageContent:
    ordered_blocks = sort_processed_blocks(blocks)
    page_text = build_page_text(ordered_blocks)
    presentation = PagePresentation(
        default_view="readable",
        readable_text=build_readable_text(ordered_blocks),
        tex_preview=build_tex_preview(ordered_blocks),
    )
    needs_review_count = sum(1 for block in ordered_blocks if block.needs_review)

    return PageContent(
        page_content_id=page_content_id or uuid4().hex,
        page_number=page_number,
        page_text=page_text,
        needs_review_count=needs_review_count,
        presentation=presentation,
        blocks=ordered_blocks,
    )
