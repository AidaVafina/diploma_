from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Iterable

from app.schemas import (
    ArticleContent,
    LatexBuildResult,
    LatexDocumentResult,
    LatexMetadata,
    PageContent,
    ProcessedBlock,
)

TEXTUAL_BLOCK_TYPES = {"text", "title", "header", "footer"}
MATH_TAG_RE = re.compile(r"<math\b[^>]*>(.*?)</math>", re.IGNORECASE | re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
MULTIBREAK_RE = re.compile(r"\n{3,}")
SAFE_DOCUMENTCLASS_RE = re.compile(r"^[A-Za-z0-9_-]+$")
AUTHOR_INITIALS_RE = re.compile(
    r"\b(?:[A-ZА-ЯЁ]\.\s*[A-ZА-ЯЁ]\.|"
    r"[A-ZА-ЯЁ]\.\s*(?:[A-ZА-ЯЁ]\.\s*)?[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё-]+)"
)
AUTHOR_NAME_TOKEN_RE = re.compile(r"\b[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё-]+\b")
AUTHOR_ALLOWED_TEXT_RE = re.compile(r"^[A-Za-zА-Яа-яЁё.\-\s()]+$")

LATEX_TEXT_ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


@dataclass(frozen=True)
class ArticleLatexAssembly:
    body: str
    metadata: LatexMetadata
    latex_preview: str
    latex_document: str
    needs_review: bool


@dataclass(frozen=True)
class _ArticleHead:
    title: str | None
    author: str | None
    title_block_id: str | None = None
    author_block_id: str | None = None


def sort_processed_blocks(blocks: Iterable[ProcessedBlock]) -> list[ProcessedBlock]:
    return sorted(
        list(blocks),
        key=lambda block: (
            block.reading_order,
            block.bbox[1],
            block.bbox[0],
            block.block_id,
        ),
    )


def _strip_html_tags(text: str) -> str:
    return HTML_TAG_RE.sub("", text)


def _normalize_plain_text(text: str) -> str:
    normalized = html.unescape(text or "")
    normalized = normalized.replace("\u00a0", " ").replace("\ufeff", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _strip_html_tags(normalized)
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n[ \t]+", "\n", normalized)
    normalized = MULTISPACE_RE.sub(" ", normalized)
    normalized = MULTIBREAK_RE.sub("\n\n", normalized)
    return normalized


def _normalize_compare_text(text: str | None) -> str:
    normalized = _normalize_plain_text(text or "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().casefold()


def _first_nonempty_line(text: str | None) -> str:
    normalized = _normalize_plain_text(text or "")
    for line in normalized.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _nonempty_lines(text: str | None) -> list[str]:
    normalized = _normalize_plain_text(text or "")
    return [line.strip() for line in normalized.splitlines() if line.strip()]


def _escape_latex_text(text: str) -> str:
    return "".join(LATEX_TEXT_ESCAPE_MAP.get(char, char) for char in text)


def _strip_latex_wrappers(latex: str) -> str:
    stripped = latex.strip()
    wrappers = [
        (r"\[", r"\]"),
        ("$$", "$$"),
        (r"\(", r"\)"),
        ("$", "$"),
    ]

    changed = True
    while changed:
        changed = False
        for left, right in wrappers:
            if stripped.startswith(left) and stripped.endswith(right):
                stripped = stripped[len(left) : len(stripped) - len(right)].strip()
                changed = True
    return stripped


def clean_latex(latex: str | None) -> str:
    if not latex:
        return ""

    cleaned = html.unescape(str(latex))
    cleaned = cleaned.replace("\u00a0", " ").replace("\ufeff", "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = MATH_TAG_RE.sub(lambda match: match.group(1), cleaned)
    cleaned = _strip_html_tags(cleaned)
    cleaned = _strip_latex_wrappers(cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = MULTISPACE_RE.sub(" ", cleaned)
    cleaned = MULTIBREAK_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def _clean_inline_math(match: re.Match[str]) -> str:
    latex = clean_latex(match.group(1))
    return f"${latex}$" if latex else ""


def text_to_latex(text: str | None) -> str:
    if not text:
        return ""

    source = html.unescape(str(text))
    parts: list[str] = []
    cursor = 0

    for match in MATH_TAG_RE.finditer(source):
        plain_part = _normalize_plain_text(source[cursor : match.start()])
        if plain_part:
            parts.append(_escape_latex_text(plain_part))
        parts.append(_clean_inline_math(match))
        cursor = match.end()

    tail = _normalize_plain_text(source[cursor:])
    if tail:
        parts.append(_escape_latex_text(tail))

    result = "".join(parts)
    result = MULTISPACE_RE.sub(" ", result)
    result = MULTIBREAK_RE.sub("\n\n", result)
    return result.strip()


def _block_latex_source(block: ProcessedBlock) -> str:
    if block.latex:
        return block.latex
    if block.formula_result and block.formula_result.latex:
        return block.formula_result.latex
    return ""


def _block_comment(kind: str, block: ProcessedBlock) -> str:
    return f"% TODO: {kind} {block.block_id}"


def build_blocks_latex_body(
    blocks: Iterable[ProcessedBlock],
    *,
    skip_block_ids: set[str] | None = None,
    skip_text_values: set[str] | None = None,
) -> tuple[str, bool]:
    parts: list[str] = []
    needs_review = False
    skip_block_ids = skip_block_ids or set()
    skip_text_values = skip_text_values or set()

    for block in sort_processed_blocks(blocks):
        if block.block_id in skip_block_ids:
            continue

        if block.needs_review:
            needs_review = True

        if block.type in TEXTUAL_BLOCK_TYPES and block.content:
            if _normalize_compare_text(block.content) in skip_text_values:
                continue
            latex_text = text_to_latex(block.content)
            if latex_text:
                parts.append(latex_text)
            continue

        if block.type == "page_number":
            continue

        if block.type == "formula":
            latex_body = clean_latex(_block_latex_source(block))
            if latex_body:
                parts.append("\\[\n" + latex_body + "\n\\]")
            else:
                parts.append(_block_comment("FORMULA", block))
                needs_review = True
            continue

        if block.type == "table":
            parts.append(_block_comment("TABLE", block))
            continue

        if block.type == "image":
            parts.append(_block_comment("IMAGE", block))

    return "\n\n".join(part for part in parts if part).strip(), needs_review


def build_latex_document(
    body: str,
    metadata: LatexMetadata | None = None,
) -> str:
    metadata = metadata or LatexMetadata()
    documentclass = metadata.documentclass or "article"
    if not SAFE_DOCUMENTCLASS_RE.fullmatch(documentclass):
        documentclass = "article"

    lines = [
        f"\\documentclass{{{documentclass}}}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T2A]{fontenc}",
        r"\usepackage[russian,english]{babel}",
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
    ]

    has_title = bool(metadata.title and metadata.title.strip())
    has_author = bool(metadata.author and metadata.author.strip())
    if has_title:
        lines.append(f"\\title{{{text_to_latex(metadata.title)}}}")
    if has_author:
        lines.append(f"\\author{{{text_to_latex(metadata.author)}}}")

    lines.append("")
    lines.append(r"\begin{document}")

    if has_title or has_author:
        lines.append("")
        lines.append(r"\maketitle")
        lines.append("")

    if body.strip():
        lines.append(body.strip())
        lines.append("")

    lines.append(r"\end{document}")
    return "\n".join(lines)


def build_page_latex_body(page_content: PageContent) -> str:
    body, _ = build_blocks_latex_body(page_content.blocks)
    return body


def build_page_latex_preview(page_content: PageContent) -> str:
    return build_page_latex_body(page_content)


def build_page_latex(page_content: PageContent) -> LatexBuildResult:
    latex_preview, needs_review = build_blocks_latex_body(page_content.blocks)
    latex_document = build_latex_document(latex_preview)
    return LatexBuildResult(
        latex_preview=latex_preview,
        latex_document=latex_document,
        needs_review=needs_review,
    )


def _resolve_article_metadata(
    article_content: ArticleContent,
    metadata: LatexMetadata | None,
) -> LatexMetadata:
    resolved = metadata or article_content.metadata or LatexMetadata()
    title = resolved.title or article_content.title
    author = resolved.author or article_content.author
    return resolved.model_copy(
        update={
            "title": title,
            "author": author,
            "documentclass": resolved.documentclass or "article",
        }
    )


def _block_text(block: ProcessedBlock | None) -> str:
    return ((block.content or block.latex or "") if block else "").strip()


def _ordered_article_blocks(article_content: ArticleContent) -> list[ProcessedBlock]:
    if article_content.pages:
        blocks: list[ProcessedBlock] = []
        for page in sorted(
            article_content.pages,
            key=lambda page: (
                page.page_number if page.page_number is not None else 10**9,
                page.page_content_id,
            ),
        ):
            blocks.extend(sort_processed_blocks(page.blocks))
        return blocks

    if article_content.structured_content:
        return sort_processed_blocks(article_content.structured_content)

    if article_content.blocks:
        return sort_processed_blocks(article_content.blocks)

    return []


def _is_author_like(text: str | None) -> bool:
    candidate = _first_nonempty_line(text)
    if not candidate or len(candidate) > 120:
        return False
    if "<" in candidate or ">" in candidate:
        return False
    if re.search(r"\d", candidate):
        return False
    if not AUTHOR_ALLOWED_TEXT_RE.fullmatch(candidate):
        return False

    if AUTHOR_INITIALS_RE.search(candidate):
        return True

    tokens = AUTHOR_NAME_TOKEN_RE.findall(candidate)
    if 2 <= len(tokens) <= 6:
        return True

    return False


def _find_first_title_block(
    blocks: list[ProcessedBlock],
) -> tuple[int | None, ProcessedBlock | None]:
    for index, block in enumerate(blocks):
        if block.type == "title" and _block_text(block):
            return index, block
    return None, None


def _find_block_by_text(
    blocks: list[ProcessedBlock],
    text: str | None,
) -> tuple[int | None, ProcessedBlock | None]:
    target = _normalize_compare_text(text)
    if not target:
        return None, None

    for index, block in enumerate(blocks):
        if block.type not in TEXTUAL_BLOCK_TYPES:
            continue
        if _normalize_compare_text(_block_text(block)) == target:
            return index, block

    return None, None


def _find_author_block_after_title(
    blocks: list[ProcessedBlock],
    title_index: int | None,
) -> tuple[ProcessedBlock | None, str | None]:
    if title_index is None:
        return None, None

    for block in blocks[title_index + 1 :]:
        if block.type in {"page_number", "header", "footer"}:
            continue

        if block.type != "text":
            return None, None

        text = _first_nonempty_line(_block_text(block))
        if not text:
            continue

        if _is_author_like(text):
            return block, text
        return None, None

    return None, None


def _extract_author_from_article_text(
    article_text: str,
    title: str | None,
) -> str | None:
    lines = _nonempty_lines(article_text)
    if not lines:
        return None

    title_norm = _normalize_compare_text(title)
    for line in lines[:3]:
        if title_norm and _normalize_compare_text(line) == title_norm:
            continue
        if _is_author_like(line):
            return line
        break

    return None


def _extract_article_head(
    article_content: ArticleContent,
    metadata: LatexMetadata | None,
) -> _ArticleHead:
    blocks = _ordered_article_blocks(article_content)
    incoming_metadata = metadata or article_content.metadata or LatexMetadata()

    title_block_index, title_block = _find_first_title_block(blocks)
    title_block_text = _block_text(title_block) if title_block else ""

    title = (
        incoming_metadata.title
        or title_block_text
        or article_content.title
        or _first_nonempty_line(article_content.article_text)
        or None
    )

    if title_block is None and title:
        title_block_index, title_block = _find_block_by_text(blocks, title)

    author_block = None
    extracted_author = None
    if title_block is not None:
        author_block, extracted_author = _find_author_block_after_title(
            blocks,
            title_block_index,
        )

    author = (
        incoming_metadata.author
        or article_content.author
        or extracted_author
        or _extract_author_from_article_text(article_content.article_text, title)
        or None
    )

    if author_block is None and author:
        _, author_block = _find_block_by_text(blocks, author)

    return _ArticleHead(
        title=title,
        author=author,
        title_block_id=title_block.block_id if title_block else None,
        author_block_id=author_block.block_id if author_block else None,
    )


def _article_skip_sets(head: _ArticleHead) -> tuple[set[str], set[str]]:
    skip_block_ids = {
        block_id
        for block_id in (head.title_block_id, head.author_block_id)
        if block_id
    }
    skip_text_values = {
        normalized
        for normalized in (
            _normalize_compare_text(head.title),
            _normalize_compare_text(head.author),
        )
        if normalized
    }
    return skip_block_ids, skip_text_values


def _article_text_body_without_head(
    article_text: str,
    head: _ArticleHead,
) -> tuple[str, bool]:
    skip_text_values = _article_skip_sets(head)[1]
    body_lines = [
        line
        for line in _nonempty_lines(article_text)
        if _normalize_compare_text(line) not in skip_text_values
    ]
    return text_to_latex("\n\n".join(body_lines)), False


def _article_body_from_content(
    article_content: ArticleContent,
    head: _ArticleHead,
) -> tuple[str, bool]:
    body_parts: list[str] = []
    needs_review = False
    skip_block_ids, skip_text_values = _article_skip_sets(head)

    if article_content.pages:
        for page in sorted(
            article_content.pages,
            key=lambda page: (
                page.page_number if page.page_number is not None else 10**9,
                page.page_content_id,
            ),
        ):
            page_body, page_needs_review = build_blocks_latex_body(
                page.blocks,
                skip_block_ids=skip_block_ids,
                skip_text_values=skip_text_values,
            )
            if page_body:
                body_parts.append(page_body)
            needs_review = needs_review or page_needs_review
        return "\n\n".join(body_parts).strip(), needs_review

    if article_content.structured_content:
        return build_blocks_latex_body(
            article_content.structured_content,
            skip_block_ids=skip_block_ids,
            skip_text_values=skip_text_values,
        )

    if article_content.blocks:
        return build_blocks_latex_body(
            article_content.blocks,
            skip_block_ids=skip_block_ids,
            skip_text_values=skip_text_values,
        )

    if article_content.article_text:
        return _article_text_body_without_head(article_content.article_text, head)

    return "", False


def _build_article_preview(body: str, metadata: LatexMetadata) -> str:
    parts: list[str] = []
    if metadata.title:
        parts.append(f"\\title{{{text_to_latex(metadata.title)}}}")
    if metadata.author:
        parts.append(f"\\author{{{text_to_latex(metadata.author)}}}")
    if metadata.title or metadata.author:
        parts.append(r"\maketitle")
    if body:
        parts.append(body)
    return "\n\n".join(parts).strip()


def assemble_article_latex(
    article_content: ArticleContent,
    metadata: LatexMetadata | None = None,
) -> ArticleLatexAssembly:
    head = _extract_article_head(article_content, metadata)
    resolved_metadata = _resolve_article_metadata(article_content, metadata).model_copy(
        update={
            "title": head.title,
            "author": head.author,
        }
    )
    body, needs_review = _article_body_from_content(article_content, head)
    latex_preview = _build_article_preview(body, resolved_metadata)
    latex_document = build_latex_document(body, resolved_metadata)
    return ArticleLatexAssembly(
        body=body,
        metadata=resolved_metadata,
        latex_preview=latex_preview,
        latex_document=latex_document,
        needs_review=needs_review,
    )


def build_article_latex(
    article_content: ArticleContent,
    metadata: LatexMetadata | None = None,
) -> LatexDocumentResult:
    assembly = assemble_article_latex(article_content, metadata)
    return LatexDocumentResult(
        latex_preview=assembly.latex_preview,
        latex_document=assembly.latex_document,
        needs_review=assembly.needs_review,
        metadata=assembly.metadata,
    )
