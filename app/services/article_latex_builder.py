from __future__ import annotations

import logging
from collections import OrderedDict

from app.schemas import (
    ArticleContent,
    ArticleLatexDocumentResult,
    ArticleLatexPreviewResult,
    LatexMetadata,
    PageContent,
)
from app.services.latex_builder import assemble_article_latex

logger = logging.getLogger(__name__)

_ARTICLE_LATEX_CACHE: OrderedDict[str, ArticleLatexDocumentResult] = OrderedDict()
_MAX_ARTICLE_LATEX_CACHE_ITEMS = 32


class ArticleLatexNotFoundError(RuntimeError):
    pass


def _cache_article_latex(result: ArticleLatexDocumentResult) -> None:
    if not result.article_id:
        return

    _ARTICLE_LATEX_CACHE[result.article_id] = result
    _ARTICLE_LATEX_CACHE.move_to_end(result.article_id)

    while len(_ARTICLE_LATEX_CACHE) > _MAX_ARTICLE_LATEX_CACHE_ITEMS:
        _ARTICLE_LATEX_CACHE.popitem(last=False)


def get_cached_article_latex(article_id: str) -> ArticleLatexDocumentResult:
    result = _ARTICLE_LATEX_CACHE.get(article_id)
    if result is None:
        raise ArticleLatexNotFoundError(
            f"LaTeX-документ статьи {article_id} не найден или уже очищен из памяти."
        )
    _ARTICLE_LATEX_CACHE.move_to_end(article_id)
    return result


def _page_numbers_from_pages(pages: list[PageContent]) -> list[int]:
    return [
        page.page_number
        for page in sorted(
            pages,
            key=lambda page: (
                page.page_number if page.page_number is not None else 10**9,
                page.page_content_id,
            ),
        )
        if page.page_number is not None
    ]


def _resolve_page_numbers(article_content: ArticleContent) -> list[int]:
    if article_content.page_numbers:
        return article_content.page_numbers
    if article_content.pages:
        return _page_numbers_from_pages(article_content.pages)
    return []


def build_article_latex_preview(
    article_content: ArticleContent,
    metadata: LatexMetadata | None = None,
) -> ArticleLatexPreviewResult:
    assembly = assemble_article_latex(article_content, metadata)
    return ArticleLatexPreviewResult(
        article_id=article_content.article_id,
        page_numbers=_resolve_page_numbers(article_content),
        latex_preview=assembly.latex_preview,
        needs_review=assembly.needs_review,
        metadata=assembly.metadata,
    )


def build_article_latex_document(
    article_content: ArticleContent,
    metadata: LatexMetadata | None = None,
) -> ArticleLatexDocumentResult:
    assembly = assemble_article_latex(article_content, metadata)

    result = ArticleLatexDocumentResult(
        article_id=article_content.article_id,
        page_numbers=_resolve_page_numbers(article_content),
        latex_preview=assembly.latex_preview,
        latex_document=assembly.latex_document,
        needs_review=assembly.needs_review,
        metadata=assembly.metadata,
    )
    _cache_article_latex(result)
    logger.info("Built article-level LaTeX for article_id=%s", result.article_id)
    return result
