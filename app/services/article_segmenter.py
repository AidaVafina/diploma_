from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.schemas import (
    ArticleContent,
    ArticleBoundary,
    ArticlePreview,
    ArticleSegmentationResult,
    PageContent,
    ProcessedBlock,
)
from app.services.article_latex_builder import build_article_latex_document

logger = logging.getLogger(__name__)

SENTENCE_END_RE = re.compile(r"[.!?\u2026:;,][\"'\u00bb\u201d)\]]*\s*$")
PLACEHOLDER_RE = re.compile(r"<<(?:FORMULA|TABLE|IMAGE)_[A-Za-z0-9_.-]+>>")
MULTISPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ArticleStartFeatures:
    page_number: int
    first_text_block: ProcessedBlock | None
    first_text_top_gap_ratio: float
    first_text_has_top_gap: bool
    previous_page_end_punctuation: bool
    title_preview: str


class ArticleSegmenter:
    def __init__(
        self,
        *,
        top_gap_ratio: float = 0.10,
    ) -> None:
        self.top_gap_ratio = top_gap_ratio

    def _page_height(self, page_content: PageContent) -> int:
        max_y = 1
        for block in page_content.blocks:
            if len(block.bbox) == 4:
                max_y = max(max_y, int(block.bbox[3]))
        return max_y

    def _sorted_blocks(self, page_content: PageContent) -> list[ProcessedBlock]:
        return sorted(
            list(page_content.blocks),
            key=lambda block: (
                block.reading_order,
                block.bbox[1],
                block.bbox[0],
                block.block_id,
            ),
        )

    def _block_text(self, block: ProcessedBlock | None) -> str:
        return ((block.content or block.latex or "") if block else "").strip()

    def _first_text_block(self, page_content: PageContent) -> ProcessedBlock | None:
        for block in self._sorted_blocks(page_content):
            if block.type != "text":
                continue
            if self._block_text(block):
                return block
        return None

    def _clean_page_end_text(self, text: str) -> str:
        cleaned = PLACEHOLDER_RE.sub("", text or "").strip()
        return MULTISPACE_RE.sub(" ", cleaned)

    def _page_text(self, page_content: PageContent) -> str:
        if page_content.page_text:
            return page_content.page_text
        if page_content.presentation and page_content.presentation.readable_text:
            return page_content.presentation.readable_text
        return "\n".join(
            self._block_text(block)
            for block in self._sorted_blocks(page_content)
            if self._block_text(block)
        )

    def _previous_page_ends_with_punctuation(
        self,
        previous_page: PageContent | None,
    ) -> bool:
        if previous_page is None:
            return False

        previous_text = self._clean_page_end_text(self._page_text(previous_page))
        if not previous_text:
            return False

        return bool(SENTENCE_END_RE.search(previous_text))

    def extract_page_features(
        self,
        page_content: PageContent,
        previous_page: PageContent | None,
    ) -> ArticleStartFeatures:
        page_height = self._page_height(page_content)
        first_text_block = self._first_text_block(page_content)
        top_gap_ratio = 0.0

        if first_text_block is not None:
            top_gap_ratio = first_text_block.bbox[1] / max(1, page_height)

        return ArticleStartFeatures(
            page_number=page_content.page_number or 0,
            first_text_block=first_text_block,
            first_text_top_gap_ratio=top_gap_ratio,
            first_text_has_top_gap=top_gap_ratio >= self.top_gap_ratio,
            previous_page_end_punctuation=self._previous_page_ends_with_punctuation(
                previous_page
            ),
            title_preview=self._block_text(first_text_block),
        )

    def score_article_start(
        self,
        page_content: PageContent,
        previous_page: PageContent | None,
    ) -> tuple[float, dict[str, object], ArticleStartFeatures]:
        features = self.extract_page_features(page_content, previous_page)

        if previous_page is None:
            debug_info = {
                "first_text_has_top_gap": features.first_text_has_top_gap,
                "first_text_top_gap_ratio": round(features.first_text_top_gap_ratio, 3),
                "previous_page_end_punctuation": features.previous_page_end_punctuation,
                "decision": "article_start",
                "final_score": 1.0,
            }
            return 1.0, debug_info, features

        score = 0.0
        if features.first_text_has_top_gap:
            score += 0.75
        if features.previous_page_end_punctuation:
            score += 0.25

        debug_info = {
            "first_text_has_top_gap": features.first_text_has_top_gap,
            "first_text_top_gap_ratio": round(features.first_text_top_gap_ratio, 3),
            "previous_page_end_punctuation": features.previous_page_end_punctuation,
            "decision": "continue_article",
            "final_score": score,
        }
        return score, debug_info, features

    def is_article_start(
        self,
        page_content: PageContent,
        previous_page: PageContent | None,
    ) -> tuple[ArticleBoundary, ArticleStartFeatures]:
        score, debug_info, features = self.score_article_start(
            page_content,
            previous_page,
        )

        if previous_page is None:
            return (
                ArticleBoundary(
                    page_number=page_content.page_number or 1,
                    is_article_start=True,
                    score=score,
                    needs_review=False,
                    debug_info=debug_info,
                ),
                features,
            )

        is_start = (
            features.first_text_block is not None
            and features.first_text_has_top_gap
        )
        needs_review = (
            (is_start and not features.previous_page_end_punctuation)
            or (not is_start and features.previous_page_end_punctuation)
        )

        if is_start:
            debug_info["decision"] = (
                "article_start_review"
                if needs_review
                else "article_start"
            )
        elif needs_review:
            debug_info["decision"] = "review_boundary"
        else:
            debug_info["decision"] = "continue_article"

        return (
            ArticleBoundary(
                page_number=page_content.page_number or 1,
                is_article_start=is_start,
                score=score,
                needs_review=needs_review,
                debug_info=debug_info,
            ),
            features,
        )

    def _resolve_article_title(
        self,
        page_content: PageContent,
        features: ArticleStartFeatures,
    ) -> str:
        if features.title_preview:
            return features.title_preview[:240]

        first_line = next(
            (
                line.strip()
                for line in self._page_text(page_content).splitlines()
                if line.strip()
            ),
            "",
        )
        return first_line[:240]

    def _readable_page_text(self, page_content: PageContent) -> str:
        if page_content.presentation and page_content.presentation.readable_text:
            return page_content.presentation.readable_text.strip()
        return self._page_text(page_content).strip()

    def _build_article_text(self, article_pages: list[PageContent]) -> str:
        parts: list[str] = []
        for page in article_pages:
            page_text = self._readable_page_text(page)
            if page_text:
                parts.append(page_text)
        return "\n\n".join(parts).strip()

    def build_article_preview(
        self,
        article_index: int,
        article_pages: list[PageContent],
        start_boundary: ArticleBoundary,
        start_features: ArticleStartFeatures,
    ) -> ArticlePreview:
        page_numbers = [
            page.page_number for page in article_pages if page.page_number is not None
        ]
        title_preview = (
            (article_pages[0].article_title if article_pages else "")
            or start_features.title_preview
        )

        article_id = f"article_{article_index:03d}"
        article_latex = build_article_latex_document(
            ArticleContent(
                article_id=article_id,
                page_numbers=page_numbers,
                title=title_preview[:240],
                pages=article_pages,
            )
        )

        return ArticlePreview(
            article_id=article_id,
            start_page=page_numbers[0] if page_numbers else article_index,
            end_page=page_numbers[-1] if page_numbers else article_index,
            page_numbers=page_numbers,
            title_preview=title_preview[:240],
            needs_review=start_boundary.needs_review,
            boundary_confidence=start_boundary.score,
            article_text=self._build_article_text(article_pages),
            article_latex_preview=article_latex.latex_preview,
            article_latex_document=article_latex.latex_document,
            debug_info=start_boundary.debug_info,
        )

    def group_pages_into_articles(
        self,
        pages: list[PageContent],
    ) -> ArticleSegmentationResult:
        ordered_pages = sorted(
            list(pages),
            key=lambda page: (
                page.page_number if page.page_number is not None else 10**9,
                page.page_content_id,
            ),
        )
        if not ordered_pages:
            return ArticleSegmentationResult(
                total_pages=0,
                article_count=0,
                needs_review_count=0,
            )

        boundaries: list[ArticleBoundary] = []
        articles: list[ArticlePreview] = []

        current_pages: list[PageContent] = []
        current_start_boundary: ArticleBoundary | None = None
        current_start_features: ArticleStartFeatures | None = None
        current_article_title: str | None = None

        previous_page: PageContent | None = None
        for page in ordered_pages:
            boundary, features = self.is_article_start(page, previous_page)

            if not current_pages or boundary.is_article_start:
                if current_pages and current_start_boundary and current_start_features:
                    articles.append(
                        self.build_article_preview(
                            len(articles) + 1,
                            current_pages,
                            current_start_boundary,
                            current_start_features,
                        )
                    )
                current_article_title = self._resolve_article_title(page, features) or None
                current_pages = [page]
                current_start_boundary = boundary
                current_start_features = features
            else:
                current_pages.append(page)

            page.article_title = current_article_title
            boundary.article_title = current_article_title
            boundaries.append(boundary)
            previous_page = page

        if current_pages and current_start_boundary and current_start_features:
            articles.append(
                self.build_article_preview(
                    len(articles) + 1,
                    current_pages,
                    current_start_boundary,
                    current_start_features,
                )
            )

        return ArticleSegmentationResult(
            total_pages=len(ordered_pages),
            article_count=len(articles),
            needs_review_count=sum(1 for boundary in boundaries if boundary.needs_review),
            boundaries=boundaries,
            articles=articles,
        )

    def segment_document_into_articles(
        self,
        pages: list[PageContent],
    ) -> ArticleSegmentationResult:
        logger.info("Segmenting document into articles using %s pages", len(pages))
        return self.group_pages_into_articles(pages)


def segment_document_into_articles(pages: list[PageContent]) -> ArticleSegmentationResult:
    return ArticleSegmenter().segment_document_into_articles(pages)
