from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import Any, Iterable

from app.schemas import (
    ArticleBoundary,
    ArticlePreview,
    ArticleSegmentationResult,
    PageContent,
    ProcessedBlock,
)

logger = logging.getLogger(__name__)

AUTHOR_LINE_RE = re.compile(
    r"^(?:von|by|par|от|из|под редакцией|проф\.?|акад\.?|чл\.-корр\.?)\b",
    re.IGNORECASE,
)
AUTHOR_NAME_RE = re.compile(
    r"^(?:[A-ZА-ЯЁ]\.\s*){1,3}[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё-]+(?:\s*\([^)]{2,50}\))?$"
)
TOP_TEXT_NORMALIZE_RE = re.compile(r"[^a-zа-яё0-9]+", re.IGNORECASE)
SENTENCE_END_RE = re.compile(r"[.!?…:;]\s*$")
LOWERCASE_START_RE = re.compile(r"^[a-zа-яё]")
PLACEHOLDER_RE = re.compile(r"<<(?:FORMULA|TABLE|IMAGE)_[A-Za-z0-9_.-]+>>")


@dataclass
class PageFeatures:
    page_number: int
    page_height: int
    title_block: ProcessedBlock | None
    title_relative_y: float
    title_is_top: bool
    title_after_header: bool
    title_is_repeated_header: bool
    spacing_before_title: float
    spacing_after_title: float
    author_block: ProcessedBlock | None
    author_distance_from_title: float | None
    first_meaningful_block: ProcessedBlock | None
    first_blocks_are_text_continuation: bool
    first_text_block_length: int
    first_text_starts_lowercase: bool
    previous_page_end_punctuation: bool
    previous_page_ends_cleanly: bool
    layout_shift_score: float
    header_similarity_score: float
    title_uniqueness_score: float
    continuation_score: float
    candidate_start: bool
    looks_like_service_page: bool
    textual_block_count: int
    title_preview: str
    author_preview: str


class ArticleSegmenter:
    def __init__(
        self,
        *,
        start_threshold: float = 0.56,
        review_margin: float = 0.12,
        rescue_margin: float = 0.08,
        title_top_ratio: float = 0.28,
        repeated_top_ratio: float = 0.18,
        repeated_header_similarity_threshold: float = 0.86,
    ) -> None:
        self.start_threshold = start_threshold
        self.review_margin = review_margin
        self.rescue_margin = rescue_margin
        self.title_top_ratio = title_top_ratio
        self.repeated_top_ratio = repeated_top_ratio
        self.repeated_header_similarity_threshold = repeated_header_similarity_threshold

    def _page_height(self, page_content: PageContent) -> int:
        max_y = 1
        for block in page_content.blocks:
            if len(block.bbox) == 4:
                max_y = max(max_y, int(block.bbox[3]))
        return max_y

    def _sorted_blocks(self, page_content: PageContent) -> list[ProcessedBlock]:
        return sorted(
            list(page_content.blocks),
            key=lambda block: (block.reading_order, block.bbox[1], block.bbox[0], block.block_id),
        )

    def _normalize_top_text(self, text: str) -> str:
        normalized = TOP_TEXT_NORMALIZE_RE.sub(" ", (text or "").casefold()).strip()
        return re.sub(r"\s{2,}", " ", normalized)

    def _block_text(self, block: ProcessedBlock | None) -> str:
        return ((block.content or block.latex or "") if block else "").strip()

    def _meaningful_blocks(self, page_content: PageContent) -> list[ProcessedBlock]:
        meaningful: list[ProcessedBlock] = []
        for block in self._sorted_blocks(page_content):
            if block.type in {"footer"}:
                continue
            if not self._block_text(block):
                continue
            meaningful.append(block)
        return meaningful

    def _first_textual_block(self, page_content: PageContent) -> ProcessedBlock | None:
        for block in self._meaningful_blocks(page_content):
            if block.type in {"text", "title", "header", "page_number"}:
                return block
        return None

    def _layout_signature(self, page_content: PageContent) -> str:
        parts: list[str] = []
        for block in self._meaningful_blocks(page_content)[:6]:
            text = self._block_text(block)
            text_marker = "short" if len(text) < 60 else "long"
            parts.append(f"{block.type}:{text_marker}")
        return "|".join(parts)

    def _fuzzy_ratio(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0

        try:
            from rapidfuzz import fuzz
        except ImportError:
            return SequenceMatcher(None, left, right).ratio()

        return float(fuzz.ratio(left, right)) / 100.0

    def _clean_page_end_text(self, text: str) -> str:
        cleaned = PLACEHOLDER_RE.sub("", text or "").strip()
        return re.sub(r"\s+", " ", cleaned)

    def _previous_page_end_flags(self, previous_page: PageContent | None) -> tuple[bool, bool]:
        if previous_page is None:
            return False, False

        previous_text = self._clean_page_end_text(previous_page.page_text or "")
        if not previous_text:
            return False, False

        end_punctuation = bool(SENTENCE_END_RE.search(previous_text))
        ends_cleanly = end_punctuation and not previous_text.endswith("-")
        return end_punctuation, ends_cleanly

    def _layout_shift_score(
        self,
        current_page: PageContent,
        previous_page: PageContent | None,
        current_has_title: bool,
    ) -> float:
        if previous_page is None:
            return 1.0

        current_signature = self._layout_signature(current_page)
        previous_signature = self._layout_signature(previous_page)
        similarity = self._fuzzy_ratio(current_signature, previous_signature)
        shift = 1.0 - similarity

        previous_has_title = any(block.type == "title" for block in previous_page.blocks)
        if current_has_title != previous_has_title:
            shift += 0.18

        current_textual_count = len(self._meaningful_blocks(current_page))
        previous_textual_count = len(self._meaningful_blocks(previous_page))
        if abs(current_textual_count - previous_textual_count) >= 2:
            shift += 0.08

        current_first = self._first_textual_block(current_page)
        previous_first = self._first_textual_block(previous_page)
        if current_first and previous_first and current_first.type != previous_first.type:
            shift += 0.12

        return max(0.0, min(1.0, shift))

    def _collect_repeated_top_lines(self, pages: Iterable[PageContent]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for page in pages:
            page_height = self._page_height(page)
            for block in self._sorted_blocks(page):
                text = self._block_text(block)
                if not text:
                    continue
                if (block.bbox[1] / max(1, page_height)) > self.repeated_top_ratio:
                    continue
                normalized = self._normalize_top_text(text)
                if len(normalized) < 5:
                    continue
                counts[normalized] = counts.get(normalized, 0) + 1
        return counts

    def is_probable_repeated_header(
        self,
        text: str,
        repeated_top_lines: dict[str, int],
    ) -> tuple[bool, float]:
        normalized = self._normalize_top_text(text)
        if len(normalized) < 5:
            return False, 0.0

        best_similarity = 0.0
        best_count = 0
        for known_line, count in repeated_top_lines.items():
            if count < 2:
                continue
            similarity = self._fuzzy_ratio(normalized, known_line)
            if similarity > best_similarity:
                best_similarity = similarity
                best_count = count

        is_repeated = (
            best_count >= 2 and best_similarity >= self.repeated_header_similarity_threshold
        )
        return is_repeated, best_similarity

    def _title_context(
        self,
        page_content: PageContent,
        title_block: ProcessedBlock | None,
    ) -> tuple[float, bool, float, float]:
        page_height = self._page_height(page_content)
        if title_block is None:
            return 1.0, False, 0.0, 0.0

        ordered_blocks = self._sorted_blocks(page_content)
        title_relative_y = title_block.bbox[1] / max(1, page_height)
        title_after_header = False
        spacing_before_title = title_relative_y
        spacing_after_title = 0.0

        try:
            title_index = next(
                index for index, block in enumerate(ordered_blocks) if block.block_id == title_block.block_id
            )
        except StopIteration:
            return title_relative_y, False, spacing_before_title, spacing_after_title

        previous_blocks = ordered_blocks[:title_index]
        next_blocks = ordered_blocks[title_index + 1 :]

        for block in reversed(previous_blocks):
            if block.type in {"header", "page_number"} and block.bbox[3] <= title_block.bbox[1]:
                title_after_header = True
                break

        previous_visible = next(
            (
                block
                for block in reversed(previous_blocks)
                if self._block_text(block) and block.type != "footer"
            ),
            None,
        )
        if previous_visible is not None:
            spacing_before_title = max(
                0.0,
                (title_block.bbox[1] - previous_visible.bbox[3]) / max(1, page_height),
            )

        next_visible = next(
            (
                block
                for block in next_blocks
                if self._block_text(block) and block.type not in {"footer", "page_number"}
            ),
            None,
        )
        if next_visible is not None:
            spacing_after_title = max(
                0.0,
                (next_visible.bbox[1] - title_block.bbox[3]) / max(1, page_height),
            )

        return title_relative_y, title_after_header, spacing_before_title, spacing_after_title

    def detect_title_block(
        self,
        page_content: PageContent,
        repeated_top_lines: dict[str, int],
    ) -> tuple[ProcessedBlock | None, bool, bool, float]:
        best_block: ProcessedBlock | None = None
        best_score = -10.0
        best_is_top = False
        best_is_repeated = False
        best_similarity = 0.0

        for block in self._sorted_blocks(page_content):
            if block.type != "title":
                continue

            text = self._block_text(block)
            if len(text) < 6:
                continue

            title_relative_y, title_after_header, _, _ = self._title_context(page_content, block)
            is_repeated, similarity = self.is_probable_repeated_header(text, repeated_top_lines)
            is_top = title_relative_y <= self.title_top_ratio

            score = 0.0
            score += 0.35
            score += 0.15 * max(0.0, 1.0 - min(title_relative_y, 0.7) / 0.7)
            if title_after_header:
                score += 0.08
            score += 0.12 * min(len(text), 120) / 120
            score -= 0.40 * similarity
            if is_repeated:
                score -= 0.35

            if score > best_score:
                best_score = score
                best_block = block
                best_is_top = is_top
                best_is_repeated = is_repeated
                best_similarity = similarity

        return best_block, best_is_top, best_is_repeated, best_similarity

    def detect_author_block(
        self,
        page_content: PageContent,
        title_block: ProcessedBlock | None,
    ) -> tuple[ProcessedBlock | None, float | None]:
        if title_block is None:
            return None, None

        page_height = self._page_height(page_content)
        sorted_blocks = self._sorted_blocks(page_content)
        try:
            title_index = next(
                index for index, block in enumerate(sorted_blocks) if block.block_id == title_block.block_id
            )
        except StopIteration:
            return None, None

        title_bottom = title_block.bbox[3]
        max_author_bottom = title_bottom + int(page_height * 0.22)

        for block in sorted_blocks[title_index + 1 : title_index + 6]:
            if block.type not in {"text", "title"}:
                continue
            text = self._block_text(block)
            if not text or len(text) > 140:
                continue
            if block.bbox[1] < title_bottom:
                continue
            if block.bbox[3] > max_author_bottom:
                continue
            if AUTHOR_LINE_RE.search(text) or AUTHOR_NAME_RE.search(text):
                distance = max(0.0, (block.bbox[1] - title_bottom) / max(1, page_height))
                return block, distance
        return None, None

    def extract_page_features(
        self,
        page_content: PageContent,
        previous_page: PageContent | None,
        repeated_top_lines: dict[str, int],
    ) -> PageFeatures:
        page_height = self._page_height(page_content)
        (
            title_block,
            title_is_top,
            title_is_repeated,
            header_similarity_score,
        ) = self.detect_title_block(
            page_content,
            repeated_top_lines,
        )
        title_relative_y, title_after_header, spacing_before_title, spacing_after_title = (
            self._title_context(page_content, title_block)
        )
        author_block, author_distance_from_title = self.detect_author_block(page_content, title_block)

        meaningful_blocks = self._meaningful_blocks(page_content)
        first_meaningful_block = meaningful_blocks[0] if meaningful_blocks else None
        first_two_types = [block.type for block in meaningful_blocks[:2]]
        first_blocks_are_text_continuation = (
            title_block is None
            and first_two_types
            and all(block_type == "text" for block_type in first_two_types)
        )

        first_text_block = next(
            (block for block in meaningful_blocks if block.type in {"text", "title"}),
            None,
        )
        first_text = self._block_text(first_text_block)
        first_text_block_length = len(first_text)
        first_text_starts_lowercase = bool(first_text and LOWERCASE_START_RE.search(first_text))

        previous_page_end_punctuation, previous_page_ends_cleanly = self._previous_page_end_flags(
            previous_page
        )
        layout_shift_score = self._layout_shift_score(
            page_content,
            previous_page,
            current_has_title=title_block is not None,
        )
        title_uniqueness_score = max(0.0, min(1.0, 1.0 - header_similarity_score))

        continuation_score = 0.0
        if first_blocks_are_text_continuation:
            continuation_score += 0.55
        if first_text_starts_lowercase:
            continuation_score += 0.25
        if first_text_block_length >= 140:
            continuation_score += 0.20
        continuation_score = max(0.0, min(1.0, continuation_score))

        looks_like_service_page = (
            title_block is None
            and len(meaningful_blocks) <= 2
            and len((page_content.page_text or "").strip()) < 80
        )

        features = PageFeatures(
            page_number=page_content.page_number or 0,
            page_height=page_height,
            title_block=title_block,
            title_relative_y=title_relative_y,
            title_is_top=title_is_top,
            title_after_header=title_after_header,
            title_is_repeated_header=title_is_repeated,
            spacing_before_title=spacing_before_title,
            spacing_after_title=spacing_after_title,
            author_block=author_block,
            author_distance_from_title=author_distance_from_title,
            first_meaningful_block=first_meaningful_block,
            first_blocks_are_text_continuation=first_blocks_are_text_continuation,
            first_text_block_length=first_text_block_length,
            first_text_starts_lowercase=first_text_starts_lowercase,
            previous_page_end_punctuation=previous_page_end_punctuation,
            previous_page_ends_cleanly=previous_page_ends_cleanly,
            layout_shift_score=layout_shift_score,
            header_similarity_score=header_similarity_score,
            title_uniqueness_score=title_uniqueness_score,
            continuation_score=continuation_score,
            candidate_start=False,
            looks_like_service_page=looks_like_service_page,
            textual_block_count=len(meaningful_blocks),
            title_preview=self._block_text(title_block),
            author_preview=self._block_text(author_block),
        )
        candidate_start = self.detect_candidate_article_start(
            page_content,
            previous_page,
            repeated_top_lines,
            features,
        )
        return replace(features, candidate_start=candidate_start)

    def detect_candidate_article_start(
        self,
        page_content: PageContent,
        previous_page: PageContent | None,
        repeated_top_lines: dict[str, int],
        features: PageFeatures | None = None,
    ) -> bool:
        page_features = features or self.extract_page_features(
            page_content,
            previous_page,
            repeated_top_lines,
        )

        if page_features.title_block is None:
            return False
        if page_features.title_is_repeated_header:
            return False

        author_near_title = (
            page_features.author_block is not None
            and (
                page_features.author_distance_from_title is None
                or page_features.author_distance_from_title <= 0.10
            )
        )
        strong_layout_break = page_features.layout_shift_score >= 0.42
        visual_gap = (
            page_features.spacing_before_title >= 0.015
            or page_features.spacing_after_title >= 0.015
        )
        title_reasonable_position = page_features.title_relative_y <= 0.58

        return bool(
            title_reasonable_position
            and (author_near_title or strong_layout_break or visual_gap)
            and page_features.title_uniqueness_score >= 0.10
        )

    def score_article_start(
        self,
        page_content: PageContent,
        previous_page: PageContent | None,
        repeated_top_lines: dict[str, int],
    ) -> tuple[float, dict[str, Any], PageFeatures]:
        features = self.extract_page_features(page_content, previous_page, repeated_top_lines)

        if previous_page is None:
            debug_info = {
                "title_block": bool(features.title_block),
                "title_top": features.title_is_top,
                "title_after_header": features.title_after_header,
                "author_after_title": bool(features.author_block),
                "header_similarity_score": round(features.header_similarity_score, 3),
                "continuation_score": round(features.continuation_score, 3),
                "layout_shift_score": round(features.layout_shift_score, 3),
                "candidate_start": True,
                "title_relative_y": round(features.title_relative_y, 3),
                "spacing_before_title": round(features.spacing_before_title, 3),
                "spacing_after_title": round(features.spacing_after_title, 3),
                "title_uniqueness_score": round(features.title_uniqueness_score, 3),
                "decision": "article_start",
                "final_score": 1.0,
            }
            return 1.0, debug_info, replace(features, candidate_start=True)

        score = 0.0

        if features.title_block is not None:
            score += 0.24
        if features.title_is_top:
            score += 0.05
        score += 0.07 * max(0.0, 1.0 - min(features.title_relative_y, 0.8) / 0.8)
        if features.title_after_header:
            score += 0.09
        if features.author_block is not None:
            score += 0.18
        if features.spacing_before_title >= 0.015:
            score += 0.05
        elif features.spacing_before_title >= 0.008:
            score += 0.025
        if features.spacing_after_title >= 0.018:
            score += 0.09
        elif features.spacing_after_title >= 0.010:
            score += 0.05

        score += 0.18 * features.layout_shift_score

        if features.previous_page_ends_cleanly:
            score += 0.08
        elif features.previous_page_end_punctuation:
            score += 0.04

        score += 0.10 * features.title_uniqueness_score
        if features.candidate_start:
            score += 0.04

        if features.title_is_repeated_header:
            score -= 0.30
            score -= 0.18 * features.header_similarity_score

        score -= 0.10 * features.continuation_score
        if features.looks_like_service_page:
            score -= 0.15

        score = max(0.0, min(1.0, score))
        debug_info = {
            "title_block": bool(features.title_block),
            "title_top": features.title_is_top,
            "title_after_header": features.title_after_header,
            "author_after_title": bool(features.author_block),
            "header_similarity_score": round(features.header_similarity_score, 3),
            "continuation_score": round(features.continuation_score, 3),
            "layout_shift_score": round(features.layout_shift_score, 3),
            "candidate_start": features.candidate_start,
            "title_relative_y": round(features.title_relative_y, 3),
            "spacing_before_title": round(features.spacing_before_title, 3),
            "spacing_after_title": round(features.spacing_after_title, 3),
            "author_distance_from_title": (
                round(features.author_distance_from_title, 3)
                if features.author_distance_from_title is not None
                else None
            ),
            "first_text_block_length": features.first_text_block_length,
            "first_text_starts_lowercase": features.first_text_starts_lowercase,
            "previous_page_end_punctuation": features.previous_page_end_punctuation,
            "previous_page_ends_cleanly": features.previous_page_ends_cleanly,
            "title_uniqueness_score": round(features.title_uniqueness_score, 3),
            "decision": "continue_article",
            "final_score": round(score, 3),
        }
        return score, debug_info, features

    def is_article_start(
        self,
        page_content: PageContent,
        previous_page: PageContent | None,
        repeated_top_lines: dict[str, int],
    ) -> tuple[ArticleBoundary, PageFeatures]:
        score, debug_info, features = self.score_article_start(
            page_content,
            previous_page,
            repeated_top_lines,
        )

        if previous_page is None:
            boundary = ArticleBoundary(
                page_number=page_content.page_number or 0,
                is_article_start=True,
                score=score,
                needs_review=False,
                debug_info=debug_info,
            )
            return boundary, features

        rescue_start = features.candidate_start and score >= (
            self.start_threshold - self.rescue_margin
        )
        is_start = score >= self.start_threshold or rescue_start
        needs_review = False

        if is_start:
            if rescue_start and score < self.start_threshold:
                needs_review = True
                debug_info["decision"] = "article_start_rescue"
            elif score < (self.start_threshold + self.review_margin):
                needs_review = True
                debug_info["decision"] = "article_start_review"
            else:
                debug_info["decision"] = "article_start"
        else:
            debug_info["decision"] = "continue_article"

        boundary = ArticleBoundary(
            page_number=page_content.page_number or 0,
            is_article_start=is_start,
            score=score,
            needs_review=needs_review,
            debug_info=debug_info,
        )
        return boundary, features

    def build_article_preview(
        self,
        article_index: int,
        article_pages: list[PageContent],
        start_boundary: ArticleBoundary,
        start_features: PageFeatures,
    ) -> ArticlePreview:
        page_numbers = [page.page_number for page in article_pages if page.page_number is not None]
        title_preview = (article_pages[0].article_title if article_pages else "") or start_features.title_preview
        author_preview = start_features.author_preview

        if not title_preview and article_pages:
            presentation = article_pages[0].presentation
            title_preview = (
                presentation.readable_text.splitlines()[0].strip()
                if presentation and presentation.readable_text
                else ""
            )

        return ArticlePreview(
            article_id=f"article_{article_index:03d}",
            start_page=page_numbers[0] if page_numbers else 0,
            end_page=page_numbers[-1] if page_numbers else 0,
            page_numbers=page_numbers,
            title_preview=title_preview[:240],
            author_preview=author_preview[:160],
            needs_review=start_boundary.needs_review,
            boundary_confidence=start_boundary.score,
            debug_info=start_boundary.debug_info,
        )

    def _resolve_article_title(
        self,
        page_content: PageContent,
        features: PageFeatures,
    ) -> str:
        if features.title_preview:
            return features.title_preview[:240]

        if page_content.article_title:
            return page_content.article_title[:240]

        if page_content.presentation and page_content.presentation.readable_text:
            first_line = next(
                (
                    line.strip()
                    for line in page_content.presentation.readable_text.splitlines()
                    if line.strip()
                ),
                "",
            )
            if first_line:
                return first_line[:240]

        first_line = next(
            (line.strip() for line in (page_content.page_text or "").splitlines() if line.strip()),
            "",
        )
        return first_line[:240]

    def group_pages_into_articles(self, pages: list[PageContent]) -> ArticleSegmentationResult:
        ordered_pages = sorted(
            list(pages),
            key=lambda page: (
                page.page_number if page.page_number is not None else 10**9,
                page.page_content_id,
            ),
        )
        if not ordered_pages:
            return ArticleSegmentationResult(total_pages=0, article_count=0, needs_review_count=0)

        repeated_top_lines = self._collect_repeated_top_lines(ordered_pages)
        boundaries: list[ArticleBoundary] = []
        articles: list[ArticlePreview] = []

        current_pages: list[PageContent] = []
        current_start_boundary: ArticleBoundary | None = None
        current_start_features: PageFeatures | None = None
        current_article_title: str | None = None

        previous_page: PageContent | None = None
        for page in ordered_pages:
            boundary, features = self.is_article_start(page, previous_page, repeated_top_lines)

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
            needs_review_count=sum(1 for article in articles if article.needs_review),
            boundaries=boundaries,
            articles=articles,
        )

    def segment_document_into_articles(self, pages: list[PageContent]) -> ArticleSegmentationResult:
        logger.info("Segmenting document into articles using %s pages", len(pages))
        return self.group_pages_into_articles(pages)


def segment_document_into_articles(pages: list[PageContent]) -> ArticleSegmentationResult:
    return ArticleSegmenter().segment_document_into_articles(pages)
