from __future__ import annotations

import logging
import re
from collections import Counter

from app.schemas import (
    ArticleContent,
    ArticleMetadata,
    ArticleReference,
    ProcessedBlock,
)

logger = logging.getLogger(__name__)

TEXTUAL_BLOCK_TYPES = {"title", "text", "header", "footer"}
IGNORED_BLOCK_TYPES = {"page_number"}
YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
PLACEHOLDER_RE = re.compile(r"<<(?:FORMULA|TABLE|IMAGE)_[A-Za-z0-9_.-]+>>")
MULTISPACE_RE = re.compile(r"[ \t]+")
AUTHOR_INITIALS_RE = re.compile(
    r"(?:[A-ZА-ЯЁ]\.\s*){1,3}[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё-]+|"
    r"[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё-]+(?:\s+(?:[A-ZА-ЯЁ]\.\s*){1,3})"
)
SHORT_SECTION_RE = re.compile(
    r"^(?:abstract|summary|keywords?|references|bibliography|"
    r"аннотация|резюме|ключевые слова|литература|библиография)\b",
    re.IGNORECASE,
)
REFERENCE_SECTION_RE = re.compile(
    r"^(?:references|bibliography|литература|библиография)\b",
    re.IGNORECASE,
)
ABSTRACT_SECTION_RE = re.compile(r"^(?:abstract|summary|аннотация|резюме)\b", re.IGNORECASE)
KEYWORDS_SECTION_RE = re.compile(r"^(?:keywords?|ключевые слова)\b", re.IGNORECASE)
KEYWORDS_INLINE_RE = re.compile(
    r"^(?:keywords?|ключевые слова)\s*[:\-–]\s*(.+)$",
    re.IGNORECASE,
)
NUMBERED_REFERENCE_RE = re.compile(r"^\s*(?:\[\d+\]|\d+[.)])\s+")
QUOTE_TITLE_RE = re.compile(r"[\"“”«](.+?)[\"“”»]")
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё-]{3,}")

STOPWORDS_RU = {
    "автор", "авторы", "без", "более", "бы", "был", "была", "были", "было",
    "в", "вам", "вас", "весь", "во", "вот", "все", "всего", "всех", "вы",
    "где", "да", "даже", "для", "до", "его", "ее", "если", "есть", "еще",
    "же", "за", "здесь", "и", "из", "или", "им", "их", "к", "как", "ко",
    "когда", "который", "ли", "либо", "между", "менее", "мы", "на", "над",
    "наш", "не", "него", "нее", "нет", "ни", "них", "но", "ну", "о", "об",
    "однако", "он", "она", "они", "оно", "от", "по", "под", "при", "с",
    "со", "так", "также", "такой", "там", "те", "тем", "то", "того", "той",
    "только", "том", "у", "уже", "хотя", "чего", "чей", "чем", "что", "чтобы",
    "эта", "эти", "это", "я",
}
STOPWORDS_EN = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "of", "on", "or", "that", "the", "their", "this",
    "to", "with",
}
KEYWORD_STOPWORDS = STOPWORDS_RU | STOPWORDS_EN | {
    "abstract", "article", "authors", "bibliography", "keywords", "references",
    "summary", "аннотация", "библиография", "ключевые", "литература",
    "references", "статья", "статьи",
}


def _block_text(block: ProcessedBlock | None) -> str:
    return ((block.content or block.latex or "") if block else "").strip()


def _normalize_text(text: str) -> str:
    text = PLACEHOLDER_RE.sub(" ", text or "")
    normalized_lines = [
        MULTISPACE_RE.sub(" ", line).strip()
        for line in text.replace("\r", "\n").split("\n")
    ]
    return "\n".join(line for line in normalized_lines if line)


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in _normalize_text(text).splitlines() if line.strip()]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip(" \t,;.")
        if not cleaned:
            continue
        marker = cleaned.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        result.append(cleaned)
    return result


def _meaningful_tokens(text: str) -> list[str]:
    tokens = []
    for token in WORD_RE.findall((text or "").lower()):
        cleaned = token.strip("-")
        if len(cleaned) < 3 or cleaned in KEYWORD_STOPWORDS:
            continue
        tokens.append(cleaned)
    return tokens


class MetadataExtractor:
    def _ordered_blocks(self, article_content: ArticleContent) -> list[ProcessedBlock]:
        if article_content.pages:
            blocks: list[ProcessedBlock] = []
            for page in sorted(
                article_content.pages,
                key=lambda page: (
                    page.page_number if page.page_number is not None else 10**9,
                    page.page_content_id,
                ),
            ):
                blocks.extend(
                    sorted(
                        page.blocks,
                        key=lambda block: (
                            block.reading_order,
                            block.bbox[1],
                            block.bbox[0],
                            block.block_id,
                        ),
                    )
                )
            return blocks

        if article_content.structured_content:
            return sorted(
                article_content.structured_content,
                key=lambda block: (
                    block.reading_order,
                    block.bbox[1],
                    block.bbox[0],
                    block.block_id,
                ),
            )

        if article_content.blocks:
            return sorted(
                article_content.blocks,
                key=lambda block: (
                    block.reading_order,
                    block.bbox[1],
                    block.bbox[0],
                    block.block_id,
                ),
            )

        return []

    def _article_text(self, article_content: ArticleContent, blocks: list[ProcessedBlock]) -> str:
        if article_content.article_text.strip():
            return article_content.article_text.strip()

        if article_content.pages:
            page_texts: list[str] = []
            for page in sorted(
                article_content.pages,
                key=lambda page: (
                    page.page_number if page.page_number is not None else 10**9,
                    page.page_content_id,
                ),
            ):
                source = (
                    (page.presentation.readable_text if page.presentation else "")
                    or page.page_text
                    or "\n".join(
                        _block_text(block)
                        for block in page.blocks
                        if block.type in TEXTUAL_BLOCK_TYPES and _block_text(block)
                    )
                )
                if source.strip():
                    page_texts.append(source.strip())
            if page_texts:
                return "\n\n".join(page_texts)

        parts = [
            _block_text(block)
            for block in blocks
            if block.type in TEXTUAL_BLOCK_TYPES and block.type not in IGNORED_BLOCK_TYPES
        ]
        return "\n".join(part for part in parts if part).strip()

    def _head_lines(self, text: str, limit: int = 10) -> list[str]:
        return _nonempty_lines(text)[:limit]

    def _find_first_title_block(
        self,
        blocks: list[ProcessedBlock],
    ) -> tuple[int | None, str | None]:
        for index, block in enumerate(blocks):
            if block.type != "title":
                continue
            text = _block_text(block)
            if text:
                return index, text.splitlines()[0].strip()
        return None, None

    def _looks_like_section_header(self, text: str) -> bool:
        candidate = (text or "").strip().strip(":")
        if not candidate:
            return False
        if SHORT_SECTION_RE.match(candidate):
            return True
        if len(candidate) <= 40 and candidate.isupper():
            return True
        return False

    def _is_author_like(self, text: str) -> bool:
        candidate = (text or "").strip().strip(",;")
        if not candidate or len(candidate) > 120:
            return False
        if any(ch.isdigit() for ch in candidate):
            return False
        if self._looks_like_section_header(candidate):
            return False
        if "@" in candidate or "http" in candidate.lower():
            return False
        if AUTHOR_INITIALS_RE.search(candidate):
            return True

        tokens = re.findall(r"[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё-]+", candidate)
        return 2 <= len(tokens) <= 5 and len(tokens) == len(candidate.split())

    def _split_authors(self, text: str) -> list[str]:
        candidate = text.strip()
        split_attempts = [
            re.split(r"\s*;\s*", candidate),
            re.split(r"\s+\b(?:and|и)\b\s+", candidate, flags=re.IGNORECASE),
        ]
        comma_split = [part.strip() for part in re.split(r"\s*,\s*", candidate) if part.strip()]
        if len(comma_split) > 1 and all(self._is_author_like(part) for part in comma_split):
            split_attempts.append(comma_split)

        for parts in split_attempts:
            cleaned_parts = [part.strip() for part in parts if part.strip()]
            if len(cleaned_parts) > 1 and all(self._is_author_like(part) for part in cleaned_parts):
                return _dedupe_preserve_order(cleaned_parts)

        return [candidate]

    def _extract_title(
        self,
        article_content: ArticleContent,
        blocks: list[ProcessedBlock],
        article_text: str,
    ) -> tuple[str | None, str | None, int | None]:
        title_index, title_block_text = self._find_first_title_block(blocks)
        if title_block_text:
            return title_block_text[:300], "title_block", title_index

        if article_content.title and article_content.title.strip():
            return article_content.title.strip()[:300], "article_content", None

        for line in self._head_lines(article_text, limit=8):
            if len(line) < 6 or self._is_author_like(line) or self._looks_like_section_header(line):
                continue
            return line[:300], "leading_line", None

        return None, None, None

    def _extract_authors_from_blocks(
        self,
        blocks: list[ProcessedBlock],
        title_index: int | None,
    ) -> list[str]:
        if title_index is None:
            return []

        authors: list[str] = []
        for block in blocks[title_index + 1 : title_index + 5]:
            if block.type in {"page_number", "header", "footer"}:
                continue
            text = _block_text(block).splitlines()[0].strip() if _block_text(block) else ""
            if not text:
                continue
            if not self._is_author_like(text):
                if authors:
                    break
                continue
            authors.extend(self._split_authors(text))
        return _dedupe_preserve_order(authors)

    def _extract_authors_from_head(self, article_text: str, title: str | None) -> list[str]:
        authors: list[str] = []
        title_marker = (title or "").casefold().strip()
        for line in self._head_lines(article_text, limit=8):
            if title_marker and line.casefold().strip() == title_marker:
                continue
            if self._is_author_like(line):
                authors.extend(self._split_authors(line))
                continue
            if authors:
                break
        return _dedupe_preserve_order(authors)

    def _detect_language(self, normalized_text: str) -> tuple[str | None, str | None]:
        lowered = normalized_text.lower()
        cyrillic = sum(1 for char in lowered if "а" <= char <= "я" or char == "ё")
        latin = sum(1 for char in lowered if "a" <= char <= "z")
        if cyrillic > latin * 1.5 and cyrillic >= 20:
            return "ru", "script_ratio"
        if latin > cyrillic * 1.5 and latin >= 20:
            return "en", "script_ratio"

        ru_hits = sum(1 for token in STOPWORDS_RU if f" {token} " in f" {lowered} ")
        en_hits = sum(1 for token in STOPWORDS_EN if f" {token} " in f" {lowered} ")
        if ru_hits > en_hits and ru_hits >= 3:
            return "ru", "stopwords"
        if en_hits > ru_hits and en_hits >= 3:
            return "en", "stopwords"
        return None, None

    def _extract_section(
        self,
        text: str,
        section_pattern: re.Pattern[str],
        stop_patterns: tuple[re.Pattern[str], ...] = (),
        max_lines: int = 8,
        max_chars: int = 1400,
    ) -> str | None:
        raw_lines = text.replace("\r", "\n").split("\n")
        for index, raw_line in enumerate(raw_lines):
            line = MULTISPACE_RE.sub(" ", raw_line).strip()
            if not line:
                continue
            match = section_pattern.match(line.rstrip(":"))
            if not match:
                continue

            inline_match = re.match(r"^[^:]+[:\-–]\s*(.+)$", line)
            collected: list[str] = []
            if inline_match and inline_match.group(1).strip():
                collected.append(inline_match.group(1).strip())

            for next_line in raw_lines[index + 1 :]:
                normalized = MULTISPACE_RE.sub(" ", next_line).strip()
                if not normalized:
                    if collected:
                        break
                    continue
                if any(pattern.match(normalized.rstrip(":")) for pattern in stop_patterns):
                    break
                if self._looks_like_section_header(normalized) and collected:
                    break
                collected.append(normalized)
                if len(collected) >= max_lines or len(" ".join(collected)) >= max_chars:
                    break

            section_text = " ".join(collected).strip(" \t:;-")
            if section_text:
                return section_text
        return None

    def _extract_keywords(self, title: str | None, abstract: str | None, text: str) -> tuple[list[str], str | None]:
        raw_lines = text.replace("\r", "\n").split("\n")
        for raw_line in raw_lines:
            line = MULTISPACE_RE.sub(" ", raw_line).strip()
            if not line:
                continue
            inline_match = KEYWORDS_INLINE_RE.match(line)
            if inline_match:
                parts = re.split(r"\s*[,;]\s*", inline_match.group(1))
                keywords = _dedupe_preserve_order([part for part in parts if part.strip()])
                if keywords:
                    return keywords[:7], "keywords_section"

        section_text = self._extract_section(
            text,
            KEYWORDS_SECTION_RE,
            stop_patterns=(REFERENCE_SECTION_RE,),
            max_lines=3,
            max_chars=400,
        )
        if section_text:
            parts = re.split(r"\s*[,;]\s*", section_text)
            keywords = _dedupe_preserve_order([part for part in parts if part.strip()])
            if keywords:
                return keywords[:7], "keywords_section"

        corpus = " ".join(part for part in [title or "", abstract or "", text[:1200]] if part).strip()
        tokens = _meaningful_tokens(corpus)
        if not tokens:
            return [], None

        token_counts = Counter(tokens)
        phrase_scores: Counter[str] = Counter()

        title_tokens = _meaningful_tokens(title or "")
        for size in (3, 2):
            for start in range(0, max(0, len(title_tokens) - size + 1)):
                phrase = " ".join(title_tokens[start : start + size])
                if phrase:
                    phrase_scores[phrase] += 10 - size

        corpus_tokens = _meaningful_tokens(corpus)
        for size in (2, 3):
            for start in range(0, max(0, len(corpus_tokens) - size + 1)):
                window = corpus_tokens[start : start + size]
                if len(window) != size:
                    continue
                phrase = " ".join(window)
                phrase_scores[phrase] += 1

        for token, count in token_counts.items():
            phrase_scores[token] += count

        keywords: list[str] = []
        for phrase, _score in phrase_scores.most_common(20):
            if phrase in KEYWORD_STOPWORDS:
                continue
            if any(phrase in existing or existing in phrase for existing in keywords):
                continue
            keywords.append(phrase)
            if len(keywords) == 5:
                break

        return keywords, ("keyword_heuristics" if keywords else None)

    def _extract_year(self, text: str) -> tuple[int | None, str | None]:
        pre_reference_lines: list[str] = []
        for raw_line in text.replace("\r", "\n").split("\n"):
            normalized = MULTISPACE_RE.sub(" ", raw_line).strip()
            if normalized and REFERENCE_SECTION_RE.match(normalized.rstrip(":")):
                break
            pre_reference_lines.append(raw_line)

        head_excerpt = "\n".join(_nonempty_lines("\n".join(pre_reference_lines))[:12])[:1500]
        matches = [int(match.group(1)) for match in YEAR_RE.finditer(head_excerpt)]
        if matches:
            return matches[0], "head_regex"
        return None, None

    def _contains_year(self, text: str) -> bool:
        return bool(YEAR_RE.search(text))

    def _looks_like_reference_start(self, text: str) -> bool:
        candidate = NUMBERED_REFERENCE_RE.sub("", text).strip()
        return bool(candidate) and candidate[0].isupper()

    def _split_reference_entries(self, section_text: str) -> list[str]:
        raw_lines = section_text.replace("\r", "\n").split("\n")
        entries: list[str] = []
        current: list[str] = []

        for raw_line in raw_lines:
            line = MULTISPACE_RE.sub(" ", raw_line).strip()
            if not line:
                continue

            is_new_entry = NUMBERED_REFERENCE_RE.match(line) is not None
            if not is_new_entry and current:
                current_text = " ".join(current)
                if (
                    self._contains_year(current_text)
                    and self._contains_year(line)
                    and self._looks_like_reference_start(line)
                ):
                    is_new_entry = True

            if is_new_entry and current:
                entries.append(" ".join(current).strip())
                current = []

            current.append(line)

        if current:
            entries.append(" ".join(current).strip())

        return [entry for entry in entries if entry]

    def _parse_reference(self, raw_text: str) -> ArticleReference:
        cleaned = NUMBERED_REFERENCE_RE.sub("", raw_text).strip()
        year_match = YEAR_RE.search(cleaned)
        year = int(year_match.group(1)) if year_match else None

        authors: list[str] = []
        title: str | None = None

        quote_match = QUOTE_TITLE_RE.search(cleaned)
        if quote_match:
            title = quote_match.group(1).strip()
            authors_candidate = cleaned[: quote_match.start()].strip(" .,-")
            if authors_candidate:
                authors = self._split_authors(authors_candidate)
        else:
            parts = [part.strip() for part in re.split(r"\.\s+", cleaned) if part.strip()]
            if parts:
                if self._is_author_like(parts[0]) or AUTHOR_INITIALS_RE.search(parts[0]):
                    authors = self._split_authors(parts[0])
                    if len(parts) > 1:
                        title = parts[1].strip(" .")
                elif len(parts) > 0:
                    title = parts[0].strip(" .")

        return ArticleReference(
            raw_text=cleaned,
            authors=_dedupe_preserve_order(authors),
            title=title,
            year=year,
        )

    def _extract_references(self, text: str) -> tuple[list[ArticleReference], str | None]:
        raw_lines = text.replace("\r", "\n").split("\n")
        for index, raw_line in enumerate(raw_lines):
            line = MULTISPACE_RE.sub(" ", raw_line).strip()
            if not line or not REFERENCE_SECTION_RE.match(line.rstrip(":")):
                continue

            section_text = "\n".join(raw_lines[index + 1 :]).strip()
            if not section_text:
                return [], None

            entries = [
                self._parse_reference(entry)
                for entry in self._split_reference_entries(section_text)
            ]
            entries = [entry for entry in entries if entry.raw_text]
            return entries, ("references_section" if entries else None)

        return [], None

    def extract(self, article_content: ArticleContent) -> ArticleMetadata:
        blocks = self._ordered_blocks(article_content)
        article_text = self._article_text(article_content, blocks)
        normalized_text = article_content.normalized_text.strip() or _normalize_text(article_text)

        title, title_source, title_index = self._extract_title(article_content, blocks, article_text)

        authors = self._extract_authors_from_blocks(blocks, title_index)
        author_source = "post_title_blocks" if authors else None
        if not authors:
            authors = self._extract_authors_from_head(article_text, title)
            author_source = "leading_lines" if authors else None

        abstract = self._extract_section(
            article_text,
            ABSTRACT_SECTION_RE,
            stop_patterns=(KEYWORDS_SECTION_RE, REFERENCE_SECTION_RE),
        )
        abstract_source = "abstract_section" if abstract else None

        keywords, keywords_source = self._extract_keywords(title, abstract, article_text)
        references, references_source = self._extract_references(article_text)
        year, year_source = self._extract_year(article_text)
        language, language_source = self._detect_language(normalized_text)

        field_sources = {
            key: value
            for key, value in {
                "title": title_source,
                "authors": author_source,
                "abstract": abstract_source,
                "keywords": keywords_source,
                "references": references_source,
                "year": year_source,
                "language": language_source,
            }.items()
            if value
        }
        confidence = {
            "title": 0.97 if title_source == "title_block" else (0.86 if title else 0.0),
            "authors": 0.93 if author_source == "post_title_blocks" else (0.8 if authors else 0.0),
            "abstract": 0.9 if abstract else 0.0,
            "keywords": 0.84 if keywords_source == "keywords_section" else (0.62 if keywords else 0.0),
            "references": 0.88 if references else 0.0,
            "year": 0.78 if year else 0.0,
            "language": 0.9 if language_source == "script_ratio" else (0.75 if language else 0.0),
        }

        needs_review = not title or not authors or language is None

        return ArticleMetadata(
            title=title,
            authors=authors,
            language=language,
            year=year,
            abstract=abstract,
            keywords=keywords,
            references=references,
            needs_review=needs_review,
            field_sources=field_sources,
            confidence=confidence,
        )


def extract_article_metadata(article_content: ArticleContent) -> ArticleMetadata:
    return MetadataExtractor().extract(article_content)
