from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from functools import lru_cache
from typing import Any

from app.core.config import settings
from app.schemas import (
    SuspiciousToken,
    TextCorrection,
    TextPostprocessResponse,
)

logger = logging.getLogger(__name__)

PLACEHOLDER_RE = re.compile(r"<<(?:FORMULA|TABLE|IMAGE)_[A-Za-z0-9_.-]+>>")
LATEX_SPAN_PATTERNS = [
    re.compile(r"\\\[[\s\S]*?\\\]"),
    re.compile(r"\\\([\s\S]*?\\\)"),
    re.compile(r"\$\$[\s\S]*?\$\$"),
    re.compile(r"(?<!\$)\$[^$\n]+\$(?!\$)"),
]
INLINE_MATHML_TAG_RE = re.compile(
    r"</?(?:math|mrow|mi|mn|mo|msub|msup|msubsup|mfrac|semantics|annotation|annotation-xml)[^>]*>",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(
    r"(__PROTECTED_\d+__|\n+|[ \t]+|[A-Za-zА-Яа-яЁё0-9]+(?:[-'][A-Za-zА-Яа-яЁё0-9]+)*|.)"
)
WORD_RE = re.compile(r"^[A-Za-zА-Яа-яЁё0-9]+(?:[-'][A-Za-zА-Яа-яЁё0-9]+)*$")
CYRILLIC_OR_LATIN_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]{2,}")
MIXED_DIGIT_WORD_RE = re.compile(r"(?=.*[A-Za-zА-Яа-яЁё])(?=.*\d)")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")
SPACE_AROUND_NEWLINE_RE = re.compile(r"[ \t]*\n[ \t]*")
HYPHENATED_LINEBREAK_RE = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
PUNCT_OPEN_SPACE_RE = re.compile(r"([(\[{])\s+")
PUNCT_CLOSE_SPACE_RE = re.compile(r"\s+([)\]}])")
PLACEHOLDER_TOKEN_RE = re.compile(r"^__PROTECTED_\d+__$")

DEFAULT_FREQUENCIES: dict[str, int] = {
    "теория": 5000,
    "текст": 4500,
    "формула": 4000,
    "страницы": 3800,
    "страница": 3700,
    "документ": 3600,
    "научный": 3400,
    "журнал": 3200,
    "ретродокумент": 3000,
    "уравнение": 2900,
    "функция": 2800,
    "интеграл": 2700,
    "точка": 2600,
    "точки": 2500,
    "значения": 2400,
    "соответствующие": 2300,
    "можно": 2200,
    "показать": 2100,
    "говорит": 2000,
    "теоремы": 1900,
    "первого": 1800,
    "абзац": 1700,
    "пример": 1600,
    "введение": 1500,
    "статья": 1400,
    "работа": 1300,
    "особенность": 1200,
    "особенностей": 1100,
    "переменные": 1000,
    "соответствует": 950,
}


@dataclass
class _Token:
    value: str
    kind: str
    index: int


class TextPostprocessorError(RuntimeError):
    pass


class TextPostprocessor:
    def __init__(
        self,
        *,
        word_frequencies: dict[str, int] | None = None,
        symspell: Any | None = None,
        morph_analyzer: Any | None = None,
        fuzzy_threshold: int | None = None,
        min_token_length: int | None = None,
        max_edit_distance: int | None = None,
        suspicious_ratio: float | None = None,
    ) -> None:
        self.word_frequencies = {
            key.lower(): int(value)
            for key, value in {**DEFAULT_FREQUENCIES, **(word_frequencies or {})}.items()
        }
        self.fuzzy_threshold = fuzzy_threshold or settings.text_postprocess_fuzzy_threshold
        self.min_token_length = (
            min_token_length or settings.text_postprocess_min_token_length
        )
        self.max_edit_distance = (
            max_edit_distance or settings.text_postprocess_max_edit_distance
        )
        self.suspicious_ratio = (
            suspicious_ratio or settings.text_postprocess_suspicious_ratio
        )
        self.symspell = symspell if symspell is not None else self._build_symspell()
        self.morph_analyzer = (
            morph_analyzer if morph_analyzer is not None else self._build_morph_analyzer()
        )
        self.lexicon = tuple(self.word_frequencies.keys())

    def _build_symspell(self) -> Any | None:
        try:
            from symspellpy import SymSpell
        except ImportError:
            logger.info("SymSpellPy is unavailable, using fallback candidate search")
            return None

        symspell = SymSpell(
            max_dictionary_edit_distance=self.max_edit_distance,
            prefix_length=7,
        )
        for word, frequency in self.word_frequencies.items():
            symspell.create_dictionary_entry(word, frequency)
        return symspell

    def _build_morph_analyzer(self) -> Any | None:
        try:
            from pymorphy3 import MorphAnalyzer
        except ImportError:
            logger.info("pymorphy3 is unavailable, morphology validation will be limited")
            return None

        try:
            return MorphAnalyzer()
        except Exception:  # pragma: no cover - runtime dependent
            logger.exception("Failed to initialize pymorphy3 MorphAnalyzer")
            return None

    def protect_placeholders(self, text: str) -> tuple[str, dict[str, str]]:
        protected_text = text
        mapping: dict[str, str] = {}
        protected_index = 0

        def replace_match(match: re.Match[str]) -> str:
            nonlocal protected_index
            key = f"__PROTECTED_{protected_index}__"
            protected_index += 1
            mapping[key] = match.group(0)
            return key

        protected_text = PLACEHOLDER_RE.sub(replace_match, protected_text)
        for pattern in LATEX_SPAN_PATTERNS:
            protected_text = pattern.sub(replace_match, protected_text)

        return protected_text, mapping

    def restore_placeholders(self, text: str, mapping: dict[str, str]) -> str:
        restored = text
        for key, original in mapping.items():
            restored = restored.replace(key, original)
        return restored

    def normalize_whitespace(self, text: str) -> str:
        normalized = text.replace("\u00a0", " ").replace("\ufeff", "")
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = SPACE_AROUND_NEWLINE_RE.sub("\n", normalized)
        normalized = MULTISPACE_RE.sub(" ", normalized)
        normalized = EXCESS_NEWLINES_RE.sub("\n\n", normalized)
        return normalized.strip()

    def normalize_punctuation(self, text: str) -> str:
        normalized = (
            text.replace("–", "-")
            .replace("—", "-")
            .replace("−", "-")
            .replace("‑", "-")
            .replace("…", "...")
            .replace("«", '"')
            .replace("»", '"')
            .replace("“", '"')
            .replace("”", '"')
            .replace("„", '"')
            .replace("’", "'")
            .replace("‘", "'")
        )
        normalized = SPACE_BEFORE_PUNCT_RE.sub(r"\1", normalized)
        normalized = PUNCT_OPEN_SPACE_RE.sub(r"\1", normalized)
        normalized = PUNCT_CLOSE_SPACE_RE.sub(r"\1", normalized)
        return normalized

    def merge_hyphenated_words(self, text: str) -> str:
        return HYPHENATED_LINEBREAK_RE.sub("", text)

    def strip_inline_mathml(self, text: str) -> str:
        normalized = html.unescape(text)
        normalized = INLINE_MATHML_TAG_RE.sub("", normalized)
        normalized = MULTISPACE_RE.sub(" ", normalized)
        return normalized

    def tokenize_text(self, text: str) -> list[_Token]:
        tokens: list[_Token] = []
        for index, match in enumerate(TOKEN_RE.finditer(text)):
            value = match.group(0)
            if PLACEHOLDER_TOKEN_RE.fullmatch(value):
                kind = "protected"
            elif value.isspace():
                kind = "whitespace"
            elif WORD_RE.fullmatch(value):
                kind = "word"
            else:
                kind = "punct"
            tokens.append(_Token(value=value, kind=kind, index=index))
        return tokens

    def _word_is_known(self, token: str) -> bool:
        lowered = token.lower()
        if lowered in self.word_frequencies:
            return True
        if self.morph_analyzer is None:
            return False
        try:
            parses = self.morph_analyzer.parse(lowered)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Morphology parse failed for token %s", token)
            return False
        for parse in parses or []:
            if getattr(parse, "is_known", False):
                return True
            if (
                getattr(parse, "score", 0.0) >= 0.9
                and getattr(parse, "normal_form", None) == lowered
            ):
                return True
        return False

    def _fuzzy_ratio(self, left: str, right: str) -> float:
        try:
            from rapidfuzz import fuzz
        except ImportError:
            return SequenceMatcher(None, left, right).ratio() * 100

        return float(fuzz.ratio(left, right))

    def _rapidfuzz_candidates(self, token: str, limit: int = 5) -> list[tuple[str, float]]:
        try:
            from rapidfuzz import fuzz, process
        except ImportError:
            candidates = get_close_matches(
                token,
                list(self.lexicon),
                n=limit,
                cutoff=self.fuzzy_threshold / 100,
            )
            return [(candidate, self._fuzzy_ratio(token, candidate)) for candidate in candidates]

        return [
            (candidate, float(score))
            for candidate, score, _ in process.extract(
                token,
                self.lexicon,
                scorer=fuzz.ratio,
                limit=limit,
            )
            if score >= self.fuzzy_threshold
        ]

    def suggest_correction(self, token: str) -> tuple[str | None, str | None, list[str]]:
        lowered = token.lower()
        if len(lowered) < self.min_token_length:
            return None, None, []
        if PLACEHOLDER_TOKEN_RE.fullmatch(token):
            return None, None, []
        if self._word_is_known(lowered):
            return None, None, []

        candidates: list[tuple[str, float, str]] = []

        if self.symspell is not None:
            try:
                from symspellpy import Verbosity

                for suggestion in self.symspell.lookup(
                    lowered,
                    Verbosity.CLOSEST,
                    max_edit_distance=self.max_edit_distance,
                ):
                    candidates.append(
                        (
                            suggestion.term,
                            100.0 - float(suggestion.distance * 10),
                            "symspell",
                        )
                    )
            except Exception:  # pragma: no cover - runtime dependent
                logger.exception("SymSpell lookup failed for token %s", token)

        for candidate, score in self._rapidfuzz_candidates(lowered):
            candidates.append((candidate, score, "rapidfuzz"))

        if not candidates:
            return None, None, []

        unique_candidates: dict[str, tuple[float, str]] = {}
        for candidate, score, reason in candidates:
            if candidate == lowered:
                continue
            previous = unique_candidates.get(candidate)
            if previous is None or score > previous[0]:
                unique_candidates[candidate] = (score, reason)

        if not unique_candidates:
            return None, None, []

        ranked = sorted(
            unique_candidates.items(),
            key=lambda item: (
                item[1][0],
                self.word_frequencies.get(item[0], 0),
                -abs(len(item[0]) - len(lowered)),
            ),
            reverse=True,
        )
        suggestions = [candidate for candidate, _ in ranked[:3]]
        best_candidate, (best_score, best_reason) = ranked[0]

        if best_score < self.fuzzy_threshold:
            return None, None, suggestions

        return best_candidate, best_reason, suggestions

    def validate_with_morphology(self, token: str, candidate: str) -> bool:
        token_lower = token.lower()
        candidate_lower = candidate.lower()

        if token_lower == candidate_lower:
            return False

        original_is_valid = self._word_is_known(token_lower)
        candidate_is_valid = self._word_is_known(candidate_lower)

        if original_is_valid:
            return False
        if candidate_is_valid:
            return True

        return self.word_frequencies.get(candidate_lower, 0) > self.word_frequencies.get(
            token_lower, 0
        )

    def correct_tokens(
        self,
        tokens: list[_Token],
    ) -> tuple[list[_Token], list[TextCorrection], list[SuspiciousToken]]:
        corrections: list[TextCorrection] = []
        suspicious_tokens: list[SuspiciousToken] = []
        corrected_tokens: list[_Token] = []

        for token in tokens:
            if token.kind != "word":
                corrected_tokens.append(token)
                continue

            suggestion, reason, suggestions = self.suggest_correction(token.value)
            if suggestion and self.validate_with_morphology(token.value, suggestion):
                corrections.append(
                    TextCorrection(
                        original=token.value,
                        corrected=suggestion,
                        reason=reason or "dictionary",
                        token_index=token.index,
                    )
                )
                logger.info(
                    "Text postprocessor corrected token %s -> %s (%s)",
                    token.value,
                    suggestion,
                    reason,
                )
                corrected_tokens.append(
                    _Token(value=suggestion, kind=token.kind, index=token.index)
                )
                continue

            suspicious_reasons: list[str] = []
            ratio = 1.0
            if suggestions:
                ratio = self._fuzzy_ratio(token.value.lower(), suggestions[0].lower()) / 100

            if MIXED_DIGIT_WORD_RE.search(token.value):
                suspicious_reasons.append("mixed_digits_letters")
            if not self._word_is_known(token.value) and len(token.value) >= self.min_token_length:
                suspicious_reasons.append("unknown_token")
            if suggestions and ratio < self.suspicious_ratio:
                suspicious_reasons.append("low_confidence_suggestion")

            if suspicious_reasons:
                suspicious_tokens.append(
                    SuspiciousToken(
                        token=token.value,
                        reason=",".join(dict.fromkeys(suspicious_reasons)),
                        token_index=token.index,
                        suggestions=suggestions,
                    )
                )

            corrected_tokens.append(token)

        return corrected_tokens, corrections, suspicious_tokens

    def rebuild_text(self, tokens: list[_Token]) -> str:
        return "".join(token.value for token in tokens)

    def postprocess_text(self, text: str) -> TextPostprocessResponse:
        protected_text, mapping = self.protect_placeholders(text)
        normalized = self.strip_inline_mathml(protected_text)
        normalized = self.merge_hyphenated_words(normalized)
        normalized = self.normalize_whitespace(normalized)
        normalized = self.normalize_punctuation(normalized)
        normalized = self.normalize_whitespace(normalized)

        tokens = self.tokenize_text(normalized)
        corrected_tokens, corrections, suspicious_tokens = self.correct_tokens(tokens)
        rebuilt = self.rebuild_text(corrected_tokens)
        restored = self.restore_placeholders(rebuilt, mapping)
        cleaned_text = self.normalize_whitespace(restored)

        return TextPostprocessResponse(
            cleaned_text=cleaned_text,
            corrections=corrections,
            suspicious_tokens=suspicious_tokens,
            needs_review=bool(suspicious_tokens),
        )


@lru_cache(maxsize=1)
def get_text_postprocessor() -> TextPostprocessor:
    return TextPostprocessor()


def postprocess_text(text: str) -> TextPostprocessResponse:
    return get_text_postprocessor().postprocess_text(text)
