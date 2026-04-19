from __future__ import annotations

import unittest

from app.services.text_postprocessor import TextPostprocessor


class FakeParse:
    def __init__(self, is_known: bool, score: float = 0.0, normal_form: str | None = None):
        self.is_known = is_known
        self.score = score
        self.normal_form = normal_form


class FakeMorphAnalyzer:
    def __init__(self, mapping: dict[str, list[FakeParse]] | None = None):
        self.mapping = mapping or {}

    def parse(self, token: str) -> list[FakeParse]:
        return self.mapping.get(token.lower(), [FakeParse(False, 0.0, None)])


class TextPostprocessorTests(unittest.TestCase):
    def test_protect_placeholders_and_restore_keeps_original_tokens(self) -> None:
        processor = TextPostprocessor()
        raw_text = "Текст <<FORMULA_block_001>> и <<TABLE_block_002>>."

        protected, mapping = processor.protect_placeholders(raw_text)
        restored = processor.restore_placeholders(protected, mapping)

        self.assertIn("__PROTECTED_0__", protected)
        self.assertIn("__PROTECTED_1__", protected)
        self.assertEqual(restored, raw_text)

    def test_merge_hyphenated_words_joins_soft_linebreaks(self) -> None:
        processor = TextPostprocessor()

        merged = processor.merge_hyphenated_words("Это тео-\nрия и функ-\nция.")

        self.assertEqual(merged, "Это теория и функция.")

    def test_strip_inline_mathml_removes_tags_but_keeps_content(self) -> None:
        processor = TextPostprocessor()

        cleaned = processor.strip_inline_mathml("где <math>x, y, z</math> параметры")

        self.assertEqual(cleaned, "где x, y, z параметры")

    def test_postprocess_text_corrects_probable_ocr_error_without_touching_placeholder(
        self,
    ) -> None:
        processor = TextPostprocessor(word_frequencies={"теория": 5000})

        result = processor.postprocess_text("Это теоря и <<FORMULA_block_001>>.")

        self.assertEqual(result.cleaned_text, "Это теория и <<FORMULA_block_001>>.")
        self.assertEqual(len(result.corrections), 1)
        self.assertEqual(result.corrections[0].original, "теоря")
        self.assertEqual(result.corrections[0].corrected, "теория")

    def test_validate_with_morphology_prefers_known_candidate_and_keeps_valid_original(
        self,
    ) -> None:
        processor = TextPostprocessor(
            word_frequencies={"теория": 5000, "мир": 4500},
            morph_analyzer=FakeMorphAnalyzer(
                {
                    "теоря": [FakeParse(False, 0.0, None)],
                    "теория": [FakeParse(True, 1.0, "теория")],
                    "мир": [FakeParse(True, 1.0, "мир")],
                    "мор": [FakeParse(False, 0.0, None)],
                }
            ),
        )

        self.assertTrue(processor.validate_with_morphology("теоря", "теория"))
        self.assertFalse(processor.validate_with_morphology("мир", "мор"))


if __name__ == "__main__":
    unittest.main()
