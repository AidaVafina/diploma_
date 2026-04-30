from __future__ import annotations

import unittest

from app.schemas import ArticleContent, LatexMetadata, PageContent, ProcessedBlock
from app.services.latex_builder import (
    build_article_latex,
    build_blocks_latex_body,
    build_page_latex,
)


def make_block(
    block_id: str,
    block_type: str,
    reading_order: int,
    content: str | None = None,
    latex: str | None = None,
) -> ProcessedBlock:
    route_to = "text_pipeline"
    if block_type == "formula":
        route_to = "formula_pipeline"
    elif block_type == "table":
        route_to = "table_pipeline"
    elif block_type == "image":
        route_to = "image_pipeline"

    return ProcessedBlock(
        block_id=block_id,
        type=block_type,
        reading_order=reading_order,
        bbox=[10, reading_order * 20, 200, reading_order * 20 + 10],
        route_to=route_to,
        content=content,
        confidence=0.95,
        needs_review=False,
        crop_data_url=None,
        crop_path=None,
        latex=latex,
        formula_result=None,
        formula_backend=None,
        ocr_result=None,
        ocr_backend="none",
    )


def make_page(blocks: list[ProcessedBlock]) -> PageContent:
    return PageContent(
        page_content_id="page_content_001",
        page_number=1,
        page_text="",
        needs_review_count=0,
        result_json_path=None,
        presentation=None,
        blocks=blocks,
    )


class LatexBuilderTests(unittest.TestCase):
    def test_converts_inline_math_tags_to_latex(self) -> None:
        body, needs_review = build_blocks_latex_body(
            [
                make_block(
                    "b1",
                    "text",
                    1,
                    "Координаты <math>x, y, z</math> и <math>\\pi + d\\pi</math>.",
                )
            ]
        )

        self.assertFalse(needs_review)
        self.assertIn("$x, y, z$", body)
        self.assertIn("$\\pi + d\\pi$", body)
        self.assertNotIn("<math", body)

    def test_keeps_display_formula_as_latex_block(self) -> None:
        body, needs_review = build_blocks_latex_body(
            [make_block("f1", "formula", 1, latex=r"\int_a^b f(x)\,dx")]
        )

        self.assertFalse(needs_review)
        self.assertIn("\\[\n\\int_a^b f(x)\\,dx\n\\]", body)

    def test_preserves_text_and_formula_order(self) -> None:
        body, _ = build_blocks_latex_body(
            [
                make_block("b1", "text", 1, "До формулы."),
                make_block("f1", "formula", 2, latex="x^2"),
                make_block("b2", "text", 3, "После формулы."),
            ]
        )

        self.assertLess(body.index("До формулы."), body.index("\\[\nx^2\n\\]"))
        self.assertLess(body.index("\\[\nx^2\n\\]"), body.index("После формулы."))

    def test_does_not_emit_html_or_mathml_tags(self) -> None:
        page = make_page(
            [
                make_block("b1", "text", 1, "Значение <math><mi>x</mi></math>."),
                make_block("f1", "formula", 2, latex="<math>\\pi</math>"),
            ]
        )

        result = build_page_latex(page)

        self.assertIn("$x$", result.latex_preview)
        self.assertIn("\\[\n\\pi\n\\]", result.latex_preview)
        self.assertNotIn("<mi>", result.latex_document)
        self.assertNotIn("</math>", result.latex_document)

    def test_document_builder_creates_latex_2e_structure(self) -> None:
        article = ArticleContent(
            title="Название статьи",
            author="А. Б. Автор",
            pages=[
                make_page(
                    [
                        make_block("t1", "title", 1, "Название статьи"),
                        make_block("a1", "text", 2, "А. Б. Автор"),
                        make_block("b1", "text", 3, "Текст статьи."),
                    ]
                )
            ],
        )

        result = build_article_latex(article, LatexMetadata(documentclass="article"))

        self.assertIn("\\documentclass{article}", result.latex_document)
        self.assertIn("\\usepackage{amsmath}", result.latex_document)
        self.assertIn("\\usepackage{amssymb}", result.latex_document)
        self.assertIn("\\begin{document}", result.latex_document)
        self.assertIn("\\title{Название статьи}", result.latex_document)
        self.assertIn("\\author{А. Б. Автор}", result.latex_document)
        self.assertIn("\\maketitle", result.latex_document)
        self.assertLess(
            result.latex_document.index("\\author{А. Б. Автор}"),
            result.latex_document.index("\\begin{document}"),
        )
        self.assertTrue(result.latex_document.strip().endswith("\\end{document}"))

    def test_article_title_and_author_are_not_duplicated_in_body(self) -> None:
        article = ArticleContent(
            pages=[
                make_page(
                    [
                        make_block("t1", "title", 1, "Название статьи"),
                        make_block("a1", "text", 2, "А. Б. Автор"),
                        make_block("b1", "text", 3, "Основной текст."),
                    ]
                )
            ],
        )

        result = build_article_latex(article)
        body_after_maketitle = result.latex_document.split("\\maketitle", 1)[1]

        self.assertIn("\\title{Название статьи}", result.latex_document)
        self.assertIn("\\author{А. Б. Автор}", result.latex_document)
        self.assertNotIn("\\section*{Название статьи}", result.latex_preview)
        self.assertNotIn("Название статьи", body_after_maketitle)
        self.assertNotIn("А. Б. Автор", body_after_maketitle)
        self.assertEqual(body_after_maketitle.count("Основной текст."), 1)

    def test_fallback_title_is_removed_from_body_when_block_matches(self) -> None:
        article = ArticleContent(
            title="Название из сегментации",
            pages=[
                make_page(
                    [
                        make_block("b1", "text", 1, "Название из сегментации"),
                        make_block("b2", "text", 2, "И. Автор"),
                        make_block("b3", "text", 3, "Текст без дубля заголовка."),
                    ]
                )
            ],
        )

        result = build_article_latex(article)
        body_after_maketitle = result.latex_document.split("\\maketitle", 1)[1]

        self.assertIn("\\title{Название из сегментации}", result.latex_document)
        self.assertIn("\\author{И. Автор}", result.latex_document)
        self.assertNotIn("Название из сегментации", body_after_maketitle)
        self.assertNotIn("И. Автор", body_after_maketitle)

    def test_table_and_image_placeholders_are_inserted(self) -> None:
        body, needs_review = build_blocks_latex_body(
            [
                make_block("table_123", "table", 1),
                make_block("image_456", "image", 2),
            ]
        )

        self.assertFalse(needs_review)
        self.assertIn("% TODO: TABLE table_123", body)
        self.assertIn("% TODO: IMAGE image_456", body)


if __name__ == "__main__":
    unittest.main()
