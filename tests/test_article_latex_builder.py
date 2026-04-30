from __future__ import annotations

import unittest

from app.schemas import ArticleContent, LatexMetadata, PageContent, ProcessedBlock
from app.services.article_latex_builder import (
    build_article_latex_document,
    build_article_latex_preview,
    get_cached_article_latex,
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


def make_page(page_number: int, blocks: list[ProcessedBlock]) -> PageContent:
    return PageContent(
        page_content_id=f"page_content_{page_number}",
        page_number=page_number,
        page_text="",
        needs_review_count=0,
        result_json_path=None,
        presentation=None,
        blocks=blocks,
    )


class ArticleLatexBuilderTests(unittest.TestCase):
    def test_builds_article_from_one_page(self) -> None:
        article = ArticleContent(
            article_id="article_one_page",
            title="О статье",
            author="А. Автор",
            pages=[
                make_page(
                    1,
                    [
                        make_block("p1_t1", "title", 1, "О статье"),
                        make_block("p1_a1", "text", 2, "А. Автор"),
                        make_block("p1_b1", "text", 3, "Текст статьи."),
                    ],
                )
            ],
        )

        result = build_article_latex_document(article)

        self.assertEqual(result.article_id, "article_one_page")
        self.assertEqual(result.page_numbers, [1])
        self.assertIn("\\title{О статье}", result.latex_preview)
        self.assertIn("\\author{А. Автор}", result.latex_preview)
        self.assertNotIn("\\section*{О статье}", result.latex_preview)
        self.assertIn("Текст статьи.", result.latex_document)
        self.assertNotIn(
            "О статье",
            result.latex_document.split("\\maketitle", 1)[1],
        )
        self.assertNotIn(
            "А. Автор",
            result.latex_document.split("\\maketitle", 1)[1],
        )

    def test_builds_article_from_multiple_pages_in_page_order(self) -> None:
        article = ArticleContent(
            article_id="article_many_pages",
            pages=[
                make_page(2, [make_block("p2_b1", "text", 1, "Вторая страница.")]),
                make_page(1, [make_block("p1_b1", "text", 1, "Первая страница.")]),
            ],
        )

        result = build_article_latex_document(article)

        self.assertEqual(result.page_numbers, [1, 2])
        self.assertLess(
            result.latex_preview.index("Первая страница."),
            result.latex_preview.index("Вторая страница."),
        )

    def test_inserts_display_formula(self) -> None:
        article = ArticleContent(
            article_id="article_formula",
            structured_content=[
                make_block("b1", "text", 1, "До формулы."),
                make_block("f1", "formula", 2, latex=r"\int_a^b f(x)\,dx"),
            ],
        )

        result = build_article_latex_document(article)

        self.assertIn("\\[\n\\int_a^b f(x)\\,dx\n\\]", result.latex_preview)

    def test_converts_inline_math_to_latex(self) -> None:
        article = ArticleContent(
            article_id="article_inline",
            structured_content=[
                make_block("b1", "text", 1, "Пусть <math>x, y, z</math> заданы."),
            ],
        )

        result = build_article_latex_preview(article)

        self.assertIn("$x, y, z$", result.latex_preview)
        self.assertNotIn("<math", result.latex_preview)

    def test_removes_html_and_mathml_from_document(self) -> None:
        article = ArticleContent(
            article_id="article_html",
            structured_content=[
                make_block("b1", "text", 1, "Значение <math><mi>x</mi></math>."),
                make_block("f1", "formula", 2, latex="<math>\\pi</math>"),
            ],
        )

        result = build_article_latex_document(article)

        self.assertIn("$x$", result.latex_document)
        self.assertIn("\\[\n\\pi\n\\]", result.latex_document)
        self.assertNotIn("<mi>", result.latex_document)
        self.assertNotIn("</math>", result.latex_document)

    def test_inserts_title_author_and_valid_document_structure(self) -> None:
        article = ArticleContent(
            article_id="article_metadata",
            pages=[
                make_page(
                    1,
                    [
                        make_block("t1", "title", 1, "Название"),
                        make_block("a1", "text", 2, "И. Автор"),
                        make_block("b1", "text", 3, "Текст."),
                    ],
                )
            ],
            metadata=LatexMetadata(
                title="Название",
                author="И. Автор",
                documentclass="revtex4-1",
            ),
        )

        result = build_article_latex_document(article)

        self.assertIn("\\documentclass{revtex4-1}", result.latex_document)
        self.assertIn("\\usepackage[utf8]{inputenc}", result.latex_document)
        self.assertIn("\\usepackage[T2A]{fontenc}", result.latex_document)
        self.assertIn("\\usepackage{amsmath}", result.latex_document)
        self.assertIn("\\usepackage{amssymb}", result.latex_document)
        self.assertIn("\\begin{document}", result.latex_document)
        self.assertIn("\\title{Название}", result.latex_document)
        self.assertIn("\\author{И. Автор}", result.latex_document)
        self.assertIn("\\maketitle", result.latex_document)
        self.assertLess(
            result.latex_document.index("\\author{И. Автор}"),
            result.latex_document.index("\\begin{document}"),
        )
        self.assertNotIn(
            "Название",
            result.latex_document.split("\\maketitle", 1)[1],
        )
        self.assertNotIn(
            "И. Автор",
            result.latex_document.split("\\maketitle", 1)[1],
        )
        self.assertTrue(result.latex_document.strip().endswith("\\end{document}"))

    def test_extracts_title_and_author_from_structured_blocks(self) -> None:
        article = ArticleContent(
            article_id="article_head",
            structured_content=[
                make_block("title_1", "title", 1, "Об одном уравнении"),
                make_block("author_1", "text", 2, "Д. С. Синцов"),
                make_block("text_1", "text", 3, "Первый абзац статьи."),
            ],
        )

        result = build_article_latex_document(article)
        body_after_maketitle = result.latex_document.split("\\maketitle", 1)[1]

        self.assertIn("\\title{Об одном уравнении}", result.latex_document)
        self.assertIn("\\author{Д. С. Синцов}", result.latex_document)
        self.assertNotIn("Об одном уравнении", body_after_maketitle)
        self.assertNotIn("Д. С. Синцов", body_after_maketitle)
        self.assertIn("Первый абзац статьи.", body_after_maketitle)

    def test_table_and_image_placeholders_are_comments(self) -> None:
        article = ArticleContent(
            article_id="article_assets",
            structured_content=[
                make_block("table_123", "table", 1),
                make_block("image_456", "image", 2),
            ],
        )

        result = build_article_latex_document(article)

        self.assertIn("% TODO: TABLE table_123", result.latex_preview)
        self.assertIn("% TODO: IMAGE image_456", result.latex_preview)

    def test_caches_built_article_latex_by_article_id(self) -> None:
        article = ArticleContent(
            article_id="article_cached",
            pages=[make_page(1, [make_block("b1", "text", 1, "Кэшируемый текст.")])],
        )

        result = build_article_latex_document(article)
        cached = get_cached_article_latex("article_cached")

        self.assertEqual(cached.latex_document, result.latex_document)


if __name__ == "__main__":
    unittest.main()
