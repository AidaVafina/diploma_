from __future__ import annotations

import unittest

from app.schemas import PageContent, ProcessedBlock
from app.services.article_segmenter import ArticleSegmenter


def make_block(
    block_id: str,
    block_type: str,
    reading_order: int,
    bbox: list[int],
    content: str,
    route_to: str | None = None,
) -> ProcessedBlock:
    inferred_route = route_to
    if inferred_route is None:
        if block_type == "formula":
            inferred_route = "formula_pipeline"
        elif block_type == "table":
            inferred_route = "table_pipeline"
        elif block_type == "image":
            inferred_route = "image_pipeline"
        else:
            inferred_route = "text_pipeline"

    return ProcessedBlock(
        block_id=block_id,
        type=block_type,
        reading_order=reading_order,
        bbox=bbox,
        route_to=inferred_route,
        content=content,
        confidence=0.95,
        needs_review=False,
        crop_data_url=None,
        crop_path=None,
        latex=None,
        formula_result=None,
        formula_backend=None,
        ocr_result=None,
        ocr_backend="none",
    )


def make_page(
    page_number: int,
    blocks: list[ProcessedBlock],
    page_text: str | None = None,
) -> PageContent:
    return PageContent(
        page_content_id=f"page_content_{page_number}",
        page_number=page_number,
        page_text=page_text or "\n".join(block.content or "" for block in blocks).strip(),
        needs_review_count=0,
        result_json_path=None,
        presentation=None,
        blocks=blocks,
    )


class ArticleSegmenterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.segmenter = ArticleSegmenter(top_gap_ratio=0.14)

    def test_marks_first_page_as_article_start(self) -> None:
        pages = [
            make_page(
                1,
                [
                    make_block("p1_b1", "text", 1, [80, 40, 820, 120], "Первая статья"),
                    make_block("p1_b2", "text", 2, [80, 140, 900, 520], "Текст первой статьи."),
                ],
            )
        ]

        result = self.segmenter.segment_document_into_articles(pages)

        self.assertEqual(result.article_count, 1)
        self.assertTrue(result.boundaries[0].is_article_start)
        self.assertEqual(result.articles[0].start_page, 1)
        self.assertEqual(pages[0].article_title, "Первая статья")

    def test_detects_new_article_when_first_text_has_top_gap_and_previous_page_ends_with_punctuation(
        self,
    ) -> None:
        pages = [
            make_page(
                1,
                [
                    make_block("p1_b1", "text", 1, [80, 40, 900, 180], "Текст первой статьи."),
                    make_block("p1_b2", "footer", 2, [80, 580, 900, 620], "1"),
                ],
                page_text="Текст первой статьи.",
            ),
            make_page(
                2,
                [
                    make_block("p2_b1", "header", 1, [80, 20, 900, 45], "Журнал"),
                    make_block("p2_b2", "text", 2, [90, 150, 900, 220], "Вторая статья"),
                    make_block("p2_b3", "text", 3, [90, 240, 900, 620], "Текст второй статьи."),
                ],
            ),
        ]

        result = self.segmenter.segment_document_into_articles(pages)

        self.assertEqual(result.article_count, 2)
        self.assertTrue(result.boundaries[1].is_article_start)
        self.assertEqual(result.boundaries[1].score, 1.0)
        self.assertTrue(result.boundaries[1].debug_info["first_text_has_top_gap"])
        self.assertTrue(result.boundaries[1].debug_info["previous_page_end_punctuation"])
        self.assertEqual(result.articles[1].title_preview, "Вторая статья")

    def test_keeps_page_inside_article_without_top_gap(self) -> None:
        pages = [
            make_page(
                1,
                [
                    make_block("p1_b1", "text", 1, [80, 40, 900, 220], "Начало статьи."),
                    make_block("p1_b2", "footer", 2, [80, 580, 900, 620], "1"),
                ],
                page_text="Начало статьи.",
            ),
            make_page(
                2,
                [
                    make_block("p2_b1", "text", 1, [80, 50, 900, 230], "Продолжение статьи"),
                    make_block("p2_b2", "footer", 2, [80, 580, 900, 620], "2"),
                ],
            ),
        ]

        result = self.segmenter.segment_document_into_articles(pages)

        self.assertEqual(result.article_count, 1)
        self.assertFalse(result.boundaries[1].is_article_start)
        self.assertTrue(result.boundaries[1].needs_review)
        self.assertEqual(result.boundaries[1].debug_info["decision"], "review_boundary")
        self.assertIn("Начало статьи.", result.articles[0].article_text)
        self.assertIn("Продолжение статьи", result.articles[0].article_text)

    def test_keeps_page_inside_article_when_previous_page_has_no_final_punctuation(
        self,
    ) -> None:
        pages = [
            make_page(
                1,
                [
                    make_block("p1_b1", "text", 1, [80, 40, 900, 220], "Текст первой статьи"),
                    make_block("p1_b2", "footer", 2, [80, 580, 900, 620], "1"),
                ],
                page_text="Текст первой статьи",
            ),
            make_page(
                2,
                [
                    make_block("p2_b1", "text", 1, [90, 150, 900, 230], "Продолжение"),
                    make_block("p2_b2", "footer", 2, [80, 580, 900, 620], "2"),
                ],
            ),
        ]

        result = self.segmenter.segment_document_into_articles(pages)

        self.assertEqual(result.article_count, 2)
        self.assertTrue(result.boundaries[1].is_article_start)
        self.assertTrue(result.boundaries[1].needs_review)
        self.assertEqual(result.boundaries[1].debug_info["decision"], "article_start_review")
        self.assertFalse(result.boundaries[1].debug_info["previous_page_end_punctuation"])

    def test_detects_article_starts_on_each_page_with_top_gap(self) -> None:
        pages = [
            make_page(
                1,
                [
                    make_block("p1_b1", "text", 1, [90, 150, 900, 220], "Первая статья"),
                    make_block("p1_b2", "text", 2, [90, 240, 900, 620], "Текст первой статьи"),
                ],
                page_text="Текст первой статьи",
            ),
            make_page(
                2,
                [
                    make_block("p2_b1", "text", 1, [90, 150, 900, 220], "Вторая статья"),
                    make_block("p2_b2", "text", 2, [90, 240, 900, 620], "Текст второй статьи"),
                ],
                page_text="Текст второй статьи",
            ),
        ]

        result = self.segmenter.segment_document_into_articles(pages)

        self.assertEqual(result.article_count, 2)
        self.assertTrue(result.boundaries[1].is_article_start)
        self.assertEqual(result.articles[0].title_preview, "Первая статья")
        self.assertEqual(result.articles[1].title_preview, "Вторая статья")


if __name__ == "__main__":
    unittest.main()
