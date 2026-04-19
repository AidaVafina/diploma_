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


def make_page(page_number: int, blocks: list[ProcessedBlock], page_text: str | None = None) -> PageContent:
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
        self.segmenter = ArticleSegmenter()

    def test_marks_page_one_as_first_article(self) -> None:
        pages = [
            make_page(
                1,
                [
                    make_block("p1_b1", "title", 1, [80, 40, 820, 120], "Об одном уравнении Poincare"),
                    make_block("p1_b2", "text", 2, [120, 130, 760, 180], "Д. М. Синцов (Харьков)."),
                    make_block("p1_b3", "text", 3, [80, 210, 900, 500], "Начало статьи."),
                ],
            ),
            make_page(
                2,
                [
                    make_block("p2_b1", "text", 1, [80, 120, 900, 480], "Продолжение статьи на второй странице."),
                ],
            ),
        ]

        result = self.segmenter.segment_document_into_articles(pages)

        self.assertEqual(result.article_count, 1)
        self.assertEqual(result.articles[0].start_page, 1)
        self.assertEqual(result.articles[0].end_page, 2)
        self.assertEqual(pages[0].article_title, "Об одном уравнении Poincare")
        self.assertEqual(pages[1].article_title, "Об одном уравнении Poincare")
        self.assertEqual(result.boundaries[0].article_title, "Об одном уравнении Poincare")
        self.assertEqual(result.boundaries[1].article_title, "Об одном уравнении Poincare")

    def test_detects_new_article_in_middle_of_journal(self) -> None:
        pages = [
            make_page(
                1,
                [
                    make_block("p1_b1", "title", 1, [90, 35, 820, 115], "Первая статья"),
                    make_block("p1_b2", "text", 2, [120, 125, 760, 170], "И. И. Иванов"),
                    make_block("p1_b3", "text", 3, [80, 220, 900, 500], "Текст первой статьи."),
                ],
            ),
            make_page(
                2,
                [
                    make_block("p2_b1", "text", 1, [80, 120, 900, 500], "Продолжение первой статьи."),
                ],
            ),
            make_page(
                3,
                [
                    make_block("p3_b1", "title", 1, [95, 42, 830, 125], "Вторая статья"),
                    make_block("p3_b2", "text", 2, [130, 138, 780, 185], "П. П. Петров"),
                    make_block("p3_b3", "text", 3, [80, 220, 900, 500], "Начало второй статьи."),
                ],
            ),
        ]

        result = self.segmenter.segment_document_into_articles(pages)

        self.assertEqual(result.article_count, 2)
        self.assertEqual(result.articles[0].page_numbers, [1, 2])
        self.assertEqual(result.articles[1].page_numbers, [3])
        self.assertEqual(result.articles[1].title_preview, "Вторая статья")
        self.assertEqual(pages[1].article_title, "Первая статья")
        self.assertEqual(pages[2].article_title, "Вторая статья")

    def test_keeps_continuation_page_inside_current_article(self) -> None:
        pages = [
            make_page(
                1,
                [
                    make_block("p1_b1", "title", 1, [90, 30, 840, 120], "Статья"),
                    make_block("p1_b2", "text", 2, [120, 130, 780, 180], "А. А. Автор"),
                    make_block("p1_b3", "text", 3, [80, 220, 900, 500], "Текст статьи."),
                ],
            ),
            make_page(
                2,
                [
                    make_block("p2_b1", "text", 1, [80, 100, 900, 300], "продолжение текста статьи"),
                    make_block("p2_b2", "text", 2, [80, 320, 900, 540], "ещё один абзац продолжения"),
                ],
            ),
        ]

        result = self.segmenter.segment_document_into_articles(pages)

        self.assertEqual(result.article_count, 1)
        self.assertFalse(result.boundaries[1].is_article_start)
        self.assertEqual(pages[1].article_title, "Статья")
        self.assertEqual(result.boundaries[1].debug_info["decision"], "continue_article")

    def test_detects_title_after_header_as_article_start(self) -> None:
        pages = [
            make_page(
                1,
                [
                    make_block("p1_b1", "title", 1, [90, 34, 840, 120], "Первая статья"),
                    make_block("p1_b2", "text", 2, [120, 132, 760, 178], "И. И. Иванов"),
                    make_block("p1_b3", "text", 3, [80, 220, 900, 500], "Текст первой статьи."),
                ],
            ),
            make_page(
                2,
                [
                    make_block("p2_b1", "header", 1, [120, 18, 760, 54], "Журнал теоретической физики"),
                    make_block("p2_b2", "page_number", 2, [820, 20, 860, 48], "17"),
                    make_block("p2_b3", "title", 3, [90, 86, 840, 150], "Новая статья в середине тома"),
                    make_block("p2_b4", "text", 4, [130, 162, 780, 206], "П. П. Петров"),
                    make_block("p2_b5", "text", 5, [80, 240, 900, 520], "Начало новой статьи."),
                ],
            ),
        ]

        result = self.segmenter.segment_document_into_articles(pages)

        self.assertEqual(result.article_count, 2)
        self.assertTrue(result.boundaries[1].is_article_start)
        self.assertTrue(result.boundaries[1].debug_info["title_after_header"])
        self.assertTrue(result.boundaries[1].debug_info["candidate_start"])
        self.assertEqual(pages[1].article_title, "Новая статья в середине тома")

    def test_ignores_repeated_header_like_title(self) -> None:
        pages = [
            make_page(
                1,
                [
                    make_block("p1_b1", "title", 1, [120, 24, 760, 70], "Вестник Академии Наук"),
                    make_block("p1_b2", "text", 2, [80, 120, 900, 480], "Продолжение статьи."),
                ],
            ),
            make_page(
                2,
                [
                    make_block("p2_b1", "title", 1, [120, 24, 760, 70], "Вестник Академии Наук"),
                    make_block("p2_b2", "text", 2, [80, 120, 900, 480], "Ещё одна страница той же статьи."),
                ],
            ),
        ]

        result = self.segmenter.segment_document_into_articles(pages)

        self.assertEqual(result.article_count, 1)
        self.assertFalse(result.boundaries[1].is_article_start)
        self.assertGreaterEqual(result.boundaries[1].debug_info["header_similarity_score"], 0.86)
        self.assertEqual(result.boundaries[1].debug_info["decision"], "continue_article")

    def test_marks_ambiguous_article_boundary_for_review(self) -> None:
        pages = [
            make_page(
                1,
                [
                    make_block("p1_b1", "title", 1, [90, 30, 840, 120], "Первая статья"),
                    make_block("p1_b2", "text", 2, [80, 220, 900, 500], "Текст первой статьи."),
                ],
            ),
            make_page(
                2,
                [
                    make_block("p2_b1", "title", 1, [90, 320, 840, 390], "Сомнительный новый заголовок"),
                    make_block("p2_b2", "text", 2, [80, 420, 900, 600], "Текст после сомнительного заголовка."),
                ],
            ),
        ]

        result = self.segmenter.segment_document_into_articles(pages)

        self.assertEqual(result.article_count, 2)
        self.assertTrue(result.articles[1].needs_review)
        self.assertIn(
            result.articles[1].debug_info["decision"],
            {"article_start_rescue", "article_start_review"},
        )
        self.assertEqual(pages[1].article_title, "Сомнительный новый заголовок")


if __name__ == "__main__":
    unittest.main()
