from __future__ import annotations

import unittest

from app.schemas import ArticleContent, PageContent, ProcessedBlock
from app.services.metadata_extractor import extract_article_metadata


def make_block(
    block_id: str,
    block_type: str,
    reading_order: int,
    content: str | None = None,
) -> ProcessedBlock:
    return ProcessedBlock(
        block_id=block_id,
        type=block_type,
        reading_order=reading_order,
        bbox=[10, reading_order * 20, 200, reading_order * 20 + 10],
        route_to="text_pipeline",
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


def make_page(page_number: int, blocks: list[ProcessedBlock], page_text: str) -> PageContent:
    return PageContent(
        page_content_id=f"page_content_{page_number}",
        page_number=page_number,
        article_title=None,
        page_text=page_text,
        needs_review_count=0,
        result_json_path=None,
        presentation=None,
        blocks=blocks,
    )


class MetadataExtractorTests(unittest.TestCase):
    def test_extracts_full_metadata_from_article_content(self) -> None:
        article_text = """
Дифференциальное уравнение в конической форме
Д. М. Синцов
Казань, 1910

Аннотация
Рассматривается дифференциальное уравнение в конической форме и его линейчатое представление.

Ключевые слова: дифференциальное уравнение, коническая форма, линейчатая форма

Основной текст статьи.

Литература
1. Синцов Д. М. О линейчатых формах. 1909.
2. Darboux G. Lecons sur les systemes orthogonaux. 1908.
        """.strip()

        article = ArticleContent(
            article_id="article_meta",
            article_text=article_text,
            pages=[
                make_page(
                    1,
                    [
                        make_block("title_1", "title", 1, "Дифференциальное уравнение в конической форме"),
                        make_block("author_1", "text", 2, "Д. М. Синцов"),
                        make_block("text_1", "text", 3, "Аннотация"),
                    ],
                    article_text,
                )
            ],
        )

        metadata = extract_article_metadata(article)

        self.assertEqual(metadata.title, "Дифференциальное уравнение в конической форме")
        self.assertEqual(metadata.authors, ["Д. М. Синцов"])
        self.assertEqual(metadata.language, "ru")
        self.assertEqual(metadata.year, 1910)
        self.assertEqual(
            metadata.abstract,
            "Рассматривается дифференциальное уравнение в конической форме и его линейчатое представление.",
        )
        self.assertEqual(
            metadata.keywords,
            ["дифференциальное уравнение", "коническая форма", "линейчатая форма"],
        )
        self.assertEqual(len(metadata.references), 2)
        self.assertEqual(metadata.references[0].year, 1909)
        self.assertEqual(metadata.field_sources["title"], "title_block")
        self.assertEqual(metadata.field_sources["authors"], "post_title_blocks")

    def test_falls_back_to_leading_lines_and_detects_english(self) -> None:
        article_text = """
On Differential Equations of the Second Order
J. Smith
Cambridge, 1924

Abstract
This paper studies a differential equation and its geometric interpretation.

The main text follows.
        """.strip()

        article = ArticleContent(
            article_id="article_meta_en",
            article_text=article_text,
            normalized_text=article_text,
        )

        metadata = extract_article_metadata(article)

        self.assertEqual(metadata.title, "On Differential Equations of the Second Order")
        self.assertEqual(metadata.authors, ["J. Smith"])
        self.assertEqual(metadata.language, "en")
        self.assertEqual(metadata.year, 1924)
        self.assertIn("differential equation", " ".join(metadata.keywords))


if __name__ == "__main__":
    unittest.main()
