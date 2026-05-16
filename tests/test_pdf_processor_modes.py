from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from app.schemas import (
    ArticleSegmentationResult,
    Block,
    LayoutAnalysisResponse,
    LayoutBlock,
    OCRResult,
    PageContent,
    ProcessedBlock,
)
from app.services import pdf_processor


class FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_text(self, mode: str) -> str:
        if mode != "text":
            raise AssertionError(f"Unexpected text mode: {mode}")
        return self._text


class FakeDocument:
    def __init__(self, pages: list[FakePage]) -> None:
        self._pages = pages
        self.page_count = len(pages)

    def __iter__(self):
        return iter(self._pages)

    def __enter__(self) -> FakeDocument:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def make_layout_analysis() -> LayoutAnalysisResponse:
    return LayoutAnalysisResponse(
        analysis_id="analysis_001",
        visualization_url="/visualization?analysis_id=analysis_001",
        visualization_data_url="data:image/png;base64,layout",
        result_json_path="/tmp/layout.json",
        blocks=[
            LayoutBlock(
                type="text",
                bbox=[0, 0, 100, 40],
                confidence=0.98,
                reading_order=1,
                text="Seed text",
                latex=None,
            )
        ],
    )


def make_page_content(page_number: int, text: str) -> PageContent:
    return PageContent(
        page_content_id=f"page_content_{page_number}",
        page_number=page_number,
        page_text=text,
        needs_review_count=0,
        result_json_path="/tmp/page-content.json",
        presentation=None,
        blocks=[
            ProcessedBlock(
                block_id=f"page_{page_number}_block_001",
                type="text",
                reading_order=1,
                bbox=[0, 0, 100, 100],
                route_to="text_pipeline",
                content=text,
                confidence=0.95,
                needs_review=False,
                crop_data_url=None,
                crop_path=None,
                latex=None,
                formula_result=None,
                formula_backend=None,
                ocr_result=OCRResult(text=text, confidence=0.95),
                ocr_backend="paddleocr",
            )
        ],
    )


class PDFProcessorModeTests(unittest.TestCase):
    def test_full_mode_uses_preprocessing_for_pages_without_text(self) -> None:
        fake_document = FakeDocument([FakePage("")])
        pixmap = object()
        image_rgb = np.zeros((60, 40, 3), dtype=np.uint8)
        layout_analysis = make_layout_analysis()
        routed_blocks = [
            Block(
                block_id="page_1_block_001",
                type="text",
                bbox=[0, 0, 100, 40],
                reading_order=1,
                route_to="text_pipeline",
                seed_text="Seed text",
                seed_latex=None,
                seed_confidence=0.98,
            )
        ]
        page_content = make_page_content(1, "Recognized text")

        with (
            mock.patch("app.services.pdf_processor.fitz.open", return_value=fake_document),
            mock.patch("app.services.pdf_processor.render_page", return_value=pixmap),
            mock.patch("app.services.pdf_processor.pixmap_to_data_url", return_value="page-preview"),
            mock.patch("app.services.pdf_processor.pixmap_to_numpy_rgb", return_value=image_rgb),
            mock.patch(
                "app.services.pdf_processor.preprocess_page_image_to_data_url",
                return_value="ocr-preview",
            ) as preprocess_mock,
            mock.patch(
                "app.services.pdf_processor.analyze_page_layout",
                return_value=layout_analysis,
            ) as analyze_mock,
            mock.patch(
                "app.services.pdf_processor.build_routed_blocks_from_layout",
                return_value=routed_blocks,
            ),
            mock.patch(
                "app.services.pdf_processor.process_text_blocks",
                return_value=page_content,
            ),
            mock.patch(
                "app.services.pdf_processor.process_formula_blocks",
                return_value=page_content,
            ) as formula_mock,
            mock.patch(
                "app.services.pdf_processor.segment_document_into_articles",
                return_value=ArticleSegmentationResult(
                    total_pages=1,
                    article_count=1,
                    needs_review_count=0,
                ),
            ) as segment_mock,
        ):
            result = pdf_processor.process_pdf_document(b"pdf", processing_mode="full")

        self.assertEqual(result.processing_mode, "full")
        self.assertEqual(result.pages[0].processing_mode, "full")
        self.assertEqual(result.pages[0].page_image_data_url, "page-preview")
        self.assertEqual(result.pages[0].image_data_url, "ocr-preview")
        preprocess_mock.assert_called_once_with(pixmap)
        analyze_mock.assert_called_once_with(image_rgb)
        formula_mock.assert_called_once()
        segment_mock.assert_called_once()

    def test_no_preprocessing_mode_skips_ocr_preprocessing(self) -> None:
        fake_document = FakeDocument([FakePage("")])
        pixmap = object()
        image_rgb = np.zeros((60, 40, 3), dtype=np.uint8)
        layout_analysis = make_layout_analysis()
        page_content = make_page_content(1, "Recognized text")

        with (
            mock.patch("app.services.pdf_processor.fitz.open", return_value=fake_document),
            mock.patch("app.services.pdf_processor.render_page", return_value=pixmap),
            mock.patch("app.services.pdf_processor.pixmap_to_data_url", return_value="page-preview"),
            mock.patch("app.services.pdf_processor.pixmap_to_numpy_rgb", return_value=image_rgb),
            mock.patch(
                "app.services.pdf_processor.preprocess_page_image_to_data_url"
            ) as preprocess_mock,
            mock.patch(
                "app.services.pdf_processor.analyze_page_layout",
                return_value=layout_analysis,
            ),
            mock.patch(
                "app.services.pdf_processor.build_routed_blocks_from_layout",
                return_value=[],
            ),
            mock.patch(
                "app.services.pdf_processor.process_text_blocks",
                return_value=page_content,
            ),
            mock.patch(
                "app.services.pdf_processor.process_formula_blocks",
                return_value=page_content,
            ),
            mock.patch(
                "app.services.pdf_processor.segment_document_into_articles",
                return_value=ArticleSegmentationResult(
                    total_pages=1,
                    article_count=1,
                    needs_review_count=0,
                ),
            ),
        ):
            result = pdf_processor.process_pdf_document(
                b"pdf",
                processing_mode="no_preprocessing",
            )

        self.assertEqual(result.processing_mode, "no_preprocessing")
        self.assertEqual(result.pages[0].processing_mode, "no_preprocessing")
        self.assertEqual(result.pages[0].page_image_data_url, "page-preview")
        self.assertIsNone(result.pages[0].image_data_url)
        preprocess_mock.assert_not_called()

    def test_text_only_mode_uses_embedded_text_without_rendering(self) -> None:
        embedded_text = "Это уже хороший встроенный текст. " * 8
        fake_document = FakeDocument([FakePage(embedded_text)])

        with (
            mock.patch("app.services.pdf_processor.fitz.open", return_value=fake_document),
            mock.patch("app.services.pdf_processor.render_page") as render_mock,
            mock.patch(
                "app.services.pdf_processor.segment_document_into_articles"
            ) as segment_mock,
        ):
            result = pdf_processor.process_pdf_document(
                b"pdf",
                processing_mode="text_only",
            )

        self.assertEqual(result.processing_mode, "text_only")
        self.assertEqual(result.pages[0].processing_mode, "text_only")
        self.assertIsNone(result.pages[0].page_image_data_url)
        self.assertIsNone(result.pages[0].layout_analysis)
        self.assertIsNone(result.article_segmentation)
        self.assertIsNotNone(result.pages[0].text_block_content)
        self.assertIn("Это уже хороший встроенный текст.", result.pages[0].text_block_content.page_text)
        self.assertEqual(
            result.pages[0].text_block_content.blocks[0].ocr_backend,
            "pdf_text",
        )
        render_mock.assert_not_called()
        segment_mock.assert_not_called()

    def test_text_only_mode_falls_back_to_surya_when_paddle_returns_empty_text(self) -> None:
        fake_document = FakeDocument([FakePage("")])
        pixmap = object()
        image_rgb = np.zeros((60, 40, 3), dtype=np.uint8)

        with (
            mock.patch("app.services.pdf_processor.fitz.open", return_value=fake_document),
            mock.patch("app.services.pdf_processor.render_page", return_value=pixmap),
            mock.patch("app.services.pdf_processor.pixmap_to_data_url", return_value="page-preview"),
            mock.patch("app.services.pdf_processor.pixmap_to_numpy_rgb", return_value=image_rgb),
            mock.patch(
                "app.services.pdf_processor.recognize_text_block",
                return_value=OCRResult(text="", confidence=0.0),
            ) as paddle_mock,
            mock.patch(
                "app.services.pdf_processor.recognize_page_text",
                return_value=OCRResult(text="Recognized via Surya", confidence=0.93),
            ) as surya_mock,
            mock.patch(
                "app.services.pdf_processor.segment_document_into_articles"
            ) as segment_mock,
        ):
            result = pdf_processor.process_pdf_document(
                b"pdf",
                processing_mode="text_only",
            )

        page = result.pages[0]
        self.assertEqual(result.processing_mode, "text_only")
        self.assertEqual(page.page_image_data_url, "page-preview")
        self.assertIsNotNone(page.text_block_content)
        self.assertEqual(page.text_block_content.page_text, "Recognized via Surya")
        self.assertEqual(page.text_block_content.blocks[0].ocr_backend, "surya_page")
        paddle_mock.assert_called_once_with(image_rgb)
        surya_mock.assert_called_once_with(image_rgb)
        segment_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
