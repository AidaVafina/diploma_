from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from typing import Any

import fitz

from app.core.config import settings
from app.schemas import DocumentProcessingResult, PageProcessingResult
from app.services.article_segmenter import segment_document_into_articles
from app.services.formula_block_processor import process_formula_blocks
from app.services.image_processor import (
    ImageProcessingError,
    pixmap_to_data_url,
    pixmap_to_numpy_rgb,
    preprocess_page_image_to_data_url,
)
from app.services.layout_analysis_surya import (
    LayoutAnalysisError,
    SuryaNotAvailableError,
    analyze_page_layout,
)
from app.services.text_block_processor import (
    build_routed_blocks_from_layout,
    process_text_blocks,
)

logger = logging.getLogger(__name__)

WHITESPACE_RE = re.compile(r"\s+")


class PDFProcessingError(RuntimeError):
    pass


class InvalidPDFError(PDFProcessingError):
    pass


ProgressEvent = dict[str, Any]


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub("", text)


def has_enough_text(text: str, threshold: int | None = None) -> bool:
    return len(normalize_text(text)) >= (threshold or settings.text_length_threshold)

# рендер страницы в изображение, так как layout-анализ, OCR блоков и распознавание формул работают по изображению страницы
def render_page(page: fitz.Page, dpi: int | None = None) -> fitz.Pixmap:
    scale = (dpi or settings.render_dpi) / 72
    matrix = fitz.Matrix(scale, scale)
    return page.get_pixmap(matrix=matrix, alpha=False)


def iter_pdf_processing_events(pdf_bytes: bytes) -> Iterator[ProgressEvent]:
    results: list[PageProcessingResult] = []

    try:
        yield {
            "type": "progress",
            "message": "Открываем PDF-документ",
        }
        # открытие документа
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except fitz.FileDataError as exc:
        logger.exception("Invalid PDF data received")
        raise InvalidPDFError(
            "Не удалось открыть PDF: файл поврежден или имеет неверный формат."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive branch
        logger.exception("Unexpected error while opening PDF document")
        raise PDFProcessingError("Ошибка открытия PDF-документа.") from exc

    with document:
        logger.info("Opened PDF document with %s pages", document.page_count)
        total_pages = document.page_count
        yield {
            "type": "progress",
            "message": f"Документ открыт: {total_pages} страниц",
        }

        # проход по всем страницам PDF
        for page_number, page in enumerate(document, start=1):
            logger.info("Processing page %s", page_number)
            page_label = f"страница {page_number}/{total_pages}"
            yield {
                "type": "progress",
                "message": f"Извлекаем встроенный текст, {page_label}",
                "page": page_number,
                "total_pages": total_pages,
            }
            extracted_text = (page.get_text("text") or "").strip()
            text_is_enough = has_enough_text(extracted_text)
            page_image_data_url: str | None = None
            image_data_url: str | None = None
            layout_analysis = None
            layout_error: str | None = None
            text_block_content = None
            text_block_error: str | None = None
            formula_block_error: str | None = None

            try:
                yield {
                    "type": "progress",
                    "message": f"Рендерим страницу в изображение, {page_label}",
                    "page": page_number,
                    "total_pages": total_pages,
                }
                pixmap = render_page(page)
                page_image_data_url = pixmap_to_data_url(pixmap)
                page_image_rgb = pixmap_to_numpy_rgb(pixmap)

                if not text_is_enough:
                    logger.info("Page %s marked for OCR preparation", page_number)
                    yield {
                        "type": "progress",
                        "message": f"Готовим изображение для OCR, {page_label}",
                        "page": page_number,
                        "total_pages": total_pages,
                    }
                    image_data_url = preprocess_page_image_to_data_url(pixmap)
                try:
                    yield {
                        "type": "progress",
                        "message": f"Выполняем layout-анализ и Surya OCR, {page_label}",
                        "page": page_number,
                        "total_pages": total_pages,
                    }
                    layout_analysis = analyze_page_layout(page_image_rgb)
                except SuryaNotAvailableError as exc:
                    logger.exception("Surya is unavailable for page %s", page_number)
                    layout_error = str(exc)
                    yield {
                        "type": "progress",
                        "message": f"Layout-анализ недоступен, {page_label}",
                        "page": page_number,
                        "total_pages": total_pages,
                    }
                except LayoutAnalysisError as exc:
                    logger.exception("Layout analysis failed for page %s", page_number)
                    layout_error = str(exc)
                    yield {
                        "type": "progress",
                        "message": f"Ошибка layout-анализа, {page_label}",
                        "page": page_number,
                        "total_pages": total_pages,
                    }

                if layout_analysis is not None:
                    try:
                        yield {
                            "type": "progress",
                            "message": f"Распознаём текстовые блоки, {page_label}",
                            "page": page_number,
                            "total_pages": total_pages,
                        }
                        routed_blocks = build_routed_blocks_from_layout(
                            layout_analysis.blocks,
                            block_id_prefix=f"page_{page_number}",
                        )
                        text_block_content = process_text_blocks(
                            page_image_rgb,
                            routed_blocks,
                            page_number=page_number,
                        )
                        yield {
                            "type": "progress",
                            "message": f"Распознаём формулы, {page_label}",
                            "page": page_number,
                            "total_pages": total_pages,
                        }
                        text_block_content = process_formula_blocks(
                            page_image_rgb,
                            text_block_content.blocks,
                            page_number=page_number,
                            page_content_id=text_block_content.page_content_id,
                        )
                    except Exception as exc:  # pragma: no cover - fallback branch
                        logger.exception(
                            "Automatic text/formula block processing failed for page %s",
                            page_number,
                        )
                        if text_block_content is None:
                            text_block_error = str(exc)
                        else:
                            formula_block_error = str(exc)
                        yield {
                            "type": "progress",
                            "message": f"Ошибка OCR или формул, {page_label}",
                            "page": page_number,
                            "total_pages": total_pages,
                        }
            except PDFProcessingError:
                raise
            except ImageProcessingError as exc:
                raise PDFProcessingError(
                    f"Ошибка подготовки изображения для страницы {page_number}."
                ) from exc
            except Exception as exc:  # pragma: no cover - defensive branch
                logger.exception("Unexpected page rendering failure on page %s", page_number)
                raise PDFProcessingError(
                    f"Ошибка рендера страницы {page_number}."
                ) from exc

            results.append(
                PageProcessingResult(
                    page_number=page_number,
                    has_text=text_is_enough,
                    text=extracted_text,
                    page_image_data_url=page_image_data_url,
                    image_path=None,
                    image_data_url=image_data_url,
                    layout_analysis=layout_analysis,
                    layout_error=layout_error,
                    text_block_content=text_block_content,
                    text_block_error=text_block_error,
                    formula_block_error=formula_block_error,
                )
            )

    article_segmentation = None
    processed_page_contents = [
        page.text_block_content for page in results if page.text_block_content is not None
    ]
    if processed_page_contents:
        try:
            yield {
                "type": "progress",
                "message": "Разделяем документ на статьи",
            }
            article_segmentation = segment_document_into_articles(processed_page_contents)
        except Exception:  # pragma: no cover - defensive branch
            logger.exception("Article segmentation failed after PDF processing")
            yield {
                "type": "progress",
                "message": "Не удалось разделить документ на статьи",
            }

    logger.info("Finished processing PDF document")
    result = DocumentProcessingResult(
        pages=results,
        article_segmentation=article_segmentation,
    )
    yield {
        "type": "result",
        "message": "Обработка завершена",
        "data": result.model_dump(mode="json"),
    }


def process_pdf_document(pdf_bytes: bytes) -> DocumentProcessingResult:
    result: DocumentProcessingResult | None = None
    for event in iter_pdf_processing_events(pdf_bytes):
        if event.get("type") == "result":
            result = DocumentProcessingResult.model_validate(event["data"])

    if result is None:  # pragma: no cover - defensive branch
        raise PDFProcessingError("Обработка PDF завершилась без результата.")

    return result
