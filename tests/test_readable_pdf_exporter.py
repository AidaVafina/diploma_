from __future__ import annotations

import base64
import io
import unittest

import fitz
from PIL import Image, ImageDraw

from app.schemas import ReadablePdfBlock, ReadablePdfExportRequest, ReadablePdfSection
from app.services.readable_pdf_exporter import (
    export_readable_pdf,
    resolve_readable_pdf_filename,
)


def extract_pdf_text(pdf_bytes: bytes) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        return "\n".join(page.get_text("text") for page in document)


def build_formula_image_data_url() -> str:
    image = Image.new("RGB", (120, 40), "white")
    draw = ImageDraw.Draw(image)
    draw.line((12, 28, 108, 28), fill="black", width=2)
    draw.text((42, 8), "x", fill="black")
    draw.text((70, 8), "y", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


FORMULA_IMAGE_DATA_URL = build_formula_image_data_url()


class ReadablePdfExporterTests(unittest.TestCase):
    def test_exports_readable_pdf_with_document_and_article_content(self) -> None:
        payload = ReadablePdfExportRequest(
            title="Читаемый текст документа",
            subtitle="Источник: tome1.pdf | Страниц: 2",
            sections=[
                ReadablePdfSection(
                    heading="Страница 1",
                    note="Заголовок статьи: Об одном уравнении",
                    blocks=[
                        ReadablePdfBlock(kind="title", text="Об одном уравнении"),
                        ReadablePdfBlock(kind="author", text="Д. С. Синцов"),
                        ReadablePdfBlock(kind="paragraph", text="Первый абзац статьи."),
                        ReadablePdfBlock(kind="formula", text=r"\int_a^b f(x)\,dx"),
                        ReadablePdfBlock(kind="placeholder", text="[TABLE]"),
                    ],
                )
            ],
        )

        pdf_bytes = export_readable_pdf(payload)
        extracted_text = extract_pdf_text(pdf_bytes)

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertIn("Об одном уравнении", extracted_text)
        self.assertIn("Д. С. Синцов", extracted_text)
        self.assertIn("Первый абзац статьи.", extracted_text)
        self.assertIn(r"\int_a^b f(x)\,dx", extracted_text)
        self.assertIn("[TABLE]", extracted_text)
        self.assertNotIn("Читаемый текст документа", extracted_text)
        self.assertNotIn("Страница 1", extracted_text)
        self.assertNotIn("Заголовок статьи:", extracted_text)

    def test_splits_long_text_across_multiple_pdf_pages(self) -> None:
        payload = ReadablePdfExportRequest(
            title="Длинный текст",
            sections=[
                ReadablePdfSection(
                    heading="Страница 7",
                    blocks=[
                        ReadablePdfBlock(
                            kind="paragraph",
                            text=("Многострочный фрагмент " * 2500).strip(),
                        )
                    ],
                )
            ],
        )

        pdf_bytes = export_readable_pdf(payload)

        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            self.assertGreater(document.page_count, 2)
            extracted_text = "\n".join(page.get_text("text") for page in document)

        self.assertIn("Многострочный фрагмент", extracted_text)
        self.assertNotIn("Длинный текст", extracted_text)
        self.assertNotIn("Страница 7", extracted_text)

    def test_embeds_formula_image_when_provided(self) -> None:
        payload = ReadablePdfExportRequest(
            title="Формула",
            sections=[
                ReadablePdfSection(
                    blocks=[
                        ReadablePdfBlock(kind="paragraph", text="До формулы."),
                        ReadablePdfBlock(
                            kind="formula",
                            text=r"\int_a^b f(x)\,dx",
                            image_data_url=FORMULA_IMAGE_DATA_URL,
                            image_width=120,
                            image_height=40,
                        ),
                        ReadablePdfBlock(kind="paragraph", text="После формулы."),
                    ]
                )
            ],
        )

        pdf_bytes = export_readable_pdf(payload)

        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            page = document[0]
            extracted_text = page.get_text("text")
            embedded_images = page.get_images(full=True)

        self.assertIn("До формулы.", extracted_text)
        self.assertIn("После формулы.", extracted_text)
        self.assertNotIn(r"\int_a^b f(x)\,dx", extracted_text)
        self.assertGreater(len(embedded_images), 0)

    def test_sanitizes_export_filename(self) -> None:
        filename = resolve_readable_pdf_filename('  article:/\\"01"  ', "fallback")

        self.assertEqual(filename, "article_01.pdf")


if __name__ == "__main__":
    unittest.main()
