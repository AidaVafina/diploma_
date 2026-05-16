from __future__ import annotations

import io
import json
import unittest
import zipfile
from unittest.mock import patch

from app.schemas import (
    ArticleMetadata,
    CollectionExportArticleRequest,
    CollectionExportRequest,
    ReadablePdfBlock,
    ReadablePdfExportRequest,
    ReadablePdfSection,
)
from app.services.collection_exporter import (
    CollectionExportError,
    export_collection_archive,
    resolve_collection_archive_filename,
)


def make_readable_pdf_payload(title: str, filename: str) -> ReadablePdfExportRequest:
    return ReadablePdfExportRequest(
        title=title,
        filename=filename,
        sections=[
            ReadablePdfSection(
                blocks=[
                    ReadablePdfBlock(kind="title", text=title),
                    ReadablePdfBlock(kind="paragraph", text="Текст статьи."),
                ]
            )
        ],
    )


class CollectionExporterTests(unittest.TestCase):
    def test_exports_collection_archive_with_expected_files(self) -> None:
        payload = CollectionExportRequest(
            title="Коллекция статей",
            filename="journal-collection",
            source_document_name="journal.pdf",
            processing_mode="full",
            articles=[
                CollectionExportArticleRequest(
                    article_id="article_001",
                    page_numbers=[1, 2],
                    title="Об одном уравнении",
                    author="Д. С. Синцов",
                    article_text="Текст статьи.",
                    article_latex_document="\\documentclass{article}\n\\begin{document}Текст статьи.\\end{document}",
                    article_metadata=ArticleMetadata(
                        title="Об одном уравнении",
                        authors=["Д. С. Синцов"],
                        language="ru",
                        year=1910,
                        keywords=["уравнение", "геометрия"],
                    ),
                    readable_pdf=make_readable_pdf_payload(
                        "Об одном уравнении",
                        "article_001-readable.pdf",
                    ),
                )
            ],
        )

        with patch(
            "app.services.collection_exporter._export_article_pdf",
            return_value=b"%PDF-test%",
        ):
            archive_bytes = export_collection_archive(payload)

        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
            names = set(archive.namelist())
            self.assertIn("Коллекция статей/manifest.json", names)
            self.assertIn("Коллекция статей/catalog.csv", names)
            self.assertIn(
                "Коллекция статей/articles/001_article_001/metadata.json",
                names,
            )
            self.assertIn(
                "Коллекция статей/articles/001_article_001/article.txt",
                names,
            )
            self.assertIn(
                "Коллекция статей/articles/001_article_001/article.tex",
                names,
            )
            self.assertIn(
                "Коллекция статей/articles/001_article_001/article_001-readable.pdf",
                names,
            )

            manifest = json.loads(
                archive.read("Коллекция статей/manifest.json").decode("utf-8")
            )
            metadata = json.loads(
                archive.read(
                    "Коллекция статей/articles/001_article_001/metadata.json"
                ).decode("utf-8")
            )
            catalog = archive.read("Коллекция статей/catalog.csv").decode("utf-8")
            pdf_bytes = archive.read(
                "Коллекция статей/articles/001_article_001/article_001-readable.pdf"
            )

        self.assertEqual(manifest["article_count"], 1)
        self.assertEqual(manifest["processing_mode"], "full")
        self.assertEqual(metadata["title"], "Об одном уравнении")
        self.assertEqual(metadata["authors"], ["Д. С. Синцов"])
        self.assertEqual(metadata["language"], "ru")
        self.assertIn("article_001", catalog)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_rejects_empty_collection(self) -> None:
        payload = CollectionExportRequest(
            title="Пустая коллекция",
            articles=[],
        )

        with self.assertRaises(CollectionExportError):
            export_collection_archive(payload)

    def test_resolves_collection_filename(self) -> None:
        filename = resolve_collection_archive_filename('  collection:/\\"01"  ', "fallback")

        self.assertEqual(filename, "collection_01.zip")


if __name__ == "__main__":
    unittest.main()
