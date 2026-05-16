from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from app.schemas import CollectionExportArticleRequest, CollectionExportRequest

FILENAME_SANITIZE_RE = re.compile(r'[\\/:*?"<>|]+')
MULTISPACE_RE = re.compile(r"\s+")
DEFAULT_ARCHIVE_FILENAME = "collection-export.zip"


class CollectionExportError(RuntimeError):
    pass


def _sanitize_filename_stem(raw_value: str | None, fallback: str) -> str:
    value = (raw_value or fallback or "").strip().replace("\x00", "")
    value = FILENAME_SANITIZE_RE.sub("_", value)
    value = MULTISPACE_RE.sub(" ", value).strip(" ._")
    return value or fallback


def resolve_collection_archive_filename(
    raw_filename: str | None,
    fallback_title: str = "collection-export",
) -> str:
    filename = _sanitize_filename_stem(raw_filename, fallback_title or "collection-export")
    if not filename.lower().endswith(".zip"):
        filename = f"{filename}.zip"
    return filename


def _article_authors(article: CollectionExportArticleRequest) -> list[str]:
    if article.article_metadata and article.article_metadata.authors:
        return [author.strip() for author in article.article_metadata.authors if author.strip()]
    if article.author.strip():
        return [article.author.strip()]
    return []


def _article_title(article: CollectionExportArticleRequest) -> str:
    if article.article_metadata and article.article_metadata.title:
        return article.article_metadata.title.strip()
    if article.title.strip():
        return article.title.strip()
    return article.article_id


def _build_article_metadata_payload(
    article: CollectionExportArticleRequest,
) -> dict[str, object]:
    article_metadata = article.article_metadata.model_dump(mode="json") if article.article_metadata else {}
    payload: dict[str, object] = {
        "article_id": article.article_id,
        "title": _article_title(article),
        "authors": _article_authors(article),
        "page_numbers": article.page_numbers,
        "article_text_available": bool(article.article_text.strip()),
        "latex_available": bool(
            article.article_latex_document.strip() or article.article_latex_preview.strip()
        ),
    }
    payload.update(article_metadata)
    payload["title"] = payload.get("title") or _article_title(article)
    payload["authors"] = payload.get("authors") or _article_authors(article)
    payload["page_numbers"] = payload.get("page_numbers") or article.page_numbers
    return payload


def _build_catalog_csv(articles: list[CollectionExportArticleRequest]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=";")
    writer.writerow(
        [
            "article_id",
            "title",
            "authors",
            "language",
            "year",
            "pages",
            "keywords",
            "needs_review",
        ]
    )

    for article in articles:
        metadata = _build_article_metadata_payload(article)
        writer.writerow(
            [
                article.article_id,
                metadata.get("title", ""),
                ", ".join(metadata.get("authors", [])),
                metadata.get("language", "") or "",
                metadata.get("year", "") or "",
                ", ".join(str(page) for page in article.page_numbers),
                ", ".join(metadata.get("keywords", [])),
                "true" if metadata.get("needs_review") else "false",
            ]
        )

    return buffer.getvalue()


def _article_folder_name(article: CollectionExportArticleRequest, index: int) -> str:
    stem = _sanitize_filename_stem(article.article_id or _article_title(article), f"article_{index:03d}")
    return f"{index:03d}_{stem}"


def _article_latex_text(article: CollectionExportArticleRequest) -> str:
    latex_text = article.article_latex_document.strip() or article.article_latex_preview.strip()
    return latex_text or "% LaTeX статьи недоступен.\n"


def _export_article_pdf(article: CollectionExportArticleRequest) -> bytes:
    from app.services.readable_pdf_exporter import ReadablePdfExportError, export_readable_pdf

    try:
        return export_readable_pdf(article.readable_pdf)
    except ReadablePdfExportError as exc:
        raise CollectionExportError(
            f"Не удалось сформировать PDF для статьи {article.article_id}: {exc}"
        ) from exc


def export_collection_archive(payload: CollectionExportRequest) -> bytes:
    if not payload.articles:
        raise CollectionExportError("В коллекции нет статей для экспорта.")

    archive_root = _sanitize_filename_stem(payload.title, "collection")
    exported_at = datetime.now(timezone.utc).isoformat()
    manifest_articles: list[dict[str, object]] = []
    archive_buffer = io.BytesIO()

    with zipfile.ZipFile(
        archive_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(f"{archive_root}/", "")
        archive.writestr(f"{archive_root}/articles/", "")

        for index, article in enumerate(payload.articles, start=1):
            folder_name = _article_folder_name(article, index)
            article_root = f"{archive_root}/articles/{folder_name}"
            metadata_payload = _build_article_metadata_payload(article)
            pdf_filename = Path(article.readable_pdf.filename or f"{article.article_id}.pdf").name
            pdf_bytes = _export_article_pdf(article)

            archive.writestr(f"{article_root}/", "")
            archive.writestr(
                f"{article_root}/metadata.json",
                json.dumps(metadata_payload, ensure_ascii=False, indent=2),
            )
            archive.writestr(
                f"{article_root}/article.txt",
                (article.article_text or "").strip(),
            )
            archive.writestr(
                f"{article_root}/article.tex",
                _article_latex_text(article),
            )
            archive.writestr(f"{article_root}/{pdf_filename}", pdf_bytes)

            manifest_articles.append(
                {
                    "article_id": article.article_id,
                    "title": metadata_payload.get("title", article.article_id),
                    "authors": metadata_payload.get("authors", []),
                    "page_numbers": article.page_numbers,
                    "folder": f"articles/{folder_name}",
                    "files": {
                        "metadata": f"articles/{folder_name}/metadata.json",
                        "text": f"articles/{folder_name}/article.txt",
                        "latex": f"articles/{folder_name}/article.tex",
                        "readable_pdf": f"articles/{folder_name}/{pdf_filename}",
                    },
                }
            )

        manifest = {
            "title": payload.title,
            "source_document_name": payload.source_document_name,
            "processing_mode": payload.processing_mode,
            "article_count": len(payload.articles),
            "exported_at": exported_at,
            "articles": manifest_articles,
        }
        archive.writestr(
            f"{archive_root}/manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
        archive.writestr(
            f"{archive_root}/catalog.csv",
            _build_catalog_csv(payload.articles),
        )

    return archive_buffer.getvalue()
