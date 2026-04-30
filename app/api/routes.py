from __future__ import annotations

import logging
import json

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import HTMLResponse
from fastapi.responses import Response

from app.core.config import settings
from app.schemas import (
    ArticleContent,
    ArticleLatexDocumentResult,
    ArticleSegmentationRequest,
    ArticleSegmentationResult,
    Block,
    DocumentProcessingResult,
    LatexBuildResult,
    LayoutAnalysisResponse,
    PageContent,
    PageTextResponse,
    TextPostprocessRequest,
    TextPostprocessResponse,
)
from app.services.article_segmenter import segment_document_into_articles
from app.services.formula_block_processor import (
    FormulaBlockProcessingError,
    FormulaCropNotFoundError,
    get_formula_crop_bytes,
    process_formula_blocks,
)
from app.services.article_latex_builder import (
    ArticleLatexNotFoundError,
    build_article_latex_document,
    get_cached_article_latex,
)
from app.services.latex_builder import build_page_latex
from app.services.layout_analysis_surya import (
    LayoutAnalysisError,
    SuryaNotAvailableError,
    VisualizationNotFoundError,
    analyze_page_layout,
    get_cached_visualization,
    load_image_bytes,
)
from app.services.pdf_processor import (
    InvalidPDFError,
    PDFProcessingError,
    process_pdf_document,
)
from app.services.text_postprocessor import postprocess_text
from app.services.text_block_processor import (
    PaddleOCRNotAvailableError,
    PageTextNotFoundError,
    TextBlockProcessingError,
    get_cached_page_content,
    get_cached_page_text,
    process_text_blocks,
)

logger = logging.getLogger(__name__)
router = APIRouter()


async def read_uploaded_pdf(file: UploadFile) -> bytes:
    file_bytes = await file.read()
    await file.close()

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Загруженный файл пуст.",
        )

    return file_bytes


def parse_blocks_payload(blocks_json: str) -> list[Block]:
    try:
        raw_blocks = json.loads(blocks_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Поле blocks должно быть валидным JSON-массивом.",
        ) from exc

    if not isinstance(raw_blocks, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Поле blocks должно содержать список блоков.",
        )

    try:
        return [Block.model_validate(raw_block) for raw_block in raw_blocks]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Некорректная структура routed blocks: {exc}",
        ) from exc


async def read_uploaded_image(file: UploadFile) -> bytes:
    file_bytes = await file.read()
    await file.close()

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Загруженное изображение пусто.",
        )

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Загрузите изображение страницы.",
        )

    return file_bytes


@router.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(
        content=(settings.static_dir / "index.html").read_text(encoding="utf-8")
    )

# получение файла пользователя с фронта
@router.post(
    "/api/process-pdf",
    response_model=DocumentProcessingResult,
    status_code=status.HTTP_200_OK,
)
async def process_pdf(file: UploadFile = File(...)) -> DocumentProcessingResult:
    logger.info(
        "Received file upload: name=%s, content_type=%s",
        file.filename,
        file.content_type,
    )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не удалось определить имя файла.",
        )

    if (
        not file.filename.lower().endswith(".pdf")
        and file.content_type != "application/pdf"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Загрузите PDF-файл.",
        )

    pdf_bytes = await read_uploaded_pdf(file)

    try:
        return process_pdf_document(pdf_bytes)
    except InvalidPDFError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except PDFProcessingError as exc:
        logger.exception("Failed to process uploaded PDF")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.post(
    "/segment-articles",
    response_model=ArticleSegmentationResult,
    status_code=status.HTTP_200_OK,
)
async def segment_articles_endpoint(
    payload: ArticleSegmentationRequest,
) -> ArticleSegmentationResult:
    return segment_document_into_articles(payload.pages)


@router.post(
    "/build-latex-page",
    response_model=LatexBuildResult,
    status_code=status.HTTP_200_OK,
)
async def build_latex_page_endpoint(
    page_content: PageContent,
) -> LatexBuildResult:
    return build_page_latex(page_content)


@router.post(
    "/build-latex-article",
    response_model=ArticleLatexDocumentResult,
    status_code=status.HTTP_200_OK,
)
async def build_latex_article_endpoint(
    article_content: ArticleContent,
) -> ArticleLatexDocumentResult:
    return build_article_latex_document(article_content, article_content.metadata)


@router.get(
    "/article-latex/{article_id}",
    response_model=ArticleLatexDocumentResult,
    status_code=status.HTTP_200_OK,
)
async def get_article_latex_endpoint(article_id: str) -> ArticleLatexDocumentResult:
    try:
        return get_cached_article_latex(article_id)
    except ArticleLatexNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.post(
    "/analyze-layout",
    response_model=LayoutAnalysisResponse,
    status_code=status.HTTP_200_OK,
)
async def analyze_layout_endpoint(
    file: UploadFile = File(...),
) -> LayoutAnalysisResponse:
    logger.info(
        "Received image upload for layout analysis: name=%s, content_type=%s",
        file.filename,
        file.content_type,
    )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не удалось определить имя изображения.",
        )

    image_bytes = await read_uploaded_image(file)

    try:
        image = load_image_bytes(image_bytes)
        return analyze_page_layout(image)
    except SuryaNotAvailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except LayoutAnalysisError as exc:
        logger.exception("Layout analysis failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.get("/visualization")
async def get_visualization(analysis_id: str) -> Response:
    try:
        visualization_png = get_cached_visualization(analysis_id)
        return Response(content=visualization_png, media_type="image/png")
    except VisualizationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.post(
    "/process-text-blocks",
    response_model=PageContent,
    status_code=status.HTTP_200_OK,
)
async def process_text_blocks_endpoint(
    file: UploadFile = File(...),
    blocks: str = Form(...),
    page_number: int | None = Form(default=None),
) -> PageContent:
    logger.info(
        "Received image upload for text block processing: name=%s, content_type=%s",
        file.filename,
        file.content_type,
    )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не удалось определить имя изображения.",
        )

    image_bytes = await read_uploaded_image(file)
    routed_blocks = parse_blocks_payload(blocks)

    try:
        return process_text_blocks(image_bytes, routed_blocks, page_number=page_number)
    except PaddleOCRNotAvailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except TextBlockProcessingError as exc:
        logger.exception("Text block processing failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.post(
    "/process-formula-blocks",
    response_model=PageContent,
    status_code=status.HTTP_200_OK,
)
async def process_formula_blocks_endpoint(
    file: UploadFile = File(...),
    blocks: str = Form(...),
    page_number: int | None = Form(default=None),
) -> PageContent:
    logger.info(
        "Received image upload for formula block processing: name=%s, content_type=%s",
        file.filename,
        file.content_type,
    )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не удалось определить имя изображения.",
        )

    image_bytes = await read_uploaded_image(file)
    routed_blocks = parse_blocks_payload(blocks)

    try:
        return process_formula_blocks(
            image_bytes,
            routed_blocks,
            page_number=page_number,
        )
    except FormulaBlockProcessingError as exc:
        logger.exception("Formula block processing failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.get("/formula-crop/{block_id}")
async def get_formula_crop(block_id: str) -> Response:
    try:
        crop_png = get_formula_crop_bytes(block_id)
        return Response(content=crop_png, media_type="image/png")
    except FormulaCropNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get(
    "/page-content",
    response_model=PageContent,
    status_code=status.HTTP_200_OK,
)
async def get_page_content(
    page_content_id: str = Query(...),
) -> PageContent:
    try:
        return get_cached_page_content(page_content_id)
    except PageTextNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.post(
    "/postprocess-text",
    response_model=TextPostprocessResponse,
    status_code=status.HTTP_200_OK,
)
async def postprocess_text_endpoint(
    payload: TextPostprocessRequest,
) -> TextPostprocessResponse:
    return postprocess_text(payload.raw_text)


@router.get(
    "/page-text",
    response_model=PageTextResponse,
    status_code=status.HTTP_200_OK,
)
async def get_page_text(page_content_id: str) -> PageTextResponse:
    try:
        return get_cached_page_text(page_content_id)
    except PageTextNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
