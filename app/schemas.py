from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PageProcessingResult(BaseModel):
    page_number: int = Field(..., description="Page number inside the source PDF")
    has_text: bool = Field(..., description="Whether the page has enough embedded text")
    text: str = Field("", description="Extracted embedded text")
    page_image_data_url: str | None = Field(
        default=None,
        description="Rendered page image preview used for automatic layout analysis",
    )
    image_path: str | None = Field(
        default=None,
        description="Deprecated compatibility field; images are not persisted to disk",
    )
    image_data_url: str | None = Field(
        default=None,
        description="Inline PNG preview for pages that require OCR preprocessing",
    )
    layout_analysis: LayoutAnalysisResponse | None = None
    layout_error: str | None = None
    text_block_content: PageContent | None = None
    text_block_error: str | None = None
    formula_block_error: str | None = None


BlockType = Literal[
    "text",
    "title",
    "formula",
    "table",
    "image",
    "header",
    "footer",
    "page_number",
]


class LayoutBlock(BaseModel):
    type: BlockType
    bbox: list[int] = Field(..., min_length=4, max_length=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reading_order: int = Field(..., ge=1)
    text: str | None = None
    latex: str | None = None


class LayoutAnalysisResponse(BaseModel):
    analysis_id: str
    visualization_url: str
    visualization_data_url: str | None = None
    result_json_path: str | None = None
    blocks: list[LayoutBlock]


RouteTarget = Literal[
    "text_pipeline",
    "formula_pipeline",
    "table_pipeline",
    "image_pipeline",
]

PageViewMode = Literal["readable", "tex", "structure"]


class Block(BaseModel):
    block_id: str | None = None
    type: BlockType
    bbox: list[int] = Field(..., min_length=4, max_length=4)
    reading_order: int = Field(..., ge=1)
    route_to: RouteTarget | None = None
    seed_text: str | None = None
    seed_latex: str | None = None
    seed_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class OCRResult(BaseModel):
    text: str = ""
    confidence: float = Field(..., ge=0.0, le=1.0)


class FormulaOCRResult(BaseModel):
    latex: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


FormulaBackend = Literal["surya", "pix2tex", "none"]


class FormulaBlockResult(BaseModel):
    block_id: str
    reading_order: int = Field(..., ge=1)
    bbox: list[int] = Field(..., min_length=4, max_length=4)
    latex: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    needs_review: bool = False
    crop_path: str | None = None
    crop_data_url: str | None = None
    formula_result: FormulaOCRResult | None = None
    formula_backend: FormulaBackend = "none"


class ProcessedBlock(BaseModel):
    block_id: str
    type: BlockType
    reading_order: int = Field(..., ge=1)
    bbox: list[int] = Field(..., min_length=4, max_length=4)
    route_to: RouteTarget
    content: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    needs_review: bool = False
    crop_data_url: str | None = None
    crop_path: str | None = None
    latex: str | None = None
    formula_result: FormulaOCRResult | None = None
    formula_backend: FormulaBackend | None = None
    ocr_result: OCRResult | None = None
    ocr_backend: str | None = None


class PagePresentation(BaseModel):
    default_view: PageViewMode = "readable"
    available_views: list[PageViewMode] = Field(
        default_factory=lambda: ["readable", "tex", "structure"]
    )
    readable_text: str = ""
    tex_preview: str = ""


class PageContent(BaseModel):
    page_content_id: str
    page_number: int | None = None
    article_title: str | None = None
    page_text: str = ""
    needs_review_count: int = 0
    result_json_path: str | None = None
    presentation: PagePresentation | None = None
    blocks: list[ProcessedBlock]


class PageTextResponse(BaseModel):
    page_content_id: str
    page_text: str = ""


class ArticleBoundary(BaseModel):
    page_number: int = Field(..., ge=1)
    article_title: str | None = None
    is_article_start: bool = False
    score: float = Field(..., ge=0.0, le=1.0)
    needs_review: bool = False
    debug_info: dict[str, Any] = Field(default_factory=dict)


class ArticlePreview(BaseModel):
    article_id: str
    start_page: int = Field(..., ge=1)
    end_page: int = Field(..., ge=1)
    page_numbers: list[int] = Field(default_factory=list)
    title_preview: str = ""
    author_preview: str = ""
    article_text: str = ""
    needs_review: bool = False
    boundary_confidence: float = Field(..., ge=0.0, le=1.0)
    debug_info: dict[str, Any] = Field(default_factory=dict)


class ArticleSegmentationResult(BaseModel):
    total_pages: int = Field(..., ge=0)
    article_count: int = Field(..., ge=0)
    needs_review_count: int = Field(..., ge=0)
    boundaries: list[ArticleBoundary] = Field(default_factory=list)
    articles: list[ArticlePreview] = Field(default_factory=list)


class TextCorrection(BaseModel):
    original: str
    corrected: str
    reason: str
    token_index: int = Field(..., ge=0)


class SuspiciousToken(BaseModel):
    token: str
    reason: str
    token_index: int = Field(..., ge=0)
    suggestions: list[str] = Field(default_factory=list)


class TextPostprocessRequest(BaseModel):
    raw_text: str = Field(..., min_length=1)


class TextPostprocessResponse(BaseModel):
    cleaned_text: str
    corrections: list[TextCorrection] = Field(default_factory=list)
    suspicious_tokens: list[SuspiciousToken] = Field(default_factory=list)
    needs_review: bool = False


class ArticleSegmentationRequest(BaseModel):
    pages: list[PageContent] = Field(default_factory=list)


class DocumentProcessingResult(BaseModel):
    pages: list[PageProcessingResult] = Field(default_factory=list)
    article_segmentation: ArticleSegmentationResult | None = None
