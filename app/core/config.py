from __future__ import annotations

import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        self.base_dir = Path(__file__).resolve().parents[2]
        self.static_dir = self.base_dir / "static"
        self.text_length_threshold = int(os.getenv("TEXT_LENGTH_THRESHOLD", "40"))
        self.render_dpi = int(os.getenv("PDF_RENDER_DPI", "200"))
        self.ocr_denoise_strength = int(os.getenv("OCR_DENOISE_STRENGTH", "10"))
        self.text_block_ocr_lang = os.getenv("TEXT_BLOCK_OCR_LANG", "en")
        self.text_block_review_confidence = float(
            os.getenv("TEXT_BLOCK_REVIEW_CONFIDENCE", "0.8")
        )
        self.text_block_min_length = int(os.getenv("TEXT_BLOCK_MIN_LENGTH", "6"))
        self.text_block_formula_placeholder = os.getenv(
            "TEXT_BLOCK_FORMULA_PLACEHOLDER", "[FORMULA]"
        )
        self.formula_block_crop_padding = int(
            os.getenv("FORMULA_BLOCK_CROP_PADDING", "8")
        )
        self.formula_block_surya_confidence_threshold = float(
            os.getenv("FORMULA_BLOCK_SURYA_CONFIDENCE_THRESHOLD", "0.55")
        )
        self.text_postprocess_min_token_length = int(
            os.getenv("TEXT_POSTPROCESS_MIN_TOKEN_LENGTH", "4")
        )
        self.text_postprocess_fuzzy_threshold = int(
            os.getenv("TEXT_POSTPROCESS_FUZZY_THRESHOLD", "88")
        )
        self.text_postprocess_max_edit_distance = int(
            os.getenv("TEXT_POSTPROCESS_MAX_EDIT_DISTANCE", "2")
        )
        self.text_postprocess_suspicious_ratio = float(
            os.getenv("TEXT_POSTPROCESS_SUSPICIOUS_RATIO", "0.72")
        )

    def ensure_directories(self) -> None:
        self.static_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
