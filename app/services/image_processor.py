from __future__ import annotations

import base64
import logging
from io import BytesIO

import cv2
import fitz
import numpy as np
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)


class ImageProcessingError(RuntimeError):
    pass


def pixmap_to_pil_image(pixmap: fitz.Pixmap) -> Image.Image:
    mode = "RGBA" if pixmap.alpha else "RGB"
    image = Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
    return image.convert("RGB")


def pixmap_to_numpy_rgb(pixmap: fitz.Pixmap) -> np.ndarray:
    return np.array(pixmap_to_pil_image(pixmap))


def denoise_page_image(grayscale: np.ndarray) -> np.ndarray:
    logger.info("Applying OCR noise reduction")
    denoised = cv2.fastNlMeansDenoising(
        grayscale,
        None,
        h=settings.ocr_denoise_strength,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    return cv2.medianBlur(denoised, 3)


def preprocess_page_image(pixmap: fitz.Pixmap) -> Image.Image:
    logger.info("Converting rendered page to numpy array")
    pil_image = pixmap_to_pil_image(pixmap)
    rgb_array = np.array(pil_image)

    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    grayscale = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2GRAY)
    denoised = denoise_page_image(grayscale)
    normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    thresholded = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return Image.fromarray(thresholded)


def pil_image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded_image}"


def numpy_image_to_data_url(image_array: np.ndarray) -> str:
    if image_array.ndim == 2:
        image = Image.fromarray(image_array)
    else:
        image = Image.fromarray(image_array.astype(np.uint8))
    return pil_image_to_data_url(image)


def pixmap_to_data_url(pixmap: fitz.Pixmap) -> str:
    return pil_image_to_data_url(pixmap_to_pil_image(pixmap))


def preprocess_page_image_to_data_url(pixmap: fitz.Pixmap) -> str:
    try:
        processed_image = preprocess_page_image(pixmap)
        logger.info("Prepared in-memory preview for OCR page")
        return pil_image_to_data_url(processed_image)
    except Exception as exc:  # pragma: no cover - defensive branch
        logger.exception("Failed to preprocess page image")
        raise ImageProcessingError("Ошибка подготовки изображения страницы.") from exc
