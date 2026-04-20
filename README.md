# Scientific Journal Scan Prototype

Прототип принимает PDF-документ, разбивает его на страницы и автоматически выполняет оба этапа: PDF-анализ и Surya layout-анализ по каждой странице. На выходе он определяет, где уже есть текст, где нужен OCR, а также строит блоки структуры страницы, порядок чтения, текст и LaTeX для формул.

Обработка выполняется в памяти: загруженный PDF, извлечённый текст и подготовленные изображения не сохраняются в проекте.

## Что умеет

- загружать PDF через FastAPI API;
- извлекать текст из каждой страницы через PyMuPDF;
- проверять, достаточно ли извлечённого текста;
- рендерить страницы в изображения и предобрабатывать их через Pillow + OpenCV;
- возвращать текст, рендер страницы и OCR-превью прямо в JSON-ответе;
- анализировать layout страницы через Surya;
- автоматически распознавать routed text-блоки через PaddleOCR с сохранением общего reading order;
- возвращать блоки `text/title/formula/table/image/header/footer/page_number`;
- строить визуализацию блоков и порядка чтения;
- разделять документ на статьи по верхнему отступу первого `text`-блока страницы с учётом пунктуации в конце предыдущей страницы;
- показывать результаты обоих этапов в одном веб-интерфейсе без повторной загрузки страницы.

Предобработка для OCR сейчас включает перевод в `grayscale`, удаление шума, нормализацию и адаптивную бинаризацию.

## Структура

```text
app/
  api/routes.py
  core/config.py
  core/logging.py
  services/image_processor.py
  services/article_segmenter.py
  services/layout_analysis_surya.py
  services/pdf_processor.py
  main.py
static/
  index.html
  styles.css
  app.js
```

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

После запуска интерфейс будет доступен по адресу [http://127.0.0.1:8000](http://127.0.0.1:8000).

## API

### `POST /api/process-pdf`

Принимает `multipart/form-data` с полем `file` и автоматически запускает:

- извлечение текста;
- OCR-подготовку для нужных страниц;
- рендер страниц;
- layout-анализ Surya по каждой странице;
- OCR текстовых блоков по каждой странице;
- разделение страниц на статьи.

Пример ответа:

```json
[
  {
    "page_number": 1,
    "has_text": true,
    "text": "Example text",
    "page_image_data_url": "data:image/png;base64,iVBORw0KGgoAAA...",
    "image_path": null,
    "image_data_url": null,
    "layout_analysis": {
      "analysis_id": "abc123",
      "visualization_url": "/visualization?analysis_id=abc123",
      "visualization_data_url": "data:image/png;base64,iVBORw0KGgoAAA...",
      "result_json_path": "/tmp/surya-layout-results/abc123.json",
      "blocks": [
        {
          "type": "text",
          "bbox": [50, 120, 600, 200],
          "confidence": 0.98,
          "reading_order": 1,
          "text": "Example paragraph",
          "latex": null
        }
      ]
    },
    "layout_error": null,
    "text_block_content": {
      "page_content_id": "5d2b4d...",
      "page_text": "Example paragraph\n[FORMULA]\nNext paragraph",
      "result_json_path": "/tmp/text-block-results/5d2b4d....json",
      "blocks": []
    },
    "text_block_error": null
  },
  {
    "page_number": 2,
    "has_text": false,
    "text": "",
    "page_image_data_url": "data:image/png;base64,iVBORw0KGgoAAA...",
    "image_path": null,
    "image_data_url": "data:image/png;base64,iVBORw0KGgoAAA...",
    "layout_analysis": null,
    "layout_error": "После постобработки не осталось валидных блоков макета.",
    "text_block_content": null,
    "text_block_error": null
  }
]
```

### `POST /analyze-layout`

Принимает `multipart/form-data` с полем `file` и возвращает найденные блоки страницы.

Пример ответа:

```json
{
  "analysis_id": "c2f3c1...",
  "visualization_url": "/visualization?analysis_id=c2f3c1...",
  "result_json_path": "/tmp/surya-layout-results/c2f3c1....json",
  "blocks": [
    {
      "type": "title",
      "bbox": [54, 70, 1160, 188],
      "confidence": 0.98,
      "reading_order": 1,
      "text": "Proceedings of ...",
      "latex": null
    },
    {
      "type": "formula",
      "bbox": [180, 620, 820, 760],
      "confidence": 0.93,
      "reading_order": 6,
      "text": null,
      "latex": "\\int_0^1 x^2 dx"
    }
  ]
}
```

### `GET /visualization?analysis_id=<id>`

Возвращает PNG-изображение страницы с нарисованными bounding boxes.

### `POST /process-text-blocks`

Принимает `multipart/form-data`:

- `file` — изображение страницы;
- `blocks` — JSON-массив routed blocks.

OCR выполняется только для блоков с `route_to = "text_pipeline"`. Остальные блоки сохраняются в общем списке как есть, чтобы не терять порядок `reading_order`. Если PaddleOCR backend недоступен, сервис автоматически использует текст, уже извлечённый Surya из layout-блоков, и помечает такие блоки для ручной проверки.

Пример ответа:

```json
{
  "page_content_id": "5d2b4d...",
  "page_text": "Рассмотрим функцию...\n[FORMULA]\nСледовательно...",
  "result_json_path": "/tmp/text-block-results/5d2b4d....json",
  "blocks": [
    {
      "block_id": "block_001",
      "type": "text",
      "reading_order": 1,
      "bbox": [40, 120, 620, 220],
      "route_to": "text_pipeline",
      "content": "Рассмотрим функцию...",
      "confidence": 0.94,
      "needs_review": false,
      "crop_data_url": "data:image/png;base64,iVBORw0KGgoAAA...",
      "ocr_result": {
        "text": "Рассмотрим функцию...",
        "confidence": 0.94
      }
    },
    {
      "block_id": "block_002",
      "type": "formula",
      "reading_order": 2,
      "bbox": [100, 260, 580, 340],
      "route_to": "formula_pipeline",
      "content": null,
      "confidence": null,
      "needs_review": false,
      "crop_data_url": null,
      "ocr_result": null
    }
  ]
}
```

### `GET /page-text?page_content_id=<id>`

Возвращает собранный текст страницы для ранее обработанного набора блоков.

## Настраиваемые параметры

- `TEXT_LENGTH_THRESHOLD` — минимальное количество символов без пробелов, чтобы страница считалась текстовой (по умолчанию `40`);
- `PDF_RENDER_DPI` — DPI рендера страницы перед OCR-предобработкой;
- `OCR_DENOISE_STRENGTH` — интенсивность шумоподавления перед бинаризацией;
- `TEXT_BLOCK_OCR_LANG` — язык PaddleOCR для текстовых блоков;
- `TEXT_BLOCK_REVIEW_CONFIDENCE` — порог confidence для `needs_review`;
- `TEXT_BLOCK_MIN_LENGTH` — минимальная длина текста без ручной проверки;
- `TEXT_BLOCK_FORMULA_PLACEHOLDER` — placeholder формулы при сборке `page_text`.

## Примечание по Surya

Если `layout-analysis` не запускается и в логах встречается ошибка вида `pad_token_id`, это обычно означает несовместимую версию `transformers`. Для этого проекта зависимость зафиксирована как `transformers>=4.56.0,<5.0.0`.

## Примечание по PaddleOCR

Для работы text block OCR нужен установленный `paddleocr` и совместимый Paddle backend. Если endpoint `/process-text-blocks` возвращает ошибку инициализации PaddleOCR, установите или обновите зависимости Paddle для вашей платформы.

## Surya

Используются официальные компоненты Surya:

- `LayoutPredictor` для layout analysis и reading order;
- `RecognitionPredictor` + `DetectionPredictor` для OCR текста;
- `RecognitionPredictor` в режиме `block_without_boxes` для формул в LaTeX.
