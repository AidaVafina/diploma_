const form = document.getElementById("upload-form");
const fileInput = document.getElementById("pdf-file");
const processingModeSelect = document.getElementById("processing-mode");
const selectedFileNameNode = document.getElementById("selected-file-name");
const processingModeDescriptionNode = document.getElementById("processing-mode-description");
const processingModeTitleNode = document.getElementById("processing-mode-title");
const processingModeNoteNode = document.getElementById("processing-mode-note");
const processButton = document.getElementById("process-button");
const statusNode = document.getElementById("status");
const processingIndicatorNode = document.getElementById("processing-indicator");
const processingStageNode = document.getElementById("processing-stage");
const summaryNode = document.getElementById("summary");
const resultsNode = document.getElementById("results");
const PDF_FORMULA_MAX_WIDTH = 430;
const PDF_FORMULA_FONT_SIZE_PX = 13;
const PDF_FORMULA_RENDER_SCALE = 2;
const PDF_FORMULA_X_PADDING = 4;
const PDF_FORMULA_Y_PADDING = 3;
const PDF_FORMULA_RENDER_CACHE = new Map();
const PROCESSING_MODE_LABELS = {
  full: "Полная обработка",
  no_preprocessing: "Без предобработки",
  text_only: "Только текст",
};
const PROCESSING_MODE_META = {
  full: {
    description:
      "Максимально точный режим: рендер, OCR-предобработка, layout-анализ, текстовые блоки и формулы.",
    note: "Подходит для сложных страниц, сканов среднего качества и документов с формулами.",
  },
  no_preprocessing: {
    description:
      "Быстрый режим для качественных PDF и чистых сканов: layout и распознавание сохраняются, но предобработка пропускается.",
    note: "Лучший выбор, когда документ уже чёткий и нужно сократить время без потери структуры.",
  },
  text_only: {
    description:
      "Минимальный по времени сценарий: извлекается только текст, а layout, формулы и OCR-предобработка отключаются.",
    note: "Подходит для текстовых документов, когда важна скорость, а структурный разбор не нужен.",
  },
};

function getProcessingModeLabel(mode) {
  return PROCESSING_MODE_LABELS[mode] || PROCESSING_MODE_LABELS.full;
}

function getSelectedProcessingMode() {
  return processingModeSelect?.value || "full";
}

function updateSelectedFileName() {
  if (!selectedFileNameNode) {
    return;
  }

  const selectedName = fileInput.files?.[0]?.name?.trim();
  selectedFileNameNode.textContent = selectedName || "Файл ещё не выбран";
}

function updateProcessingModePresentation() {
  const mode = getSelectedProcessingMode();
  const modeMeta = PROCESSING_MODE_META[mode] || PROCESSING_MODE_META.full;
  const modeLabel = getProcessingModeLabel(mode);

  if (processingModeDescriptionNode) {
    processingModeDescriptionNode.textContent = modeMeta.description;
  }

  if (processingModeTitleNode) {
    processingModeTitleNode.textContent = modeLabel;
  }

  if (processingModeNoteNode) {
    processingModeNoteNode.textContent = modeMeta.note;
  }
}

function setStatus(message, isError = false) {
  statusNode.textContent = message;
  statusNode.style.color = isError ? "var(--warning)" : "var(--muted)";
}

function setProcessingStage(message, state = "active") {
  if (!processingIndicatorNode || !processingStageNode) {
    return;
  }

  processingIndicatorNode.hidden = false;
  processingIndicatorNode.classList.toggle("processing-indicator--active", state === "active");
  processingIndicatorNode.classList.toggle("processing-indicator--done", state === "done");
  processingIndicatorNode.classList.toggle("processing-indicator--error", state === "error");
  processingStageNode.textContent = message;
}

async function readProcessingStream(response) {
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Браузер не поддерживает live-progress для этого запроса.");
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let documentResult = null;

  function handleLine(line) {
    if (!line.trim()) {
      return;
    }

    const event = JSON.parse(line);
    if (event.type === "error") {
      setProcessingStage(event.message || "Ошибка обработки", "error");
      throw new Error(event.message || "Не удалось обработать PDF.");
    }

    if (event.message) {
      setProcessingStage(event.message, event.type === "result" ? "done" : "active");
    }

    if (event.type === "result") {
      documentResult = event.data;
    }
  }

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    lines.forEach(handleLine);

    if (done) {
      break;
    }
  }

  handleLine(buffer);

  if (!documentResult) {
    throw new Error("Сервер завершил live-progress без результата обработки.");
  }

  return documentResult;
}

function summarizeText(text, limit = 300) {
  const trimmed = (text || "").trim();
  if (!trimmed) {
    return "Встроенный текст не найден или слишком короткий.";
  }

  return trimmed.length > limit ? `${trimmed.slice(0, limit).trim()}...` : trimmed;
}

function summarizeBlockContent(block, limit = 140) {
  const source = (block.latex || block.text || "").trim();
  if (!source) {
    return "—";
  }

  return source.length > limit ? `${source.slice(0, limit).trim()}...` : source;
}

function formatBBox(bbox) {
  if (!Array.isArray(bbox) || bbox.length !== 4) {
    return "—";
  }

  return `[${bbox.join(", ")}]`;
}

function createBadge(hasText) {
  const badge = document.createElement("span");
  badge.className = `badge ${hasText ? "badge--text" : "badge--ocr"}`;
  badge.textContent = hasText ? "Текст достаточен" : "Нужен OCR";
  return badge;
}

function createMetaChip(label, value, modifier = "") {
  const chip = document.createElement("div");
  chip.className = `meta-chip${modifier ? ` meta-chip--${modifier}` : ""}`;

  const labelNode = document.createElement("span");
  labelNode.className = "meta-chip__label";
  labelNode.textContent = label;

  const valueNode = document.createElement("strong");
  valueNode.textContent = value;

  chip.appendChild(labelNode);
  chip.appendChild(valueNode);
  return chip;
}

function createPreviewPanel(title, imageSource, placeholderText) {
  const panel = document.createElement("div");
  panel.className = "preview-panel";

  const heading = document.createElement("h4");
  heading.textContent = title;
  panel.appendChild(heading);

  const frame = document.createElement("div");
  frame.className = "preview-panel__frame";

  if (imageSource) {
    const image = document.createElement("img");
    image.src = imageSource;
    image.alt = title;
    frame.appendChild(image);
  } else {
    const placeholder = document.createElement("div");
    placeholder.className = "page-card__placeholder";
    placeholder.textContent = placeholderText;
    frame.appendChild(placeholder);
  }

  panel.appendChild(frame);
  return panel;
}

function createBlockTypeBadge(type) {
  const badge = document.createElement("span");
  badge.className = `table-type table-type--${type}`;
  badge.textContent = type;
  return badge;
}

function createSmallBadge(text, modifier = "") {
  const badge = document.createElement("span");
  badge.className = `small-badge${modifier ? ` small-badge--${modifier}` : ""}`;
  badge.textContent = text;
  return badge;
}

function summarizeArticlePreview(text, fallback) {
  const value = (text || "").trim();
  return value || fallback;
}

function getPageReadableText(page) {
  const pageContent = page.text_block_content || {};
  const presentation = pageContent.presentation || {};
  return (
    presentation.readable_text ||
    pageContent.page_text ||
    page.text ||
    ""
  ).trim();
}

function getArticlePages(article, pages) {
  const pageNumbers = new Set(article.page_numbers || []);
  if (!pageNumbers.size && article.start_page && article.end_page) {
    for (let pageNumber = article.start_page; pageNumber <= article.end_page; pageNumber += 1) {
      pageNumbers.add(pageNumber);
    }
  }
  return pages.filter((page) => pageNumbers.has(page.page_number));
}

function collectArticleText(article, pages) {
  const preparedArticleText = (article.article_text || "").trim();
  if (preparedArticleText) {
    return preparedArticleText;
  }

  const articlePages = getArticlePages(article, pages);
  const parts = articlePages
    .map((page) => getPageReadableText(page))
    .filter(Boolean);

  return parts.join("\n\n").trim();
}

function getSourceDocumentName() {
  return fileInput.files?.[0]?.name || "document.pdf";
}

function getSourceDocumentStem() {
  const sourceName = getSourceDocumentName().trim();
  if (!sourceName) {
    return "document";
  }

  return sourceName.replace(/\.[^.]+$/, "") || "document";
}

function sanitizeFilenamePart(value, fallback = "document") {
  const cleaned = String(value || "")
    .replace(/[\\/:*?"<>|]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  return cleaned || fallback;
}

async function readErrorMessage(response) {
  let rawText = "";

  try {
    rawText = await response.text();
    const payload = JSON.parse(rawText);
    if (payload?.detail) {
      return payload.detail;
    }
  } catch (error) {
    console.warn("Unable to parse error payload:", error);
  }

  return rawText || "Не удалось сформировать PDF.";
}

function extractFilenameFromDisposition(headerValue) {
  if (!headerValue) {
    return "";
  }

  const encodedMatch = headerValue.match(/filename\*=UTF-8''([^;]+)/i);
  if (encodedMatch?.[1]) {
    try {
      return decodeURIComponent(encodedMatch[1]);
    } catch (error) {
      console.warn("Unable to decode attachment filename:", error);
    }
  }

  const plainMatch = headerValue.match(/filename="([^"]+)"/i);
  return plainMatch?.[1] || "";
}

function waitForNextFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

async function loadImageElement(sourceUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Не удалось загрузить изображение формулы."));
    image.src = sourceUrl;
  });
}

async function ensureMathJaxForPdfExport() {
  if (!window.MathJax) {
    return null;
  }

  if (window.MathJax.startup?.promise) {
    try {
      await window.MathJax.startup.promise;
    } catch (error) {
      console.warn("MathJax startup failed:", error);
      return null;
    }
  }

  return typeof window.MathJax.tex2svgPromise === "function" ? window.MathJax : null;
}

async function renderFormulaLatexToPngData(latex) {
  const normalizedLatex = String(latex || "").trim();
  if (!normalizedLatex) {
    return null;
  }

  if (PDF_FORMULA_RENDER_CACHE.has(normalizedLatex)) {
    return PDF_FORMULA_RENDER_CACHE.get(normalizedLatex);
  }

  const renderPromise = (async () => {
    const mathJax = await ensureMathJaxForPdfExport();
    if (!mathJax) {
      return null;
    }

    let mathContainer = null;
    try {
      mathContainer = await mathJax.tex2svgPromise(normalizedLatex, {
        display: true,
      });
    } catch (error) {
      console.warn("MathJax formula render failed:", error);
      return null;
    }

    const host = document.createElement("div");
    host.style.position = "fixed";
    host.style.left = "-10000px";
    host.style.top = "0";
    host.style.opacity = "0";
    host.style.pointerEvents = "none";
    host.style.display = "inline-block";
    host.style.padding = `${PDF_FORMULA_Y_PADDING}px ${PDF_FORMULA_X_PADDING}px`;
    host.style.background = "#ffffff";
    host.style.color = "#1f2d33";
    host.style.fontSize = `${PDF_FORMULA_FONT_SIZE_PX}px`;
    host.style.lineHeight = "1";
    host.appendChild(mathContainer);
    document.body.appendChild(host);

    try {
      await waitForNextFrame();

      const svgNode = mathContainer.querySelector("svg");
      if (!svgNode) {
        return null;
      }

      const formulaWidth = Math.max(1, Math.ceil(svgNode.getBoundingClientRect().width));
      const formulaHeight = Math.max(1, Math.ceil(svgNode.getBoundingClientRect().height));
      const maxFormulaWidth = Math.max(1, PDF_FORMULA_MAX_WIDTH - PDF_FORMULA_X_PADDING * 2);
      const scaleFactor = formulaWidth > maxFormulaWidth ? maxFormulaWidth / formulaWidth : 1;
      const scaledFormulaWidth = Math.max(1, Math.ceil(formulaWidth * scaleFactor));
      const scaledFormulaHeight = Math.max(1, Math.ceil(formulaHeight * scaleFactor));
      const exportWidth = scaledFormulaWidth + PDF_FORMULA_X_PADDING * 2;
      const exportHeight = scaledFormulaHeight + PDF_FORMULA_Y_PADDING * 2;

      const svgClone = svgNode.cloneNode(true);
      svgClone.removeAttribute("style");
      svgClone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      svgClone.setAttribute("x", String(PDF_FORMULA_X_PADDING));
      svgClone.setAttribute("y", String(PDF_FORMULA_Y_PADDING));
      svgClone.setAttribute("width", String(scaledFormulaWidth));
      svgClone.setAttribute("height", String(scaledFormulaHeight));

      const serializedFormulaSvg = new XMLSerializer().serializeToString(svgClone);
      const svgMarkup = [
        `<svg xmlns="http://www.w3.org/2000/svg" width="${exportWidth}" height="${exportHeight}" viewBox="0 0 ${exportWidth} ${exportHeight}">`,
        `<rect width="${exportWidth}" height="${exportHeight}" fill="white"/>`,
        serializedFormulaSvg,
        "</svg>",
      ].join("");
      const svgBlob = new Blob([svgMarkup], {
        type: "image/svg+xml;charset=utf-8",
      });
      const svgUrl = URL.createObjectURL(svgBlob);

      try {
        const formulaImage = await loadImageElement(svgUrl);
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.ceil(exportWidth * PDF_FORMULA_RENDER_SCALE));
        canvas.height = Math.max(1, Math.ceil(exportHeight * PDF_FORMULA_RENDER_SCALE));

        const context = canvas.getContext("2d");
        if (!context) {
          return null;
        }

        context.scale(PDF_FORMULA_RENDER_SCALE, PDF_FORMULA_RENDER_SCALE);
        context.fillStyle = "#ffffff";
        context.fillRect(0, 0, exportWidth, exportHeight);
        context.drawImage(formulaImage, 0, 0, exportWidth, exportHeight);

        return {
          imageDataUrl: canvas.toDataURL("image/png"),
          width: exportWidth,
          height: exportHeight,
        };
      } finally {
        URL.revokeObjectURL(svgUrl);
      }
    } finally {
      host.remove();
    }
  })();

  PDF_FORMULA_RENDER_CACHE.set(normalizedLatex, renderPromise);

  try {
    return await renderPromise;
  } catch (error) {
    PDF_FORMULA_RENDER_CACHE.delete(normalizedLatex);
    throw error;
  }
}

async function prepareReadableBlocksForPdfExport(readableBlocks) {
  const preparedBlocks = [];

  for (const readableBlock of readableBlocks) {
    if (readableBlock.kind !== "formula" || readableBlock.isError) {
      preparedBlocks.push({
        kind: readableBlock.kind,
        text: readableBlock.text,
      });
      continue;
    }

    try {
      const formulaImage = await renderFormulaLatexToPngData(readableBlock.text);
      if (formulaImage?.imageDataUrl) {
        preparedBlocks.push({
          kind: readableBlock.kind,
          text: readableBlock.text,
          image_data_url: formulaImage.imageDataUrl,
          image_width: formulaImage.width,
          image_height: formulaImage.height,
        });
        continue;
      }
    } catch (error) {
      console.warn("Unable to render formula for PDF export:", error);
    }

    preparedBlocks.push({
      kind: readableBlock.kind,
      text: readableBlock.text,
    });
  }

  return preparedBlocks;
}

async function downloadReadablePdf(payloadSource, triggerButton, successMessage) {
  const initialLabel = triggerButton?.textContent || "";

  if (triggerButton) {
    triggerButton.disabled = true;
    triggerButton.textContent = "Готовим PDF...";
  }

  try {
    setStatus("Подготавливаем содержимое для PDF...");
    const payload =
      typeof payloadSource === "function" ? await payloadSource() : await payloadSource;

    setStatus("Формируем PDF с читаемым текстом...");
    const response = await fetch("/api/export-readable-pdf", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }

    const blob = await response.blob();
    const filename =
      extractFilenameFromDisposition(response.headers.get("Content-Disposition")) ||
      payload.filename ||
      "readable-text.pdf";
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(objectUrl);
    setStatus(successMessage || "PDF с читаемым текстом сформирован.");
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Не удалось сформировать PDF.", true);
  } finally {
    if (triggerButton) {
      triggerButton.disabled = false;
      triggerButton.textContent = initialLabel;
    }
  }
}

async function downloadCollectionArchive(payloadSource, triggerButton, successMessage) {
  const initialLabel = triggerButton?.textContent || "";

  if (triggerButton) {
    triggerButton.disabled = true;
    triggerButton.textContent = "Собираем архив...";
  }

  try {
    setStatus("Подготавливаем статьи для экспорта коллекции...");
    const payload =
      typeof payloadSource === "function" ? await payloadSource() : await payloadSource;

    setStatus("Формируем ZIP-архив коллекции...");
    const response = await fetch("/api/export-collection", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }

    const blob = await response.blob();
    const filename =
      extractFilenameFromDisposition(response.headers.get("Content-Disposition")) ||
      payload.filename ||
      "collection-export.zip";
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(objectUrl);
    setStatus(successMessage || "Архив коллекции сформирован.");
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Не удалось сформировать архив коллекции.", true);
  } finally {
    if (triggerButton) {
      triggerButton.disabled = false;
      triggerButton.textContent = initialLabel;
    }
  }
}

function summarizeProcessedContent(block, limit = 110) {
  const source =
    (
      block.content ||
      block.latex ||
      block.ocr_result?.text ||
      block.formula_result?.latex ||
      (block.type === "formula" ? "[FORMULA]" : "")
    ).trim();

  if (!source) {
    if (block.route_to === "text_pipeline") {
      return "Пустой результат OCR";
    }
    if (block.route_to === "formula_pipeline") {
      return "LaTeX не распознан";
    }
    return "Без OCR";
  }

  return source.length > limit ? `${source.slice(0, limit).trim()}...` : source;
}

function createOrderFlow(pageContent) {
  const flow = document.createElement("div");
  flow.className = "order-flow";

  pageContent.blocks.forEach((block) => {
    const item = document.createElement("div");
    item.className = "order-flow__item";

    const number = document.createElement("strong");
    number.textContent = String(block.reading_order);

    const label = document.createElement("span");
    label.textContent = block.type;

    item.appendChild(number);
    item.appendChild(label);
    flow.appendChild(item);
  });

  return flow;
}

function createBlockDetailRow(label, value) {
  const row = document.createElement("div");
  row.className = "block-item__meta-row";

  const labelNode = document.createElement("span");
  labelNode.className = "block-item__meta-label";
  labelNode.textContent = label;

  const valueNode = document.createElement("strong");
  valueNode.className = "block-item__meta-value";
  valueNode.textContent = value;

  row.appendChild(labelNode);
  row.appendChild(valueNode);
  return row;
}

function createProcessedBlockList(pageContent) {
  const wrapper = document.createElement("div");
  wrapper.className = "page-layout page-layout--compact-list";

  const title = document.createElement("h4");
  title.textContent = "Блоки страницы";
  wrapper.appendChild(title);

  const helper = document.createElement("p");
  helper.className = "page-layout__empty";
  helper.textContent =
    "Нажми на нужный блок, чтобы открыть подробный результат OCR или LaTeX.";
  wrapper.appendChild(helper);

  wrapper.appendChild(createOrderFlow(pageContent));

  const list = document.createElement("div");
  list.className = "block-list";

  pageContent.blocks.forEach((block) => {
    const details = document.createElement("details");
    details.className = "block-item";

    const summary = document.createElement("summary");
    summary.className = "block-item__summary";

    const main = document.createElement("div");
    main.className = "block-item__summary-main";

    const order = document.createElement("div");
    order.className = "block-item__order";
    order.textContent = String(block.reading_order);

    const headline = document.createElement("div");
    headline.className = "block-item__headline";

    const headlineTop = document.createElement("div");
    headlineTop.className = "block-item__headline-top";
    headlineTop.appendChild(createBlockTypeBadge(block.type));
    const pipelineLabel =
      block.route_to === "text_pipeline"
        ? "text OCR"
        : block.route_to === "formula_pipeline"
          ? "formula OCR"
          : "skip OCR";
    headlineTop.appendChild(
      createSmallBadge(
        pipelineLabel,
        block.route_to === "text_pipeline" || block.route_to === "formula_pipeline" ? "success" : ""
      )
    );

    const preview = document.createElement("p");
    preview.className = "block-item__preview-text";
    preview.textContent = summarizeProcessedContent(block);

    headline.appendChild(headlineTop);
    headline.appendChild(preview);

    main.appendChild(order);
    main.appendChild(headline);

    const aside = document.createElement("div");
    aside.className = "block-item__summary-aside";
    if (block.confidence !== null && block.confidence !== undefined) {
      aside.appendChild(createSmallBadge(`conf ${Number(block.confidence).toFixed(3)}`));
    }

    summary.appendChild(main);
    summary.appendChild(aside);
    details.appendChild(summary);

    const body = document.createElement("div");
    body.className = "block-item__body";

    const meta = document.createElement("div");
    meta.className = "block-item__meta";
    meta.appendChild(createBlockDetailRow("Order", String(block.reading_order)));
    meta.appendChild(createBlockDetailRow("BBox", formatBBox(block.bbox)));
    meta.appendChild(
      createBlockDetailRow(
        "Confidence",
        block.confidence === null || block.confidence === undefined
          ? "—"
          : Number(block.confidence).toFixed(3)
      )
    );
    body.appendChild(meta);

    const detailGrid = document.createElement("div");
    detailGrid.className = "block-item__detail-grid";

    const contentPanel = document.createElement("div");
    contentPanel.className = "block-item__panel";
    const contentTitle = document.createElement("h5");
    contentTitle.textContent = block.type === "formula" ? "LaTeX" : "Content";
    const contentText = document.createElement("pre");
    contentText.className = "block-item__content";
    contentText.textContent = block.content || block.latex || "—";
    contentPanel.appendChild(contentTitle);
    contentPanel.appendChild(contentText);
    detailGrid.appendChild(contentPanel);

    const rawResult =
      (block.type === "formula" ? block.formula_result?.latex : block.ocr_result?.text) || "";
    const normalizedResult = block.content || block.latex || "";
    if (rawResult && rawResult !== normalizedResult) {
      const rawPanel = document.createElement("div");
      rawPanel.className = "block-item__panel";
      const rawTitle = document.createElement("h5");
      rawTitle.textContent = block.type === "formula" ? "Raw LaTeX" : "Raw OCR";
      const rawText = document.createElement("pre");
      rawText.className = "block-item__content";
      rawText.textContent = rawResult;
      rawPanel.appendChild(rawTitle);
      rawPanel.appendChild(rawText);
      detailGrid.appendChild(rawPanel);
    }

    if (block.crop_data_url) {
      const cropPanel = document.createElement("div");
      cropPanel.className = "block-item__panel";
      const cropTitle = document.createElement("h5");
      cropTitle.textContent = "Crop";
      const cropImage = document.createElement("img");
      cropImage.className = "block-item__crop";
      cropImage.src = block.crop_data_url;
      cropImage.alt = `${block.block_id} crop`;
      cropPanel.appendChild(cropTitle);
      cropPanel.appendChild(cropImage);
      detailGrid.appendChild(cropPanel);
    }

    body.appendChild(detailGrid);
    details.appendChild(body);

    list.appendChild(details);
  });

  wrapper.appendChild(list);
  return wrapper;
}

function createViewMetaLine(pageContent, fallbackText = "JSON-путь недоступен.") {
  const meta = document.createElement("p");
  meta.className = "page-text-summary__meta";
  meta.textContent = pageContent.result_json_path
    ? `JSON сохранён: ${pageContent.result_json_path}`
    : fallbackText;
  return meta;
}

function createTextViewPanel(titleText, contentText, pageContent, modifier = "", options = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = "page-text-summary";

  const title = document.createElement("h4");
  title.textContent = titleText;
  wrapper.appendChild(title);
  if (options.showMeta !== false) {
    wrapper.appendChild(createViewMetaLine(pageContent));
  }

  const textNode = document.createElement("pre");
  textNode.className = `page-text-summary__content${modifier ? ` page-text-summary__content--${modifier}` : ""}`;
  textNode.textContent = contentText;
  wrapper.appendChild(textNode);

  return wrapper;
}

function decodeHtmlEntities(text) {
  return String(text || "")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'");
}

function sanitizeLatexForReadableView(latex) {
  const source = decodeHtmlEntities(latex || "");
  if (!source.trim()) {
    return "";
  }

  let cleaned = source
    .replace(/<math\b[^>]*>/gi, "")
    .replace(/<\/math>/gi, "")
    .replace(/<\/?(mrow|mi|mn|mo|msub|msup|msubsup|mfrac|semantics|annotation|annotation-xml)[^>]*>/gi, "")
    .replace(/\r/g, "\n")
    .trim();

  cleaned = cleaned.replace(/^\\\[/, "").replace(/\\\]$/, "").trim();
  cleaned = cleaned.replace(/^\$\$/, "").replace(/\$\$$/, "").trim();
  cleaned = cleaned.replace(/[ \t]+\n/g, "\n").replace(/\n[ \t]+/g, "\n");
  cleaned = cleaned.replace(/[ \t]{2,}/g, " ");
  return cleaned.trim();
}

function sanitizeReadableText(text) {
  const source = decodeHtmlEntities(text || "");
  return source
    .replace(/<math\b[^>]*>/gi, "")
    .replace(/<\/math>/gi, "")
    .replace(/<\/?(mrow|mi|mn|mo|msub|msup|msubsup|mfrac|semantics|annotation|annotation-xml)[^>]*>/gi, "")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

function normalizeReadableTextForCompare(text) {
  return sanitizeReadableText(text).replace(/\s+/g, " ").trim().toLowerCase();
}

function isReadableTextBlock(block) {
  return ["text", "title", "header", "footer", "page_number"].includes(block.type);
}

function isArticleTitleParagraph(block, cleanedText, pageContent) {
  if (block.type === "title") {
    return true;
  }

  const articleTitle = normalizeReadableTextForCompare(pageContent.article_title);
  const paragraphText = normalizeReadableTextForCompare(cleanedText);
  if (!articleTitle || !paragraphText) {
    return false;
  }

  return articleTitle === paragraphText;
}

function looksLikeAuthorLine(text) {
  const cleanedText = sanitizeReadableText(text).replace(/\s+/g, " ").trim();
  if (!cleanedText || cleanedText.length > 160) {
    return false;
  }

  const wordCount = cleanedText.split(/\s+/).filter(Boolean).length;
  if (wordCount > 16 || /[=\\]/.test(cleanedText)) {
    return false;
  }

  return /(?:^|[\s,(])(?:[A-ZА-ЯЁ]\.\s*){1,3}/.test(cleanedText);
}

function validateLatexForReadableView(latex) {
  if (!latex) {
    return false;
  }

  if (!window.katex || typeof window.katex.renderToString !== "function") {
    return null;
  }

  try {
    window.katex.renderToString(latex, {
      displayMode: true,
      throwOnError: true,
    });
    return true;
  } catch (error) {
    console.warn("KaTeX validation failed:", error);
    return false;
  }
}

function renderReadableMath(container) {
  if (!container || typeof window.renderMathInElement !== "function") {
    return;
  }

  window.renderMathInElement(container, {
    delimiters: [
      { left: "\\[", right: "\\]", display: true },
    ],
    throwOnError: false,
  });
}

function buildReadableBlocks(pageContent) {
  let expectAuthorAfterTitle = false;
  const readableBlocks = [];

  (pageContent.blocks || []).forEach((block) => {
    if (block.type === "formula") {
      const cleanedLatex = sanitizeLatexForReadableView(
        block.latex || block.formula_result?.latex || ""
      );
      const latexState = validateLatexForReadableView(cleanedLatex);

      if (!cleanedLatex || latexState === false) {
        readableBlocks.push({
          kind: "formula",
          text: "[FORMULA ERROR]",
          isError: true,
          isPending: false,
        });
      } else {
        readableBlocks.push({
          kind: "formula",
          text: cleanedLatex,
          isError: false,
          isPending: latexState === null,
        });
      }
      return;
    }

    if (isReadableTextBlock(block)) {
      const cleanedText = sanitizeReadableText(block.content);
      if (!cleanedText) {
        return;
      }

      const isArticleTitle = isArticleTitleParagraph(block, cleanedText, pageContent);
      const isAuthor = expectAuthorAfterTitle && looksLikeAuthorLine(cleanedText);
      if (isArticleTitle) {
        expectAuthorAfterTitle = true;
        readableBlocks.push({
          kind: "title",
          text: cleanedText,
        });
      } else if (isAuthor) {
        expectAuthorAfterTitle = false;
        readableBlocks.push({
          kind: "author",
          text: cleanedText,
        });
      } else if (!["header", "footer", "page_number"].includes(block.type)) {
        expectAuthorAfterTitle = false;
        readableBlocks.push({
          kind: "paragraph",
          text: cleanedText,
        });
      }
      return;
    }

    if (block.type === "table" || block.type === "image") {
      readableBlocks.push({
        kind: "placeholder",
        text: block.type === "table" ? "[TABLE]" : "[IMAGE]",
      });
    }
  });

  return readableBlocks;
}

function buildFallbackReadableBlocks(text, emptyMessage) {
  const cleanedText = String(text || "").trim();
  if (!cleanedText) {
    return emptyMessage
      ? [
          {
            kind: "placeholder",
            text: emptyMessage,
          },
        ]
      : [];
  }

  return [
    {
      kind: "paragraph",
      text: cleanedText,
    },
  ];
}

function buildReadableBlocksForPage(page, articleTitle = "") {
  if (page.text_block_content?.blocks?.length) {
    const pageContent = {
      ...page.text_block_content,
      article_title: page.text_block_content.article_title || articleTitle,
    };
    return buildReadableBlocks(pageContent);
  }

  return buildFallbackReadableBlocks(getPageReadableText(page), "Текст страницы недоступен.");
}

function buildReadableBlocksForArticle(article, pages) {
  const articlePages = getArticlePages(article, pages);
  let hasPageContent = false;
  const readableBlocks = [];

  articlePages.forEach((page) => {
    const pageBlocks = buildReadableBlocksForPage(page, article.title_preview);
    if (!pageBlocks.length) {
      return;
    }

    hasPageContent = true;
    readableBlocks.push(...pageBlocks);
  });

  if (hasPageContent) {
    return readableBlocks;
  }

  return buildFallbackReadableBlocks(
    collectArticleText(article, pages),
    "Текст статьи недоступен."
  );
}

function appendReadableBlock(content, readableBlock) {
  if (readableBlock.kind === "formula") {
    const formulaNode = document.createElement("div");
    formulaNode.className = "formula";

    if (readableBlock.isError) {
      formulaNode.classList.add("formula--error");
      formulaNode.textContent = readableBlock.text || "[FORMULA ERROR]";
    } else {
      formulaNode.textContent = `\\[${readableBlock.text}\\]`;
      if (readableBlock.isPending) {
        formulaNode.classList.add("formula--pending");
      }
    }

    content.appendChild(formulaNode);
    return;
  }

  if (readableBlock.kind === "placeholder") {
    const placeholder = document.createElement("div");
    placeholder.className = "page-readable-view__placeholder";
    placeholder.textContent = readableBlock.text;
    content.appendChild(placeholder);
    return;
  }

  const paragraph = document.createElement("p");
  paragraph.className = "page-readable-view__paragraph";

  if (readableBlock.kind === "title") {
    paragraph.classList.add(
      "page-readable-view__paragraph--title",
      "page-readable-view__paragraph--article-title"
    );
  } else if (readableBlock.kind === "author") {
    paragraph.classList.add("page-readable-view__paragraph--author");
  }

  paragraph.textContent = readableBlock.text;
  content.appendChild(paragraph);
}

function appendReadableBlocks(content, readableBlocks) {
  readableBlocks.forEach((readableBlock) => {
    appendReadableBlock(content, readableBlock);
  });
}

function renderReadableView(pageContent) {
  const wrapper = document.createElement("div");
  wrapper.className = "page-readable-view";

  const title = document.createElement("h4");
  title.textContent = "Текст страницы для чтения";
  wrapper.appendChild(title);

  const content = document.createElement("div");
  content.className = "page-readable-view__content";
  appendReadableBlocks(content, buildReadableBlocks(pageContent));

  wrapper.appendChild(content);
  requestAnimationFrame(() => renderReadableMath(content));
  return wrapper;
}

function createArticleReadableText(article, pages) {
  const content = document.createElement("div");
  content.className = "article-item__text-content";

  appendReadableBlocks(content, buildReadableBlocksForArticle(article, pages));
  requestAnimationFrame(() => renderReadableMath(content));
  return content;
}

function createStructureViewPanel(pageContent) {
  const wrapper = document.createElement("div");
  wrapper.className = "page-result-structure";

  const summary = document.createElement("div");
  summary.className = "page-result-structure__summary";
  summary.appendChild(createSmallBadge(`${pageContent.blocks.length} блоков`));
  summary.appendChild(createSmallBadge(`${pageContent.needs_review_count || 0} требуют проверки`));
  wrapper.appendChild(summary);
  wrapper.appendChild(createViewMetaLine(pageContent));
  wrapper.appendChild(createProcessedBlockList(pageContent));
  return wrapper;
}

function createPageContentViewer(pageContent) {
  const viewer = document.createElement("div");
  viewer.className = "page-result-viewer";

  const presentation = pageContent.presentation || {};
  const configuredViewModes = presentation.available_views || ["readable", "tex", "structure"];
  const viewModes = configuredViewModes.filter((mode) => mode !== "tex");
  if (!viewModes.length) {
    viewModes.push("readable");
  }
  const defaultView = viewModes.includes(presentation.default_view)
    ? presentation.default_view
    : "readable";

  const buttonsWrap = document.createElement("div");
  buttonsWrap.className = "page-result-switcher";

  const panelsWrap = document.createElement("div");
  panelsWrap.className = "page-result-panels";

  const viewDefinitions = {
    readable: {
      label: "Текст",
      panel: renderReadableView(pageContent),
    },
    structure: {
      label: "Структура",
      panel: createStructureViewPanel(pageContent),
    },
  };

  const buttons = new Map();
  const panels = new Map();

  function activateView(mode) {
    buttons.forEach((button, buttonMode) => {
      button.classList.toggle("page-result-switcher__button--active", buttonMode === mode);
    });
    panels.forEach((panel, panelMode) => {
      panel.hidden = panelMode !== mode;
    });
  }

  viewModes.forEach((mode) => {
    const view = viewDefinitions[mode];
    if (!view) {
      return;
    }

    const button = document.createElement("button");
    button.type = "button";
    button.className = "page-result-switcher__button";
    button.textContent = view.label;
    button.addEventListener("click", () => activateView(mode));
    buttons.set(mode, button);
    buttonsWrap.appendChild(button);

    const panel = document.createElement("div");
    panel.className = "page-result-panel";
    panel.appendChild(view.panel);
    panels.set(mode, panel);
    panelsWrap.appendChild(panel);
  });

  viewer.appendChild(buttonsWrap);
  viewer.appendChild(panelsWrap);
  activateView(viewDefinitions[defaultView] ? defaultView : "readable");
  return viewer;
}

function createTabbedViewer(viewDefinitions, viewOrder, defaultView = "") {
  const viewer = document.createElement("div");
  viewer.className = "page-result-viewer article-result-viewer";

  const buttonsWrap = document.createElement("div");
  buttonsWrap.className = "page-result-switcher";

  const panelsWrap = document.createElement("div");
  panelsWrap.className = "page-result-panels";

  const buttons = new Map();
  const panels = new Map();

  function activateView(mode) {
    buttons.forEach((button, buttonMode) => {
      button.classList.toggle("page-result-switcher__button--active", buttonMode === mode);
    });
    panels.forEach((panel, panelMode) => {
      panel.hidden = panelMode !== mode;
    });
  }

  viewOrder.forEach((mode) => {
    const view = viewDefinitions[mode];
    if (!view) {
      return;
    }

    const button = document.createElement("button");
    button.type = "button";
    button.className = "page-result-switcher__button";
    button.textContent = view.label;
    button.addEventListener("click", () => activateView(mode));
    buttons.set(mode, button);
    buttonsWrap.appendChild(button);

    const panel = document.createElement("div");
    panel.className = "page-result-panel";
    panel.appendChild(view.panel);
    panels.set(mode, panel);
    panelsWrap.appendChild(panel);
  });

  viewer.appendChild(buttonsWrap);
  viewer.appendChild(panelsWrap);
  activateView(viewDefinitions[defaultView] ? defaultView : viewOrder[0]);
  return viewer;
}

function createArticleTextPanel(article, pages) {
  const textBlock = document.createElement("div");
  textBlock.className = "article-item__text";

  const textTitle = document.createElement("h5");
  textTitle.textContent = "Текст статьи";

  const textContent = createArticleReadableText(article, pages);

  textBlock.appendChild(textTitle);
  textBlock.appendChild(textContent);
  return textBlock;
}

function createArticleLatexPanel(article) {
  const latexBlock = document.createElement("div");
  latexBlock.className = "article-item__latex";

  const latexTitle = document.createElement("h5");
  latexTitle.textContent = "LaTeX статьи";

  const latexContent = document.createElement("pre");
  latexContent.className = "article-item__latex-content";
  latexContent.textContent = article.article_latex_preview || "% LaTeX статьи недоступен.";

  latexBlock.appendChild(latexTitle);
  latexBlock.appendChild(latexContent);
  return latexBlock;
}

function createArticleContentViewer(article, pages) {
  return createTabbedViewer(
    {
      text: {
        label: "Текст",
        panel: createArticleTextPanel(article, pages),
      },
      latex: {
        label: "LaTeX",
        panel: createArticleLatexPanel(article),
      },
    },
    ["text", "latex"],
    "text"
  );
}

function createSecondaryButton(label, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "secondary-button";
  button.textContent = label;
  button.addEventListener("click", onClick);
  return button;
}

async function buildDocumentReadablePdfPayload(pages) {
  const sourceStem = sanitizeFilenamePart(getSourceDocumentStem(), "document");
  const sortedPages = pages
    .slice()
    .sort((left, right) => left.page_number - right.page_number);
  const sections = [];

  for (const page of sortedPages) {
    sections.push({
      heading: null,
      note: null,
      blocks: await prepareReadableBlocksForPdfExport(buildReadableBlocksForPage(page)),
    });
  }

  return {
    title: sourceStem,
    subtitle: null,
    filename: `${sourceStem}-readable.pdf`,
    sections,
  };
}

async function buildArticleReadablePdfPayload(article, pages) {
  const sourceStem = sanitizeFilenamePart(getSourceDocumentStem(), "document");
  const articleTitle = summarizeArticlePreview(article.title_preview, article.article_id);

  return {
    title: articleTitle,
    subtitle: null,
    filename: `${sourceStem}-${sanitizeFilenamePart(article.article_id, "article")}.pdf`,
    sections: [
      {
        heading: null,
        note: null,
        blocks: await prepareReadableBlocksForPdfExport(buildReadableBlocksForArticle(article, pages)),
      },
    ],
  };
}

async function buildCollectionExportPayload(articleSegmentation, pages, processingMode = "full") {
  const sourceStem = sanitizeFilenamePart(getSourceDocumentStem(), "document");
  const articles = [];

  for (const article of articleSegmentation?.articles || []) {
    articles.push({
      article_id: article.article_id,
      page_numbers: article.page_numbers || [],
      title: article.title_preview || "",
      author: article.author_preview || "",
      article_text: collectArticleText(article, pages),
      article_latex_preview: article.article_latex_preview || "",
      article_latex_document: article.article_latex_document || "",
      article_metadata: article.article_metadata || null,
      readable_pdf: await buildArticleReadablePdfPayload(article, pages),
    });
  }

  return {
    title: sourceStem,
    filename: `${sourceStem}-collection.zip`,
    source_document_name: getSourceDocumentName(),
    processing_mode: processingMode,
    articles,
  };
}

function createReadableExportSection(pages, articleSegmentation, processingMode = "full") {
  const section = document.createElement("section");
  section.className = "readable-export";

  const header = document.createElement("div");
  header.className = "readable-export__header";

  const titleWrap = document.createElement("div");
  const title = document.createElement("h3");
  title.textContent = "Экспорт читаемого текста";
  titleWrap.appendChild(title);

  const description = document.createElement("p");
  description.className = "readable-export__description";
  description.textContent =
    "Скачайте объединённый PDF с тем читаемым представлением документа, которое показано в интерфейсе.";
  titleWrap.appendChild(description);

  const actions = document.createElement("div");
  actions.className = "readable-export__actions";

  const exportButton = createSecondaryButton("Скачать PDF документа", async () => {
    downloadReadablePdf(
      () => buildDocumentReadablePdfPayload(pages),
      exportButton,
      "PDF с читаемым текстом документа сформирован."
    );
  });

  actions.appendChild(exportButton);

  if (articleSegmentation?.articles?.length) {
    const exportCollectionButton = createSecondaryButton(
      "Скачать коллекцию",
      async () => {
        downloadCollectionArchive(
          () => buildCollectionExportPayload(articleSegmentation, pages, processingMode),
          exportCollectionButton,
          "Архив электронной коллекции сформирован."
        );
      }
    );
    actions.appendChild(exportCollectionButton);
  }

  header.appendChild(titleWrap);
  header.appendChild(actions);
  section.appendChild(header);
  return section;
}

function createTextBlockProcessorSection(page) {
  const section = document.createElement("section");
  section.className = "page-ocr-section";

  const header = document.createElement("div");
  header.className = "page-ocr-section__header";

  const titleWrap = document.createElement("div");
  const title = document.createElement("h4");
  title.textContent = "Финальное представление страницы";
  titleWrap.appendChild(title);

  const description = document.createElement("p");
  description.className = "page-ocr-section__description";
  description.textContent =
    "На основе упорядоченных блоков страницы формируется единый PageContent: машинная структура и читаемый текст.";
  titleWrap.appendChild(description);

  const status = document.createElement("p");
  status.className = "page-ocr-section__status";
  status.textContent = "Ожидаем финальный результат представления страницы.";

  const content = document.createElement("div");
  content.className = "page-ocr-section__content";

  header.appendChild(titleWrap);
  section.appendChild(header);
  section.appendChild(status);
  section.appendChild(content);

  if (page.text_block_error) {
    status.textContent = page.text_block_error;
    status.classList.add("page-ocr-section__status--error");
    return section;
  }

  if (page.text_block_content) {
    if (page.formula_block_error) {
      status.textContent = `PageContent сформирован частично: ${page.formula_block_error}`;
      status.classList.add("page-ocr-section__status--error");
    } else {
      status.textContent =
        "Сформирован единый PageContent: машинная структура страницы и человекочитаемый текст доступны в переключаемых режимах.";
    }
    content.appendChild(createPageContentViewer(page.text_block_content));
    return section;
  }

  status.textContent = "OCR блоков не был выполнен для этой страницы.";
  status.classList.add("page-ocr-section__status--error");
  return section;
}

function createArticleSegmentationSection(segmentation, pages) {
  const section = document.createElement("section");
  section.className = "article-segmentation";

  const header = document.createElement("div");
  header.className = "article-segmentation__header";

  const titleWrap = document.createElement("div");
  const title = document.createElement("h3");
  title.textContent = "Найденные статьи";
  titleWrap.appendChild(title);

  const description = document.createElement("p");
  description.className = "article-segmentation__description";
  description.textContent =
    "Граница статьи определяется по отступу первого text-блока сверху; пунктуация в конце предыдущей страницы повышает уверенность.";
  titleWrap.appendChild(description);

  const summary = document.createElement("div");
  summary.className = "article-segmentation__summary";
  summary.appendChild(createSmallBadge(`${segmentation.article_count || 0} статей`, "success"));
  summary.appendChild(createSmallBadge(`${segmentation.total_pages || 0} страниц`));

  header.appendChild(titleWrap);
  header.appendChild(summary);
  section.appendChild(header);

  const list = document.createElement("div");
  list.className = "article-list";

  (segmentation.articles || []).forEach((article) => {
    const articleMetadata = article.article_metadata || null;
    const details = document.createElement("details");
    details.className = "article-item";

    const summaryNode = document.createElement("summary");
    summaryNode.className = "article-item__summary";

    const main = document.createElement("div");
    main.className = "article-item__summary-main";

    const idBadge = document.createElement("div");
    idBadge.className = "article-item__id";
    idBadge.textContent = article.article_id.replace("article_", "A");

    const headline = document.createElement("div");
    headline.className = "article-item__headline";

    const headlineTop = document.createElement("div");
    headlineTop.className = "article-item__headline-top";
    headlineTop.appendChild(createSmallBadge(`${article.start_page}-${article.end_page}`));

    const titleNode = document.createElement("strong");
    titleNode.className = "article-item__title";
    titleNode.textContent = summarizeArticlePreview(article.title_preview, "Без уверенного начала");

    headline.appendChild(headlineTop);
    headline.appendChild(titleNode);

    main.appendChild(idBadge);
    main.appendChild(headline);

    const aside = document.createElement("div");
    aside.className = "article-item__summary-aside";
    aside.appendChild(
      createSmallBadge(`score ${Number(article.boundary_confidence || 0).toFixed(2)}`)
    );

    summaryNode.appendChild(main);
    summaryNode.appendChild(aside);
    details.appendChild(summaryNode);

    const body = document.createElement("div");
    body.className = "article-item__body";

    const actions = document.createElement("div");
    actions.className = "article-item__actions";
    const exportButton = createSecondaryButton("Скачать PDF статьи", async () => {
      downloadReadablePdf(
        () => buildArticleReadablePdfPayload(article, pages),
        exportButton,
        `PDF статьи "${summarizeArticlePreview(article.title_preview, article.article_id)}" сформирован.`
      );
    });
    actions.appendChild(exportButton);
    body.appendChild(actions);

    const meta = document.createElement("div");
    meta.className = "article-item__meta";
    meta.appendChild(createBlockDetailRow("Article ID", article.article_id));
    meta.appendChild(createBlockDetailRow("Pages", (article.page_numbers || []).join(", ") || "—"));
    meta.appendChild(createBlockDetailRow("Title", article.title_preview || "—"));
    meta.appendChild(createBlockDetailRow("Author", article.author_preview || "—"));
    if (articleMetadata?.language) {
      meta.appendChild(createBlockDetailRow("Language", articleMetadata.language));
    }
    if (articleMetadata?.year) {
      meta.appendChild(createBlockDetailRow("Year", String(articleMetadata.year)));
    }
    if (articleMetadata?.keywords?.length) {
      meta.appendChild(createBlockDetailRow("Keywords", articleMetadata.keywords.join(", ")));
    }
    if (articleMetadata?.abstract) {
      meta.appendChild(createBlockDetailRow("Abstract", summarizeText(articleMetadata.abstract, 220)));
    }
    if (articleMetadata?.references?.length) {
      meta.appendChild(
        createBlockDetailRow("References", String(articleMetadata.references.length))
      );
    }
    meta.appendChild(createBlockDetailRow("Start", String(article.start_page)));
    meta.appendChild(createBlockDetailRow("End", String(article.end_page)));
    body.appendChild(meta);

    body.appendChild(createArticleContentViewer(article, pages));

    details.appendChild(body);
    list.appendChild(details);
  });

  section.appendChild(list);
  return section;
}

function createPageCard(page, documentMode = "full") {
  const processingMode = page.processing_mode || documentMode || "full";
  const card = document.createElement("article");
  card.className = "page-card page-card--rich";

  const header = document.createElement("div");
  header.className = "page-card__header";

  const titleWrap = document.createElement("div");
  const title = document.createElement("h3");
  title.textContent = `Страница ${page.page_number}`;
  titleWrap.appendChild(title);
  header.appendChild(titleWrap);
  header.appendChild(createBadge(page.has_text));

  const meta = document.createElement("div");
  meta.className = "page-card__meta";
  meta.appendChild(createMetaChip("Режим", getProcessingModeLabel(processingMode)));
  meta.appendChild(createMetaChip("PDF", page.has_text ? "text" : "ocr"));

  if (page.layout_analysis) {
    meta.appendChild(createMetaChip("Layout", `${page.layout_analysis.blocks.length} блоков`, "success"));
  } else if (page.layout_error) {
    meta.appendChild(createMetaChip("Layout", "ошибка", "error"));
  } else {
    meta.appendChild(createMetaChip("Layout", "нет данных"));
  }

  if (page.text_block_content) {
    meta.appendChild(
      createMetaChip(
        "PageContent",
        `${page.text_block_content.blocks.length} блоков`,
        page.formula_block_error ? "error" : "success"
      )
    );
  } else if (page.text_block_error) {
    meta.appendChild(createMetaChip("PageContent", "ошибка", "error"));
  }

  if (page.formula_block_error) {
    meta.appendChild(createMetaChip("Formula", "ошибка", "error"));
  }

  if (page.layout_analysis?.result_json_path) {
    meta.appendChild(createMetaChip("JSON", "saved"));
  }

  const previews = document.createElement("div");
  previews.className = "page-card__previews";
  const renderPlaceholder =
    processingMode === "text_only" && !page.page_image_data_url
      ? "Рендер страницы был пропущен, потому что встроенного текста оказалось достаточно."
      : "Рендер страницы недоступен.";
  const ocrPreviewPlaceholder =
    processingMode === "full"
      ? "OCR-предобработка для этой страницы не потребовалась."
      : "Предобработка отключена для выбранного режима.";
  const layoutPlaceholder =
    processingMode === "text_only"
      ? 'Layout-анализ пропущен в режиме "Только текст".'
      : page.layout_error || "Layout-анализ не вернул визуализацию.";
  previews.appendChild(
    createPreviewPanel(
      "Рендер страницы",
      page.page_image_data_url,
      renderPlaceholder
    )
  );
  previews.appendChild(
    createPreviewPanel(
      "OCR-превью",
      page.image_data_url,
      ocrPreviewPlaceholder
    )
  );
  previews.appendChild(
    createPreviewPanel(
      "Layout-разметка",
      page.layout_analysis?.visualization_data_url || null,
      layoutPlaceholder
    )
  );

  const excerpt = document.createElement("p");
  excerpt.className = "page-card__text";
  excerpt.textContent = summarizeText(page.text);

  card.appendChild(header);
  card.appendChild(meta);
  card.appendChild(previews);
  card.appendChild(excerpt);

  if (page.layout_error) {
    const errorNode = document.createElement("p");
    errorNode.className = "page-card__error";
    errorNode.textContent = `Layout-анализ: ${page.layout_error}`;
    card.appendChild(errorNode);
  }

  if (page.layout_analysis || page.text_block_content || page.text_block_error) {
    card.appendChild(createTextBlockProcessorSection(page));
  }

  return card;
}

function renderResults(documentResult) {
  const pages = Array.isArray(documentResult) ? documentResult : documentResult.pages || [];
  const processingMode = Array.isArray(documentResult)
    ? "full"
    : documentResult.processing_mode || documentResult.pages?.[0]?.processing_mode || "full";
  const articleSegmentation = Array.isArray(documentResult)
    ? null
    : documentResult.article_segmentation || null;
  resultsNode.innerHTML = "";

  if (!pages.length) {
    resultsNode.innerHTML = `
      <article class="empty-state">
        <h3>Страницы не найдены</h3>
        <p>Сервис не вернул данных по загруженному документу.</p>
      </article>
    `;
    summaryNode.textContent = "Нет данных";
    return;
  }

  const pagesWithText = pages.filter((page) => page.has_text).length;
  const pagesForOcr = pages.length - pagesWithText;
  const pagesWithLayout = pages.filter((page) => page.layout_analysis).length;
  const pagesWithLayoutError = pages.filter((page) => page.layout_error).length;
  const pagesWithTextBlocks = pages.filter(
    (page) => page.text_block_content && !page.formula_block_error
  ).length;
  const pagesWithTextBlockError = pages.filter(
    (page) => page.text_block_error || page.formula_block_error
  ).length;
  const articleSummary = articleSegmentation
    ? ` | Статей: ${articleSegmentation.article_count}`
    : "";
  summaryNode.textContent =
    `Режим: ${getProcessingModeLabel(processingMode)} | Всего: ${pages.length} | С текстом: ${pagesWithText} | Для OCR: ${pagesForOcr} | ` +
    `Layout OK: ${pagesWithLayout} | Layout ошибок: ${pagesWithLayoutError} | ` +
    `PageContent OK: ${pagesWithTextBlocks} | PageContent ошибок: ${pagesWithTextBlockError}` +
    articleSummary;

  resultsNode.appendChild(createReadableExportSection(pages, articleSegmentation, processingMode));

  if (articleSegmentation) {
    resultsNode.appendChild(createArticleSegmentationSection(articleSegmentation, pages));
  }

  pages.forEach((page) => {
    resultsNode.appendChild(createPageCard(page, processingMode));
  });
}

async function handleSubmit(event) {
  event.preventDefault();

  const [file] = fileInput.files;
  if (!file) {
    setStatus("Сначала выберите PDF-файл.", true);
    return;
  }

  const formData = new FormData();
  const processingMode = getSelectedProcessingMode();
  formData.append("file", file);
  formData.append("processing_mode", processingMode);

  processButton.disabled = true;
  setProcessingStage(`Отправляем PDF на сервер (${getProcessingModeLabel(processingMode)})`, "active");
  setStatus(`Идёт обработка PDF: ${getProcessingModeLabel(processingMode)}.`);
  summaryNode.textContent = "Live-progress запущен";

  try {
    const response = await fetch("/api/process-pdf-stream", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.detail || "Не удалось обработать PDF.");
    }

    const payload = await readProcessingStream(response);
    renderResults(payload);
    setProcessingStage("Обработка завершена", "done");
    setStatus(`PDF-этап завершён: ${getProcessingModeLabel(payload.processing_mode || processingMode)}.`);
  } catch (error) {
    console.error(error);
    setProcessingStage(error.message || "Ошибка обработки", "error");
    setStatus(error.message || "Произошла ошибка при обработке файла.", true);
    summaryNode.textContent = "Ошибка обработки";
  } finally {
    processButton.disabled = false;
  }
}

fileInput.addEventListener("change", () => {
  updateSelectedFileName();

  if (fileInput.files?.[0]?.name) {
    setStatus(`Файл выбран: ${fileInput.files[0].name}`);
  } else {
    setStatus("Файл ещё не загружен.");
  }
});

processingModeSelect?.addEventListener("change", () => {
  updateProcessingModePresentation();
});

updateSelectedFileName();
updateProcessingModePresentation();
form.addEventListener("submit", handleSubmit);
