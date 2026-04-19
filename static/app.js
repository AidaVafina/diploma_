const form = document.getElementById("upload-form");
const fileInput = document.getElementById("pdf-file");
const processButton = document.getElementById("process-button");
const statusNode = document.getElementById("status");
const summaryNode = document.getElementById("summary");
const resultsNode = document.getElementById("results");

function setStatus(message, isError = false) {
  statusNode.textContent = message;
  statusNode.style.color = isError ? "var(--warning)" : "var(--muted)";
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

function formatDebugValue(value) {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (Array.isArray(value)) {
    return value.join(", ");
  }
  return String(value);
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

function createTextViewPanel(titleText, contentText, pageContent, modifier = "") {
  const wrapper = document.createElement("div");
  wrapper.className = "page-text-summary";

  const title = document.createElement("h4");
  title.textContent = titleText;
  wrapper.appendChild(title);
  wrapper.appendChild(createViewMetaLine(pageContent));

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

function renderReadableView(pageContent) {
  const wrapper = document.createElement("div");
  wrapper.className = "page-readable-view";

  const title = document.createElement("h4");
  title.textContent = "Текст страницы для чтения";
  wrapper.appendChild(title);

  const content = document.createElement("div");
  content.className = "page-readable-view__content";

  (pageContent.blocks || []).forEach((block) => {
    if (block.type === "formula") {
      const formulaNode = document.createElement("div");
      formulaNode.className = "formula";

      const cleanedLatex = sanitizeLatexForReadableView(
        block.latex || block.formula_result?.latex || ""
      );
      const latexState = validateLatexForReadableView(cleanedLatex);

      if (!cleanedLatex || latexState === false) {
        formulaNode.classList.add("formula--error");
        formulaNode.textContent = "[FORMULA ERROR]";
      } else {
        formulaNode.textContent = `\\[${cleanedLatex}\\]`;
        if (latexState === null) {
          formulaNode.classList.add("formula--pending");
        }
      }

      content.appendChild(formulaNode);
      return;
    }

    if (["text", "title", "header", "footer", "page_number"].includes(block.type)) {
      const cleanedText = sanitizeReadableText(block.content);
      if (!cleanedText) {
        return;
      }

      const paragraph = document.createElement("p");
      paragraph.className =
        block.type === "title" ? "page-readable-view__paragraph page-readable-view__paragraph--title" :
        "page-readable-view__paragraph";
      paragraph.textContent = cleanedText;
      content.appendChild(paragraph);
      return;
    }

    if (block.type === "table" || block.type === "image") {
      const placeholder = document.createElement("div");
      placeholder.className = "page-readable-view__placeholder";
      placeholder.textContent = block.type === "table" ? "[TABLE]" : "[IMAGE]";
      content.appendChild(placeholder);
    }
  });

  wrapper.appendChild(content);
  requestAnimationFrame(() => renderReadableMath(content));
  return wrapper;
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
  const viewModes = presentation.available_views || ["readable", "tex", "structure"];
  const defaultView = presentation.default_view || "readable";

  const buttonsWrap = document.createElement("div");
  buttonsWrap.className = "page-result-switcher";

  const panelsWrap = document.createElement("div");
  panelsWrap.className = "page-result-panels";

  const viewDefinitions = {
    readable: {
      label: "Текст",
      panel: renderReadableView(pageContent),
    },
    tex: {
      label: "TeX",
      panel: createTextViewPanel(
        "Предварительный TeX страницы",
        presentation.tex_preview || "% TeX preview пока недоступен.",
        pageContent,
        "tex"
      ),
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
    "На основе упорядоченных блоков страницы формируется единый PageContent: машинная структура, читаемый текст и предварительный TeX.";
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
        "Сформирован единый PageContent: машинная структура страницы, человекочитаемый текст и предварительный TeX доступны в переключаемых режимах.";
    }
    content.appendChild(createPageContentViewer(page.text_block_content));
    return section;
  }

  status.textContent = "OCR блоков не был выполнен для этой страницы.";
  status.classList.add("page-ocr-section__status--error");
  return section;
}

function createArticleSegmentationSection(segmentation) {
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
    "Сегментация выполнена поверх готовых PageContent: показаны предполагаемые границы статей и краткие превью.";
  titleWrap.appendChild(description);

  const summary = document.createElement("div");
  summary.className = "article-segmentation__summary";
  summary.appendChild(createSmallBadge(`${segmentation.article_count || 0} статей`, "success"));
  summary.appendChild(createSmallBadge(`${segmentation.total_pages || 0} страниц`));
  summary.appendChild(
    createSmallBadge(
      `${segmentation.needs_review_count || 0} требуют проверки`,
      segmentation.needs_review_count ? "warning" : ""
    )
  );

  header.appendChild(titleWrap);
  header.appendChild(summary);
  section.appendChild(header);

  const list = document.createElement("div");
  list.className = "article-list";

  if (!segmentation.articles?.length) {
    const empty = document.createElement("p");
    empty.className = "article-segmentation__description";
    empty.textContent = "Сегментатор не нашёл уверенных границ статей.";
    section.appendChild(empty);
    return section;
  }

  segmentation.articles.forEach((article) => {
    const details = document.createElement("details");
    details.className = "article-item";
    if (article.needs_review) {
      details.classList.add("article-item--review");
    }

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
    if (article.needs_review) {
      headlineTop.appendChild(createSmallBadge("warning", "warning"));
    }

    const titleNode = document.createElement("strong");
    titleNode.className = "article-item__title";
    titleNode.textContent = summarizeArticlePreview(article.title_preview, "Без уверенного заголовка");

    const subtitle = document.createElement("p");
    subtitle.className = "article-item__subtitle";
    subtitle.textContent = summarizeArticlePreview(article.author_preview, "Автор не определён");

    headline.appendChild(headlineTop);
    headline.appendChild(titleNode);
    headline.appendChild(subtitle);

    main.appendChild(idBadge);
    main.appendChild(headline);

    const aside = document.createElement("div");
    aside.className = "article-item__summary-aside";
    aside.appendChild(
      createSmallBadge(`conf ${Number(article.boundary_confidence || 0).toFixed(2)}`)
    );

    summaryNode.appendChild(main);
    summaryNode.appendChild(aside);
    details.appendChild(summaryNode);

    const body = document.createElement("div");
    body.className = "article-item__body";

    const meta = document.createElement("div");
    meta.className = "article-item__meta";
    meta.appendChild(createBlockDetailRow("Article ID", article.article_id));
    meta.appendChild(createBlockDetailRow("Pages", (article.page_numbers || []).join(", ") || "—"));
    meta.appendChild(createBlockDetailRow("Start", String(article.start_page)));
    meta.appendChild(createBlockDetailRow("End", String(article.end_page)));
    meta.appendChild(
      createBlockDetailRow(
        "Confidence",
        Number(article.boundary_confidence || 0).toFixed(3)
      )
    );
    body.appendChild(meta);

    if (article.debug_info && Object.keys(article.debug_info).length) {
      const debug = document.createElement("div");
      debug.className = "article-item__debug";
      const debugTitle = document.createElement("h5");
      debugTitle.textContent = "Debug info";
      const debugList = document.createElement("dl");
      debugList.className = "article-item__debug-grid";
      Object.entries(article.debug_info).forEach(([key, value]) => {
        const label = document.createElement("dt");
        label.textContent = key;
        const valueNode = document.createElement("dd");
        valueNode.textContent = formatDebugValue(value);
        debugList.appendChild(label);
        debugList.appendChild(valueNode);
      });
      debug.appendChild(debugTitle);
      debug.appendChild(debugList);
      body.appendChild(debug);
    }

    details.appendChild(body);
    list.appendChild(details);
  });

  section.appendChild(list);
  return section;
}

function createPageCard(page) {
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
  previews.appendChild(
    createPreviewPanel(
      "Рендер страницы",
      page.page_image_data_url,
      "Рендер страницы недоступен."
    )
  );
  previews.appendChild(
    createPreviewPanel(
      "OCR-превью",
      page.image_data_url,
      "OCR-предобработка для этой страницы не потребовалась."
    )
  );
  previews.appendChild(
    createPreviewPanel(
      "Layout-разметка",
      page.layout_analysis?.visualization_data_url || null,
      page.layout_error || "Layout-анализ не вернул визуализацию."
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
  } else if (page.layout_analysis) {
    card.appendChild(createTextBlockProcessorSection(page));
  }

  return card;
}

function renderResults(documentResult) {
  const pages = Array.isArray(documentResult) ? documentResult : documentResult.pages || [];
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
    ? ` | Статей: ${articleSegmentation.article_count} | Границ на проверку: ${articleSegmentation.needs_review_count}`
    : "";
  summaryNode.textContent =
    `Всего: ${pages.length} | С текстом: ${pagesWithText} | Для OCR: ${pagesForOcr} | ` +
    `Layout OK: ${pagesWithLayout} | Layout ошибок: ${pagesWithLayoutError} | ` +
    `PageContent OK: ${pagesWithTextBlocks} | PageContent ошибок: ${pagesWithTextBlockError}` +
    articleSummary;

  if (articleSegmentation) {
    resultsNode.appendChild(createArticleSegmentationSection(articleSegmentation));
  }

  pages.forEach((page) => {
    resultsNode.appendChild(createPageCard(page));
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
  formData.append("file", file);

  processButton.disabled = true;
  setStatus(
    "Загружаем PDF, обрабатываем страницы, запускаем layout-анализ и собираем финальный PageContent..."
  );
  summaryNode.textContent = "Комбинированная обработка запущена";

  try {
    const response = await fetch("/api/process-pdf", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Не удалось обработать PDF.");
    }

    renderResults(payload);
    setStatus("PDF-этап завершён. PageContent сформирован для каждой страницы, журнал разбит на статьи.");
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Произошла ошибка при обработке файла.", true);
    summaryNode.textContent = "Ошибка обработки";
  } finally {
    processButton.disabled = false;
  }
}

form.addEventListener("submit", handleSubmit);
