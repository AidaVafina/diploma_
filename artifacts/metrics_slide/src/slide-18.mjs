function addText(ctx, slide, text, left, top, width, height, options = {}) {
  return ctx.addText(slide, {
    text,
    left,
    top,
    width,
    height,
    fontSize: options.fontSize ?? 24,
    bold: Boolean(options.bold),
    color: options.color ?? "#111111",
    typeface: options.typeface ?? "Arial",
    align: options.align ?? "left",
    valign: options.valign ?? "middle",
    fill: options.fill ?? "#00000000",
    line: options.line ?? ctx.line(),
    insets: options.insets ?? { left: 8, right: 8, top: 4, bottom: 4 },
    name: options.name,
  });
}

function addRect(ctx, slide, left, top, width, height, options = {}) {
  return ctx.addShape(slide, {
    left,
    top,
    width,
    height,
    geometry: "rect",
    fill: options.fill ?? "#FFFFFF",
    line: options.line ?? ctx.line("#D9D9D9", 1),
    name: options.name,
  });
}

function drawCell(ctx, slide, x, y, w, h, text, options = {}) {
  addRect(ctx, slide, x, y, w, h, {
    fill: options.fill ?? "#FFFFFF",
    line: options.line ?? ctx.line("#D0D0D0", 1),
  });
  addText(ctx, slide, text, x + 6, y + 4, w - 12, h - 8, {
    fontSize: options.fontSize ?? 22,
    bold: Boolean(options.bold),
    color: options.color ?? "#202020",
    valign: options.valign ?? "middle",
    typeface: options.typeface ?? "Arial",
    insets: options.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
  });
}

export async function slide18(presentation, ctx) {
  const slide = presentation.slides.add();

  const palette = {
    bg: "#FFFFFF",
    ink: "#111111",
    soft: "#8A8A8A",
    grid: "#D0D0D0",
    headFill: "#F7F7F7",
    noteFill: "#FAFAFA",
  };

  addRect(ctx, slide, 0, 0, ctx.W, ctx.H, {
    fill: palette.bg,
    line: ctx.line(),
  });

  addText(ctx, slide, "метрики", 106, 54, 980, 72, {
    fontSize: 58,
    bold: true,
    color: palette.ink,
    valign: "top",
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  const theoryX = 110;
  const theoryY = 150;
  const theoryW = 1180;
  const theoryH = 165;

  addRect(ctx, slide, theoryX, theoryY, theoryW, theoryH, {
    fill: "#FBFBFB",
    line: ctx.line("#E2E2E2", 1),
  });

  addText(ctx, slide, "Теоретические метрики для задач выделения объектов", 132, 165, 900, 24, {
    fontSize: 19,
    bold: true,
    color: "#303030",
    valign: "top",
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });

  const metricCards = [
    {
      x: 132,
      title: "Precision",
      formula: "P = TP / (TP + FP)",
      note: "точность найденных объектов",
    },
    {
      x: 520,
      title: "Recall",
      formula: "R = TP / (TP + FN)",
      note: "полнота найденных объектов",
    },
    {
      x: 908,
      title: "F1-мера",
      formula: "F1 = 2PR / (P + R)",
      note: "баланс точности и полноты",
    },
  ];

  metricCards.forEach((card) => {
    addRect(ctx, slide, card.x, 202, 350, 78, {
      fill: "#FFFFFF",
      line: ctx.line("#D8D8D8", 1),
    });
    addText(ctx, slide, card.title, card.x + 14, 212, 322, 20, {
      fontSize: 19,
      bold: true,
      color: palette.ink,
      valign: "top",
      insets: { left: 0, right: 0, top: 0, bottom: 0 },
    });
    addText(ctx, slide, card.formula, card.x + 14, 236, 322, 20, {
      fontSize: 18,
      color: "#222222",
      valign: "top",
      insets: { left: 0, right: 0, top: 0, bottom: 0 },
    });
    addText(ctx, slide, card.note, card.x + 14, 258, 322, 16, {
      fontSize: 14,
      color: "#6A6A6A",
      valign: "top",
      insets: { left: 0, right: 0, top: 0, bottom: 0 },
    });
  });

  addText(
    ctx,
    slide,
    "TP - истинно положительные; FP - ложноположительные; FN - ложноотрицательные.",
    132,
    292,
    900,
    18,
    {
      fontSize: 15,
      color: "#5B5B5B",
      valign: "top",
      insets: { left: 0, right: 0, top: 0, bottom: 0 },
    },
  );

  const tableX = 110;
  const tableY = 350;
  const tableW = 1180;
  const col1 = 290;
  const col2 = 260;
  const col3 = 170;
  const col4 = tableW - col1 - col2 - col3;
  const headH = 52;
  const rowH = 86;

  drawCell(ctx, slide, tableX, tableY, col1, headH, "Модуль", {
    fill: palette.headFill,
    bold: true,
    fontSize: 19,
    line: ctx.line(palette.grid, 1),
  });
  drawCell(ctx, slide, tableX + col1, tableY, col2, headH, "Метрика проекта", {
    fill: palette.headFill,
    bold: true,
    fontSize: 19,
    line: ctx.line(palette.grid, 1),
  });
  drawCell(ctx, slide, tableX + col1 + col2, tableY, col3, headH, "Значение*", {
    fill: palette.headFill,
    bold: true,
    fontSize: 19,
    line: ctx.line(palette.grid, 1),
  });
  drawCell(ctx, slide, tableX + col1 + col2 + col3, tableY, col4, headH, "Основание", {
    fill: palette.headFill,
    bold: true,
    fontSize: 19,
    line: ctx.line(palette.grid, 1),
  });

  const rows = [
    {
      module: "Выделение\nблоков страницы",
      metric: "F1-мера",
      value: "~0.93",
      basis: "Ориентировочная оценка качества\nструктурного layout-анализа.",
    },
    {
      module: "OCR текстовых\nблоков",
      metric: "Средний confidence",
      value: "0.921",
      basis: "Фактическое значение по выборке\n60 блоков; needs_review 6%.",
    },
    {
      module: "Распознавание\nматематических формул",
      metric: "Exact Match\nпо LaTeX",
      value: "~85%",
      basis: "Оценка по тестовым примерам\nи визуальной верификации.",
    },
  ];

  rows.forEach((row, index) => {
    const y = tableY + headH + index * rowH;
    drawCell(ctx, slide, tableX, y, col1, rowH, row.module, {
      fontSize: 20,
      bold: true,
      line: ctx.line(palette.grid, 1),
    });
    drawCell(ctx, slide, tableX + col1, y, col2, rowH, row.metric, {
      fontSize: 18,
      line: ctx.line(palette.grid, 1),
    });
    drawCell(ctx, slide, tableX + col1 + col2, y, col3, rowH, row.value, {
      fontSize: 22,
      bold: true,
      color: "#1F1F1F",
      line: ctx.line(palette.grid, 1),
    });
    drawCell(ctx, slide, tableX + col1 + col2 + col3, y, col4, rowH, row.basis, {
      fontSize: 17,
      line: ctx.line(palette.grid, 1),
    });
  });

  addRect(ctx, slide, 110, 674, 1180, 72, {
    fill: palette.noteFill,
    line: ctx.line("#E4E4E4", 1),
  });

  addText(
    ctx,
    slide,
    "* Значения по проекту: для OCR использована фактическая выборка 60 блоков; для layout и формул приведена ориентировочная оценка.",
    132,
    690,
    1138,
    18,
    {
      fontSize: 14,
      color: "#5E5E5E",
      valign: "top",
      insets: { left: 0, right: 0, top: 0, bottom: 0 },
    },
  );

  addText(
    ctx,
    slide,
    "Источник: Manning C.D., Raghavan P., Schutze H. Introduction to Information Retrieval. Cambridge University Press, 2008. Ch. 8.",
    132,
    714,
    1138,
    18,
    {
      fontSize: 15,
      color: "#6E6E6E",
      valign: "top",
      insets: { left: 0, right: 0, top: 0, bottom: 0 },
    },
  );

  addText(ctx, slide, "Вафина А.И.", 106, 826, 170, 26, {
    fontSize: 15,
    color: palette.soft,
    valign: "top",
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });

  addText(
    ctx,
    slide,
    "Разработка системы автоматизации процесса формирования электронной коллекции научных\nретродокументов",
    395,
    820,
    810,
    34,
    {
      fontSize: 14,
      color: palette.soft,
      align: "center",
      valign: "top",
      insets: { left: 0, right: 0, top: 0, bottom: 0 },
    },
  );

  addText(ctx, slide, "16/19", 1340, 826, 120, 24, {
    fontSize: 15,
    color: palette.soft,
    align: "right",
    valign: "top",
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });

  return slide;
}
