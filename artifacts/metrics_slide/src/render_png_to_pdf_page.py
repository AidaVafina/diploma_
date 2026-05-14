from __future__ import annotations

import sys
from pathlib import Path

from reportlab.pdfgen import canvas


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: render_png_to_pdf_page.py <input.png> <output.pdf> <page_width_pt> <page_height_pt>"
        )

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    page_width = float(sys.argv[3])
    page_height = float(sys.argv[4])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))
    pdf.drawImage(str(input_path), 0, 0, width=page_width, height=page_height, mask="auto")
    pdf.showPage()
    pdf.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
