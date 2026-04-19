from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from app.schemas import Block
from app.services.formula_block_processor import process_formula_blocks


class FormulaBlockProcessorTests(unittest.TestCase):
    def test_process_formula_blocks_reuses_seed_latex_without_rerunning_ocr(self) -> None:
        image = np.ones((200, 400, 3), dtype="uint8") * 255
        blocks = [
            Block(
                block_id="page_1_block_001",
                type="formula",
                bbox=[10, 20, 150, 80],
                reading_order=1,
                route_to="formula_pipeline",
                seed_latex="x^2 + y^2 = z^2",
            )
        ]

        with patch(
            "app.services.formula_block_processor.recognize_formula_block",
            side_effect=AssertionError("recognize_formula_block should not be called"),
        ):
            page_content = process_formula_blocks(image, blocks, page_number=1)

        self.assertEqual(len(page_content.blocks), 1)
        self.assertEqual(page_content.blocks[0].latex, "x^2 + y^2 = z^2")
        self.assertEqual(page_content.blocks[0].formula_backend, "surya")


if __name__ == "__main__":
    unittest.main()
