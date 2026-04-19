from __future__ import annotations

import logging
import os


def setup_logging() -> None:
    if getattr(setup_logging, "_configured", False):
        return

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    setup_logging._configured = True

