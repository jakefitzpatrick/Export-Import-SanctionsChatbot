from __future__ import annotations
"""
Centralized logging configuration for the chatbot system.
Produces structured logs (JSON or plain) with console and optional file handlers.
"""
import os
import sys
import logging

from pythonjsonlogger import jsonlogger


def setup_logger(name = None) -> logging.Logger:
    """
    Configure and return a logger instance.
    Reads LOG_LEVEL, LOG_FORMAT ('json' or 'text'), and optional LOG_FILE from environment.
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    fmt_style = os.getenv("LOG_FORMAT", "json").lower()

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if fmt_style == "json":
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    log_file = os.getenv("LOG_FILE")
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
