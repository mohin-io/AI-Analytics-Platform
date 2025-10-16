"""
Logging utility for the Unified AI Analytics Platform

This module provides a centralized logging configuration for the entire platform.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "unified_ai_platform",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    This function creates a logger that outputs to both console and a log file,
    with formatted timestamps and log levels. The log file is created in the
    specified directory with a timestamp in the filename.

    Args:
        name: Name of the logger
        log_file: Optional specific log file name. If None, auto-generates based on timestamp
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_dir: Directory to store log files

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("my_module", level=logging.DEBUG)
        >>> logger.info("Starting process")
        >>> logger.error("An error occurred")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{name}_{timestamp}.log"

    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path / log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.

    This function retrieves a logger that was previously set up. If the logger
    doesn't exist, it creates a new one with default settings.

    Args:
        name: Name of the logger to retrieve

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger("unified_ai_platform")
        >>> logger.info("Processing data")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger