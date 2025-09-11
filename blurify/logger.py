"""
Simple logging wrapper for Blurify.

Provides consistent logging across all modules with configurable levels.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def get_logger(
    name: str, 
    level: str = "INFO",
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_root_logger(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """
    Setup root logger for the entire application.
    
    Args:
        level: Log level for root logger
        log_file: Optional file to write all logs to
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


class LoggerMixin:
    """Mixin class to add logging capability to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)
    
    def log_debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)
    
    def log_info(self, message: str, *args, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, *args, **kwargs)
    
    def log_warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, *args, **kwargs)
    
    def log_error(self, message: str, *args, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, *args, **kwargs)
    
    def log_critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(message, *args, **kwargs)
