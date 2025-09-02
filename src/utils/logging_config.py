"""Logging configuration for the H200 system."""

import logging
import sys
from typing import Optional
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "created", "filename", 
                          "funcName", "levelname", "levelno", "lineno", 
                          "module", "exc_info", "exc_text", "stack_info",
                          "pathname", "processName", "process", "threadName",
                          "thread", "relativeCreated", "msecs", "getMessage"]:
                log_data[key] = value
        
        return json.dumps(log_data)


def setup_logging(
    name: Optional[str] = None,
    level: str = "INFO",
    use_json: bool = True,
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        name: Logger name (defaults to root logger)
        level: Logging level
        use_json: Whether to use JSON formatting
    
    Returns:
        Configured logger
    """
    # Get logger
    logger = logging.getLogger(name)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set level
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    
    # Set formatter
    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    
    handler.setFormatter(formatter)
    
    # Add handler
    logger.addHandler(handler)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger by name."""
    return setup_logging(name)