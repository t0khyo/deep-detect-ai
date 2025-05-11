import logging
import logging.handlers
import os
from pathlib import Path
from .config import get_config

def setup_logging():
    config = get_config()
    
    # Create logs directory if it doesn't exist
    log_dir = Path(config.BASE_DIR) / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),  # Console handler
            logging.handlers.RotatingFileHandler(
                log_dir / "app.log",
                maxBytes=10485760,  # 10MB
                backupCount=5
            )
        ]
    )
    
    # Create logger for the application
    logger = logging.getLogger("deep_detect_ai")
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    return logger 