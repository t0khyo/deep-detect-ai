import logging
from .config import get_config

def setup_logging():
    config = get_config()
    
    # Configure root logger with console handler only
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler()  # Console handler only
        ]
    )
    
    # Create logger for the application
    logger = logging.getLogger("deep_detect_ai")
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    return logger 