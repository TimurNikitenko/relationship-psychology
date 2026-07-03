import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger():
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    logger = logging.getLogger("relationships_bot")
    logger.setLevel(log_level)
    
    # Clear any existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (rotates at 10MB, keeps 5 backup files)
    file_handler = RotatingFileHandler(
        "logs/bot.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()
