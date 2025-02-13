"""
Centralized logging configuration for the token monitoring system.
"""

import os
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting and handlers.
    
    Args:
        name: The name of the logger
        log_file: Optional log file path. If not provided, will use name.log
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Set up logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler with rotation
    if not log_file:
        log_file = f"logs/{name}_{datetime.now().strftime('%Y%m%d')}.log"
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Add handlers if they don't exist
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.
    
    Args:
        name: The name of the logger to get or create
        
    Returns:
        logging.Logger: The requested logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger 