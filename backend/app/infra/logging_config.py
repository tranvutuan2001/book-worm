import logging
import argparse
import os
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar

# Context variable to track current request ID across async operations
current_request_id: ContextVar[str] = ContextVar('request_id', default=None)

class RequestFormatter(logging.Formatter):
    """Custom formatter that includes request ID in log messages"""
    
    def format(self, record):
        # Add request ID to the record if available
        request_id = current_request_id.get(None)
        if request_id:
            record.request_id = request_id
        else:
            record.request_id = "no-request"
        return super().format(record)

def setup_logging():
    """
    Setup logging system with separate handlers for app and LLM logs
    
    Args:
        enable_logging: If True, enables file logging. If False, only console logging.
    """
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Common formatter with request ID
    formatter = RequestFormatter(
        '%(asctime)s | %(name)s | %(levelname)s | REQ:%(request_id)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (always active)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # App log handler - for main application logs
    app_log_handler = RotatingFileHandler(
        os.path.join(logs_dir, "app.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    app_log_handler.setLevel(logging.INFO)
    app_log_handler.setFormatter(formatter)
    
    # LLM log handler - for LLM-specific logs
    llm_log_handler = RotatingFileHandler(
        os.path.join(logs_dir, "llm.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    llm_log_handler.setLevel(logging.INFO)
    llm_log_handler.setFormatter(formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(app_log_handler)
    
    # Setup LLM-specific logger
    llm_logger = logging.getLogger('app.llm_connector')
    llm_logger.handlers.clear()  # Remove existing handlers
    llm_logger.addHandler(llm_log_handler)
    llm_logger.addHandler(console_handler)  # Also log to console
    llm_logger.setLevel(logging.INFO)
    llm_logger.propagate = False  # Don't propagate to root to avoid duplicate logs
    

def get_request_logger(logger_name: str = "app"):
    """Get a logger for request-specific logging"""
    return logging.getLogger(logger_name)

def start_request_logging(endpoint: str = None, user_query: str = None):
    """
    Start logging for a new customer request
    
    Returns:
        str: Request ID for tracking
    """
    request_id = str(uuid.uuid4())[:8]
    current_request_id.set(request_id)
    
    logger = get_request_logger("app.request")
    logger.info("=" * 80)
    logger.info(f"🚀 NEW CUSTOMER REQUEST STARTED")
    if endpoint:
        logger.info(f"📍 Endpoint: {endpoint}")
    if user_query:
        logger.info(f"💬 Query: {user_query[:200]}...")
    logger.info("=" * 80)
    
    return request_id

def end_request_logging(response_summary: str = None, success: bool = True):
    """
    End logging for a customer request
    
    Args:
        response_summary: Brief summary of the response
        success: Whether the request was successful
    """
    request_id = current_request_id.get("unknown")
    logger = get_request_logger("app.request")
    
    logger.info("=" * 80)
    status_emoji = "✅" if success else "❌"
    logger.info(f"{status_emoji} CUSTOMER REQUEST COMPLETED")
    
    if response_summary:
        logger.info(f"📝 Response: {response_summary[:300]}...")
    
    logger.info(f"🏁 REQUEST {request_id} FINISHED")
    logger.info("=" * 80)
    logger.info("")  # Empty line for separation
    
    # Clear the request ID from context
    current_request_id.set(None)

def get_llm_logger():
    """Get the LLM-specific logger"""
    return logging.getLogger('app.llm_connector')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Book Worm API Server")
    parser.add_argument(
        '--enable-logging', 
        action='store_true',
        help='Enable file logging for app and LLM logs'
    )
    return parser.parse_args()