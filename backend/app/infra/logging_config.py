import logging
import os
import uuid
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar

# Context variable to track current request ID across async operations
current_request_id: ContextVar[str] = ContextVar('request_id', default=None)


class RequestFormatter(logging.Formatter):
    """Custom formatter that includes request ID in log messages."""

    def format(self, record):
        request_id = current_request_id.get(None)
        record.request_id = request_id if request_id else "no-request"
        return super().format(record)


def setup_logging():
    """Setup file-only logging with separate handlers for app and LLM logs."""
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    formatter = RequestFormatter(
        '%(asctime)s | %(name)s | %(levelname)s | REQ:%(request_id)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger — file only, no console output
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    app_log_handler = RotatingFileHandler(
        os.path.join(logs_dir, "app.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    app_log_handler.setLevel(logging.INFO)
    app_log_handler.setFormatter(formatter)
    root_logger.addHandler(app_log_handler)

    # LLM-specific logger writes to its own file and does not propagate to root
    llm_log_handler = RotatingFileHandler(
        os.path.join(logs_dir, "llm.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    llm_log_handler.setLevel(logging.INFO)
    llm_log_handler.setFormatter(formatter)

    llm_logger = logging.getLogger('app.llm_connector')
    llm_logger.handlers.clear()
    llm_logger.addHandler(llm_log_handler)
    llm_logger.setLevel(logging.INFO)
    llm_logger.propagate = False


def get_request_logger(logger_name: str = "app") -> logging.Logger:
    """Return a logger for the given name."""
    return logging.getLogger(logger_name)


def start_request_logging(endpoint: str = None, user_query: str = None) -> str:
    """
    Assign a new request ID to the current context and log request start.

    Returns:
        str: The generated request ID.
    """
    request_id = str(uuid.uuid4())[:8]
    current_request_id.set(request_id)

    logger = logging.getLogger("app.request")
    logger.info("=" * 80)
    logger.info("NEW REQUEST STARTED")
    if endpoint:
        logger.info(f"Endpoint: {endpoint}")
    if user_query:
        logger.info(f"Query: {user_query[:200]}...")
    logger.info("=" * 80)

    return request_id


def end_request_logging(response_summary: str = None, success: bool = True):
    """Log request completion and clear the request ID from context."""
    request_id = current_request_id.get("unknown")
    logger = logging.getLogger("app.request")

    status = "COMPLETED" if success else "FAILED"
    logger.info("=" * 80)
    logger.info(f"REQUEST {request_id} {status}")
    if response_summary:
        logger.info(f"Response: {response_summary[:300]}...")
    logger.info("=" * 80)
    logger.info("")

    current_request_id.set(None)