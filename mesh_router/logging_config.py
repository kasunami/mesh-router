import logging
import os
import sys

try:
    from pythonjsonlogger import jsonlogger

    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False


def setup_logging(level: int = logging.INFO, force_json: bool = False, service_name: str | None = None):
    """Configure plain-text or JSON logging for mesh-router."""
    enable_json = force_json or (
        os.getenv("MESH_LOG_JSON", "false").lower() == "true"
        or os.getenv("LOG_JSON", "false").lower() == "true"
    )

    if enable_json and not JSON_LOGGER_AVAILABLE:
        print(
            "WARNING: JSON logging requested but python-json-logger is not installed; using text logs.",
            file=sys.stderr,
        )
        enable_json = False

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if enable_json:
        fmt = "%(asctime)s %(levelname)s %(name)s %(message)s %(pathname)s %(funcName)s %(lineno)d"
        formatter = jsonlogger.JsonFormatter(fmt, rename_fields={"name": "logger"}) if service_name else jsonlogger.JsonFormatter(fmt)
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    return root_logger
