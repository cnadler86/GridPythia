"""GridPythia web GUI entry point (thin launcher).

All application logic lives in :mod:`GridPythia.server`.  This module only
parses CLI arguments and starts uvicorn.

Usage::

    uv run python -m utils.webgui
    uv run python -m utils.webgui --config /path/to/config.yaml --port 8080
    # then open http://localhost:8080
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from structlog import get_logger

logger = get_logger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"
# Module-level app reference used by uvicorn when --reload is active.
_reload_config_path: Path = _DEFAULT_CONFIG


def _make_app():  # -> FastAPI (imported lazily to keep this module light)
    """App factory called by uvicorn in reload mode (``factory=True``)."""
    from GridPythia.server import create_app

    return create_app(_reload_config_path)


def run() -> None:
    """Parse CLI arguments and start the uvicorn server."""
    global _reload_config_path  # noqa: PLW0603

    parser = argparse.ArgumentParser(description="GridPythia web GUI")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="Path to config.yaml (default: %(default)s)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="TCP port (default: 8080)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    _reload_config_path = Path(args.config).expanduser().resolve()

    import structlog

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    import uvicorn

    logger.info(
        "webgui_starting", config=str(_reload_config_path), url=f"http://localhost:{args.port}"
    )

    if args.reload:
        uvicorn.run(
            "utils.webgui:_make_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
            log_level="warning",
        )
    else:
        from GridPythia.server import create_app

        uvicorn.run(
            create_app(_reload_config_path),
            host=args.host,
            port=args.port,
            log_level="warning",
        )


if __name__ == "__main__":
    run()
