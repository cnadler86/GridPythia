"""GridPythia web GUI entry point (thin launcher).

All application logic lives in :mod:`GridPythia.server`.  This module only
parses CLI arguments and starts uvicorn.

Usage::

    uv run python -m utils.webgui
    uv run python -m utils.webgui --config /path/to/config.yaml --port 8080
    # then open http://localhost:8080
"""

from __future__ import annotations

# ── ARM / libatomic compatibility ──────────────────────────────────────────
# On ARMv7/ARMv6 (e.g. Raspberry Pi) the HiGHS solver (via highspy / CVXPY)
# requires libatomic to be loaded into the process BEFORE highspy's shared
# library is dlopen()'d.
# Setting os.environ["LD_PRELOAD"] mid-process only affects child processes,
# NOT dlopen() in the current process.  Using ctypes.CDLL with RTLD_GLOBAL
# inserts the symbols into the process-wide dynamic-linking namespace,
# so any subsequent dlopen() call (e.g. highspy loading libhighs.so) can
# resolve the atomic symbols without crashing.
#
# Manual steps required on the target once (if the library path differs):
#   find /usr/lib -name 'libatomic*' 2>/dev/null
#   # update LIBATOMIC_PATH below if needed, or install: apt-get install libatomic1
import ctypes as _ctypes
from platform import machine as _machine

if _machine() in ("armv7l", "armv6l"):
    _LIBATOMIC = "/usr/lib/arm-linux-gnueabihf/libatomic.so.1"
    try:
        _ctypes.CDLL(_LIBATOMIC, mode=_ctypes.RTLD_GLOBAL)
        print(f"Preloaded {_LIBATOMIC} for ARM HiGHS compatibility.")
    except OSError as _e:
        print(f"Warning: could not preload {_LIBATOMIC}: {_e}")
# ─────────────────────────────────────────────────────────────────────────

import argparse
import logging
from pathlib import Path

from structlog import get_logger

logger = get_logger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8080
_DEFAULT_LOG_LEVEL = "warning"
# Module-level app reference used by uvicorn when --reload is active.
_reload_config_path: Path = _DEFAULT_CONFIG


def _load_bind_from_config(config_path: Path) -> tuple[str, int]:
    """Load bind host/port defaults from AppConfig server section."""
    from GridPythia.config import AppConfig

    cfg = AppConfig.from_yaml_file(config_path)
    return cfg.server.bind_host, cfg.server.bind_port


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
    parser.add_argument(
        "--host", default=None, help="Bind address (overrides config server.bind_host)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="TCP port (overrides config server.bind_port)",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    _reload_config_path = Path(args.config).expanduser().resolve()
    cfg_host, cfg_port = _load_bind_from_config(_reload_config_path)
    bind_host = args.host or cfg_host or _DEFAULT_HOST
    bind_port = args.port if args.port is not None else cfg_port

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
        "app_starting", config=str(_reload_config_path), url=f"http://{bind_host}:{bind_port}"
    )

    if args.reload:
        uvicorn.run(
            "main:_make_app",
            factory=True,
            host=bind_host,
            port=bind_port,
            reload=True,
            log_level=_DEFAULT_LOG_LEVEL,
        )
    else:
        from GridPythia.server import create_app

        uvicorn.run(
            create_app(_reload_config_path),
            host=bind_host,
            port=bind_port,
            log_level=_DEFAULT_LOG_LEVEL,
        )


if __name__ == "__main__":
    run()
