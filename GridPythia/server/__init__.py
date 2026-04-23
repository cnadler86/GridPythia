"""GridPythia HTTP API server package.

Usage::

    from pathlib import Path
    from GridPythia.server import create_app

    app = create_app(Path("config.yaml"))
"""

from GridPythia.server.app import create_app

__all__ = ["create_app"]
