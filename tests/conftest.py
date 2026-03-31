"""Test configuration for HEMS2."""

import sys

import pytest
import structlog


@pytest.fixture(autouse=True)
def suppress_loguru():
    """Suppress verbose loguru output during tests."""
    structlog.configure(
        processors=[
            structlog.processors.KeyValueRenderer(key_order=["event", "logger", "level", "timestamp"]),
        ]
    )   
    yield
