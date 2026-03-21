"""Test configuration for HEMS2."""

import sys

import pytest
from loguru import logger


@pytest.fixture(autouse=True)
def suppress_loguru():
    """Suppress verbose loguru output during tests."""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    yield
