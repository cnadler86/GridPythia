"""Tests for global state helpers (optimizer lock thread safety)."""

from __future__ import annotations

import asyncio
import threading

import pytest


class TestOptimizerLockThreadSafety:
    """Verify that get_optimizer_lock() returns the same instance from multiple threads."""

    def test_concurrent_lock_creation_returns_same_instance(self):
        """Multiple threads calling get_optimizer_lock() should all get the same Lock."""
        import GridPythia.server.state as state

        # Reset to force lazy init
        state._optimizer_lock = None

        results: list[asyncio.Lock] = []
        barrier = threading.Barrier(4)

        def _get():
            barrier.wait()
            results.append(state.get_optimizer_lock())

        threads = [threading.Thread(target=_get) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(results) == 4
        # All must be the exact same Lock instance
        assert all(r is results[0] for r in results)
