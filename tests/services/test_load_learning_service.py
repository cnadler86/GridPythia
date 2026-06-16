"""Tests for the LoadLearningService provider re-attach on config reload."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import GridPythia.server.state as state
from GridPythia.prediction.load.adaptive import AdaptiveLoadProvider
from GridPythia.prediction.load.config import AdaptiveLoadConfig, LoadProfileConfig
from GridPythia.server.plugins.load_learning.service import LoadLearningService


def _write_csv(path: Path) -> None:
    rows = ["time,weekday,weekend"]
    for h in range(24):
        rows.append(f"{h:02d}:00,200.0,100.0")
    path.write_text("\n".join(rows), encoding="utf-8")


def _make_provider(tmp_path: Path, name: str) -> AdaptiveLoadProvider:
    csv = tmp_path / f"{name}.csv"
    _write_csv(csv)
    adaptive_cfg = AdaptiveLoadConfig(enabled=True, db_path=str(tmp_path / f"{name}.sqlite"))
    cfg = LoadProfileConfig(path=csv, adaptive=adaptive_cfg)
    return AdaptiveLoadProvider(cfg, adaptive_cfg)


def _make_service(adaptive_cfg: AdaptiveLoadConfig) -> LoadLearningService:
    # Lightweight stand-in for AppConfig: only the attributes the service reads.
    fake_config = SimpleNamespace(
        prediction=SimpleNamespace(load=SimpleNamespace(adaptive=adaptive_cfg)),
        server=SimpleNamespace(mqtt=SimpleNamespace(enabled=False)),
    )
    return LoadLearningService(fake_config)  # type: ignore[arg-type]


@pytest.fixture
def _clean_state(monkeypatch):
    monkeypatch.setattr(state, "providers", None, raising=False)
    yield


def test_live_provider_resolves_from_state(tmp_path, _clean_state):
    prov = _make_provider(tmp_path, "p1")
    state.providers = SimpleNamespace(load=prov)
    service = _make_service(prov._adaptive_cfg)

    assert service.provider is prov


def test_reattach_on_provider_rebuild_preserves_runtime_state(tmp_path, _clean_state):
    prov1 = _make_provider(tmp_path, "p1")
    state.providers = SimpleNamespace(load=prov1)
    service = _make_service(prov1._adaptive_cfg)

    # Apply runtime-only state through the service.
    service.vacation_mode = True
    service.update_active_forecast_appliances({"dishwasher"})
    assert service.provider is prov1
    assert prov1.vacation_mode is True

    # Simulate a config reload: state.providers is rebuilt with a fresh provider.
    prov2 = _make_provider(tmp_path, "p2")
    state.providers = SimpleNamespace(load=prov2)

    # Next access reattaches and re-applies the runtime state onto prov2.
    assert service.provider is prov2
    assert prov2.vacation_mode is True
    assert prov2._active_forecast_appliances == {"dishwasher"}


def test_reattach_flushes_old_provider_pending_buckets(tmp_path, _clean_state):
    prov1 = _make_provider(tmp_path, "p1")
    state.providers = SimpleNamespace(load=prov1)
    service = _make_service(prov1._adaptive_cfg)

    # Ingest two past-dated samples into distinct closed buckets.  At least the
    # second stays buffered in RAM (the flush interval has not elapsed yet).
    service.ingest_power(500.0, ts=1_000_000)
    service.ingest_power(600.0, ts=1_002_000)
    assert prov1.get_stats()["pending_accumulator_buckets"] > 0

    # Rebuild provider → reattach must flush prov1 so the sample is not lost.
    prov2 = _make_provider(tmp_path, "p2")
    state.providers = SimpleNamespace(load=prov2)
    _ = service.provider  # triggers reattach

    assert prov1.get_stats()["pending_accumulator_buckets"] == 0
    # Both providers share nothing here (separate db files); the point is the
    # old provider was drained rather than silently abandoned.
    assert prov1.db.count("load_w") >= 1
