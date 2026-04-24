"""Global application configuration model and YAML parsing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from GridPythia.config.optimization import OptimizationConfig
from GridPythia.config.prediction import PredictionConfig
from GridPythia.config.server import ServerConfig


class AppConfig(BaseModel):
    """Root config model mirroring the config.yaml structure."""

    model_config = {"frozen": True}

    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AppConfig":
        """Build an AppConfig from a mapping payload."""
        return cls.model_validate(payload)

    @classmethod
    def from_yaml(cls, text: str) -> "AppConfig":
        """Build an AppConfig from YAML text."""
        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Root YAML node must be a mapping")
        return cls.from_dict(loaded)

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> "AppConfig":
        """Build an AppConfig from a YAML file path."""
        cfg_path = Path(path).expanduser()
        with cfg_path.open("r", encoding="utf-8") as fh:
            return cls.from_yaml(fh.read())
