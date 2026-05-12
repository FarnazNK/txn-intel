"""Model registry: versioned model artifacts with metadata.

Layout on disk:
    {model_dir}/
        churn/
            v20260506_134500/
                model.pt              # torch state_dict
                bundle.joblib         # python-side pre-processors / config
                meta.json
            CHAMPION  -> v20260506_134500
        anomaly/
            v.../
                autoencoder.keras
                head.keras
                bundle.joblib
                meta.json
        recommend/
            v.../
                two_tower.pt
                ranker.keras
                bundle.joblib
                meta.json

CHAMPION is a text file pointing to the active version. Promotion is atomic
(write to .tmp, then rename).

Each model module is responsible for implementing its own ``save_artifacts``
and ``load_artifacts`` helpers — the registry just provides versioning,
metadata, champion tracking, and GC. This keeps the registry agnostic to
which DL framework a given model uses.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import joblib

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class ModelMetadata:
    name: str
    version: str
    trained_at: str
    feature_schema: list[str]
    metrics: dict[str, float]
    training_window: dict[str, str]
    n_train: int
    n_eval: int
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = Path(base_dir or get_settings().model_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _model_dir(self, name: str) -> Path:
        return self.base_dir / name

    def version_dir(self, name: str, version: str) -> Path:
        return self._model_dir(name) / version

    @staticmethod
    def new_version() -> str:
        return "v" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def save(self, name: str, model: Any, metadata: ModelMetadata,
             saver: Callable[[Any, Path], None] | None = None) -> str:
        """Persist a model + metadata.

        ``saver``: callable(model, version_dir) responsible for writing the
        framework-specific artifacts (e.g. torch state_dict, keras model).
        If omitted, falls back to joblib.dump for backward compatibility.
        """
        version_dir = self.version_dir(name, metadata.version)
        version_dir.mkdir(parents=True, exist_ok=True)
        if saver is None:
            joblib.dump(model, version_dir / "model.joblib")
        else:
            saver(model, version_dir)
        with open(version_dir / "meta.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2)
        log.info("saved %s/%s with metrics=%s", name, metadata.version, metadata.metrics)
        return metadata.version

    def list_versions(self, name: str) -> list[str]:
        d = self._model_dir(name)
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("v"))

    def get_metadata(self, name: str, version: str) -> ModelMetadata:
        with open(self.version_dir(name, version) / "meta.json") as f:
            data = json.load(f)
        return ModelMetadata(**data)

    def load(self, name: str, version: str | None = None,
             loader: Callable[[Path], Any] | None = None) -> tuple[Any, ModelMetadata]:
        if version is None:
            version = self.champion(name)
        if version is None:
            raise FileNotFoundError(f"no model registered for {name}")
        vdir = self.version_dir(name, version)
        if loader is None:
            model_path = vdir / "model.joblib"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"no model.joblib in {vdir}; pass a loader for non-joblib artifacts"
                )
            model = joblib.load(model_path)
        else:
            model = loader(vdir)
        meta = self.get_metadata(name, version)
        return model, meta

    def champion(self, name: str) -> str | None:
        marker = self._model_dir(name) / "CHAMPION"
        if not marker.exists():
            return None
        return marker.read_text().strip()

    def promote(self, name: str, version: str) -> None:
        if not self.version_dir(name, version).exists():
            raise FileNotFoundError(f"{name}/{version} does not exist")
        marker = self._model_dir(name) / "CHAMPION"
        tmp = marker.with_suffix(".tmp")
        tmp.write_text(version)
        os.replace(tmp, marker)
        log.info("promoted %s -> %s", name, version)

    def compare(self, name: str, version_a: str, version_b: str,
                metric: str, higher_is_better: bool = True) -> str:
        a = self.get_metadata(name, version_a).metrics[metric]
        b = self.get_metadata(name, version_b).metrics[metric]
        winner = version_a if (a > b) == higher_is_better else version_b
        log.info("compare %s on %s: %s=%.4f vs %s=%.4f -> %s",
                 name, metric, version_a, a, version_b, b, winner)
        return winner

    def gc(self, name: str, keep: int = 5) -> int:
        versions = self.list_versions(name)
        champion = self.champion(name)
        keep_set = set(versions[-keep:])
        if champion:
            keep_set.add(champion)
        removed = 0
        for v in versions:
            if v not in keep_set:
                shutil.rmtree(self.version_dir(name, v))
                removed += 1
        if removed:
            log.info("gc %s: removed %d versions", name, removed)
        return removed
