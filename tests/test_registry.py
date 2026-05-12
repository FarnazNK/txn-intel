"""Registry tests.

We exercise both the legacy joblib path (with a tiny dummy object) and the
pluggable saver/loader path (with a tiny torch model) to make sure the
DL-aware additions to ``ModelRegistry`` don't regress versioning, promotion,
comparison, or GC behavior.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from app.ml.registry import ModelMetadata, ModelRegistry


@pytest.fixture
def tmp_registry():
    with tempfile.TemporaryDirectory() as d:
        yield ModelRegistry(base_dir=d)


def _make_meta(version: str, pr_auc: float) -> ModelMetadata:
    return ModelMetadata(
        name="test", version=version, trained_at="2026-01-01T00:00:00",
        feature_schema=["a", "b"], metrics={"pr_auc": pr_auc, "roc_auc": 0.7},
        training_window={"start": "2025-01-01", "end": "2025-12-31"},
        n_train=100, n_eval=20,
    )


class _TinyNet(nn.Module):
    def __init__(self, in_dim: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


def _torch_saver(model: _TinyNet, vdir: Path) -> None:
    torch.save({"state_dict": model.state_dict(), "in_dim": model.fc.in_features},
               vdir / "model.pt")


def _torch_loader(vdir: Path) -> _TinyNet:
    blob = torch.load(vdir / "model.pt", map_location="cpu")
    m = _TinyNet(in_dim=blob["in_dim"])
    m.load_state_dict(blob["state_dict"])
    m.eval()
    return m


# ── Legacy joblib path ──────────────────────────────────────────────────────
def test_save_and_load_joblib(tmp_registry):
    payload = {"weights": [1.0, 2.0], "feature_order": ["a", "b"]}
    v1 = ModelRegistry.new_version()
    tmp_registry.save("test", payload, _make_meta(v1, 0.5))
    loaded, meta = tmp_registry.load("test", v1)
    assert loaded == payload
    assert meta.version == v1
    assert meta.metrics["pr_auc"] == 0.5


# ── DL-aware (saver/loader) path ────────────────────────────────────────────
def test_save_and_load_with_custom_saver(tmp_registry):
    model = _TinyNet(in_dim=4)
    v1 = ModelRegistry.new_version()
    tmp_registry.save("torch_model", model, _make_meta(v1, 0.6), saver=_torch_saver)
    loaded, meta = tmp_registry.load("torch_model", v1, loader=_torch_loader)
    assert isinstance(loaded, _TinyNet)
    # Weights survived a round trip
    assert torch.allclose(loaded.fc.weight, model.fc.weight)
    assert meta.version == v1
    # And the joblib model.joblib must NOT exist when a custom saver was used
    vdir = tmp_registry.version_dir("torch_model", v1)
    assert (vdir / "model.pt").exists()
    assert not (vdir / "model.joblib").exists()


def test_load_without_loader_when_no_joblib_raises(tmp_registry):
    model = _TinyNet()
    v1 = ModelRegistry.new_version()
    tmp_registry.save("torch_model", model, _make_meta(v1, 0.6), saver=_torch_saver)
    # No loader passed AND no model.joblib was written → must error clearly
    with pytest.raises(FileNotFoundError):
        tmp_registry.load("torch_model", v1)


# ── Promotion / comparison / GC (framework-agnostic) ────────────────────────
def test_promote_atomicity(tmp_registry):
    payload = {"k": "v"}
    v1 = "v20260101_000000"
    v2 = "v20260102_000000"
    tmp_registry.save("test", payload, _make_meta(v1, 0.5))
    tmp_registry.save("test", payload, _make_meta(v2, 0.6))
    assert tmp_registry.champion("test") is None
    tmp_registry.promote("test", v1)
    assert tmp_registry.champion("test") == v1
    tmp_registry.promote("test", v2)
    assert tmp_registry.champion("test") == v2


def test_compare_picks_higher(tmp_registry):
    payload = {"k": "v"}
    v1 = "v20260101_000000"
    v2 = "v20260102_000000"
    tmp_registry.save("test", payload, _make_meta(v1, 0.5))
    tmp_registry.save("test", payload, _make_meta(v2, 0.7))
    assert tmp_registry.compare("test", v1, v2, "pr_auc", higher_is_better=True) == v2
    assert tmp_registry.compare("test", v1, v2, "pr_auc", higher_is_better=False) == v1


def test_gc_keeps_recent_and_champion(tmp_registry):
    payload = {"k": "v"}
    versions = [f"v202601{i:02d}_000000" for i in range(1, 11)]
    for v in versions:
        tmp_registry.save("test", payload, _make_meta(v, 0.5))
    tmp_registry.promote("test", versions[0])  # champion is the oldest
    removed = tmp_registry.gc("test", keep=3)
    remaining = tmp_registry.list_versions("test")
    assert len(remaining) == 4
    assert versions[0] in remaining
    assert versions[-1] in remaining
    assert removed == 6


def test_load_champion_when_no_version_specified(tmp_registry):
    payload = {"k": "v"}
    v1 = "v20260101_000000"
    tmp_registry.save("test", payload, _make_meta(v1, 0.5))
    tmp_registry.promote("test", v1)
    loaded, meta = tmp_registry.load("test")
    assert meta.version == v1


def test_load_raises_when_no_champion(tmp_registry):
    with pytest.raises(FileNotFoundError):
        tmp_registry.load("nothing")
