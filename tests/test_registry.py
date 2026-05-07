import tempfile
from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression

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


def test_save_and_load(tmp_registry):
    model = LogisticRegression()
    v1 = ModelRegistry.new_version()
    tmp_registry.save("test", model, _make_meta(v1, 0.5))
    loaded, meta = tmp_registry.load("test", v1)
    assert isinstance(loaded, LogisticRegression)
    assert meta.version == v1
    assert meta.metrics["pr_auc"] == 0.5


def test_promote_atomicity(tmp_registry):
    model = LogisticRegression()
    v1 = "v20260101_000000"
    v2 = "v20260102_000000"
    tmp_registry.save("test", model, _make_meta(v1, 0.5))
    tmp_registry.save("test", model, _make_meta(v2, 0.6))
    assert tmp_registry.champion("test") is None
    tmp_registry.promote("test", v1)
    assert tmp_registry.champion("test") == v1
    tmp_registry.promote("test", v2)
    assert tmp_registry.champion("test") == v2


def test_compare_picks_higher(tmp_registry):
    model = LogisticRegression()
    v1 = "v20260101_000000"
    v2 = "v20260102_000000"
    tmp_registry.save("test", model, _make_meta(v1, 0.5))
    tmp_registry.save("test", model, _make_meta(v2, 0.7))
    assert tmp_registry.compare("test", v1, v2, "pr_auc", higher_is_better=True) == v2
    assert tmp_registry.compare("test", v1, v2, "pr_auc", higher_is_better=False) == v1


def test_gc_keeps_recent_and_champion(tmp_registry):
    model = LogisticRegression()
    versions = [f"v202601{i:02d}_000000" for i in range(1, 11)]
    for v in versions:
        tmp_registry.save("test", model, _make_meta(v, 0.5))
    tmp_registry.promote("test", versions[0])  # champion is the oldest
    removed = tmp_registry.gc("test", keep=3)
    remaining = tmp_registry.list_versions("test")
    # Should keep last 3 + the champion (oldest), so 4 total
    assert len(remaining) == 4
    assert versions[0] in remaining
    assert versions[-1] in remaining
    assert removed == 6


def test_load_champion_when_no_version_specified(tmp_registry):
    model = LogisticRegression()
    v1 = "v20260101_000000"
    tmp_registry.save("test", model, _make_meta(v1, 0.5))
    tmp_registry.promote("test", v1)
    loaded, meta = tmp_registry.load("test")
    assert meta.version == v1


def test_load_raises_when_no_champion(tmp_registry):
    with pytest.raises(FileNotFoundError):
        tmp_registry.load("nothing")
