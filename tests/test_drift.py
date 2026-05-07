import numpy as np

from app.monitoring.drift import psi


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    y = rng.normal(0, 1, 5000)
    score = psi(x, y)
    # Should be small for two samples from the same distribution
    assert score < 0.05, f"PSI too large for identical dists: {score}"


def test_psi_large_for_shifted_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    y = rng.normal(2, 1, 5000)
    score = psi(x, y)
    assert score > 0.5, f"PSI too small for shifted dists: {score}"


def test_psi_handles_empty():
    assert psi(np.array([]), np.array([1, 2, 3])) == 0.0
    assert psi(np.array([1, 2, 3]), np.array([])) == 0.0


def test_psi_handles_constant():
    # Reference is constant — only one bin edge effectively
    score = psi(np.zeros(100), np.zeros(100))
    assert score == 0.0
