"""Basic tests for the demonstration project."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze import (  # noqa: E402
    fit_statsmodels_regression,
    mean_difference,
    optimize_training_allocation,
    run_scipy_tests,
)
from generate_data import generate_workers  # noqa: E402


def sample_data() -> pd.DataFrame:
    return generate_workers(n_workers=800, seed=1234)


def test_mean_difference_direction() -> None:
    data = sample_data()
    non_frontline = data.loc[data["frontline"] == 0, "ai_exposure"].to_numpy()
    frontline = data.loc[data["frontline"] == 1, "ai_exposure"].to_numpy()
    assert mean_difference(non_frontline, frontline) > 0


def test_regression_contains_interaction() -> None:
    model = fit_statsmodels_regression(sample_data())
    assert "ai_exposure:frontline" in model.params.index
    assert np.isfinite(model.params["ai_exposure:frontline"])


def test_scipy_outputs_are_valid() -> None:
    results = run_scipy_tests(sample_data(), n_resamples=299)
    assert 0 <= results["welch_t_test"]["p_value"] <= 1
    assert 0 <= results["permutation_test"]["p_value"] <= 1
    ci = results["bootstrap_95_percent_ci"]
    assert ci["low"] < ci["high"]


def test_optimization_uses_full_budget() -> None:
    allocation = optimize_training_allocation(sample_data())
    assert np.isclose(
        allocation["allocated_training_budget"].sum(),
        500_000,
        atol=0.10,
    )
    assert (allocation["allocated_training_budget"] >= 0).all()
