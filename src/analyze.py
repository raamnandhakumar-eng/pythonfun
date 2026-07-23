"""Analyze the synthetic AI-exposure worker dataset.

Statsmodels handles the econometric model and coefficient inference.
SciPy handles standalone tests, resampling, and constrained optimization.
"""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats
import statsmodels.formula.api as smf

SEED = 20260722
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_workers.csv"
OUTPUT_DIR = ROOT / "outputs"


def mean_difference(non_frontline: np.ndarray, frontline: np.ndarray) -> float:
    """Mean exposure of non-frontline minus frontline workers."""
    return float(np.mean(non_frontline) - np.mean(frontline))


def run_scipy_tests(data: pd.DataFrame, n_resamples: int = 9_999) -> dict:
    """Run parametric, permutation, and bootstrap comparisons."""
    non_frontline = data.loc[data["frontline"] == 0, "ai_exposure"].to_numpy()
    frontline = data.loc[data["frontline"] == 1, "ai_exposure"].to_numpy()

    welch = stats.ttest_ind(
        non_frontline,
        frontline,
        equal_var=False,
        alternative="two-sided",
    )

    permutation = stats.permutation_test(
        (non_frontline, frontline),
        mean_difference,
        permutation_type="independent",
        vectorized=False,
        n_resamples=n_resamples,
        alternative="two-sided",
        rng=np.random.default_rng(SEED),
    )

    bootstrap = stats.bootstrap(
        (non_frontline, frontline),
        mean_difference,
        vectorized=False,
        paired=False,
        n_resamples=n_resamples,
        confidence_level=0.95,
        method="BCa",
        rng=np.random.default_rng(SEED + 1),
    )

    return {
        "group_sizes": {
            "non_frontline": int(non_frontline.size),
            "frontline": int(frontline.size),
        },
        "mean_ai_exposure": {
            "non_frontline": float(non_frontline.mean()),
            "frontline": float(frontline.mean()),
        },
        "mean_difference_nonfrontline_minus_frontline": mean_difference(
            non_frontline, frontline
        ),
        "welch_t_test": {
            "statistic": float(welch.statistic),
            "p_value": float(welch.pvalue),
        },
        "permutation_test": {
            "statistic": float(permutation.statistic),
            "p_value": float(permutation.pvalue),
            "resamples": n_resamples,
        },
        "bootstrap_95_percent_ci": {
            "low": float(bootstrap.confidence_interval.low),
            "high": float(bootstrap.confidence_interval.high),
            "method": "BCa",
            "resamples": n_resamples,
        },
    }


def fit_statsmodels_regression(data: pd.DataFrame):
    """Fit OLS with occupation fixed effects and HC3 robust covariance."""
    formula = (
        "np.log(hourly_wage) ~ ai_exposure * frontline "
        "+ education_years + experience + I(experience ** 2) "
        "+ union_member + C(occupation_family)"
    )
    return smf.ols(formula=formula, data=data).fit(cov_type="HC3")


def coefficient_table(model) -> pd.DataFrame:
    """Create a compact coefficient table."""
    interval = model.conf_int()
    table = pd.DataFrame(
        {
            "term": model.params.index,
            "coefficient": model.params.values,
            "robust_standard_error": model.bse.values,
            "p_value": model.pvalues.values,
            "ci_95_low": interval.iloc[:, 0].values,
            "ci_95_high": interval.iloc[:, 1].values,
        }
    )
    return table


def build_prediction_grid(data: pd.DataFrame) -> pd.DataFrame:
    """Create a controlled prediction grid for the wage figure."""
    exposure_grid = np.linspace(0.05, 0.95, 100)
    common = {
        "education_years": float(data["education_years"].median()),
        "experience": float(data["experience"].median()),
        "union_member": 0,
        "occupation_family": "administrative",
    }

    rows = []
    for frontline in [0, 1]:
        for exposure in exposure_grid:
            rows.append(
                {
                    **common,
                    "frontline": frontline,
                    "ai_exposure": exposure,
                }
            )
    return pd.DataFrame(rows)


def save_prediction_figure(model, data: pd.DataFrame) -> None:
    """Save predicted hourly wage by AI exposure and frontline status."""
    grid = build_prediction_grid(data)
    prediction = model.get_prediction(grid).summary_frame(alpha=0.05)
    grid["predicted_hourly_wage"] = np.exp(prediction["mean"].to_numpy())
    grid["ci_low"] = np.exp(prediction["mean_ci_lower"].to_numpy())
    grid["ci_high"] = np.exp(prediction["mean_ci_upper"].to_numpy())

    fig, ax = plt.subplots(figsize=(8, 5))
    for status, label in [(0, "Non-frontline"), (1, "Frontline")]:
        subset = grid.loc[grid["frontline"] == status]
        ax.plot(
            subset["ai_exposure"],
            subset["predicted_hourly_wage"],
            label=label,
        )
        ax.fill_between(
            subset["ai_exposure"],
            subset["ci_low"],
            subset["ci_high"],
            alpha=0.15,
        )

    ax.set_title("Predicted hourly wage by simulated AI exposure")
    ax.set_xlabel("AI exposure index")
    ax.set_ylabel("Predicted hourly wage")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "predicted_wages.png", dpi=200)
    plt.close(fig)


def optimize_training_allocation(data: pd.DataFrame) -> pd.DataFrame:
    """Allocate a fixed illustrative training budget across occupations.

    The objective uses diminishing returns. It is a numerical demonstration,
    not a causal policy model.
    """
    summary = (
        data.groupby("occupation_family", as_index=False)
        .agg(
            workers=("worker_id", "size"),
            mean_ai_exposure=("ai_exposure", "mean"),
            frontline_share=("frontline", "mean"),
        )
        .sort_values("occupation_family")
        .reset_index(drop=True)
    )

    budget = 500_000.0
    maximum_per_group = 180_000.0

    priority = (
        summary["workers"].to_numpy()
        * (1.05 - summary["mean_ai_exposure"].to_numpy())
        * (0.75 + summary["frontline_share"].to_numpy())
    )
    scale = 70_000.0

    def objective(allocation: np.ndarray) -> float:
        benefit = np.sum(priority * np.log1p(allocation / scale))
        return -float(benefit)

    constraints = {
        "type": "eq",
        "fun": lambda allocation: float(np.sum(allocation) - budget),
    }
    bounds = [(0.0, maximum_per_group)] * len(summary)
    initial = np.full(len(summary), budget / len(summary))

    result = optimize.minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1_000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    summary["priority_score"] = priority
    summary["allocated_training_budget"] = result.x
    summary["allocated_training_budget"] = summary[
        "allocated_training_budget"
    ].round(2)
    summary["budget_share"] = (
        summary["allocated_training_budget"] / budget
    ).round(4)
    return summary


def save_allocation_figure(allocation: pd.DataFrame) -> None:
    """Save the optimized illustrative allocation."""
    plot_data = allocation.sort_values("allocated_training_budget")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        plot_data["occupation_family"],
        plot_data["allocated_training_budget"],
    )
    ax.set_title("Illustrative optimized AI-training allocation")
    ax.set_xlabel("Allocated budget")
    ax.set_ylabel("Occupation family")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_allocation.png", dpi=200)
    plt.close(fig)


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Return group-level descriptive statistics."""
    return (
        data.groupby("frontline")
        .agg(
            workers=("worker_id", "size"),
            mean_ai_exposure=("ai_exposure", "mean"),
            median_ai_exposure=("ai_exposure", "median"),
            mean_hourly_wage=("hourly_wage", "mean"),
            median_hourly_wage=("hourly_wage", "median"),
            mean_education_years=("education_years", "mean"),
            mean_experience=("experience", "mean"),
        )
        .rename(index={0: "non_frontline", 1: "frontline"})
        .reset_index(names="group")
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} does not exist. Run src/generate_data.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)

    descriptive_statistics(data).to_csv(
        OUTPUT_DIR / "descriptive_statistics.csv",
        index=False,
    )

    scipy_results = run_scipy_tests(data)
    (OUTPUT_DIR / "scipy_tests.json").write_text(
        json.dumps(scipy_results, indent=2),
        encoding="utf-8",
    )

    model = fit_statsmodels_regression(data)
    coefficient_table(model).to_csv(
        OUTPUT_DIR / "coefficient_table.csv",
        index=False,
    )
    (OUTPUT_DIR / "model_summary.txt").write_text(
        model.summary().as_text(),
        encoding="utf-8",
    )

    save_prediction_figure(model, data)

    allocation = optimize_training_allocation(data)
    allocation.to_csv(
        OUTPUT_DIR / "allocation_by_occupation.csv",
        index=False,
    )
    save_allocation_figure(allocation)

    interaction = "ai_exposure:frontline"
    print("Analysis complete.")
    print(
        "AI exposure × frontline coefficient: "
        f"{model.params[interaction]:.4f} "
        f"(robust p={model.pvalues[interaction]:.4g})"
    )
    print(
        "SciPy exposure gap: "
        f"{scipy_results['mean_difference_nonfrontline_minus_frontline']:.4f}"
    )
    print(f"Outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
