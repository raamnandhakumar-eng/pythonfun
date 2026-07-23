"""Generate a reproducible synthetic worker dataset.

The data are intentionally synthetic. They provide a realistic structure for
demonstrating regression, inference, resampling, and optimization.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260722
N_WORKERS = 3000

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "sample_workers.csv"


def generate_workers(n_workers: int = N_WORKERS, seed: int = SEED) -> pd.DataFrame:
    """Return a synthetic worker-level dataset."""
    rng = np.random.default_rng(seed)

    occupations = np.array(
        ["administrative", "healthcare", "manufacturing", "retail", "technical"]
    )
    occupation = rng.choice(
        occupations,
        size=n_workers,
        p=[0.18, 0.20, 0.22, 0.22, 0.18],
    )

    frontline_prob = {
        "administrative": 0.20,
        "healthcare": 0.68,
        "manufacturing": 0.72,
        "retail": 0.82,
        "technical": 0.15,
    }
    frontline = np.array(
        [rng.binomial(1, frontline_prob[group]) for group in occupation],
        dtype=int,
    )

    education_base = {
        "administrative": 14.5,
        "healthcare": 15.5,
        "manufacturing": 13.0,
        "retail": 12.8,
        "technical": 16.5,
    }
    education_years = np.array(
        [
            np.clip(rng.normal(education_base[group], 1.8), 10, 21)
            for group in occupation
        ]
    )

    experience = np.clip(rng.gamma(shape=3.0, scale=4.5, size=n_workers), 0, 40)

    union_prob = {
        "administrative": 0.08,
        "healthcare": 0.20,
        "manufacturing": 0.30,
        "retail": 0.06,
        "technical": 0.10,
    }
    union_member = np.array(
        [rng.binomial(1, union_prob[group]) for group in occupation],
        dtype=int,
    )

    exposure_mean = {
        "administrative": 0.70,
        "healthcare": 0.46,
        "manufacturing": 0.35,
        "retail": 0.30,
        "technical": 0.76,
    }
    ai_exposure = np.array(
        [
            np.clip(
                rng.normal(exposure_mean[group] - 0.09 * is_frontline, 0.13),
                0.02,
                0.98,
            )
            for group, is_frontline in zip(occupation, frontline)
        ]
    )

    occupation_wage_effect = {
        "administrative": 0.00,
        "healthcare": 0.18,
        "manufacturing": 0.06,
        "retail": -0.16,
        "technical": 0.34,
    }

    log_wage = (
        2.35
        + 0.055 * education_years
        + 0.025 * experience
        - 0.00045 * experience**2
        + 0.08 * union_member
        + 0.20 * ai_exposure
        - 0.10 * frontline
        - 0.15 * ai_exposure * frontline
        + np.array([occupation_wage_effect[group] for group in occupation])
        + rng.normal(0, 0.20 + 0.08 * frontline, size=n_workers)
    )

    hourly_wage = np.exp(log_wage)

    workers = pd.DataFrame(
        {
            "worker_id": np.arange(1, n_workers + 1),
            "occupation_family": occupation,
            "frontline": frontline,
            "ai_exposure": ai_exposure.round(4),
            "education_years": education_years.round(2),
            "experience": experience.round(2),
            "union_member": union_member,
            "hourly_wage": hourly_wage.round(2),
        }
    )
    return workers


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    workers = generate_workers()
    workers.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(workers):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
