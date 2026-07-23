# Statsmodels vs. SciPy: AI Exposure and Worker Outcomes

A small, reproducible economics project showing how **Statsmodels** and **SciPy**
serve different roles in a serious Python analysis.

## Core question

Do frontline and non-frontline workers differ in simulated AI exposure, and does
the association between AI exposure and wages vary between those groups?

The included dataset is **synthetic**. It is designed for demonstration,
portfolio use, and code review. Results must not be interpreted as evidence
about the real labor market.

## What each library does

| Library | Used here for |
|---|---|
| Statsmodels | OLS regression, interaction terms, occupation fixed effects, HC3 robust standard errors, coefficient inference |
| SciPy | Welch's t-test, permutation test, bootstrap confidence interval, constrained optimization |
| pandas | Data preparation and result tables |
| matplotlib | Publication-style output figures |

## Model

The regression estimates:

```text
log(hourly_wage)
  = AI exposure
  + frontline status
  + AI exposure × frontline status
  + education
  + experience
  + experience²
  + union membership
  + occupation-family fixed effects
```

The model uses **HC3 heteroskedasticity-robust standard errors**.

## Repository structure

```text
.
├── data/                         # generated locally
│   └── sample_workers.csv
├── outputs/                      # generated locally
│   ├── allocation_by_occupation.csv
│   ├── coefficient_table.csv
│   ├── descriptive_statistics.csv
│   ├── model_summary.txt
│   ├── predicted_wages.png
│   ├── scipy_tests.json
│   └── training_allocation.png
├── src/
│   ├── analyze.py
│   └── generate_data.py
├── tests/
│   └── test_analysis.py
├── .github/workflows/tests.yml
├── requirements.txt
└── README.md
```

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/generate_data.py
python src/analyze.py
pytest
```

Windows activation:

```powershell
.venv\Scripts\activate
```

## Interpretation guide

- The **Statsmodels interaction coefficient** tests whether the wage-exposure
  slope differs for frontline workers.
- The **SciPy Welch test** compares average simulated exposure between groups.
- The **permutation test** evaluates the exposure gap with fewer distributional
  assumptions.
- The **bootstrap interval** quantifies uncertainty around the mean exposure gap.
- The **optimization model** allocates a fixed training budget under bounds and
  diminishing returns. It is illustrative, not a causal policy recommendation.

## Why this is stronger than a basic notebook

- Clear separation between data generation and analysis
- Reproducible random seeds
- Robust standard errors
- Parametric and nonparametric inference
- Constrained optimization
- Saved machine-readable outputs
- Basic automated tests
- Explicit limitation against causal interpretation

## Suggested extension with real data

Replace `data/sample_workers.csv` with a worker-level or occupation-level dataset
containing:

- hourly wage or annual earnings
- AI exposure measure
- frontline indicator
- education
- experience or age
- union status
- occupation or industry

Then update column names in `src/analyze.py`. For survey data, add sampling
weights and use a design-aware method where appropriate.

## Verification

The included automated test suite passes with **4 tests** covering the simulated exposure gap, regression interaction term, SciPy inference outputs, and constrained-budget feasibility. GitHub Actions runs the same tests on every push and pull request.
