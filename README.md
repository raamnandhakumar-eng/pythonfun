# Statsmodels vs. SciPy

A small labor-economics project I built to understand when I would use **Statsmodels** and when I would use **SciPy** in the same analysis.

Rather than making two unrelated examples, I used one question throughout the project:

> Do frontline workers have lower simulated exposure to AI, and is the relationship between AI exposure and wages different for them?

The project generates a synthetic worker dataset, compares the two groups, runs an econometric model, measures uncertainty in several ways, and finishes with a simple training-budget allocation exercise.

The data are entirely simulated. The results show that the code can recover patterns deliberately built into the data; they are not findings about the real labor market.

## What the simulated run showed

The generated dataset contains **3,000 workers** across administrative, healthcare, manufacturing, retail, and technical occupations.

| Measure | Non-frontline | Frontline |
|---|---:|---:|
| Workers | 1,376 | 1,624 |
| Mean AI-exposure score | 0.610 | 0.318 |
| Median AI-exposure score | 0.638 | 0.294 |
| Mean hourly wage | $41.13 | $29.02 |
| Mean education | 14.96 years | 13.90 years |

### Frontline workers had lower simulated AI exposure

The average exposure score for non-frontline workers was **0.292 points higher** than the average for frontline workers.

I checked that gap three different ways:

- Welch t-statistic: **39.83**
- Welch p-value: **below 0.001**
- permutation-test p-value: **0.0002**
- BCa bootstrap 95% confidence interval: **0.277 to 0.306**

The methods use different assumptions, but they all point in the same direction. That is useful here because the gap was intentionally added when the data were generated.

### The wage relationship was weaker for frontline workers

The Statsmodels regression includes AI exposure, frontline status, their interaction, education, experience, experience squared, union membership, and occupation-family fixed effects.

The estimated interaction was:

> **AI exposure × frontline: -0.198**  
> **HC3 robust p-value: below 0.001**

In the simulated data, higher AI exposure is associated with higher wages, but that relationship is weaker for frontline workers. The model explains about **63.4%** of the variation in log hourly wages.

This is not a causal result. It is a check that the regression can recover a difference that was deliberately built into the dataset.

### The budget exercise favored lower-exposure frontline occupations

I used SciPy's constrained optimizer to divide a hypothetical **$500,000 AI-training budget** across occupation families.

| Occupation | Allocation | Share |
|---|---:|---:|
| Manufacturing | $180,000 | 36.0% |
| Retail | $180,000 | 36.0% |
| Healthcare | $129,488 | 25.9% |
| Administrative | $10,512 | 2.1% |
| Technical | $0 | 0.0% |

The objective gives more weight to occupations with more workers, lower simulated AI exposure, and a larger frontline share. The result is only an illustration of constrained optimization; it is not a policy recommendation.

## What I used each library for

### Statsmodels

Statsmodels handles the econometric part of the project:

- an OLS wage model;
- an AI-exposure and frontline interaction;
- occupation-family fixed effects;
- HC3 heteroskedasticity-robust standard errors;
- confidence intervals and coefficient-level inference;
- controlled wage predictions.

### SciPy

SciPy handles the standalone statistical and numerical work:

- Welch's unequal-variance t-test;
- a permutation test;
- a BCa bootstrap confidence interval;
- constrained nonlinear optimization.

The main lesson for me was that the libraries overlap, but they solve different parts of the problem. Statsmodels provides the regression framework and interpretable model output. SciPy is useful for focused tests, resampling, and optimization.

## Regression used in the project

```text
log(hourly wage)
  = AI exposure
  + frontline status
  + AI exposure × frontline status
  + education
  + experience
  + experience²
  + union membership
  + occupation-family fixed effects
```

The interaction term is the main quantity of interest because it shows whether the wage-exposure relationship changes for frontline workers.

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/generate_data.py
python src/analyze.py
pytest -q
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

Running the analysis creates the synthetic dataset and writes the statistical results, regression output, allocation table, and figures to `data/` and `outputs/`.

Those generated files are intentionally ignored by Git because they can be recreated from the source code and fixed random seed.

## Repository layout

```text
src/generate_data.py       creates the synthetic worker data
src/analyze.py             runs the tests, regression, figures, and optimization
tests/test_analysis.py     checks the main calculations and constraints
.github/workflows/tests.yml runs the test suite on GitHub Actions
requirements.txt           Python dependencies
README.md                  project explanation and reported simulated results
```

## What the tests check

The automated tests confirm that:

- the generated frontline exposure gap has the intended direction;
- the regression includes the exposure-by-frontline interaction;
- the SciPy tests return valid p-values and confidence intervals;
- the optimizer uses the full $500,000 budget without creating negative allocations.

## Limits of the project

This project demonstrates a statistical workflow, not an empirical labor-market study. A real version would need validated labor data, a defensible measure of AI exposure, survey weights where appropriate, sensitivity checks, and a credible identification strategy before making causal claims.

The optimization exercise would also need real estimates of training costs, participation, learning outcomes, worker benefits, and employer constraints.

For now, the repository is a compact example of how I use Python to connect statistics, econometrics, and an economic-policy question.