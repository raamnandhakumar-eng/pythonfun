# Statsmodels vs. SciPy

## A small labor-economics experiment in Python

I built this project because I wanted to understand where **Statsmodels** ends and **SciPy** begins in a real analysis.

The question I used was simple:

> **Do frontline workers have lower simulated exposure to AI, and does the relationship between AI exposure and wages differ for them?**

Instead of writing disconnected examples for each library, I created one reproducible workflow with a worker-level dataset, statistical tests, an econometric model, uncertainty estimates, visualizations, and a constrained policy-allocation exercise.

The data are synthetic. The results show that the code and statistical workflow work as intended. They are not claims about the real labor market.

---

## Results at a glance

The simulated dataset contains **3,000 workers** across administrative, healthcare, manufacturing, retail, and technical occupations.

| Result | Non-frontline | Frontline |
|---|---:|---:|
| Workers | 1,376 | 1,624 |
| Mean AI-exposure index | 0.610 | 0.318 |
| Median AI-exposure index | 0.638 | 0.294 |
| Mean hourly wage | $41.13 | $29.02 |
| Mean education | 14.96 years | 13.90 years |

### 1. The simulated exposure gap is large

Non-frontline workers have an average AI-exposure score **0.292 points higher** than frontline workers.

- Welch t-statistic: **39.83**
- Welch p-value: **< 0.001**
- Permutation-test p-value: **0.0002**
- BCa bootstrap 95% confidence interval: **0.277 to 0.306**

The same pattern survives a parametric test, a permutation test, and bootstrap resampling. That agreement is useful because each method relies on different assumptions.

### 2. The wage-exposure relationship differs by worker type

The Statsmodels regression estimates the interaction between AI exposure and frontline status at:

> **AI exposure × frontline coefficient: -0.198**  
> **HC3 robust p-value: < 0.001**

In this simulated model, the positive association between AI exposure and wages is substantially weaker for frontline workers. The regression controls for education, experience, experience squared, union membership, and occupation-family fixed effects.

The model explains about **63.4%** of the variation in log hourly wages within the synthetic sample.

This is not a causal estimate. It is a check that the econometric specification can recover a deliberately structured difference in the generated data.

### 3. The optimization exercise prioritizes lower-exposure frontline groups

I also used SciPy's constrained optimizer to allocate a hypothetical **$500,000 AI-training budget** across occupation families.

Under the illustrative priority rule:

| Occupation | Allocation | Budget share |
|---|---:|---:|
| Manufacturing | $180,000 | 36.0% |
| Retail | $180,000 | 36.0% |
| Healthcare | $129,488 | 25.9% |
| Administrative | $10,512 | 2.1% |
| Technical | $0 | 0.0% |

The result directs most of the budget toward occupations with larger worker populations, lower simulated AI exposure, and higher frontline shares.

This is a numerical demonstration, not a real policy recommendation. The objective function and constraints determine the result.

---

## What Statsmodels and SciPy each contribute

This project made the distinction clearer for me.

### Statsmodels

I use Statsmodels when the main task is an econometric model and I need:

- interpretable regression coefficients;
- interaction terms;
- occupation fixed effects;
- HC3 heteroskedasticity-robust standard errors;
- confidence intervals and coefficient-level inference;
- controlled wage predictions.

### SciPy

I use SciPy when I need focused statistical or numerical tools:

- Welch's unequal-variance t-test;
- permutation testing;
- BCa bootstrap confidence intervals;
- constrained nonlinear optimization.

They overlap in places, but they are not substitutes. Statsmodels handles the regression framework. SciPy handles standalone inference, resampling, and optimization.

---

## Econometric specification

The main model is:

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

The interaction coefficient is the main quantity of interest. It tests whether the slope linking AI exposure to wages differs between frontline and non-frontline workers.

---

## Why I built it this way

A notebook would have been faster, but I wanted the project to resemble a small research workflow rather than a one-time demonstration.

The repository therefore separates:

1. data generation;
2. statistical analysis;
3. saved tables and figures;
4. automated verification.

It also uses a fixed random seed, machine-readable outputs, robust standard errors, multiple inferential methods, and explicit warnings against causal interpretation.

---

## Repository structure

```text
.
├── data/
│   └── sample_workers.csv
├── outputs/
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

---

## Run the project

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/generate_data.py
python src/analyze.py
pytest
```

On Windows:

```powershell
.venv\Scripts\activate
```

The analysis writes the regression output, statistical-test results, descriptive tables, optimization results, and figures to `outputs/`.

---

## Verification

The automated test suite checks that:

- the simulated frontline exposure gap is generated correctly;
- the regression contains the intended interaction term;
- the SciPy procedures return valid inference outputs;
- the optimized allocation respects the total budget and group constraints.

GitHub Actions runs the same tests after every push and pull request.

---

## Limitations

The data-generating process deliberately creates differences in exposure and wages. The project therefore demonstrates recovery and inference, not discovery.

A real study would need:

- validated worker-level or occupation-level data;
- a defensible measure of AI exposure;
- survey weights where applicable;
- careful treatment of selection and omitted-variable bias;
- sensitivity analyses and alternative specifications;
- a credible identification strategy before making causal claims.

The optimization model would also need real estimates of training costs, uptake, learning returns, worker benefits, and employer constraints.

---

## Next step

The natural extension is to replace the synthetic file with public labor-market data and test whether the same exposure gap appears empirically. A stronger version could combine occupation-level AI-exposure measures with wage and employment data from sources such as O*NET, the Current Population Survey, or the American Community Survey.

For now, this repository is a compact example of how I use Python to connect statistical programming, econometric reasoning, and an economic policy question.