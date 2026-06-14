"""
Q1 expenditure model.

Calibrates income elasticities via log-log regression (with age fixed effects),
computes US and UK taxes, and estimates essential expenditures + disposable income
for a given demographic profile.
"""

import numpy as np

from .data import (
    GBP_TO_USD,
    uk_income, uk_exp, uk_baselines, uk_rf,
    us_baseline, regional_factors,
)

CATS = ["Food", "Housing", "Transport", "Healthcare", "Insurance"]


# ---------------------------------------------------------------------------
# Elasticity calibration
# ---------------------------------------------------------------------------

def calibrate_elasticities(verbose: bool = False) -> dict[str, float]:
    """
    Fit log-log regression E = α·ln(S) + age_dummies on UK quintile data.
    Healthcare elasticity is set to 0.500 (estimated; insufficient cross-quintile data).
    Returns dict mapping category → income elasticity α.
    """
    age_keys = list(uk_exp.keys())
    cats_to_fit = ["Food", "Housing", "Transport", "Insurance"]
    alphas: dict[str, float] = {}

    if verbose:
        print("Log-Log Regression WITH Age Fixed Effects")
        print(f"  {'Category':<14} {'α':>7} {'R²':>7} {'n':>4}")
        print("  " + "-" * 34)

    for cat in cats_to_fit:
        lnS, lnE, aid = [], [], []
        for j, age in enumerate(age_keys):
            for i in range(5):
                v = uk_exp[age][cat][i]
                if v is not None and not np.isnan(v):
                    lnS.append(np.log(uk_income[i]))
                    lnE.append(np.log(v))
                    aid.append(j)

        n = len(lnS)
        X = np.zeros((n, 1 + len(age_keys)))
        X[:, 0] = lnS
        for k in range(n):
            X[k, 1 + aid[k]] = 1.0

        y = np.array(lnE)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        r2 = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)
        alphas[cat] = round(beta[0], 3)

        if verbose:
            print(f"  {cat:<14} {beta[0]:>7.3f} {r2:>7.3f} {n:>4}")

    alphas["Healthcare"] = 0.500
    if verbose:
        print(f"  {'Healthcare':<14} {'0.500':>7} {'est.':>7} {'--':>4}")

    return alphas


# ---------------------------------------------------------------------------
# Tax functions
# ---------------------------------------------------------------------------

def us_tax(salary: float) -> float:
    """Federal income tax + FICA + estimated 5% state tax for a single filer."""
    fica = 0.062 * min(salary, 168600) + 0.0145 * salary
    taxable = max(salary - 14600, 0)
    brackets = [
        (11600, 0.10), (35550, 0.12), (53375, 0.22),
        (91425, 0.24), (51775, 0.32), (365625, 0.35), (1e18, 0.37),
    ]
    fed, rem = 0.0, taxable
    for width, rate in brackets:
        fed += min(rem, width) * rate
        rem -= min(rem, width)
        if rem <= 0:
            break
    return round(fica + fed + 0.05 * salary, 2)


def uk_tax(salary_gbp: float) -> float:
    """UK income tax + National Insurance, returned in USD."""
    it = (
        0.20 * min(max(salary_gbp - 12570, 0), 37700)
        + 0.40 * min(max(salary_gbp - 50270, 0), 74870)
        + 0.45 * max(salary_gbp - 125140, 0)
    )
    ni = (
        0.08 * min(max(salary_gbp - 12570, 0), 37700)
        + 0.02 * max(salary_gbp - 50270, 0)
    )
    return round((it + ni) * GBP_TO_USD, 2)


# ---------------------------------------------------------------------------
# Age-group lookups
# ---------------------------------------------------------------------------

def get_us_age(age: int) -> str:
    for lo, hi, label in [
        (0, 25, "<25"), (25, 35, "25-34"), (35, 45, "35-44"),
        (45, 55, "45-54"), (55, 65, "55-64"), (65, 75, "65-74"), (75, 200, "75+"),
    ]:
        if lo <= age < hi:
            return label
    raise ValueError(f"Age {age} out of range")


def get_uk_age(age: int) -> str:
    if age < 30:
        return "<30"
    if age < 50:
        return "30-49"
    return "50-64"


# ---------------------------------------------------------------------------
# Core model functions
# ---------------------------------------------------------------------------

def estimate_expenditure(
    salary_usd: float,
    age: int,
    region: str,
    country: str,
    alphas: dict[str, float],
) -> dict[str, float]:
    """
    Predict annual essential expenditures (USD) using income-elasticity scaling.

    Parameters
    ----------
    salary_usd : annual salary in USD (already converted for UK profiles)
    age        : age in years
    region     : US region or UK nation
    country    : "US" or "UK"
    alphas     : income elasticities from calibrate_elasticities()
    """
    if country == "US":
        ag = get_us_age(age)
        b = us_baseline[ag]
        ratio = salary_usd / b["income"]
        return {
            c: b[c] * ratio ** alphas[c] * regional_factors[c].get(region, 1.0)
            for c in CATS
        }
    else:
        ag = get_uk_age(age)
        # Convert weekly £ baseline to annual USD, then scale by income ratio
        weekly_mean_gbp = np.mean([272, 544, 850, 1257, 2391])  # uk_income mean
        annual_salary_gbp = salary_usd / GBP_TO_USD
        ratio = (annual_salary_gbp / 52) / weekly_mean_gbp
        return {
            c: uk_baselines[ag][c] * 52 * ratio ** alphas[c] * uk_rf[c].get(region, 1.0) * GBP_TO_USD
            for c in CATS
        }


def disposable_income(
    salary: float,
    age: int,
    region: str,
    country: str,
    alphas: dict[str, float],
) -> dict:
    """
    Compute full income breakdown for a profile.

    Parameters
    ----------
    salary  : annual salary in native currency (£ for UK, $ for US)
    country : "US" or "UK"

    Returns dict with keys: salary, tax, after_tax, expenses, total_exp, DI
    """
    salary_usd = salary if country == "US" else salary * GBP_TO_USD
    tax = us_tax(salary) if country == "US" else uk_tax(salary)
    after_tax = salary_usd - tax
    expenses = estimate_expenditure(salary_usd, age, region, country, alphas)
    total_exp = sum(expenses.values())
    return {
        "salary": salary_usd,
        "tax": tax,
        "after_tax": after_tax,
        "expenses": expenses,
        "total_exp": total_exp,
        "DI": after_tax - total_exp,
    }
