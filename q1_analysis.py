"""
Q1 — Expenditure & Disposable Income Analysis

Calibrates income elasticities via log-log regression, computes disposable income
for six demographic profiles, runs a sensitivity analysis, and generates figures 1–6.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.model import calibrate_elasticities, disposable_income, us_tax, uk_tax
from src.data import us_baseline, regional_factors, GBP_TO_USD
from src import plots

# ---------------------------------------------------------------------------
# 1. Calibrate elasticities
# ---------------------------------------------------------------------------
alphas = calibrate_elasticities(verbose=True)

CATS = ["Food", "Housing", "Transport", "Healthcare", "Insurance"]

# ---------------------------------------------------------------------------
# 2. Run profiles
# ---------------------------------------------------------------------------
# (label, age, salary_native, region, country)
PROFILES = [
    ("A", 27, 65000,  "Northeast", "US"),
    ("B", 42, 110000, "Midwest",   "US"),
    ("C", 60, 85000,  "South",     "US"),
    ("D", 22, 35000,  "West",      "US"),
    ("E", 28, 38000,  "England",   "UK"),
    ("F", 45, 55000,  "Wales",     "UK"),
]

results: dict[str, dict] = {}

print("\n" + "=" * 80)
print("RESULTS (all USD)")
print("=" * 80)
print(f"{'ID':<4} {'Salary':>10} {'Tax':>10} {'After-Tax':>10} {'Essentials':>10} {'DI':>12}")
print("-" * 80)

for nm, age, sal, reg, ctry in PROFILES:
    r = disposable_income(sal, age, reg, ctry, alphas)
    results[nm] = r
    print(
        f" {nm:<3} ${r['salary']:>9,.0f} ${r['tax']:>9,.0f} ${r['after_tax']:>9,.0f} "
        f"${r['total_exp']:>9,.0f}  ${r['DI']:>10,.0f}"
    )

# ---------------------------------------------------------------------------
# 3. Worked example — Profile A
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("WORKED EXAMPLE: Profile A (Age 27, $65K, Northeast)")
print("=" * 80)
r = results["A"]
ag = "25-34"
ratio = 65000 / us_baseline[ag]["income"]
print(f"  Tax = ${r['tax']:,.0f}  |  After-tax = ${r['after_tax']:,.0f}")
print(f"  Age group: {ag}, mean income = ${us_baseline[ag]['income']:,}, ratio = {ratio:.3f}")
for c in CATS:
    b = us_baseline[ag][c]
    a = alphas[c]
    rf = regional_factors[c]["Northeast"]
    print(f"  E_{c:<12} = {b:>6,} × {ratio:.3f}^{a:.3f} × {rf:.3f} = ${r['expenses'][c]:>8,.0f}")
print(f"  Total = ${r['total_exp']:,.0f}  →  DI = ${r['DI']:,.0f}")

# ---------------------------------------------------------------------------
# 4. Sensitivity — Profile A, α ± 0.10
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SENSITIVITY: Profile A, α ± 0.10")
print("=" * 80)
print(f"  {'Parameter':<20} {'α−0.10':>10} {'Baseline':>10} {'α+0.10':>10}")

bdi = results["A"]["DI"]
sens_lo, sens_hi = [], []
sens_cats = ["Food", "Housing", "Transport"]

for cat in sens_cats:
    orig = alphas[cat]
    alphas[cat] = orig - 0.10
    lo = disposable_income(65000, 27, "Northeast", "US", alphas)["DI"]
    alphas[cat] = orig + 0.10
    hi = disposable_income(65000, 27, "Northeast", "US", alphas)["DI"]
    alphas[cat] = orig
    sens_lo.append(lo)
    sens_hi.append(hi)
    print(f"  α_{cat:<17} ${lo:>9,.0f} ${bdi:>9,.0f} ${hi:>9,.0f}")

# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------
labels = [
    "A\n27, NE, US", "B\n42, MW, US", "C\n60, S, US",
    "D\n22, W, US",  "E\n28, Eng, UK", "F\n45, Wales, UK",
]
short = ["A", "B", "C", "D", "E", "F"]

salary   = [results[k]["salary"]    for k in ["A", "B", "C", "D", "E", "F"]]
tax      = [results[k]["tax"]       for k in ["A", "B", "C", "D", "E", "F"]]
di       = [results[k]["DI"]        for k in ["A", "B", "C", "D", "E", "F"]]
exp      = {c: [results[k]["expenses"][c] for k in ["A", "B", "C", "D", "E", "F"]] for c in CATS}

alpha_cats = ["Food", "Housing", "Transport", "Healthcare", "Insurance"]
alpha_vals = [alphas[c] for c in alpha_cats]
r2_old = [0.655, 0.153, 0.891, None, 0.889]
r2_new = [0.941, 0.865, 0.969, None, 0.967]

plots.fig1_disposable_income(labels, di)
plots.fig2_expenditure_breakdown(labels, short, exp)
plots.fig3_income_waterfall(PROFILES, salary, tax, exp, di)
plots.fig4_r2_comparison(alpha_cats, r2_old, r2_new)
plots.fig5_elasticities(alpha_cats, alpha_vals)
plots.fig6_sensitivity_tornado(sens_cats, sens_lo, sens_hi, bdi)

print("\n✓ Figures 1–6 saved to outputs/")
plt.show()
