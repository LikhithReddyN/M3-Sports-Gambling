"""
Q2 — Sports Gambling Outcome Analysis

Models individual bettor outcomes (participation, wagering, Monte Carlo P&L)
and validates against US industry revenue at the population level.
Generates figures 7–10.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.gambling import (
    get_participation,
    annual_wager,
    expected_annual_loss,
    simulate_year,
    estimate_population_losses,
    EDGE_BY_RISK,
    RISK_MULTIPLIERS,
    AVG_DI,
    AVG_RISK,
    PARTICIPATION,
    BASE_WAGER_FRAC,
)
from src import plots

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Individual profile outcomes
# ---------------------------------------------------------------------------
PROFILES = [
    # (label, age, gender, DI, risk, country)
    ("Young Male, Low Risk",  25, "M",  13431, 1, "US"),
    ("Young Male, Med Risk",  25, "M",  13431, 3, "US"),
    ("Young Male, High Risk", 25, "M",  13431, 5, "US"),
    ("Mid-Career Female",     42, "F",  28982, 2, "UK"),
    ("Senior, Conservative",  68, "M",   9876, 1, "US"),
    ("Young, Negative DI",    22, "M", -11620, 4, "US"),
]

print("=" * 95)
print("Q2 RESULTS: Expected Annual Gambling Gain/Loss")
print("=" * 95)
print(f"{'Profile':<28} {'DI':>8} {'Risk':>5} {'Wager':>9} {'Edge':>6} {'E[Loss]':>10} {'P(win)':>7}")
print("-" * 95)

sim_results: dict[str, dict] = {}
for label, age, gender, di, risk, country in PROFILES:
    p = get_participation(age, gender)
    w = annual_wager(di, risk)
    el = expected_annual_loss(w, risk)
    pnl = simulate_year(w, risk)
    p_profit = np.mean(pnl > 0)
    sim_results[label] = {"pnl": pnl, "wager": w, "el": el, "risk": risk, "di": di}
    print(
        f"  {label:<26} ${di:>7,.0f} {risk:>5} ${w:>8,.0f} "
        f"{EDGE_BY_RISK[risk]*100:>5.1f}% ${el:>9,.0f} {p_profit:>6.1%}"
    )

# ---------------------------------------------------------------------------
# 2. Population-level validation
# ---------------------------------------------------------------------------
pop = estimate_population_losses()
print("\n" + "=" * 95)
print("POPULATION-LEVEL VALIDATION")
print("=" * 95)
print(f"  Estimated US gamblers:      {pop['total_gamblers_M']:.1f} million")
print(f"  Estimated total US losses:  ${pop['total_losses_M']:,.0f} million")
print(f"  Actual US industry revenue: ~$15,000 million")
print(f"  Model / Actual ratio:       {pop['total_losses_M'] / 15000:.2f}")
print(f"  Average loss per gambler:   ${pop['avg_loss_per_gambler']:,.0f}/year")

# ---------------------------------------------------------------------------
# 3. Figures
# ---------------------------------------------------------------------------
di_example = 13431
risks = [1, 2, 3, 4, 5]
wagers = [annual_wager(di_example, r) for r in risks]
losses = [-expected_annual_loss(w, r) for w, r in zip(wagers, risks)]

# Simulate fig10 p_wins
p_wins = []
for r in risks:
    w = annual_wager(di_example, r)
    pnl = simulate_year(w, r)
    p_wins.append(np.mean(pnl > 0) * 100)

# Fig 9 — heatmap matrix
from src.gambling import US_POP
brackets = [(18, 25), (25, 35), (35, 50), (50, 65), (65, 99)]
matrix = np.zeros((2, 5))
for j, bracket in enumerate(brackets):
    di = AVG_DI[bracket]
    risk_avg = AVG_RISK[bracket]
    r_lo = int(risk_avg)
    frac = risk_avg - r_lo
    rm = RISK_MULTIPLIERS[r_lo] * (1 - frac) + RISK_MULTIPLIERS[min(r_lo + 1, 5)] * frac
    edge = EDGE_BY_RISK[r_lo] * (1 - frac) + EDGE_BY_RISK[min(r_lo + 1, 5)] * frac
    avg_loss = di * BASE_WAGER_FRAC * rm * edge
    for i, gender in enumerate(["M", "F"]):
        matrix[i, j] = avg_loss * PARTICIPATION[bracket][gender]

risk_display = [
    ("Young Male, Low Risk",  "#2ecc71"),
    ("Young Male, Med Risk",  "#e67e22"),
    ("Young Male, High Risk", "#e74c3c"),
]

plots.fig7_pnl_distribution(sim_results, risk_display)
plots.fig8_loss_vs_risk(di_example, wagers, losses)
plots.fig9_loss_heatmap(matrix)
plots.fig10_prob_profit(p_wins)

print("\n✓ Figures 7–10 saved to outputs/")
plt.show()
