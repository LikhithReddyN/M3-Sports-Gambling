"""
Q2 sports gambling model.

Models participation rates, annual wagering amounts, house edge by bet type,
and Monte Carlo simulation of annual P&L outcomes.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Participation rates — P(has active sports-betting account) by age × gender
# Source: 22% US overall; ~50% of men 18-49; scaled down for other groups
# ---------------------------------------------------------------------------

PARTICIPATION: dict[tuple, dict] = {
    (18, 25): {"M": 0.45, "F": 0.15},
    (25, 35): {"M": 0.52, "F": 0.18},
    (35, 50): {"M": 0.48, "F": 0.14},
    (50, 65): {"M": 0.25, "F": 0.08},
    (65, 99): {"M": 0.10, "F": 0.03},
}


def get_participation(age: int, gender: str) -> float:
    for (lo, hi), rates in PARTICIPATION.items():
        if lo <= age < hi:
            return rates[gender]
    return 0.05


# ---------------------------------------------------------------------------
# Wagering model
# Annual wager = DI × BASE_WAGER_FRAC × risk_multiplier
# Risk tolerance: 1 (conservative) → 5 (aggressive)
# Calibrated so that base case (~DI $15K, risk=2) ≈ $1,800/gambler,
# consistent with ~$15B US industry revenue across ~22% of adults.
# ---------------------------------------------------------------------------

BASE_WAGER_FRAC = 0.10

RISK_MULTIPLIERS = {1: 1.0, 2: 2.0, 3: 4.0, 4: 7.0, 5: 12.0}

# Effective house edge as fraction of total amount wagered
EDGE_BY_RISK = {
    1: 0.045,   # spread bets at standard -110 vig
    2: 0.055,   # spread + some prop bets
    3: 0.080,   # props + small parlays
    4: 0.120,   # parlays + live betting
    5: 0.180,   # heavy parlays, longshots
}

BETS_PER_YEAR = {1: 50, 2: 100, 3: 200, 4: 350, 5: 500}


def annual_wager(disposable_income: float, risk_tolerance: int) -> float:
    """Total amount wagered per year (not net loss — gross action)."""
    base = 2000 if disposable_income <= 0 else disposable_income * BASE_WAGER_FRAC
    return base * RISK_MULTIPLIERS[risk_tolerance]


def expected_annual_loss(wager: float, risk: int) -> float:
    """Deterministic expected annual loss = wager × house_edge."""
    return -wager * EDGE_BY_RISK[risk]


def simulate_year(wager: float, risk: int, n_sims: int = 10_000) -> np.ndarray:
    """
    Monte Carlo simulation of annual P&L for a bettor.

    Models each year as BETS_PER_YEAR[risk] independent bets.
    Payout multiplier scales with risk so that E[loss] = -edge × wager.

    Returns array of shape (n_sims,) with annual net P&L values.
    """
    n_bets = BETS_PER_YEAR[risk]
    bet_size = wager / n_bets
    edge = EDGE_BY_RISK[risk]

    # Map risk level to payout multiplier (higher risk → bigger but rarer wins)
    if risk <= 2:
        payout_mult = 1.0
    elif risk <= 3:
        payout_mult = 1.5
    elif risk <= 4:
        payout_mult = 3.0
    else:
        payout_mult = 6.0

    # p_win ensures E[pnl per bet] = -edge * bet_size
    p_win = (1 - edge) / (1 + payout_mult)

    wins = np.random.binomial(n_bets, p_win, size=n_sims)
    losses = n_bets - wins
    pnl = wins * payout_mult * bet_size - losses * bet_size
    return pnl


# ---------------------------------------------------------------------------
# Population-level constants (US)
# ---------------------------------------------------------------------------

# Adult population by age bracket (millions, approximate)
US_POP: dict[tuple, dict] = {
    (18, 25): {"M": 15.0, "F": 14.5},
    (25, 35): {"M": 23.0, "F": 22.5},
    (35, 50): {"M": 32.0, "F": 31.5},
    (50, 65): {"M": 31.0, "F": 32.0},
    (65, 99): {"M": 24.0, "F": 28.0},
}

# Average disposable income by age bracket ($)
AVG_DI: dict[tuple, float] = {
    (18, 25): 5000,
    (25, 35): 13000,
    (35, 50): 20000,
    (50, 65): 15000,
    (65, 99): 8000,
}

# Average risk tolerance by age bracket
AVG_RISK: dict[tuple, float] = {
    (18, 25): 3.2,
    (25, 35): 2.8,
    (35, 50): 2.3,
    (50, 65): 1.8,
    (65, 99): 1.3,
}


def estimate_population_losses() -> dict:
    """
    Estimate total US sports-gambling losses and number of active gamblers.
    Returns dict with total_losses_M ($M), total_gamblers_M (millions),
    avg_loss_per_gambler ($).
    """
    total_losses = 0.0
    total_gamblers = 0.0

    for bracket, pops in US_POP.items():
        di = AVG_DI[bracket]
        risk_avg = AVG_RISK[bracket]

        r_lo = int(risk_avg)
        r_hi = min(r_lo + 1, 5)
        frac = risk_avg - r_lo
        rm = RISK_MULTIPLIERS[r_lo] * (1 - frac) + RISK_MULTIPLIERS[r_hi] * frac
        edge = EDGE_BY_RISK[r_lo] * (1 - frac) + EDGE_BY_RISK[r_hi] * frac
        loss_per_gambler = di * BASE_WAGER_FRAC * rm * edge

        for gender in ("M", "F"):
            pop_m = pops[gender]
            n_gamblers = pop_m * PARTICIPATION[bracket][gender]
            total_losses += n_gamblers * loss_per_gambler
            total_gamblers += n_gamblers

    return {
        "total_losses_M": total_losses,
        "total_gamblers_M": total_gamblers,
        "avg_loss_per_gambler": total_losses / total_gamblers * 1e6,
    }
