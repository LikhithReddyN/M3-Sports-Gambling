# M3 Challenge 2026 — Sports Gambling

**Team solution to the 2026 M3 Mathematical Competition problem on sports gambling economics.**

The model answers two questions:
- **Q1** — Given a demographic profile (age, salary, region, country), what is disposable income after essential expenditures?
- **Q2** — Given disposable income and risk tolerance, what are a sports bettor's expected annual losses and the probability of coming out ahead?

---

## Repository Structure

```
M3-Sports-Gambling/
├── q1_analysis.py        # Q1 entry point: elasticity calibration, DI, figures 1–6
├── q2_gambling.py        # Q2 entry point: wagering model, Monte Carlo, figures 7–10
├── requirements.txt
├── data/
│   └── README.md         # Where to place the M3 Challenge Excel file
├── notebooks/
│   └── M3_comp.ipynb     # Original competition notebook
├── outputs/              # Generated figures (gitignored contents)
└── src/
    ├── data.py           # Raw UK/US expenditure data constants
    ├── model.py          # Income elasticity regression, tax functions, DI calculator
    ├── gambling.py       # Participation rates, wagering model, Monte Carlo simulation
    └── plots.py          # All matplotlib figure functions (figs 1–10)
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Q1: disposable income analysis
python q1_analysis.py

# Q2: sports gambling outcomes
python q2_gambling.py
```

Figures are saved to `outputs/`.

---

## Methodology

### Q1 — Expenditure Model

Essential spending is predicted with a **log-log income-elasticity model**:

```
E_cat = E_baseline × (salary / income_baseline)^α × regional_factor
```

Elasticities (`α`) are calibrated via OLS regression on UK household expenditure data across 5 income quintiles and 3 age groups, with age dummy variables as fixed effects. This raises R² from ~0.15–0.65 to 0.87–0.97 across categories.

| Category   | α (elasticity) | Interpretation                        |
|------------|---------------|---------------------------------------|
| Insurance  | 0.797         | Near-proportional to income            |
| Transport  | 0.744         | Moderately income-elastic              |
| Healthcare | 0.500         | Estimated (insufficient quintile data) |
| Food       | 0.329         | Near-necessity (inelastic)             |
| Housing    | 0.158         | Very inelastic (fixed local costs)     |

US and UK tax functions model federal/state income tax, FICA, and UK income tax + National Insurance respectively.

### Q2 — Gambling Model

Annual wagering is modeled as:

```
wager = DI × 0.10 × risk_multiplier
```

where risk tolerance (1–5) maps to bet-type preference (spreads → parlays), house edge (4.5%–18%), and number of bets per year (50–500).

Each bettor's annual P&L is simulated via **Monte Carlo** (10,000 runs) as N independent bets, with win probability set so E[loss] = house_edge × wager. The model is validated against the ~$15B US sports-betting industry revenue.

**Key result:** Even at the lowest risk level, expected loss is ~5–6% of disposable income. Higher risk dramatically increases both expected loss and variance — but does not materially increase the probability of finishing the year in profit.

---

## Data Sources

- **UK**: Office for National Statistics — *Family Spending Workbook* (household expenditure by age and income quintile)
- **US**: Bureau of Labor Statistics — *Consumer Expenditure Survey* (annual expenditure by age group and region)
- **Industry benchmark**: American Gaming Association — US sports betting handle and revenue, 2023
