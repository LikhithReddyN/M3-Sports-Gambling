"""
Raw UK and US household expenditure data transcribed from the M3 Challenge 2026 dataset.
All UK values are weekly pounds (£); US values are annual dollars ($).
"""

import numpy as np

GBP_TO_USD = 1.27

# ---------------------------------------------------------------------------
# UK Data
# ---------------------------------------------------------------------------

# Mean weekly household income by quintile (£)
uk_income = np.array([272, 544, 850, 1257, 2391])

# Weekly expenditure (£) by age group and income quintile [Q1 .. Q5]
uk_exp = {
    "<30": {
        "Food":      [35.1, 36.1, 53.1, 55.9, 58.4],
        "Housing":   [136.0, 143.7, 171.3, 158.4, 264.7],
        "Transport": [19.1, 46.5, 61.6, 90.5, 108.0],
        "Insurance": [15.0, 35.6, 54.6, 79.8, 109.5],
    },
    "30-49": {
        "Food":      [45.7, 57.3, 69.3, 76.2, 87.3],
        "Housing":   [103.8, 133.0, 124.7, 110.2, 126.4],
        "Transport": [29.6, 51.1, 74.0, 97.0, 134.1],
        "Insurance": [28.5, 46.7, 72.4, 102.4, 162.6],
    },
    "50-64": {
        "Food":      [40.2, 56.8, 68.0, 78.3, 98.6],
        "Housing":   [75.5, 91.1, 92.8, 98.1, 104.7],
        "Transport": [36.6, 60.7, 85.8, 119.4, 172.4],
        "Insurance": [34.8, 51.4, 70.9, 89.8, 141.6],
    },
}

# "All" quintile averages — weekly £ per age group
uk_baselines = {
    "<30":   {"Food": 48.9, "Housing": 173.5, "Transport": 67.4,  "Healthcare": 3.8,  "Insurance": 60.8},
    "30-49": {"Food": 71.7, "Housing": 120.7, "Transport": 88.4,  "Healthcare": 6.7,  "Insurance": 96.8},
    "50-64": {"Food": 70.8, "Housing": 93.4,  "Transport": 101.0, "Healthcare": 9.7,  "Insurance": 82.6},
}

# UK regional cost-of-living factors (relative to national average)
uk_rf = {
    "Food":       {"England": 1.003, "Wales": 0.951, "Scotland": 0.940, "N. Ireland": 1.191},
    "Housing":    {"England": 1.244, "Wales": 0.973, "Scotland": 0.955, "N. Ireland": 0.833},
    "Transport":  {"England": 0.987, "Wales": 0.993, "Scotland": 0.927, "N. Ireland": 1.093},
    "Healthcare": {"England": 1.012, "Wales": 0.896, "Scotland": 0.745, "N. Ireland": 1.326},
    "Insurance":  {"England": 1.176, "Wales": 1.024, "Scotland": 0.928, "N. Ireland": 0.858},
}

# ---------------------------------------------------------------------------
# US Data
# ---------------------------------------------------------------------------

us_age_labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]

# Mean annual income before taxes by age group ($)
us_income = [48514, 102494, 128285, 141121, 121571, 75460, 56028]

# Annual expenditure by category and age group ($)
us_exp = {
    "Food":       [7215, 9630, 12460, 12772, 10214, 8483, 7168],
    "Housing":    [16853, 26380, 30369, 30747, 27019, 22329, 21999],
    "Transport":  [9243, 12802, 15581, 17184, 15085, 11414, 6855],
    "Healthcare": [1485, 3825, 5949, 6748, 6711, 7715, 7918],
    "Insurance":  [4428, 10701, 13465, 14948, 12172, 4579, 1908],
}

# Keyed baseline lookup: age_label → {income, Food, Housing, ...}
us_baseline = {
    ag: {"income": us_income[i], **{c: us_exp[c][i] for c in us_exp}}
    for i, ag in enumerate(us_age_labels)
}

# US regional annual expenditure ($)
us_regions = ["Northeast", "Midwest", "South", "West"]
us_reg_exp = {
    "Food":       [11372, 9677, 9003, 11746],
    "Housing":    [29469, 23065, 23260, 32147],
    "Transport":  [12341, 12901, 12768, 15463],
    "Healthcare": [6289, 6246, 5870, 6660],
    "Insurance":  [10702, 9665, 8488, 11539],
}

# Regional cost factors (ratio to national mean, per category)
regional_factors = {
    cat: {r: round(v / np.mean(vals), 3) for r, v in zip(us_regions, vals)}
    for cat, vals in us_reg_exp.items()
}
