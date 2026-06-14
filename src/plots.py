"""
All matplotlib figures for the M3 2026 Sports Gambling analysis.
Figures 1–6 cover Q1 (expenditure / disposable income).
Figures 7–10 cover Q2 (gambling outcomes).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
}

_usd = mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
_usd_k = mticker.FuncFormatter(lambda x, _: f"${x / 1000:.0f}k")


def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=200, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Q1 Figures
# ---------------------------------------------------------------------------

def fig1_disposable_income(labels: list, di: list) -> plt.Figure:
    """Bar chart of disposable income by demographic profile."""
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if d < 0 else "#2ecc71" for d in di]
    bars = ax.bar(labels, di, color=colors, edgecolor="white", linewidth=1.2, width=0.6)
    ax.axhline(0, color="#2c3e50", linewidth=1.2)
    for bar, val in zip(bars, di):
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + (800 if val >= 0 else -1800),
            f"${val:,.0f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontweight="bold",
            fontsize=11,
        )
    ax.set_ylabel("Disposable Income (USD/year)")
    ax.set_title("Disposable Income by Demographic Profile")
    ax.yaxis.set_major_formatter(_usd)
    ax.set_ylim(min(di) - 5000, max(di) + 5000)
    fig.tight_layout()
    _save(fig, "fig1_disposable_income.png")
    return fig


def fig2_expenditure_breakdown(
    labels: list, short: list, exp: dict
) -> plt.Figure:
    """Stacked bar chart of essential expenditure by profile."""
    plt.rcParams.update(STYLE)
    cats = list(exp.keys())
    colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(short))
    bottom = np.zeros(len(short))
    for i, cat in enumerate(cats):
        vals = np.array(exp[cat])
        ax.bar(x, vals, bottom=bottom, label=cat, color=colors[i],
               edgecolor="white", linewidth=0.8, width=0.55)
        for j in range(len(short)):
            share = vals[j] / sum(exp[c][j] for c in cats)
            if share > 0.12:
                ax.text(x[j], bottom[j] + vals[j] / 2, f"${vals[j]/1000:.1f}k",
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        bottom += vals
    for j in range(len(short)):
        total = sum(exp[c][j] for c in cats)
        ax.text(x[j], total + 800, f"${total/1000:.1f}k",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#2c3e50")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Annual Expenditure (USD)")
    ax.set_title("Essential Expenditure Breakdown by Profile")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.yaxis.set_major_formatter(_usd)
    fig.tight_layout()
    _save(fig, "fig2_expenditure_breakdown.png")
    return fig


def fig3_income_waterfall(profile_tuples: list, salary: list, tax: list,
                           exp: dict, di: list) -> plt.Figure:
    """6-panel income waterfall (salary → tax → expenses → DI)."""
    plt.rcParams.update(STYLE)
    cats = list(exp.keys())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for idx, (nm, age, sal, reg, ctry) in enumerate(profile_tuples):
        ax = axes[idx // 3][idx % 3]
        s = salary[idx]; t = tax[idx]
        e = sum(exp[c][idx] for c in cats); d = di[idx]
        segments = ["Salary", "Tax", "Expenses", "Disposable"]
        values = [s, -t, -e, d]
        bottoms = [0, s, s - t, 0]
        tops = [s, s - t, s - t - e, d]
        colors = ["#3498db", "#e74c3c", "#e67e22", "#2ecc71" if d >= 0 else "#e74c3c"]
        for k in range(4):
            b = min(bottoms[k], tops[k])
            h = abs(values[k])
            ax.bar(segments[k], h, bottom=b, color=colors[k], edgecolor="white", width=0.5)
            ax.text(k, b + h / 2, f"${abs(values[k])/1000:.1f}k",
                    ha="center", va="center", fontsize=8, fontweight="bold", color="white")
        for k in range(3):
            ax.plot([k + 0.25, k + 0.75], [tops[k], tops[k]],
                    color="#7f8c8d", linewidth=0.8, linestyle="--")
        ax.set_title(f"Profile {nm} ({ctry})", fontsize=11)
        ax.yaxis.set_major_formatter(_usd_k)
        ax.set_ylim(min(0, d) - 5000, s + 5000)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Income Waterfall: Salary → Tax → Expenses → Disposable",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig3_waterfall.png")
    return fig


def fig4_r2_comparison(alpha_cats: list, r2_old: list, r2_new: list) -> plt.Figure:
    """Grouped bar chart showing R² improvement from adding age fixed effects."""
    plt.rcParams.update(STYLE)
    plot_cats = [c for i, c in enumerate(alpha_cats) if r2_old[i] is not None]
    old_vals = [v for v in r2_old if v is not None]
    new_vals = [r2_new[i] for i, v in enumerate(r2_old) if v is not None]
    x = np.arange(len(plot_cats))
    w = 0.3
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_old = ax.bar(x - w / 2, old_vals, w, label="Without Fixed Effects", color="#e74c3c", alpha=0.8)
    bars_new = ax.bar(x + w / 2, new_vals, w, label="With Age Fixed Effects", color="#2ecc71", alpha=0.8)
    for bar in bars_old:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, color="#e74c3c")
    for bar in bars_new:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, color="#27ae60")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_cats)
    ax.set_ylabel("R²")
    ax.set_title("Regression Fit: Impact of Age Fixed Effects")
    ax.set_ylim(0, 1.12)
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig4_r2_comparison.png")
    return fig


def fig5_elasticities(alpha_cats: list, alphas: list) -> plt.Figure:
    """Horizontal bar chart of income elasticities by category."""
    plt.rcParams.update(STYLE)
    colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(alpha_cats[::-1], alphas[::-1], color=colors[::-1], height=0.5, edgecolor="white")
    for bar, val in zip(bars, alphas[::-1]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"α = {val:.3f}", va="center", fontweight="bold", fontsize=10)
    ax.axvline(1.0, color="#95a5a6", linestyle="--", linewidth=1, label="Unit elasticity")
    ax.set_xlabel("Income Elasticity (α)")
    ax.set_title("Income Elasticity of Essential Expenditures")
    ax.set_xlim(0, 1.1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "fig5_elasticities.png")
    return fig


def fig6_sensitivity_tornado(
    sens_cats: list, sens_lo: list, sens_hi: list, sens_base: float
) -> plt.Figure:
    """Tornado chart: sensitivity of Profile A DI to ±0.10 elasticity shifts."""
    plt.rcParams.update(STYLE)
    y = np.arange(len(sens_cats))
    left = [lo - sens_base for lo in sens_lo]
    right = [hi - sens_base for hi in sens_hi]
    widths = [r - l for l, r in zip(left, right)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(y, widths, left=left, color=["#2ecc71", "#3498db", "#e67e22"],
            height=0.4, edgecolor="white")
    ax.axvline(0, color="#2c3e50", linewidth=1.2)
    for i in range(len(sens_cats)):
        ax.text(left[i] - 50, y[i], f"${sens_lo[i]:,.0f}", ha="right", va="center", fontsize=9)
        ax.text(right[i] + 50, y[i], f"${sens_hi[i]:,.0f}", ha="left", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"α_{c}" for c in sens_cats])
    ax.set_xlabel(f"Change from Baseline DI (${sens_base:,.0f})")
    ax.set_title("Sensitivity: Profile A Disposable Income to α ± 0.10")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:+,.0f}"))
    fig.tight_layout()
    _save(fig, "fig6_sensitivity.png")
    return fig


# ---------------------------------------------------------------------------
# Q2 Figures
# ---------------------------------------------------------------------------

def fig7_pnl_distribution(sim_results: dict, risk_profiles: list) -> plt.Figure:
    """Histogram of annual P&L for three risk levels (same DI)."""
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (label, color) in zip(axes, risk_profiles):
        r = sim_results[label]
        pnl = r["pnl"]
        ax.hist(pnl, bins=60, color=color, alpha=0.8, edgecolor="white", density=True)
        ax.axvline(0, color="#2c3e50", linewidth=1.5)
        ax.axvline(r["el"], color="red", linewidth=2, linestyle="--",
                   label=f"E[loss] = ${r['el']:,.0f}")
        ax.set_title(f"Risk {r['risk']}: Wager ${r['wager']:,.0f}")
        ax.set_xlabel("Annual P&L ($)")
        ax.legend(fontsize=10)
        ax.xaxis.set_major_formatter(_usd_k)
    fig.suptitle("Distribution of Annual Gambling P&L by Risk Tolerance",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig7_pnl_distribution.png")
    return fig


def fig8_loss_vs_risk(di: float, wagers: list, losses: list) -> plt.Figure:
    """Bar chart of expected annual loss by risk level."""
    plt.rcParams.update(STYLE)
    risks = [1, 2, 3, 4, 5]
    risk_labels = ["1\nConservative", "2\nCautious", "3\nModerate", "4\nAggressive", "5\nHigh-Risk"]
    colors = ["#2ecc71", "#27ae60", "#e67e22", "#e74c3c", "#c0392b"]
    loss_pct_di = [l / di * 100 for l in losses]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(risk_labels, losses, color=colors, width=0.55, edgecolor="white")
    for bar, pct in zip(bars, loss_pct_di):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{pct:.0f}% of DI", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#2c3e50")
    ax.set_ylabel("Expected Annual Loss ($)", fontsize=12)
    ax.set_xlabel("Risk Tolerance Level", fontsize=12)
    ax.set_title(f"Expected Gambling Loss by Risk Level (DI = ${di:,})")
    ax.yaxis.set_major_formatter(_usd)
    ax.set_ylim(0, max(losses) * 1.35)
    fig.tight_layout()
    _save(fig, "fig8_loss_vs_risk.png")
    return fig


def fig9_loss_heatmap(matrix: np.ndarray) -> plt.Figure:
    """Heatmap of expected annual loss per person by age × gender."""
    plt.rcParams.update(STYLE)
    age_labels = ["18-24", "25-34", "35-49", "50-64", "65+"]
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels(age_labels, fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Male", "Female"], fontsize=12)
    for i in range(2):
        for j in range(5):
            color = "white" if matrix[i, j] > 150 else "#2c3e50"
            ax.text(j, i, f"${matrix[i, j]:,.0f}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)
    ax.set_title("Expected Annual Loss per Person by Demographic ($)\n(includes non-gamblers)")
    plt.colorbar(im, ax=ax, label="$/year", shrink=0.8)
    fig.tight_layout()
    _save(fig, "fig9_loss_heatmap.png")
    return fig


def fig10_prob_profit(p_wins: list) -> plt.Figure:
    """Bar chart of probability of ending the year in profit by risk level."""
    plt.rcParams.update(STYLE)
    risk_labels = ["1\nConservative", "2\nCautious", "3\nModerate", "4\nAggressive", "5\nHigh-Risk"]
    colors = ["#2ecc71", "#27ae60", "#e67e22", "#e74c3c", "#c0392b"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(risk_labels, p_wins, color=colors, width=0.55, edgecolor="white")
    for i, pw in enumerate(p_wins):
        ax.text(i, pw + 1.5, f"{pw:.1f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#2c3e50")
    ax.axhline(50, color="#95a5a6", linestyle="--", linewidth=1, label="50% line")
    ax.set_ylabel("Probability of Profit (%)", fontsize=12)
    ax.set_xlabel("Risk Tolerance Level", fontsize=12)
    ax.set_title("Probability of Ending the Year in Profit")
    ax.set_ylim(0, 65)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, "fig10_prob_profit.png")
    return fig
