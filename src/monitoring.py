"""Grading live forecasts: which rows count, and how they score against baselines.

pandas only, so CI tests it. `08_monitor` collects `flight_delay_predictions` (a few
rows a day) and calls these; the README's forward-test figures come from the same code.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from src.faa_nas import is_general_aviation_only

LATE_MINUTES = 15   # US DOT: an arrival 15+ minutes late is delayed

# Every row of the predictions table is exactly one of these.
STATUS_MEANING = {
    "graded": "forecast made in time, flight landed: scored below",
    "awaiting": "flight not landed yet, or landed after the last outcome check",
    "no_outcome": "past the lookback and the provider never reported it Arrived",
    "retrospective": "scored after the information it claims not to have",
    "context": "an alternative on the route; the provider gives it no arrival time",
    "cancelled": "has an outcome, but the 15-minute rule cannot grade it",
}
STATUS_ORDER = list(STATUS_MEANING)


def _utc(values: pd.Series) -> pd.Series:
    """Epoch seconds or ISO strings -> tz-aware UTC timestamps."""
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, unit="s", utc=True)
    return pd.to_datetime(values, utc=True)


def _flag(values: pd.Series) -> pd.Series:
    """Nullable booleans (or 'true'/'false' strings) -> bool, NULL as False."""
    return values.astype(str).str.lower().eq("true")


def is_retrospective(df: pd.DataFrame) -> pd.Series:
    """A claim made after the information it is supposed to lack.

    `07_score` flags a score made after landing. A schedule-only forecast made after
    the scheduled departure is the other case: by then the flight's day has started,
    and a re-fetch days later must not pass for a morning forecast.
    """
    flagged = _flag(df["is_retrospective"]) if "is_retrospective" in df else False
    late_pre = ((df["recommended_model"] == "pre_departure")
                & (_utc(df["prediction_timestamp"]) > _utc(df["scheduled_departure_utc"])))
    return flagged | late_pre.fillna(False)


def claim_status(df: pd.DataFrame, now: pd.Timestamp, lookback_days: int) -> pd.Series:
    """Label every prediction row with one key of STATUS_MEANING."""
    outcome = df["arrival_delay"].notna()
    status_text = df.get("flight_status", pd.Series("", index=df.index)).fillna("")
    recent = _utc(df["scheduled_departure_utc"]) >= now - pd.Timedelta(days=lookback_days)

    out = pd.Series("no_outcome", index=df.index)
    out[recent] = "awaiting"
    out[df.get("row_kind", pd.Series("", index=df.index)) == "alternative"] = "context"
    out[outcome] = "graded"
    out[status_text.str.lower().str.contains("cancel")] = "cancelled"
    out[is_retrospective(df)] = "retrospective"
    return out


def with_outcome(graded: pd.DataFrame) -> pd.DataFrame:
    """Probability `p` (0-1) and outcome `late` (0/1) beside each graded row."""
    out = graded.copy()
    out["p"] = out["delay_probability_pct"].astype(float) / 100.0
    out["late"] = (out["arrival_delay"].astype(float) >= LATE_MINUTES).astype(int)
    return out


def roc_auc(p: pd.Series, y: pd.Series) -> Optional[float]:
    """Mann-Whitney AUC with tied scores averaged. None without both outcomes."""
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return None
    ranks = p.rank(method="average")
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for k of n. Honest at small n, unlike k/n +/- 2se."""
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    centre = (phat + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (0.0 if k == 0 else centre - half, 1.0 if k == n else centre + half)


def scorecard(scored: pd.DataFrame) -> dict:
    """Every headline number for one model variant, each beside its baseline.

    `scored` comes from `with_outcome`. The constant forecast at the observed rate is
    the best any constant could do, and is only knowable in hindsight, so beating it
    is a high bar and losing to it is the honest default for a weak model.
    """
    n = len(scored)
    if n == 0:
        return {"forecasts": 0}
    p, y = scored["p"], scored["late"]
    late = int(y.sum())
    observed = late / n
    brier = float(((p - y) ** 2).mean())
    brier_const = observed * (1 - observed)
    flagged = scored["will_be_delayed"].astype(int)
    return {
        "forecasts": n,
        "flights": int(scored["flight"].nunique()),
        "days": int(scored["flight_date"].nunique()),
        "late": late,
        "observed_rate": observed,
        "observed_ci": wilson(late, n),
        "mean_predicted": float(p.mean()),
        "brier": brier,
        "brier_constant": brier_const,
        "brier_skill": (1 - brier / brier_const) if brier_const else None,
        "auc": roc_auc(p, y),
        "accuracy_50": float((scored["advisory_flag"].astype(int) == y).mean()),
        "accuracy_always_on_time": 1 - observed,
        "called_late_50": int(scored["advisory_flag"].astype(int).sum()),
        "flagged_f1": int(flagged.sum()),
        "caught_f1": int((flagged & y).sum()),
    }


def reliability(scored: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Observed late rate in equal-count bins of predicted probability, with Wilson intervals.

    Equal counts, not equal widths: a weak model's calibrated outputs sit in a narrow
    band, so fixed-width bins leave single flights alone in a bin and their 0% or 100%
    swamps the chart. Ties are split by row order so every bin holds about n/bins.
    """
    k = max(1, min(bins, len(scored)))
    group = pd.qcut(scored["p"].rank(method="first"), k, labels=False)
    rows = []
    for _, grp in scored.groupby(group):
        late = int(grp["late"].sum())
        lo, hi = wilson(late, len(grp))
        rows.append({"forecasts": len(grp), "late": late,
                     "p_min": grp["p"].min(), "p_max": grp["p"].max(),
                     "mean_predicted": grp["p"].mean(), "observed_rate": late / len(grp),
                     "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def breakdown(scored: pd.DataFrame, by: str) -> pd.DataFrame:
    """Forecasts, late arrivals, mean predicted and observed rate per group."""
    return (scored.groupby(by)
            .agg(forecasts=("late", "size"), late=("late", "sum"),
                 mean_predicted=("p", "mean"), observed_rate=("late", "mean"))
            .reset_index())


def nas_affects_airlines(conditions: Optional[str]) -> bool:
    """True when a recorded NAS line could touch an airline flight.

    `07`/`10` join conditions with '; '. Standing closures to non-scheduled general
    aviation (LAX has one posted until 2027) are dropped: no airline flight is subject
    to them, and counting them made every LAX departure look 'degraded'.
    """
    if not conditions or (isinstance(conditions, float) and math.isnan(conditions)):
        return False
    return any(part.strip() and not is_general_aviation_only(part)
               for part in str(conditions).split("; "))


def forward_test_figure(scored: pd.DataFrame, f1_cut: float, label: str):
    """Two panels: calibration in equal-count bins, and every graded forecast.

    Used by 08_monitor and for the README image, so both show the same picture.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from src import plotting as P

    bins = reliability(scored)
    fig, (cal, strip) = plt.subplots(1, 2, figsize=(12.5, 4.8),
                                     gridspec_kw={"width_ratios": [1, 1.35]})

    top = max(0.4, float(bins["ci_hi"].max()) + 0.05, float(scored["p"].max()) + 0.05)
    cal.plot([0, top], [0, top], color=P.MUTED, linestyle="--", linewidth=1.2, label="perfect calibration")
    cal.errorbar(bins["mean_predicted"], bins["observed_rate"],
                 yerr=[bins["observed_rate"] - bins["ci_lo"], bins["ci_hi"] - bins["observed_rate"]],
                 fmt="o", color=P.ACCENT, ecolor=P.ACCENT, elinewidth=1.2, capsize=4,
                 markersize=7, label="live flights, 95% interval")
    for _, r in bins.iterrows():
        cal.annotate(f"{int(r['late'])}/{int(r['forecasts'])}", (r["mean_predicted"], r["observed_rate"]),
                     textcoords="offset points", xytext=(8, -3), fontsize=8, color=P.INK)
    cal.set_xlim(0, top)
    cal.set_ylim(0, top)
    cal.legend(frameon=False, fontsize=8, loc="upper left")
    P.style(cal, title="Predicted vs observed", xlabel="Mean predicted chance of delay",
            ylabel="Share that arrived 15+ min late")
    cal.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    cal.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    rng = np.random.RandomState(0)
    for outcome, colour, name in ((0, P.NEGATIVE, "on time"), (1, P.POSITIVE, "15+ min late")):
        pts = scored[scored["late"] == outcome]
        strip.scatter(pts["p"], outcome + rng.uniform(-0.18, 0.18, len(pts)), s=34, alpha=0.75,
                      color=colour, edgecolor="white", linewidth=0.5,
                      label=f"arrived {name} ({len(pts)})", zorder=3)
    for x, text in ((f1_cut, f"F1 cut {f1_cut:.0%}\n(metrics)"), (0.5, "50%\nLIKELY LATE call")):
        strip.axvline(x, color=P.INK if x == 0.5 else P.MUTED, linestyle=":", linewidth=1.2)
        strip.annotate(text, (x, 1.42), ha="center", va="top", fontsize=8, color=P.INK)
    strip.set_xlim(0, max(0.6, float(scored["p"].max()) + 0.05))
    strip.set_ylim(-0.5, 1.5)
    strip.set_yticks([0, 1])
    strip.set_yticklabels(["on time", "late"])
    strip.legend(frameon=False, fontsize=8, loc="lower right")
    P.style(strip, title=f"Every graded {label} forecast", xlabel="Predicted chance of delay")
    strip.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    strip.grid(axis="x", color=P.GRID, linewidth=0.8)
    strip.grid(axis="y", visible=False)

    fig.tight_layout()
    return fig
