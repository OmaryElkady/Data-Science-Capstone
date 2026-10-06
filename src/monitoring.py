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


def day_bootstrap(scored: pd.DataFrame, stat, reps: int = 2000, seed: int = 0,
                  cluster: str = "flight_date") -> Optional[tuple[float, float]]:
    """95% interval for `stat(scored)`, resampling whole days rather than flights.

    Flights on one day share weather and hub congestion, so they are not independent,
    and an interval that treats them as independent is too narrow. Resampling days keeps
    that dependence. None with fewer than three days, or when most resamples cannot
    compute the statistic (an AUC needs a late flight in the sample).
    """
    import numpy as np

    groups = [g for _, g in scored.groupby(cluster)]
    if len(groups) < 3:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))],
                           ignore_index=True)
        value = stat(sample)
        if value is not None:
            values.append(value)
    if len(values) < reps / 2:
        return None
    lo, hi = np.percentile(values, [2.5, 97.5])
    return (float(lo), float(hi))


def scorecard(scored: pd.DataFrame, bootstrap: bool = True) -> dict:
    """Every headline number for one model variant, each beside its baseline.

    `scored` comes from `with_outcome`. The constant forecast at the observed rate is
    the best any constant could do, and is only knowable in hindsight, so beating it
    is a high bar and losing to it is the honest default for a weak model. Intervals on
    the AUC and the calibration gap resample days (`day_bootstrap`).
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
    gap = float(p.mean()) - observed
    return {
        "auc_ci": day_bootstrap(scored, lambda s: roc_auc(s["p"], s["late"])) if bootstrap else None,
        "gap": gap,
        "gap_ci": (day_bootstrap(scored, lambda s: s["p"].mean() - s["late"].mean())
                   if bootstrap else None),
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


def timeline(scored: pd.DataFrame, by: str = "flight_date") -> pd.DataFrame:
    """Running totals by day: forecasts, late arrivals, mean predicted, observed rate.

    Cumulative, so the last row is the whole sample and the curve shows whether the gap
    between predicted and observed is settling or still moving with each day's weather.
    """
    daily = (scored.groupby(by)
             .agg(forecasts=("late", "size"), late=("late", "sum"), predicted=("p", "sum"))
             .sort_index().reset_index())
    daily["cum_forecasts"] = daily["forecasts"].cumsum()
    daily["cum_late"] = daily["late"].cumsum()
    daily["cum_observed"] = daily["cum_late"] / daily["cum_forecasts"]
    daily["cum_predicted"] = daily["predicted"].cumsum() / daily["cum_forecasts"]
    bounds = [wilson(int(k), int(n)) for k, n in zip(daily["cum_late"], daily["cum_forecasts"])]
    daily["ci_lo"] = [b[0] for b in bounds]
    daily["ci_hi"] = [b[1] for b in bounds]
    return daily.drop(columns="predicted")


def headline(card: dict, label: str) -> str:
    """One sentence of the scorecard, for a chart title or a report."""
    text = (f"{label}: predicted {card['mean_predicted']:.1%} late, observed "
            f"{card['observed_rate']:.1%} ({card['late']} of {card['forecasts']})")
    if card.get("gap_ci"):
        lo, hi = card["gap_ci"]
        text += f"  ·  gap {card['gap'] * 100:+.1f} pts (95% CI {lo * 100:+.1f} to {hi * 100:+.1f})"
    if card.get("auc") is not None:
        text += f"  ·  ROC-AUC {card['auc']:.2f}"
        if card.get("auc_ci"):
            text += f" ({card['auc_ci'][0]:.2f} to {card['auc_ci'][1]:.2f})"
    return text


def _percent_axis(axis):
    from matplotlib.ticker import PercentFormatter
    axis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def forward_test_figure(scored: pd.DataFrame, f1_cut: float, label: str,
                        card: Optional[dict] = None, min_route: int = 5, min_running: int = 10):
    """The forward test on one page: over time, by route, calibration, every forecast.

    Used by 08_monitor and for the README image, so both show the same picture.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from src import plotting as P

    card = card or scorecard(scored)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2),
                             gridspec_kw={"width_ratios": [1.35, 1], "hspace": 0.45, "wspace": 0.28})
    (over_time, by_route), (cal, strip) = axes
    first, last = scored["flight_date"].min(), scored["flight_date"].max()
    fig.suptitle(f"The {label} model on live flights, {first} to {last}",
                 x=0.06, ha="left", y=0.995, fontsize=15, fontweight="bold", color=P.INK)
    fig.text(0.06, 0.945, headline(card, f"{card['forecasts']} graded forecasts"),
             ha="left", fontsize=9.5, color=P.INK)

    # Over time: running predicted vs observed, with the observed rate's interval. The
    # curve starts once it covers `min_running` forecasts: a first day of one flight is
    # 0% or 100% and would set the scale for everything after it.
    t = timeline(scored)
    t = t[t["cum_forecasts"] >= min(min_running, int(t["cum_forecasts"].iloc[-1]))].reset_index(drop=True)
    x = np.arange(len(t))
    over_time.fill_between(x, t["ci_lo"], t["ci_hi"], color=P.ACCENT, alpha=0.12, linewidth=0,
                           label="observed, 95% interval")
    over_time.plot(x, t["cum_observed"], color=P.ACCENT, marker="o", markersize=5, linewidth=2,
                   label="observed late rate")
    over_time.plot(x, t["cum_predicted"], color=P.INK, linestyle="--", linewidth=1.8,
                   label="mean predicted")
    over_time.set_xticks(x)
    over_time.set_xticklabels([pd.Timestamp(d).strftime("%b %d") for d in t["flight_date"]],
                              fontsize=8)
    over_time.set_ylim(0, max(0.4, float(t["ci_hi"].max()) + 0.05))
    over_time.legend(frameon=False, fontsize=8, loc="upper right")
    P.style(over_time, title="Running total as forecasts accumulate",
            ylabel="Share of flights")
    _percent_axis(over_time.yaxis)
    over_time.annotate(f"{int(t['cum_forecasts'].iloc[-1])} forecasts", (x[-1], t["cum_observed"].iloc[-1]),
                       textcoords="offset points", xytext=(-4, -14), ha="right", fontsize=8, color=P.INK)

    # By route: predicted (hollow) against observed (filled, with interval).
    routes = breakdown(scored, "route")
    small = routes[routes["forecasts"] < min_route]
    routes = routes[routes["forecasts"] >= min_route].sort_values("forecasts")
    y = np.arange(len(routes))
    for yi, (_, r) in zip(y, routes.iterrows()):
        lo, hi = wilson(int(r["late"]), int(r["forecasts"]))
        by_route.plot([lo, hi], [yi, yi], color=P.ACCENT, alpha=0.35, linewidth=5, solid_capstyle="round")
        by_route.plot([r["mean_predicted"], r["observed_rate"]], [yi, yi], color=P.MUTED, linewidth=1)
    by_route.scatter(routes["observed_rate"], y, color=P.ACCENT, s=60, zorder=3, label="observed")
    by_route.scatter(routes["mean_predicted"], y, facecolor="white", edgecolor=P.INK, s=60,
                     linewidth=1.5, zorder=3, label="predicted")
    by_route.set_yticks(y)
    by_route.set_yticklabels([f"{r['route']}  ({int(r['late'])}/{int(r['forecasts'])})"
                              for _, r in routes.iterrows()], fontsize=9)
    by_route.set_ylim(-0.7, len(routes) - 0.3)
    by_route.set_xlim(0, max(0.4, float(routes["mean_predicted"].max()) + 0.08))
    by_route.legend(frameon=False, fontsize=8, loc="lower right")
    P.style(by_route, title="By route (late / forecasts)", xlabel="Share arriving 15+ min late")
    by_route.grid(axis="x", color=P.GRID, linewidth=0.8)
    by_route.grid(axis="y", visible=False)
    _percent_axis(by_route.xaxis)
    if len(small):
        by_route.annotate(f"Not shown: {len(small)} route(s) under {min_route} forecasts",
                          (0, -0.16), xycoords="axes fraction", fontsize=7.5, color=P.MUTED)

    # Calibration in equal-count groups.
    bins = reliability(scored)
    top = max(0.4, float(bins["ci_hi"].max()) + 0.05, float(scored["p"].max()) + 0.05)
    cal.plot([0, top], [0, top], color=P.MUTED, linestyle="--", linewidth=1.2, label="perfect calibration")
    cal.errorbar(bins["mean_predicted"], bins["observed_rate"],
                 yerr=[bins["observed_rate"] - bins["ci_lo"], bins["ci_hi"] - bins["observed_rate"]],
                 fmt="o", color=P.ACCENT, ecolor=P.ACCENT, elinewidth=1.2, capsize=4,
                 markersize=7, label="five equal-count groups, 95% interval")
    for _, r in bins.iterrows():
        cal.annotate(f"{int(r['late'])}/{int(r['forecasts'])}", (r["mean_predicted"], r["observed_rate"]),
                     textcoords="offset points", xytext=(8, -3), fontsize=8, color=P.INK)
    cal.set_xlim(0, top)
    cal.set_ylim(0, top)
    cal.legend(frameon=False, fontsize=8, loc="upper left")
    P.style(cal, title="Calibration", xlabel="Mean predicted chance of delay",
            ylabel="Share that arrived 15+ min late")
    _percent_axis(cal.xaxis)
    _percent_axis(cal.yaxis)

    # Every graded forecast, by what happened.
    rng = np.random.RandomState(0)
    for outcome, colour, name in ((0, P.NEGATIVE, "on time"), (1, P.POSITIVE, "15+ min late")):
        pts = scored[scored["late"] == outcome]
        strip.scatter(pts["p"], outcome + rng.uniform(-0.18, 0.18, len(pts)), s=34, alpha=0.75,
                      color=colour, edgecolor="white", linewidth=0.5,
                      label=f"arrived {name} ({len(pts)})", zorder=3)
    for cut, text in ((f1_cut, f"F1 cut {f1_cut:.0%}\n(metrics)"), (0.5, "50%\nLIKELY LATE")):
        strip.axvline(cut, color=P.INK if cut == 0.5 else P.MUTED, linestyle=":", linewidth=1.2)
        strip.annotate(text, (cut, 1.45), ha="center", va="top", fontsize=8, color=P.INK)
    strip.set_xlim(0, max(0.6, float(scored["p"].max()) + 0.05))
    strip.set_ylim(-0.5, 1.5)
    strip.set_yticks([0, 1])
    strip.set_yticklabels(["on time", "late"])
    strip.legend(frameon=False, fontsize=8, loc="lower right")
    P.style(strip, title="Every forecast, by outcome", xlabel="Predicted chance of delay")
    strip.grid(axis="x", color=P.GRID, linewidth=0.8)
    strip.grid(axis="y", visible=False)
    _percent_axis(strip.xaxis)

    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.06, right=0.98)
    return fig


RATE_COLUMNS = ("mean_predicted", "observed_rate", "ci_lo", "ci_hi", "p_min", "p_max")


def percent_table(df: pd.DataFrame, columns=RATE_COLUMNS) -> pd.DataFrame:
    """A copy with rate columns as '19.4%' strings, for display rather than arithmetic."""
    out = df.copy()
    for c in columns:
        if c in out:
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
    return out
