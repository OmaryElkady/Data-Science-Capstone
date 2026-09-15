"""Pure helper functions used by the Silver/Gold layers and the API pipeline.

These functions have no Spark or I/O dependencies so they can be unit-tested
without a cluster. Every temporal feature the medallion pipeline derives is
implemented here.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import holidays
import pandas as pd
import pytz

_US_HOLIDAYS = holidays.UnitedStates()

NEAR_HOLIDAY_WINDOW_DAYS = 1
HOLIDAY_PERIOD_WINDOW_DAYS = 7


def get_season(month: int) -> str:
    """Meteorological season from month number (1-12)."""
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    if month in (9, 10, 11):
        return "Fall"
    raise ValueError(f"month must be 1-12, got {month}")


def check_holiday(d: date) -> int:
    """1 if `d` is a US federal holiday, else 0."""
    return int(d in _US_HOLIDAYS)


def check_near_holiday(d: date) -> int:
    """1 if `d` is within ±1 day of a US federal holiday, else 0."""
    for offset in range(-NEAR_HOLIDAY_WINDOW_DAYS, NEAR_HOLIDAY_WINDOW_DAYS + 1):
        if (d + pd.Timedelta(days=offset)) in _US_HOLIDAYS:
            return 1
    return 0


def check_holiday_period(d: date) -> int:
    """1 if `d` is within ±7 days of a US federal holiday, else 0."""
    for offset in range(-HOLIDAY_PERIOD_WINDOW_DAYS, HOLIDAY_PERIOD_WINDOW_DAYS + 1):
        if (d + pd.Timedelta(days=offset)) in _US_HOLIDAYS:
            return 1
    return 0


def spark_day_of_week(d: date) -> int:
    """Day of week in Spark's `dayofweek` convention: 1=Sunday .. 7=Saturday.

    Python's `date.weekday()` is 0=Monday. Encoding the conversion here, once,
    is what lets the date dimension below replace Spark's own `dayofweek`
    without silently shifting the feature by a day.
    """
    return (d.weekday() + 1) % 7 + 1


def build_date_dimension(start: date, end: date) -> list[dict]:
    """One row per calendar date, with every derived temporal flag precomputed.

    Replaces four Python UDFs evaluated once per flight row. The dataset has
    roughly two million rows spanning about two thousand distinct dates, so the
    holiday lookups run ~1000x fewer times, and the join that applies them runs
    entirely in the JVM instead of serialising each row to a Python worker.

    Returned columns match the Spark builtins they replace exactly:
    `dayofweek` (1=Sunday), `weekofyear` (ISO), `quarter`, `month`.
    """
    if end < start:
        raise ValueError(f"end {end} precedes start {start}")

    rows, day = [], start
    while day <= end:
        dow = spark_day_of_week(day)
        rows.append(
            {
                "flight_date": day,
                "flight_month": day.month,
                "day_of_week": dow,
                "week_of_year": day.isocalendar()[1],
                "day_of_month": day.day,
                "quarter": (day.month - 1) // 3 + 1,
                "is_weekend": int(dow in (1, 7)),
                "is_holiday": check_holiday(day),
                "is_near_holiday": check_near_holiday(day),
                "is_holiday_period": check_holiday_period(day),
                "season": get_season(day.month),
            }
        )
        day += timedelta(days=1)   # datetime.timedelta, not pd.Timedelta:
        #                            adding a pandas offset promotes `date` to
        #                            `Timestamp` and would write a Spark
        #                            timestamp column where a date is expected.
    return rows


def convert_utc_to_hhmm(ts) -> Optional[int]:
    """UTC timestamp -> HHMM integer (e.g. 14:30 UTC -> 1430). None if unparseable."""
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    try:
        t = pd.to_datetime(ts, utc=True)
    except (ValueError, TypeError):
        return None
    if pd.isna(t):
        return None
    return int(t.hour * 100 + t.minute)


def fix_timestamp_smart(ts_str: str, tz_name: str) -> pd.Timestamp:
    """Parse a naive local timestamp string in `tz_name` and return UTC.

    Any existing offset in `ts_str` is ignored — the value is treated as local
    time in `tz_name`. Returns pd.NaT on parse failure. AviationStack returns
    timestamps in local airport time; this function is the single place that
    encodes that contract.
    """
    if not ts_str or not tz_name:
        return pd.NaT
    try:
        naive = pd.to_datetime(ts_str).tz_localize(None)
        tz = pytz.timezone(tz_name)
        localized = tz.localize(naive)
        return localized.astimezone(pytz.UTC)
    except (ValueError, TypeError, pytz.UnknownTimeZoneError):
        return pd.NaT


def calculate_elapsed(dep_hhmm: int, arr_hhmm: int) -> Optional[int]:
    """Elapsed minutes between two HHMM integers, handling midnight rollover."""
    if dep_hhmm is None or arr_hhmm is None:
        return None
    try:
        dep_min = (dep_hhmm // 100) * 60 + (dep_hhmm % 100)
        arr_min = (arr_hhmm // 100) * 60 + (arr_hhmm % 100)
    except (TypeError, ValueError):
        return None
    if arr_min < dep_min:
        arr_min += 24 * 60
    return arr_min - dep_min


def hhmm_to_hour(hhmm: Optional[int]) -> Optional[int]:
    """HHMM integer -> hour of day (0-23)."""
    if hhmm is None:
        return None
    try:
        return int(hhmm) // 100
    except (TypeError, ValueError):
        return None
