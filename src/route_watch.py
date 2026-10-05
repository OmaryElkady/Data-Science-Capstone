"""Pure helpers for the scheduled route watch (notebook 10).

No Spark and no network, so everything here is unit-tested in CI.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

_ROUTE_RE = re.compile(r"^\s*([A-Za-z]{3})\s*-\s*([A-Za-z]{3})\s*$")


def parse_routes(spec: str) -> list[tuple[str, str]]:
    """'ATL-JFK, ord-lga' -> [('ATL', 'JFK'), ('ORD', 'LGA')]."""
    routes = []
    for part in (spec or "").split(","):
        if not part.strip():
            continue
        m = _ROUTE_RE.match(part)
        if not m:
            raise ValueError(f"{part.strip()!r} is not a route. Expected ORIGIN-DEST, e.g. ATL-JFK.")
        origin, dest = m.group(1).upper(), m.group(2).upper()
        if origin == dest:
            raise ValueError(f"{part.strip()!r} starts and ends at the same airport.")
        routes.append((origin, dest))
    if not routes:
        raise ValueError("No routes given.")
    return routes


def departure_window(now_utc: datetime, tz_name: str, lead_minutes: int,
                     hours: int) -> tuple[datetime, datetime]:
    """Local, naive start and end for an airport-departures query.

    Starts `lead_minutes` from now at the origin, so every flight it returns can
    still be forecast before it leaves. AeroDataBox caps one query at 12 hours.
    """
    if hours > 12:
        raise ValueError("AeroDataBox caps one airport query at 12 hours")
    local_now = now_utc.astimezone(ZoneInfo(tz_name))
    start = (local_now + timedelta(minutes=lead_minutes)).replace(
        second=0, microsecond=0, tzinfo=None)
    return start, start + timedelta(hours=hours)


def dedupe_same_minute(rows: Sequence[dict]) -> list[dict]:
    """Drop codeshares the provider's filter missed: same route, same minute."""
    seen, out = set(), []
    for r in rows:
        key = (r["origin_airport_code"], r["destination_airport_code"],
               r["flight_date"], r["crs_dep_time"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def spread_pick(rows: Sequence[dict], n: int,
                key: str = "scheduled_departure_utc", rotation: int = 0) -> list[dict]:
    """Up to `n` rows, one from each of `n` equal bands of the window, earliest first.

    Taking the first `n` would watch only the morning bank, so the window is cut into
    bands to keep the hour-of-day mix. `rotation` picks a different flight within each
    band: pass the day number and the watch stops grading the same flights every day,
    which made 57 forecasts mostly repeats of the same 25 flight numbers.
    """
    ordered = sorted(rows, key=lambda r: r[key])
    if n <= 0:
        return []
    if len(ordered) <= n:
        return ordered
    bounds = [round(i * len(ordered) / n) for i in range(n + 1)]
    return [ordered[lo + rotation % (hi - lo)] for lo, hi in zip(bounds, bounds[1:])]


def units_remaining(quota: dict) -> Optional[int]:
    """AeroDataBox's remaining-units header as an int, or None if absent."""
    raw = (quota or {}).get("units_remaining")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def quota_exhausted(quota: dict, reserve: int) -> bool:
    """True once the remaining units have fallen to the reserve.

    An unknown count (no call made yet, or a missing header) is not exhausted.
    """
    left = units_remaining(quota)
    return left is not None and left <= reserve
