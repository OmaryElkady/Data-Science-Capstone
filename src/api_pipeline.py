"""AviationStack live-flight ingestion.

Fetches a live flight from AviationStack, projects it into the Silver schema
using the same temporal features as the historical medallion, then writes
Bronze/Silver rows to Unity Catalog. A recorded fixture is provided so the
pipeline is demonstrable offline (no key, no network) — set `USE_FIXTURE=True`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from . import config
from .features import (
    calculate_elapsed,
    check_holiday,
    check_holiday_period,
    check_near_holiday,
    convert_utc_to_hhmm,
    fix_timestamp_smart,
    get_season,
)

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "aviationstack_sample.json"

RATE_LIMIT_STATUS = 429
RETRY_STATUSES = {500, 502, 503, 504, RATE_LIMIT_STATUS}
MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 2


@dataclass
class LookupParams:
    """Query parameters accepted by AviationStack's /flights endpoint."""

    dep_iata: Optional[str] = None
    arr_iata: Optional[str] = None
    airline_iata: Optional[str] = None
    flight_iata: Optional[str] = None
    limit: int = 100

    def to_query(self, access_key: str) -> dict[str, Any]:
        q: dict[str, Any] = {"access_key": access_key, "limit": self.limit}
        for name in ("dep_iata", "arr_iata", "airline_iata", "flight_iata"):
            v = getattr(self, name)
            if v:
                q[name] = v
        return q


def load_fixture() -> dict[str, Any]:
    """Read the committed sample response so the pipeline runs without a key."""
    with FIXTURE_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def fetch_flights(params: LookupParams, access_key: str, timeout: int = 15) -> dict[str, Any]:
    """Call AviationStack with exponential backoff on transient failures."""
    url = f"{config.AVIATIONSTACK_BASE_URL}flights"
    query = params.to_query(access_key)

    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, params=query, timeout=timeout)
        if resp.status_code == 200:
            payload = resp.json()
            if "error" in payload:
                raise RuntimeError(f"AviationStack error: {payload['error']}")
            return payload
        if resp.status_code not in RETRY_STATUSES:
            resp.raise_for_status()
        time.sleep(BACKOFF_BASE_SECONDS ** attempt)

    raise RuntimeError(f"AviationStack unavailable after {MAX_RETRIES} attempts")


def project_to_silver(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten an AviationStack response into the Silver contract."""
    records = payload.get("data") or []
    rows = []
    for rec in records:
        dep = rec.get("departure") or {}
        arr = rec.get("arrival") or {}
        airline = rec.get("airline") or {}
        flight = rec.get("flight") or {}

        dep_utc = fix_timestamp_smart(dep.get("scheduled"), dep.get("timezone") or "UTC")
        arr_utc = fix_timestamp_smart(arr.get("scheduled"), arr.get("timezone") or "UTC")
        crs_dep = convert_utc_to_hhmm(dep_utc)
        crs_arr = convert_utc_to_hhmm(arr_utc)
        elapsed = calculate_elapsed(crs_dep, crs_arr) if crs_dep and crs_arr else None

        flight_date = dep_utc.date() if pd.notna(dep_utc) else None
        if flight_date is None:
            continue

        rows.append({
            "airline_name": airline.get("name"),
            "airline_code": airline.get("iata"),
            "fl_number": _safe_int(flight.get("number")),
            # ICAO flight designator, e.g. "DAL1234". This is the join key to
            # OpenSky, which broadcasts `callsign` in the same form. Keeping it
            # here means no IATA-to-ICAO mapping table is needed anywhere:
            # AviationStack already supplies the ICAO spelling.
            "flight_icao": (flight.get("icao") or "").strip().upper() or None,
            "flight_iata": (flight.get("iata") or "").strip().upper() or None,
            # AviationStack returns one row per *marketing* flight number, so a
            # single aircraft appears several times: ATL-LAX at 18:47 came back
            # as DL753, WS6993 and AM4626. `codeshared` is null only on the row
            # for the operating carrier. Without this flag the recommender
            # offers the same aeroplane as an alternative to itself.
            "is_codeshare": flight.get("codeshared") is not None,
            "operating_flight_iata": (
                ((flight.get("codeshared") or {}).get("flight_iata") or "").strip().upper()
                or None
            ),
            "origin_airport_code": dep.get("iata"),
            "destination_airport_code": arr.get("iata"),
            "flight_date": flight_date,
            "flight_month": flight_date.month,
            "flight_year": flight_date.year,
            "crs_dep_time": crs_dep,
            "crs_arr_time": crs_arr,
            "crs_elapsed_time": elapsed,
            "dep_delay": _safe_float(dep.get("delay")),
            "arrival_delay": _safe_float(arr.get("delay")),
            "distance": _safe_float(rec.get("distance_km")),
            "day_of_week": flight_date.isoweekday() % 7 + 1,
            "week_of_year": flight_date.isocalendar().week,
            "day_of_month": flight_date.day,
            "quarter": (flight_date.month - 1) // 3 + 1,
            "is_weekend": int(flight_date.weekday() >= 5),
            "is_holiday": check_holiday(flight_date),
            "is_near_holiday": check_near_holiday(flight_date),
            "is_holiday_period": check_holiday_period(flight_date),
            "season": get_season(flight_date.month),
            "flight_status": rec.get("flight_status"),
        })
    return pd.DataFrame(rows)


def _safe_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _safe_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def validate_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Cheap DQ check reported per ingestion. Written to DATA_QUALITY_LOG."""
    required = {
        "airline_code", "origin_airport_code", "destination_airport_code",
        "flight_date", "crs_dep_time", "crs_arr_time",
    }
    missing_cols = sorted(required - set(df.columns))
    total = len(df)
    null_rate = {
        col: float(df[col].isna().mean()) if col in df.columns else 1.0
        for col in required
    }
    ok = not missing_cols and total > 0 and all(v < 0.5 for v in null_rate.values())
    return {
        "checked_at": pd.Timestamp.utcnow().isoformat(),
        "row_count": int(total),
        "missing_columns": missing_cols,
        "null_rate": null_rate,
        "passed": bool(ok),
    }
