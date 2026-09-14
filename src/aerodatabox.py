"""AeroDataBox client: real schedules, real gate times, and the join keys.

Why this replaces AviationStack for the live path
-------------------------------------------------
The AviationStack plan in use returns *historical* schedules — flights dated two
weeks before the day they were fetched. That makes the live half of this project
impossible rather than merely limited: a flight from a fortnight ago cannot be
airborne now, so every attempt to match it against a live ADS-B snapshot returned
zero by construction.

AeroDataBox returns the flight that is operating today, with both scheduled and
revised times, and it carries two things that matter more than either:

- `callSign` (e.g. `"DAL1572"`) — OpenSky's callsign, supplied directly. No
  IATA-to-ICAO mapping table is needed anywhere.
- `aircraft.modeS` (e.g. `"A34729"`) — the ICAO 24-bit airframe address, which is
  OpenSky's `icao24`. That is a *unique airframe* match rather than a flight-number
  match, and it is the strongest join available between the two feeds.

One input, not four
-------------------
`/flights/number/{number}/{date}` derives the whole itinerary from the flight
number: DL1572 comes back as ATL->IAH with times, distance, aircraft and airline.
Origin and destination are outputs, not inputs, so the user supplies a flight
number and a date and nothing else.

Delay semantics
---------------
`dep_delay` here is `revisedTime - scheduledTime` on departure, which is gate
delay — the same quantity BTS records as `DEP_DELAY` and the models were trained
on. This is the reason AeroDataBox answers "how late" while OpenSky answers
"where and what phase": deriving delay from ADS-B would give wheels-off, which
differs from gate delay by taxi-out and is worst at exactly the congested
airports where delay matters most.

Quota
-----
The free RapidAPI tier is metered in "API units" as well as requests, and the two
limits differ. Every response carries both in headers; `last_quota` exposes what
the most recent call reported so a notebook can print it rather than guess.
"""

from __future__ import annotations

import re
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import requests

from src.features import (
    check_holiday,
    check_holiday_period,
    check_near_holiday,
    get_season,
    spark_day_of_week,
)

HOST = "aerodatabox.p.rapidapi.com"
BASE_URL = f"https://{HOST}"

# "dl1572", "DL 1572", "dl-1572" all mean the same flight. Accept what a person
# would type; normalise before it reaches the API or a join key.
_FLIGHT_RE = re.compile(r"^\s*([A-Za-z]{2,3})\s*[-]?\s*(\d{1,4})\s*$")


def normalise_flight_number(raw: str) -> str:
    """'dl1572' -> 'DL1572'. Raises on anything that is not a flight number."""
    if not raw:
        raise ValueError("flight number is required")
    m = _FLIGHT_RE.match(raw)
    if not m:
        raise ValueError(
            f"{raw!r} is not a flight number. Expected a carrier code followed by "
            "digits, e.g. DL1572, AA100, UA 55."
        )
    return f"{m.group(1).upper()}{int(m.group(2))}"


class AeroDataBoxClient:
    """Thin client over the endpoints this project needs."""

    def __init__(self, api_key: str, timeout: int = 40):
        if not api_key:
            raise ValueError("AeroDataBox API key is required")
        self._headers = {"x-rapidapi-host": HOST, "x-rapidapi-key": api_key}
        self._timeout = timeout
        self.last_quota: dict[str, Optional[str]] = {}

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        resp = requests.get(
            f"{BASE_URL}/{path.lstrip('/')}",
            headers=self._headers,
            params=params or {},
            timeout=self._timeout,
        )
        self.last_quota = {
            "units_remaining": resp.headers.get("X-RateLimit-API-Units-Remaining"),
            "units_limit": resp.headers.get("X-RateLimit-API-Units-Limit"),
            "requests_remaining": resp.headers.get("X-RateLimit-Requests-Remaining"),
            "requests_limit": resp.headers.get("X-RateLimit-Requests-Limit"),
        }
        if resp.status_code == 404:
            return None          # no such flight on that date; not an error
        resp.raise_for_status()
        return resp.json()

    def flight_by_number(self, number: str, on: _date) -> list[dict]:
        """Every leg of `number` operating on `on`. Route is derived, not supplied."""
        got = self._get(f"flights/number/{normalise_flight_number(number)}/{on.isoformat()}")
        return got or []

    def airport_departures(self, iata: str, start: datetime, end: datetime) -> list[dict]:
        """Departures from `iata` in a local-time window.

        The window is capped at 12 hours by the API. Codeshares are excluded at
        the source: one aircraft sold under three flight numbers would otherwise
        appear as three candidate alternatives to itself.
        """
        if end <= start:
            raise ValueError("end must be after start")
        if end - start > timedelta(hours=12):
            raise ValueError("AeroDataBox caps the window at 12 hours")
        got = self._get(
            f"flights/airports/iata/{iata.upper()}"
            f"/{start.strftime('%Y-%m-%dT%H:%M')}/{end.strftime('%Y-%m-%dT%H:%M')}",
            {
                "direction": "Departure",
                "withCodeshared": "false",
                "withCancelled": "false",
                "withCargo": "false",
                "withPrivate": "false",
            },
        )
        return (got or {}).get("departures", []) or []


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def _parse_utc(block: Optional[dict]) -> Optional[datetime]:
    """AeroDataBox spells UTC as '2026-09-14 16:17Z'."""
    if not block:
        return None
    raw = block.get("utc")
    if not raw:
        return None
    try:
        return datetime.strptime(raw.strip(), "%Y-%m-%d %H:%MZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_local(block: Optional[dict]) -> Optional[datetime]:
    """Local airport time, spelled '2026-09-14 12:17-04:00'.

    This, not UTC, is what the clock-time features are built from. BTS records
    `CRS_DEP_TIME` as local time at the origin, so the model learned hour-of-day
    risk on a local clock — `02_eda` measured the quietest hour at 05:00 and the
    worst at 19:00, both local. Feeding it a UTC hour reads a different point on
    that curve: DL1572 leaves Atlanta at 12:17 local and 16:17Z, and scoring it
    as a 16:00 departure is simply a different flight as far as the model is
    concerned.

    The same applies to the date. BTS `FL_DATE` is the local calendar date, which
    for a late-evening departure is the previous day in UTC.
    """
    if not block:
        return None
    raw = block.get("local")
    if not raw:
        return None
    try:
        return datetime.strptime(raw.strip(), "%Y-%m-%d %H:%M%z")
    except ValueError:
        return None


def _hhmm(dt: Optional[datetime]) -> Optional[int]:
    return None if dt is None else dt.hour * 100 + dt.minute


def _delay_minutes(scheduled: Optional[datetime], revised: Optional[datetime]) -> Optional[float]:
    """Gate delay in minutes. None when the provider has not revised the time.

    None is meaningful: it says the flight has not been observed departing, which
    routes it to the pre-departure model. Zero would falsely claim punctuality.
    """
    if scheduled is None or revised is None:
        return None
    return (revised - scheduled).total_seconds() / 60.0


def flight_to_row(flight: dict) -> Optional[dict]:
    """One AeroDataBox flight -> one row of the Silver contract.

    Returns None when the record cannot carry a prediction — no departure time,
    or no route. Skipping is right: a row with a null scheduled departure cannot
    produce any of the temporal features the model needs.
    """
    dep, arr = flight.get("departure") or {}, flight.get("arrival") or {}
    dep_ap, arr_ap = dep.get("airport") or {}, arr.get("airport") or {}

    # Two clocks, each for a different job.
    #   UTC   — durations and delays. Subtracting two UTC instants is unambiguous
    #           and immune to a DST boundary falling between them.
    #   local — every clock-time feature, because that is the clock BTS recorded
    #           and therefore the clock the model learned on.
    dep_sched = _parse_utc(dep.get("scheduledTime"))
    arr_sched = _parse_utc(arr.get("scheduledTime"))
    dep_local = _parse_local(dep.get("scheduledTime")) or dep_sched
    arr_local = _parse_local(arr.get("scheduledTime")) or arr_sched
    if dep_sched is None or not dep_ap.get("iata") or not arr_ap.get("iata"):
        return None

    airline = flight.get("airline") or {}
    aircraft = flight.get("aircraft") or {}
    number = (flight.get("number") or "").replace(" ", "").upper()
    digits = re.sub(r"^[A-Z]{2,3}", "", number)

    dep_delay = _delay_minutes(dep_sched, _parse_utc(dep.get("revisedTime")))
    arr_delay = _delay_minutes(arr_sched, _parse_utc(arr.get("revisedTime")))

    # Local clock times, matching BTS CRS_DEP_TIME / CRS_ARR_TIME.
    crs_dep, crs_arr = _hhmm(dep_local), _hhmm(arr_local)
    # Block time from UTC: a flight crossing three zones has no meaningful
    # local-to-local duration, and UTC gives the real elapsed minutes.
    elapsed = (
        (arr_sched - dep_sched).total_seconds() / 60.0
        if arr_sched is not None else None
    )

    # The calendar block the feature pipeline expects. Computed from the same
    # unit-tested helpers 03_silver uses, so a live row and a training row derive
    # these fields identically — including Spark's 1=Sunday day-of-week
    # convention, which is off by one from Python's if taken directly.
    # Local date, as BTS FL_DATE is. A 22:00 local departure is the next day in
    # UTC, and dating it that way would move it to the wrong day of the week.
    flight_day = dep_local.date()
    dow = spark_day_of_week(flight_day)

    return {
        "flight_iata": number or None,
        "flight_month": flight_day.month,
        "flight_year": flight_day.year,
        "day_of_week": dow,
        "week_of_year": flight_day.isocalendar()[1],
        "day_of_month": flight_day.day,
        "quarter": (flight_day.month - 1) // 3 + 1,
        "is_weekend": int(dow in (1, 7)),
        "is_holiday": check_holiday(flight_day),
        "is_near_holiday": check_near_holiday(flight_day),
        "is_holiday_period": check_holiday_period(flight_day),
        "season": get_season(flight_day.month),
        "dep_hour": None if crs_dep is None else crs_dep // 100,
        "arr_hour": None if crs_arr is None else crs_arr // 100,
        # Supplied by the provider, not reconstructed: these are the OpenSky keys.
        "flight_icao": (flight.get("callSign") or "").strip().upper() or None,
        "aircraft_icao24": (aircraft.get("modeS") or "").strip().lower() or None,
        "aircraft_reg": aircraft.get("reg"),
        "aircraft_model": aircraft.get("model"),
        "airline_name": airline.get("name"),
        "airline_code": airline.get("iata"),
        "fl_number": int(digits) if digits.isdigit() else None,
        "origin_airport_code": dep_ap.get("iata"),
        "destination_airport_code": arr_ap.get("iata"),
        "flight_date": flight_day,
        # Kept for display and for joining against the OpenSky snapshot, which is
        # timestamped in UTC. The features above are local; these two are not, and
        # naming them makes the difference impossible to miss.
        "scheduled_departure_utc": dep_sched,
        "origin_timezone": dep_ap.get("timeZone"),
        "crs_dep_time": crs_dep,
        "crs_arr_time": crs_arr,
        "crs_elapsed_time": elapsed,
        "distance": (flight.get("greatCircleDistance") or {}).get("km"),
        "dep_delay": dep_delay,
        "arrival_delay": arr_delay,
        "flight_status": flight.get("status"),
        "is_codeshare": (flight.get("codeshareStatus") or "") == "IsCodeshared",
        "data_quality": ",".join(dep.get("quality") or []),
    }


def departure_to_row(dep_record: dict, origin_iata: str) -> Optional[dict]:
    """Airport-departure record -> Silver row.

    The airport endpoint nests the *other* end of the journey under `movement`
    and omits the origin, which the caller already knows.
    """
    movement = dep_record.get("movement") or {}
    dest = (movement.get("airport") or {}).get("iata")
    dep_sched = _parse_utc(movement.get("scheduledTime"))
    if dep_sched is None or not dest:
        return None

    synthetic = {
        "departure": {
            "airport": {"iata": origin_iata.upper()},
            "scheduledTime": movement.get("scheduledTime"),
            "revisedTime": movement.get("revisedTime"),
            "quality": movement.get("quality"),
        },
        "arrival": {"airport": {"iata": dest}, "scheduledTime": None},
        "airline": dep_record.get("airline"),
        "aircraft": dep_record.get("aircraft"),
        "number": dep_record.get("number"),
        "callSign": dep_record.get("callSign"),
        "status": dep_record.get("status"),
        "codeshareStatus": dep_record.get("codeshareStatus"),
        "greatCircleDistance": {},
    }
    return flight_to_row(synthetic)
