"""OpenSky Network client: live aircraft state, and the phase split it enables.

Why this exists alongside AviationStack
---------------------------------------
`06_api_ingest` calls AviationStack once per origin/destination pair. The dataset
holds 7,675 distinct routes, so covering them once costs 7,675 calls and
refreshing every fifteen minutes costs that again, every fifteen minutes.

OpenSky's `/states/all` returns every aircraft the network is currently tracking
in a single request. Measured against the continental US bounding box: **8,166
aircraft in one 1.1 MB response**, of which 2,893 carried callsigns belonging to
US carriers present in the BTS data. Call count stops scaling with the number of
flights and scales only with polling frequency.

Each state vector also carries `on_ground`, which is exactly the split the two
models need — `True` is the pre-departure population, `False` is the in-flight
population. No extra query, no classifier.

The division of labour
----------------------
OpenSky decides *which* flights are worth asking about. AviationStack answers
*how late* they were.

That split is deliberate. The model trains on BTS `DEP_DELAY`, which is **gate**
departure delay. OpenSky observes when an airframe stops being `on_ground`, which
is **wheels-off**. The two differ by taxi-out time — minutes at a small field,
far more at a congested hub — so deriving delay from OpenSky would inject a bias
that is worst exactly where delay matters most, and would do it silently.
AviationStack's `departure.delay` is already on gate semantics, so it is the
value that gets served. See `docs/API_STRATEGY.md`.

Joining the two
---------------
AviationStack returns `flight.icao` (e.g. `"DAL1234"`). OpenSky broadcasts
`callsign` in the same ICAO form, space-padded (e.g. `"DAL1234 "`). After
stripping, they are directly comparable — no IATA-to-ICAO mapping table is
needed, because AviationStack supplies the ICAO form itself.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Optional

import requests

# Index -> name for the 17-element state vector OpenSky returns. Positional
# arrays are compact on the wire and unreadable in code; this is the one place
# the mapping lives.
STATE_FIELDS = [
    "icao24",            # 0  unique airframe address
    "callsign",          # 1  space-padded; strip before comparing
    "origin_country",    # 2
    "time_position",     # 3  unix seconds of the last position report
    "last_contact",      # 4
    "longitude",         # 5
    "latitude",          # 6
    "baro_altitude",     # 7  metres
    "on_ground",         # 8  THE phase split
    "velocity",          # 9  m/s
    "true_track",        # 10 degrees from north
    "vertical_rate",     # 11 m/s, positive is climbing
    "sensors",           # 12
    "geo_altitude",      # 13 metres
    "squawk",            # 14
    "spi",               # 15
    "position_source",   # 16
]

TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network"
    "/protocol/openid-connect/token"
)
STATES_URL = "https://opensky-network.org/api/states/all"

# Continental US. The BTS dataset is US domestic, so a global call would return
# mostly aircraft that can never match a scheduled flight in this pipeline.
CONUS_BBOX = {"lamin": 24.0, "lomin": -125.0, "lamax": 49.5, "lomax": -66.0}


def get_access_token(client_id: str, client_secret: str, timeout: int = 20) -> str:
    """Exchange client credentials for a bearer token (OAuth2 client_credentials).

    Tokens are short-lived — 1800 seconds when measured — so fetch one per run
    rather than caching across a scheduled job.
    """
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=timeout,
    )
    resp.raise_for_status()
    token = resp.json().get("access_token")
    if not token:
        raise ValueError("OpenSky token response carried no access_token")
    return token


def fetch_states(
    token: str, bbox: Optional[dict] = None, timeout: int = 60
) -> dict[str, Any]:
    """One call, the whole tracked airspace inside `bbox`."""
    resp = requests.get(
        STATES_URL,
        headers={"Authorization": f"Bearer {token}"},
        params=bbox if bbox is not None else CONUS_BBOX,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


class OpenSkyClient:
    """Holds credentials, hands out a valid bearer token, fetches state.

    What actually expires
    ---------------------
    The `clientId` / `clientSecret` pair is long-lived: store it in Databricks
    secrets once and leave it. The *access token* minted from it lasts 1,800
    seconds. Nothing about that requires re-saving a secret — the token is
    derived at runtime and never persisted.

    So why a manager? Because a token fetched at the top of a notebook is dead
    thirty minutes later, and a scoring job that outlives that window would fail
    partway with a 401 having already done most of its work. This refreshes on
    demand: every call checks the clock first, and a 401 mid-flight forces one
    re-auth and retry in case the token was revoked early.

    `token_fn` and `clock` are injectable so the expiry logic can be tested
    without network access or real waiting.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_margin_seconds: int = 120,
        token_fn: Optional[Callable[[str, str], tuple[str, int]]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not client_id or not client_secret:
            raise ValueError("OpenSky client_id and client_secret are both required")
        self._client_id = client_id
        self._client_secret = client_secret
        # Refresh early. A token that is valid when checked but expires during
        # the request is the failure this margin exists to prevent.
        self._margin = refresh_margin_seconds
        self._token_fn = token_fn or _fetch_token_with_expiry
        self._clock = clock
        self._token: Optional[str] = None
        self._expires_at: float = 0.0
        self.refresh_count = 0

    @property
    def expired(self) -> bool:
        return self._token is None or self._clock() >= self._expires_at - self._margin

    def token(self) -> str:
        """A bearer token that is valid now, minting a new one if it is not."""
        if self.expired:
            token, expires_in = self._token_fn(self._client_id, self._client_secret)
            self._token = token
            self._expires_at = self._clock() + float(expires_in)
            self.refresh_count += 1
        return self._token

    def invalidate(self) -> None:
        """Drop the cached token so the next call re-authenticates."""
        self._token = None
        self._expires_at = 0.0

    def fetch_states(self, bbox: Optional[dict] = None, timeout: int = 60) -> dict[str, Any]:
        """One call, the whole tracked airspace, with one re-auth retry on 401."""
        try:
            return fetch_states(self.token(), bbox, timeout)
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 401:
                raise
            # Valid by the clock but rejected: revoked, or the server's notion of
            # expiry differs from ours. Re-auth once, then let it fail for real.
            self.invalidate()
            return fetch_states(self.token(), bbox, timeout)


def _fetch_token_with_expiry(client_id: str, client_secret: str) -> tuple[str, int]:
    """(token, expires_in_seconds) from the OAuth2 token endpoint."""
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=20,
    )
    resp.raise_for_status()
    body = resp.json()
    token = body.get("access_token")
    if not token:
        raise ValueError("OpenSky token response carried no access_token")
    # Default low rather than high: assuming a long life and being wrong means
    # failing mid-job, assuming a short one just means refreshing sooner.
    return token, int(body.get("expires_in", 1800))


def normalise_callsign(raw: Optional[str]) -> Optional[str]:
    """Strip OpenSky's padding. Returns None for absent or blank callsigns.

    Roughly 1.3% of vectors carry no callsign at all, and those aircraft cannot
    be matched to a scheduled flight by any means.
    """
    if raw is None:
        return None
    cleaned = raw.strip().upper()
    return cleaned or None


def parse_states(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Positional state vectors -> dicts, with a normalised callsign attached.

    Short vectors are tolerated: OpenSky has added fields over time, and a
    client that hard-indexes position 16 breaks the day it returns 16 elements.
    """
    snapshot_time = payload.get("time")
    rows = []
    for vector in payload.get("states") or []:
        row = {
            name: (vector[i] if i < len(vector) else None)
            for i, name in enumerate(STATE_FIELDS)
        }
        row["callsign"] = normalise_callsign(row.get("callsign"))
        row["snapshot_time"] = snapshot_time
        rows.append(row)
    return rows


def split_by_phase(rows: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """(on_ground, airborne) — the pre-departure and in-flight populations.

    `on_ground` is None on rare vectors; those are treated as airborne, because
    the in-flight path re-checks against the schedule and the pre-departure path
    would otherwise score an aircraft that has already left.
    """
    ground, airborne = [], []
    for row in rows:
        (ground if row.get("on_ground") is True else airborne).append(row)
    return ground, airborne


def match_to_schedule(
    rows: Iterable[dict], scheduled_icao: Iterable[str]
) -> tuple[list[dict], dict[str, int]]:
    """Keep rows whose callsign matches a scheduled flight's ICAO designator.

    Returns the matched rows and a report. The report matters as much as the
    rows: match rate is a number to measure and publish, not to assume. Around
    39% of the aircraft aloft over the US are N-registered general aviation that
    will never appear in a BTS schedule, so a low overall rate is expected and
    is not by itself a defect.
    """
    wanted = {c.strip().upper() for c in scheduled_icao if c}
    rows = list(rows)

    matched = [r for r in rows if r.get("callsign") and r["callsign"] in wanted]
    with_callsign = sum(1 for r in rows if r.get("callsign"))

    report = {
        "states_seen": len(rows),
        "with_callsign": with_callsign,
        "without_callsign": len(rows) - with_callsign,
        "scheduled_flights": len(wanted),
        "matched": len(matched),
    }
    report["match_rate_of_scheduled"] = (
        len(matched) / len(wanted) if wanted else 0.0
    )
    return matched, report
