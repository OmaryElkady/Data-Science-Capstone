"""FAA National Airspace System status: live airport conditions, free, no key.

NAS causes about a fifth of delay minutes (03_eda), but there is no historical archive, so
it cannot be a model feature: a model trained on 2019-2023 never saw it. It is shown beside
a forecast as context, and recorded so 08_monitor can check whether misses cluster under
degraded conditions.

Source: https://nasstatus.faa.gov/api/airport-status-information (XML, no key).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

STATUS_URL = "https://nasstatus.faa.gov/api/airport-status-information"

# NOTAM wording for a closure that only applies to non-scheduled general aviation,
# e.g. "LAX AD AP CLSD TO NON SKED TRANSIENT GA ACFT". These stay posted for months
# and no airline flight is subject to them.
_GA_ONLY = re.compile(r"CLSD TO NON SKED|TRANSIENT GA", re.IGNORECASE)


def is_general_aviation_only(text: str) -> bool:
    """True for a closure line or NOTAM that only restricts general aviation."""
    return "closure" in (text or "").lower() and bool(_GA_ONLY.search(text or ""))


@dataclass
class AirportCondition:
    """One reported condition at one airport."""

    airport: str
    kind: str                 # ground_delay | arrival_departure_delay | closure
    reason: str = ""
    detail: str = ""
    trend: str = ""

    @property
    def affects_airlines(self) -> bool:
        """False for a standing closure to non-scheduled general aviation only."""
        return not (self.kind == "closure" and _GA_ONLY.search(self.reason or ""))

    def describe(self) -> str:
        parts = [self.detail, self.reason]
        body = " — ".join(p for p in parts if p)
        return f"{self.airport}: {self.kind.replace('_', ' ')}" + (f" ({body})" if body else "")


@dataclass
class NasStatus:
    updated: str = ""
    conditions: list[AirportCondition] = field(default_factory=list)

    def for_airport(self, iata: str) -> list[AirportCondition]:
        code = (iata or "").strip().upper()
        return [c for c in self.conditions if c.airport == code]

    @property
    def airports_affected(self) -> set[str]:
        return {c.airport for c in self.conditions}


def _text(node: Optional[ET.Element], *names: str) -> str:
    """First non-empty value among `names`, searched anywhere beneath `node`."""
    if node is None:
        return ""
    for name in names:
        found = node.find(f".//{name}")
        if found is not None and (found.text or "").strip():
            return (found.text or "").strip()
    return ""


def parse_status(xml_text: str) -> NasStatus:
    """FAA status XML -> a flat list of conditions.

    The document nests differently per delay type and repeats `Delay_type`
    elements with the same `Name`, so it is flattened on the way in rather than
    navigated at every call site.
    """
    root = ET.fromstring(xml_text)
    status = NasStatus(updated=(root.findtext("Update_Time") or "").strip())

    for delay_type in root.findall("Delay_type"):
        label = (delay_type.findtext("Name") or "").strip().lower()

        if "ground delay" in label:
            kind = "ground_delay"
        elif "closure" in label:
            kind = "closure"
        else:
            kind = "arrival_departure_delay"

        for container in delay_type:
            if container.tag == "Name":
                continue
            for entry in container:
                airport = _text(entry, "ARPT", "Arpt")
                if not airport:
                    continue

                if kind == "ground_delay":
                    avg, mx = _text(entry, "Avg"), _text(entry, "Max")
                    detail = " / ".join(x for x in (f"avg {avg}" if avg else "",
                                                    f"max {mx}" if mx else "") if x)
                elif kind == "closure":
                    start, reopen = _text(entry, "Start"), _text(entry, "Reopen")
                    detail = " ".join(x for x in (f"closed {start}" if start else "",
                                                  f"reopens {reopen}" if reopen else "") if x)
                else:
                    mn, mx = _text(entry, "Min"), _text(entry, "Max")
                    detail = " to ".join(x for x in (mn, mx) if x)

                status.conditions.append(AirportCondition(
                    airport=airport.upper(),
                    kind=kind,
                    reason=_text(entry, "Reason"),
                    detail=detail,
                    trend=_text(entry, "Trend"),
                ))

    return status


def fetch_status(timeout: int = 20) -> NasStatus:
    """Current NAS status. No key, no quota, no account."""
    resp = requests.get(STATUS_URL, timeout=timeout,
                        headers={"Accept": "application/xml"})
    resp.raise_for_status()
    return parse_status(resp.text)


def summarise(status: NasStatus, airports: Any) -> list[str]:
    """One line per condition affecting any of `airports`, in reading order.

    General-aviation-only closures are left out: they never touch an airline flight.
    """
    wanted = [a for a in dict.fromkeys(
        (x or "").strip().upper() for x in airports) if a]
    lines = []
    for code in wanted:
        for condition in status.for_airport(code):
            if condition.affects_airlines:
                lines.append(condition.describe())
    return lines
