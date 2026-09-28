"""Download BTS Reporting Carrier On-Time Performance months as CSV.

Why this replaces the Kaggle sample
-----------------------------------
`flights_sample_3m.csv` is a ~10% sample, and two of this project's documented
limitations trace directly to that:

1. **No `TAIL_NUM`.** The sample drops it, so aircraft rotation -- the single
   largest delay cause in the EDA at 38.4% of delay minutes -- cannot be built
   at all. It is named in the README as the strongest available addition and
   blocked for want of one column.
2. **Congestion counts are wrong by a factor of ten.** Counting co-occurring
   flights in a 10% sample counts *sampled* flights: `sched_deps_origin_hour`
   came back with a mean of 3.3 where a hub runs 50-80. The share features exist
   as a workaround for exactly this, because a ratio is invariant to uniform
   sampling and a count is not.

Both are fixed by the same download, and neither is fixed by sampling it again.
Rotation needs *every* leg an airframe flew that day -- drop one and the chain
breaks silently -- so scope this by time, never by sampling or by airport.

Source
------
BTS publishes stable monthly ZIPs at transtats.bts.gov/PREZIP. No key, no
account, no form. Each month is roughly 550-650k rows, ~25 MB zipped and ~250 MB
as CSV, so a four-year pull is around 12 GB unzipped and 29M rows. Check that
against your Databricks storage and compute quota *before* pulling all of it --
see `--months` and the single-month smoke test in the README.

Usage
-----
    python tools/fetch_bts.py --years 2023 --out data/bts
    python tools/fetch_bts.py --years 2019 2021 2022 2023 --out data/bts
    python tools/fetch_bts.py --years 2023 --months 1 --out data/bts   # one month

Then upload the CSVs to the Databricks volume, or point `config.SOURCE_CSV` at a
directory of them -- Spark reads a directory of CSVs as one DataFrame.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
import zipfile

BASE = ("https://transtats.bts.gov/PREZIP/"
        "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip")

# Columns worth keeping. The full table is ~110 fields; these are the ones Silver
# reads today plus the four that make this download worth doing.
#
# BTS spells them differently from the Kaggle sample, which is the whole of the
# integration work -- see the mapping in 02_silver.
WANTED = [
    "FlightDate",                        # -> FL_DATE
    "Reporting_Airline",                 # -> AIRLINE_CODE
    "Flight_Number_Reporting_Airline",   # -> FL_NUMBER
    "Origin",                            # -> ORIGIN
    "Dest",                              # -> DEST
    "CRSDepTime",                        # -> CRS_DEP_TIME
    "DepDelay",                          # -> DEP_DELAY
    "CRSArrTime",                        # -> CRS_ARR_TIME
    "ArrDelay",                          # -> ARR_DELAY
    "CRSElapsedTime",                    # -> CRS_ELAPSED_TIME
    "Distance",                          # -> DISTANCE
    "Cancelled",
    "Diverted",
    "Tail_Number",                       # NEW: the reason for this script
    "CarrierDelay", "WeatherDelay", "NASDelay",
    "SecurityDelay", "LateAircraftDelay",
]


def fetch_month(year: int, month: int, out_dir: str, keep_zip: bool = False) -> str | None:
    """Download and extract one month. Returns the CSV path, or None if skipped."""
    url = BASE.format(year=year, month=month)
    zip_path = os.path.join(out_dir, f"bts_{year}_{month:02d}.zip")
    csv_path = os.path.join(out_dir, f"bts_{year}_{month:02d}.csv")

    if os.path.exists(csv_path):
        print(f"  {year}-{month:02d}  already present, skipping")
        return csv_path

    print(f"  {year}-{month:02d}  downloading ... ", end="", flush=True)
    try:
        # BTS rejects the default urllib agent with 403.
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=300) as response, \
                open(zip_path, "wb") as handle:
            handle.write(response.read())
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}")
        if exc.code == 404:
            print(f"      No file for {year}-{month:02d}. The most recent months lag "
                  "by roughly 10 weeks.")
        return None
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"failed ({exc})")
        return None

    with zipfile.ZipFile(zip_path) as archive:
        inner = next((n for n in archive.namelist() if n.lower().endswith(".csv")), None)
        if inner is None:
            print("no CSV inside the archive")
            return None
        with archive.open(inner) as source, open(csv_path, "wb") as target:
            target.write(source.read())

    if not keep_zip:
        os.remove(zip_path)

    size_mb = os.path.getsize(csv_path) / 1024 ** 2
    print(f"{size_mb:,.0f} MB")
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    parser.add_argument("--out", default="data/bts")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    written = []
    for year in args.years:
        print(f"{year}:")
        for month in args.months:
            path = fetch_month(year, month, args.out, args.keep_zip)
            if path:
                written.append(path)

    total_gb = sum(os.path.getsize(p) for p in written) / 1024 ** 3
    print(f"\n{len(written)} month(s) in {args.out}  ({total_gb:,.1f} GB)")
    if written:
        print("\nColumns this project needs from these files:")
        print("  " + ", ".join(WANTED))
        print("\nNext: upload to the Databricks volume, then run 01_bronze against")
        print("the directory. Validate one month end to end before pulling four years.")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
