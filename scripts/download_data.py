"""Download and extract the Ergast Formula 1 CSV tables into data/raw/.

The task brief names the Kaggle dataset "Formula 1 World Championship
(1950 to 2020)". That dataset is a mirror of the Ergast Motor Racing Database
CSV export. Ergast retired its own download endpoint, so this script pulls the
identical archive from a public GitHub mirror, which keeps the repository
reproducible without Kaggle credentials.

Usage:
    python scripts/download_data.py
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/rubenv/ergast-mrd/master/f1db_csv.zip"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

REQUIRED = ["lap_times.csv", "pit_stops.csv", "results.csv", "races.csv"]


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if all((RAW_DIR / name).exists() for name in REQUIRED):
        print(f"Data already present in {RAW_DIR}. Nothing to do.")
        return 0

    print(f"Downloading {SOURCE_URL} ...")
    with urllib.request.urlopen(SOURCE_URL, timeout=180) as response:
        payload = response.read()
    print(f"  received {len(payload) / 1_048_576:.1f} MiB")

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(RAW_DIR)

    missing = [name for name in REQUIRED if not (RAW_DIR / name).exists()]
    if missing:
        print(f"ERROR: archive did not contain {missing}", file=sys.stderr)
        return 1

    print(f"Extracted {len(list(RAW_DIR.glob('*.csv')))} CSV files to {RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
