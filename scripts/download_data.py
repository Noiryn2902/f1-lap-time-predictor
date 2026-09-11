"""Put the four required CSV tables into data/raw/.

The task names the Kaggle dataset "Formula 1 World Championship (1950 to 2020)"
by Rohan Rao, and notes that the Ergast API shut down in early 2025. **No API is
used anywhere in this project.** Everything reads static CSV files from disk.

Three ways to supply those files, tried in order:

1. **Already present.** If ``data/raw/`` holds the four tables, nothing happens.
2. **A Kaggle download you point at.** Pass the archive or folder you
   downloaded from Kaggle. This is the path the task describes::

       python scripts/download_data.py --kaggle ~/Downloads/archive.zip
       python scripts/download_data.py --kaggle ~/Downloads/f1-dataset/

3. **Automatic fallback.** With no arguments and no local copy, the script
   fetches the same static CSV export from a public mirror of the Ergast
   database, which is the upstream source the Kaggle dataset republishes. This
   exists so the repository is reproducible by a reviewer who does not want to
   sign in to Kaggle. It is still a static archive, not an API.

The four tables carry identical schemas whichever route is used, so the analysis
is unchanged.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

MIRROR_URL = "https://raw.githubusercontent.com/rubenv/ergast-mrd/master/f1db_csv.zip"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

REQUIRED = ["lap_times.csv", "pit_stops.csv", "results.csv", "races.csv"]


def already_present() -> bool:
    return all((RAW_DIR / name).exists() for name in REQUIRED)


def missing() -> list[str]:
    return [name for name in REQUIRED if not (RAW_DIR / name).exists()]


def from_kaggle(source: Path) -> None:
    """Copy or extract a Kaggle download into data/raw/."""
    if not source.exists():
        raise SystemExit(f"ERROR: {source} does not exist")

    if source.is_dir():
        found = 0
        for name in REQUIRED:
            for candidate in source.rglob(name):
                shutil.copy2(candidate, RAW_DIR / name)
                found += 1
                break
        print(f"Copied {found} of {len(REQUIRED)} tables from {source}")
    elif source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            archive.extractall(RAW_DIR)
        print(f"Extracted {source} to {RAW_DIR}")
    else:
        raise SystemExit(f"ERROR: expected a .zip or a folder, got {source}")


def from_mirror() -> None:
    print(f"No local copy found. Fetching the static CSV export from:\n  {MIRROR_URL}")
    with urllib.request.urlopen(MIRROR_URL, timeout=180) as response:
        payload = response.read()
    print(f"  received {len(payload) / 1_048_576:.1f} MiB")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(RAW_DIR)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kaggle",
        type=Path,
        metavar="PATH",
        help="a .zip or folder downloaded from the Kaggle dataset page",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if already_present() and not args.kaggle:
        print(f"All four tables already in {RAW_DIR}. Nothing to do.")
        return 0

    if args.kaggle:
        from_kaggle(args.kaggle)
    else:
        from_mirror()

    if gaps := missing():
        print(f"ERROR: still missing {gaps}", file=sys.stderr)
        return 1

    print(f"Ready: {len(list(RAW_DIR.glob('*.csv')))} CSV files in {RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
