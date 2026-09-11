"""Load the Ergast/Kaggle Formula 1 CSV tables.

The published CSV dump ships a ``races.csv`` whose header row lists only the
original 8 columns while every data row carries the 18 columns Ergast added
later (practice, qualifying and sprint session times). Reading it with the
default header silently shifts every field, which turns ``year`` into a string
column of timestamps. ``RACE_COLUMNS`` below restores the real schema.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Ergast encodes missing values as a literal backslash-N.
NA_VALUES = ["\\N"]

RACE_COLUMNS = [
    "raceId", "year", "round", "circuitId", "name", "date", "time", "url",
    "fp1_date", "fp1_time", "fp2_date", "fp2_time", "fp3_date", "fp3_time",
    "quali_date", "quali_time", "sprint_date", "sprint_time",
]


def _read(name: str, **kwargs) -> pd.DataFrame:
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path, na_values=NA_VALUES, **kwargs)


def load_races() -> pd.DataFrame:
    """Races, with the broken header replaced by the true 18-column schema."""
    return _read("races", header=0, names=RACE_COLUMNS)


def load_lap_times() -> pd.DataFrame:
    """Lap times, with a ``seconds`` column added alongside ``milliseconds``."""
    laps = _read("lap_times")
    laps["seconds"] = laps["milliseconds"] / 1000.0
    return laps


def load_pit_stops() -> pd.DataFrame:
    return _read("pit_stops")


def load_results() -> pd.DataFrame:
    return _read("results")


def load_drivers() -> pd.DataFrame:
    return _read("drivers")


def find_race(year: int, name_contains: str) -> pd.Series:
    """Return the single race row matching a year and a name fragment."""
    races = load_races()
    hit = races[
        (races["year"] == year)
        & races["name"].str.contains(name_contains, case=False, na=False)
    ]
    if len(hit) != 1:
        raise ValueError(
            f"Expected exactly 1 race for {year} {name_contains!r}, got {len(hit)}."
        )
    return hit.iloc[0]


if __name__ == "__main__":
    race = find_race(2019, "Bahrain")
    laps = load_lap_times()
    laps = laps[laps["raceId"] == race["raceId"]]
    print(f"{race['name']} {race['year']} (raceId={race['raceId']})")
    print(f"  {len(laps):,} laps from {laps['driverId'].nunique()} drivers")
    print(f"  median lap {laps['seconds'].median():.3f}s")
