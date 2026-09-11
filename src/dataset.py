"""Select the race and drivers, and assemble the lap table for the analysis.

Race choice
-----------
The 2019 United States Grand Prix at the Circuit of the Americas. It was picked
by scoring every 2019 race on wetness, disruption, lap-time spikes and
mid-race attrition (see ``race_selection.py``), then choosing among the
survivors on stint structure.

It wins on three counts:

1. The lowest lap-time spike of the 2019 season, 1.073x the race median. No
   safety car flattened the pace, so slow laps are tire wear rather than
   neutralisation.
2. The second-lowest lap-time variation of the season, so the track was dry
   throughout.
3. Most finishers ran three stints. That leaves two training stints and one
   held-out final stint per driver, and a roughly 60/40 split of laps. A
   one-stop race such as the 2019 Austrian Grand Prix scores just as clean but
   gives each driver only a single training stint, and its long final stints
   would make the test set larger than the training set.

The Bahrain Grand Prix, an obvious first choice, is rejected by the same
scoring: a 1.345x spike and three mid-race retirements, reflecting the late
safety car and the leader's engine failure.

Driver choice
-------------
Drivers must have been classified, run within one lap of the full distance,
made at least two stints, and have at least ``MIN_TEST_LAPS`` laps in the final
stint and ``MIN_TRAIN_LAPS`` laps before it. The brief asks for five to ten, so
the highest-finishing qualifying drivers are taken.
"""

from __future__ import annotations

import pandas as pd

from data_loader import (
    find_race,
    load_drivers,
    load_lap_times,
    load_pit_stops,
    load_results,
)

RACE_YEAR = 2019
RACE_NAME = "United States"

MIN_STINTS = 2
MIN_TEST_LAPS = 10
MIN_TRAIN_LAPS = 20
MAX_DRIVERS = 8


def select_drivers(race_id: int, total_laps: int) -> pd.DataFrame:
    """Return the drivers whose stint structure supports a held-out final stint."""
    results = load_results()
    stops = load_pit_stops()
    drivers = load_drivers()

    res = results[results["raceId"] == race_id]
    classified = res[res["positionText"].astype(str).str.isdigit()]
    finishers = classified[classified["laps"] >= total_laps - 1]

    race_stops = stops[stops["raceId"] == race_id]

    rows = []
    for _, r in finishers.iterrows():
        pit_laps = sorted(race_stops[race_stops["driverId"] == r["driverId"]]["lap"])
        if not pit_laps:
            continue
        bounds = [0, *pit_laps, total_laps]
        lengths = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
        if len(lengths) < MIN_STINTS:
            continue
        rows.append({
            "driverId": r["driverId"],
            "grid": r["grid"],
            "finish_position": int(r["positionOrder"]),
            "n_stints": len(lengths),
            "train_laps": sum(lengths[:-1]),
            "test_laps": lengths[-1],
        })

    table = pd.DataFrame(rows)
    eligible = table[
        (table["test_laps"] >= MIN_TEST_LAPS)
        & (table["train_laps"] >= MIN_TRAIN_LAPS)
    ]
    eligible = eligible.sort_values("finish_position").head(MAX_DRIVERS)

    names = drivers[["driverId", "code", "surname"]]
    return eligible.merge(names, on="driverId", how="left").reset_index(drop=True)


def build_lap_table() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Return the race row, the selected drivers, and their raw lap table."""
    race = find_race(RACE_YEAR, RACE_NAME)
    race_id = race["raceId"]

    laps = load_lap_times()
    race_laps = laps[laps["raceId"] == race_id]
    total_laps = int(race_laps["lap"].max())

    selected = select_drivers(race_id, total_laps)

    table = race_laps[race_laps["driverId"].isin(selected["driverId"])].copy()
    table = table.merge(
        selected[["driverId", "code", "surname", "grid", "finish_position"]],
        on="driverId",
        how="left",
    )
    table["total_laps"] = total_laps
    table = table.sort_values(["driverId", "lap"]).reset_index(drop=True)
    return race, selected, table


if __name__ == "__main__":
    race, selected, table = build_lap_table()
    print(f"{race['name']} {race['year']}  raceId={race['raceId']}  {race['date']}")
    print(f"{int(table['total_laps'].iloc[0])} laps\n")
    print(selected[[
        "code", "surname", "grid", "finish_position", "n_stints",
        "train_laps", "test_laps",
    ]].to_string(index=False))
    print(f"\n{len(table):,} laps across {table['driverId'].nunique()} drivers")
    print(f"median lap {table['seconds'].median():.3f}s (uncleaned)")
