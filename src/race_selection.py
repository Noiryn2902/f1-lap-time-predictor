"""Score candidate races so the chosen one is defensible from the data.

The brief asks for a dry race with no red flags. Rather than relying on
recollection, every race in a candidate season is scored on four signals that
are visible in the lap data itself:

``wet_score``
    Coefficient of variation of each driver's lap times, averaged. Wet or
    drying races swing far more than dry ones.
``disruption``
    Share of laps slower than 1.5x the field median. Safety cars, virtual
    safety cars and red flags all inflate this.
``max_spike``
    The slowest lap of the race, as a multiple of the race median, measured on
    the field median so one struggling car cannot cause it. Safety cars, red
    flag restarts and standing starts all show up here.
``usable_drivers``
    Drivers who finished, ran the full distance, and made at least two stints,
    which is the minimum for a stint-based train/test split.

A race is a good candidate when the first three are low and the last is at
least five.

Two measurement traps this module avoids. Counting cars that stop appearing in
the lap data reads end-of-race finishing as retirement, because the leader
completes more laps than the cars they lapped; attrition is therefore measured
only up to three laps from the end. And wetness must be part of the viability
filter, not just the ranking: the 2019 German Grand Prix has low disruption and
no red flag, yet it was the wettest race of that season.
"""

from __future__ import annotations

import pandas as pd

from data_loader import load_lap_times, load_pit_stops, load_races, load_results

OUTLIER_MULTIPLIER = 1.5
MIN_STINTS_FOR_SPLIT = 2


def score_race(
    race_id: int,
    laps: pd.DataFrame,
    stops: pd.DataFrame,
    results: pd.DataFrame,
) -> dict | None:
    rl = laps[laps["raceId"] == race_id]
    if rl.empty:
        return None

    rs = results[results["raceId"] == race_id]
    rp = stops[stops["raceId"] == race_id]

    total_laps = int(rl["lap"].max())

    # Wetness proxy: mean per-driver coefficient of variation.
    cv = rl.groupby("driverId")["seconds"].agg(
        lambda s: s.std() / s.mean() if len(s) > 2 and s.mean() else float("nan")
    )
    wet_score = float(cv.mean())

    # Disruption proxy: share of laps far slower than the field median.
    median = rl["seconds"].median()
    disruption = float((rl["seconds"] > OUTLIER_MULTIPLIER * median).mean())

    # Mid-race attrition only. The last laps are excluded because the leader
    # finishes before the cars they lapped, which looks like mass retirement.
    mid = rl[rl["lap"] <= total_laps - 3]
    running = mid.groupby("lap")["driverId"].nunique().sort_index()
    attrition = int(-running.diff().min()) if len(running) > 1 else 0

    # Safety car / restart proxy: slowest field-median lap, relative to the
    # race median. Taken on the field median so one limping car cannot cause it.
    lap_medians = rl.groupby("lap")["seconds"].median()
    max_spike = round(float(lap_medians.max() / median), 3)

    # Drivers usable for a stint-based split.
    classified = rs[rs["positionText"].astype(str).str.isdigit()]
    full_distance = classified[classified["laps"] >= total_laps - 1]
    stints = rp.groupby("driverId").size() + 1
    usable = [
        d for d in full_distance["driverId"]
        if stints.get(d, 1) >= MIN_STINTS_FOR_SPLIT
    ]

    return {
        "raceId": race_id,
        "total_laps": total_laps,
        "wet_score": round(wet_score, 4),
        "disruption": round(disruption, 4),
        "max_spike": max_spike,
        "attrition": attrition,
        "usable_drivers": len(usable),
        "median_lap_s": round(float(median), 2),
    }


def rank_season(year: int) -> pd.DataFrame:
    races = load_races()
    laps = load_lap_times()
    stops = load_pit_stops()
    results = load_results()

    season = races[races["year"] == year]
    rows = []
    for _, race in season.iterrows():
        scored = score_race(race["raceId"], laps, stops, results)
        if scored:
            scored["round"] = int(race["round"])
            scored["name"] = race["name"]
            rows.append(scored)

    table = pd.DataFrame(rows)
    # Wetness is a viability gate, not merely a tiebreaker. A race can have low
    # disruption and no attrition and still have been run in the rain.
    table["viable"] = (
        (table["usable_drivers"] >= 5)
        & (table["wet_score"] <= 0.05)
        & (table["disruption"] <= 0.01)
        & (table["max_spike"] <= 1.30)
        & (table["attrition"] <= 2)
    )
    table["rank_score"] = table["wet_score"] + table["disruption"]
    return table.sort_values(["viable", "rank_score"], ascending=[False, True])


if __name__ == "__main__":
    cols = [
        "round", "name", "total_laps", "wet_score", "disruption",
        "max_spike", "attrition", "usable_drivers", "viable",
    ]
    for year in (2019,):
        print(f"\n{'=' * 100}")
        print(f"{year} season ranked. Lower wet_score, disruption and max_spike are better.")
        print("=" * 100)
        print(rank_season(year)[cols].to_string(index=False))
