"""Remove laps whose time reflects something other than green-flag race pace.

Four rules, each flagged separately so the removal count can be reported by
reason rather than as a single opaque total.

``pit_in``
    The lap on which the car entered the pit lane. It carries the pit entry,
    the stop itself and the pit exit, so it is many seconds slow.
``pit_out``
    The lap immediately after. The car begins it from pit-lane speed on cold
    tires, so it is slow for a reason unrelated to tire wear.
``outlier``
    Laps slower than ``OUTLIER_MULTIPLIER`` times that driver's own median,
    as the brief specifies. This catches safety cars and off-track moments.
``lap_one``
    The opening lap, which begins from a standstill on the grid. Optional; see
    ``DROP_LAP_ONE``.

The outlier rule is applied against each driver's own median rather than the
field median, so a slower car is not penalised for being slow.
"""

from __future__ import annotations

import pandas as pd

OUTLIER_MULTIPLIER = 1.5
DROP_LAP_ONE = True


def add_clean_flags(laps: pd.DataFrame, stops: pd.DataFrame) -> pd.DataFrame:
    """Return ``laps`` with one boolean column per removal reason."""
    out = laps.copy()

    pit_pairs = set(zip(stops["driverId"], stops["lap"]))
    out_pairs = {(d, lap + 1) for d, lap in pit_pairs}
    keys = list(zip(out["driverId"], out["lap"]))

    out["is_pit_in"] = [k in pit_pairs for k in keys]
    out["is_pit_out"] = [k in out_pairs for k in keys]

    driver_median = out.groupby("driverId")["seconds"].transform("median")
    out["driver_median"] = driver_median
    out["is_outlier"] = out["seconds"] > OUTLIER_MULTIPLIER * driver_median

    out["is_lap_one"] = out["lap"] == 1

    return out


def removal_report(flagged: pd.DataFrame, drop_lap_one: bool = DROP_LAP_ONE) -> pd.DataFrame:
    """Count removals by reason, and the overlap between reasons."""
    reasons = [
        ("pit_in", "Lap the car entered the pits", flagged["is_pit_in"]),
        ("pit_out", "Lap after a pit stop, on cold tires", flagged["is_pit_out"]),
        (
            "outlier",
            f"Slower than {OUTLIER_MULTIPLIER}x that driver's median",
            flagged["is_outlier"],
        ),
    ]
    if drop_lap_one:
        reasons.append(("lap_one", "Opening lap, standing start", flagged["is_lap_one"]))

    rows = [
        {"reason": name, "description": desc, "laps_flagged": int(mask.sum())}
        for name, desc, mask in reasons
    ]

    combined = pd.Series(False, index=flagged.index)
    for _, _, mask in reasons:
        combined |= mask

    total = len(flagged)
    removed = int(combined.sum())
    flagged_sum = sum(r["laps_flagged"] for r in rows)

    rows.append({
        "reason": "TOTAL removed",
        "description": f"unique laps ({flagged_sum - removed} flagged by two rules at once)",
        "laps_flagged": removed,
    })
    rows.append({
        "reason": "REMAINING",
        "description": f"{removed / total:.1%} of {total} laps removed",
        "laps_flagged": total - removed,
    })
    return pd.DataFrame(rows)


def clean(
    laps: pd.DataFrame,
    stops: pd.DataFrame,
    drop_lap_one: bool = DROP_LAP_ONE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the cleaned laps and the removal report."""
    flagged = add_clean_flags(laps, stops)
    report = removal_report(flagged, drop_lap_one=drop_lap_one)

    drop = flagged["is_pit_in"] | flagged["is_pit_out"] | flagged["is_outlier"]
    if drop_lap_one:
        drop |= flagged["is_lap_one"]

    cleaned = flagged[~drop].drop(
        columns=["is_pit_in", "is_pit_out", "is_outlier", "is_lap_one", "driver_median"]
    )
    return cleaned.reset_index(drop=True), report


if __name__ == "__main__":
    from data_loader import load_pit_stops
    from dataset import build_lap_table

    race, selected, laps = build_lap_table()
    stops = load_pit_stops()
    stops = stops[stops["raceId"] == race["raceId"]]

    flagged = add_clean_flags(laps, stops)

    print(f"{race['name']} {race['year']}\n")
    for drop_one in (False, True):
        cleaned, report = clean(laps, stops, drop_lap_one=drop_one)
        label = "dropping lap 1" if drop_one else "keeping lap 1"
        print(f"--- {label} ---")
        print(report.to_string(index=False))
        print(f"    spread of remaining laps: {cleaned['seconds'].std():.3f}s sd, "
              f"{cleaned['seconds'].min():.3f}-{cleaned['seconds'].max():.3f}s\n")

    lap_one = flagged[flagged["is_lap_one"]]
    rest = flagged[~flagged["is_lap_one"] & ~flagged["is_pit_in"] & ~flagged["is_pit_out"]]
    print(f"lap 1 median   {lap_one['seconds'].median():.3f}s")
    print(f"other laps     {rest['seconds'].median():.3f}s")
    print(f"lap 1 is {lap_one['seconds'].median() - rest['seconds'].median():+.3f}s slower")
