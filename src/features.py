"""Derive stint, tire age and degradation-shape features from cleaned laps.

Features
--------
``stint_number``
    Which set of tires the car is on, counting from 1. Teams run different
    compounds in different stints, so this carries compound information that
    the dataset does not expose directly.
``tire_age``
    Laps completed on the current set, resetting to 0 at every pit stop. This
    is the feature the whole project exists to test.
``tire_age_sq``
    Tire age squared. Degradation is not linear: a set loses little in its
    first laps and falls away faster once the surface is gone. A purely linear
    term cannot represent that curve.

On the fuel proxy
-----------------
A natural instinct is to add ``fuel_proxy = total_laps - lap`` so the model can
separate fuel burn from tire wear. Within a single race that feature is
worthless, and adding it would be a mistake rather than an improvement.

``total_laps`` is constant across one race, so ``total_laps - lap`` is an exact
linear transform of ``lap``: their correlation is -1.000. The two carry
identical information. In a linear model the pair is perfectly collinear and
the coefficients become unidentifiable.

The useful conclusion is the opposite of adding a feature. Within one race,
**lap number already is the fuel proxy**, along with track evolution as rubber
goes down. Both effects are monotonic in lap number and cannot be separated
from it without multi-race data. That is what makes the baseline a fair
comparison: it already controls for fuel and track evolution, so any
improvement from ``tire_age`` is attributable to the tires.

``verify_fuel_collinearity`` demonstrates this rather than asserting it.
"""

from __future__ import annotations

import pandas as pd

from data_loader import load_pit_stops


def add_features(laps: pd.DataFrame, stops: pd.DataFrame) -> pd.DataFrame:
    """Return ``laps`` with stint number, tire age and its square."""
    out = laps.sort_values(["driverId", "lap"]).copy()

    pit_laps = {
        driver: sorted(grp["lap"].tolist())
        for driver, grp in stops.groupby("driverId")
    }

    stint_numbers: list[int] = []
    tire_ages: list[int] = []

    for driver, grp in out.groupby("driverId", sort=False):
        stops_for_driver = pit_laps.get(driver, [])
        for lap in grp["lap"]:
            # Stints are 1-indexed; a stop on lap L ends the stint at lap L.
            completed = sum(1 for s in stops_for_driver if s < lap)
            stint_numbers.append(completed + 1)
            stint_start = stops_for_driver[completed - 1] if completed else 0
            tire_ages.append(int(lap - stint_start - 1))

    out["stint_number"] = stint_numbers
    out["tire_age"] = tire_ages
    out["tire_age_sq"] = out["tire_age"] ** 2
    return out.reset_index(drop=True)


def verify_fuel_collinearity(laps: pd.DataFrame) -> pd.DataFrame:
    """Show that a fuel proxy built from lap number adds nothing."""
    probe = laps[["lap"]].copy()
    probe["fuel_proxy"] = laps["total_laps"] - laps["lap"]
    corr = probe["lap"].corr(probe["fuel_proxy"])
    return pd.DataFrame([{
        "feature_a": "lap",
        "feature_b": "fuel_proxy = total_laps - lap",
        "correlation": round(float(corr), 6),
        "verdict": "perfectly collinear, fuel_proxy adds no information",
    }])


def check_tire_age_resets(featured: pd.DataFrame, stops: pd.DataFrame) -> pd.DataFrame:
    """Confirm tire age restarts at the bottom of its range in every stint.

    Tire age is 0 on the out-lap and 1 on the lap after. Cleaning removes the
    out-lap, and removes lap 1 which is the equivalent lap of the opening
    stint, so every surviving stint should begin at ``tire_age == 1``.

    One driver legitimately loses a whole stint here: Albon pitted on lap 2,
    so his opening stint consists of lap 1 alone, which cleaning removes. He
    contributes stints 2 to 4 instead of 1 to 4.
    """
    rows = []
    for (driver, stint), grp in featured.groupby(["driverId", "stint_number"]):
        first = grp.sort_values("lap").iloc[0]
        rows.append({
            "code": first["code"],
            "stint": int(stint),
            "first_lap": int(first["lap"]),
            "tire_age_at_start": int(first["tire_age"]),
            "stint_laps": len(grp),
            "max_tire_age": int(grp["tire_age"].max()),
        })
    return pd.DataFrame(rows).sort_values(["code", "stint"]).reset_index(drop=True)


if __name__ == "__main__":
    from cleaning import clean
    from dataset import build_lap_table

    race, selected, laps = build_lap_table()
    stops = load_pit_stops()
    stops = stops[stops["raceId"] == race["raceId"]]

    cleaned, _ = clean(laps, stops)
    featured = add_features(cleaned, stops)

    print("Fuel proxy check")
    print(verify_fuel_collinearity(featured).to_string(index=False))

    print("\nTire age resets per stint")
    print(check_tire_age_resets(featured, stops).to_string(index=False))

    print(f"\n{len(featured)} laps")
    print(f"tire_age range {featured['tire_age'].min()} to {featured['tire_age'].max()}")
    print(f"stints per driver: {featured.groupby('code')['stint_number'].max().to_dict()}")
