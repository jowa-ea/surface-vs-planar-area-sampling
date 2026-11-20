"""
Step 4 -- per-class and per-cell surface-area underestimation from the
global area table (paper, Sections "Aggregation and underestimation
metrics" and "Global surface versus planar area").

Cell x class records with a planar area below 100 ha are dropped first, as
are records without a positive surface area (a processing failure: a
non-empty pixel set always has a positive triangulated surface area).
The relative underestimation is area-weighted: sum(surface) / sum(planar) - 1
(columns *_inflation_* keep the development-time name).

Reads --area-stats (default results/global_inflation/global_area_stats.csv) and
writes, next
to it:
  global_class_summary.csv  one row per class plus "All classes" (paper table)
  global_cell_summary.csv   all classes combined, one row per cell (paper map
                            and per-cell statistics)
and prints the numbers quoted in the text.

Usage: python -m terrain_inflation.global_inflation.summarize [--area-stats FILE]
"""

import argparse
import os

import pandas as pd

from .. import paths

PLANAR_FLOOR_HA = 100.0
HA_PER_MHA = 1e6


def load(area_stats_csv):
    df = pd.read_csv(area_stats_csv)
    n_raw = len(df)
    df = df[df["planar_area_ha"] >= PLANAR_FLOOR_HA]
    n_floor = n_raw - len(df)
    df = df[df["surface_area_ha"] > 0]
    print(f"{n_raw} cell x class records; {n_floor} below {PLANAR_FLOOR_HA:.0f} ha, "
          f"{n_raw - n_floor - len(df)} without surface area dropped")
    return df


def _row(name, planar_ha, surface_ha, n_cells):
    return {"class_name": name,
            "planar_Mha": planar_ha / HA_PER_MHA,
            "surface_Mha": surface_ha / HA_PER_MHA,
            "absolute_inflation_Mha": (surface_ha - planar_ha) / HA_PER_MHA,
            "relative_inflation_pct": (surface_ha / planar_ha - 1) * 100,
            "n_cells": n_cells}


def class_summary(df):
    rows = [_row(name, g["planar_area_ha"].sum(), g["surface_area_ha"].sum(), g["cell_id"].nunique())
            for name, g in df.groupby("class_name")]
    rows.sort(key=lambda r: r["absolute_inflation_Mha"], reverse=True)
    rows.append(_row("All classes", df["planar_area_ha"].sum(), df["surface_area_ha"].sum(),
                     df["cell_id"].nunique()))
    return pd.DataFrame(rows)


def cell_summary(df):
    cells = (df.groupby("cell_id")
             .agg(centroid_lon=("centroid_lon", "first"), centroid_lat=("centroid_lat", "first"),
                  planar_area_ha=("planar_area_ha", "sum"), surface_area_ha=("surface_area_ha", "sum"))
             .reset_index())
    cells["relative_inflation_pct"] = (cells["surface_area_ha"] / cells["planar_area_ha"] - 1) * 100
    return cells


def main(area_stats_csv):
    out_dir = os.path.dirname(os.path.abspath(area_stats_csv))
    df = load(area_stats_csv)
    classes, cells = class_summary(df), cell_summary(df)
    for name, table in (("global_class_summary.csv", classes), ("global_cell_summary.csv", cells)):
        path = os.path.join(out_dir, name)
        table.to_csv(path, index=False, float_format="%.6f")
        print(f"Wrote {path}")

    g, pct = classes.iloc[-1], cells["relative_inflation_pct"]
    print(f"\ncells: {len(cells)}\n"
          f"planar {g['planar_Mha']:,.1f} Mha, surface {g['surface_Mha']:,.1f} Mha, "
          f"+{g['absolute_inflation_Mha']:,.1f} Mha ({g['relative_inflation_pct']:.2f}%)\n"
          f"per cell: median {pct.median():.2f}%, p90 {pct.quantile(0.9):.1f}%, "
          f"p99 {pct.quantile(0.99):.1f}%, max {pct.max():.1f}%, "
          f"> 10%: {int((pct > 10).sum())} cells ({(pct > 10).mean() * 100:.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area-stats", default=os.path.join(paths.GLOBAL_RESULTS, "global_area_stats.csv"))
    main(ap.parse_args().area_stats)
