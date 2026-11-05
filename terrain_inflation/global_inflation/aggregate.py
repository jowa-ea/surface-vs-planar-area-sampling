"""
Step 3 -- merge per-shard outputs into global tables.

Reads shard_*_{area_stats,status,provenance,runtime}.csv from --shard-dir and
writes to --out-dir (default work/global_inflation/; the published run is in
results/global_inflation/):
  global_area_stats.csv   area rows of cells whose final status is "done"
  global_status.csv       final status per cell (last row per cell wins)
  global_provenance.csv   STAC item IDs per cell
  global_runtime.csv      timing per cell

Usage: python -m terrain_inflation.global_inflation.aggregate [--shard-dir DIR] [--out-dir DIR]
"""

import argparse
import glob
import os

import pandas as pd

from .. import paths


def _concat(pattern):
    frames = [pd.read_csv(p) for p in sorted(glob.glob(pattern)) if os.path.getsize(p) > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate(shard_dir, out_dir):
    status = _concat(os.path.join(shard_dir, "shard_*_status.csv"))
    if status.empty:
        raise SystemExit(f"No shard_*_status.csv found under {shard_dir}")
    status = status.drop_duplicates("cell_id", keep="last").sort_values("cell_id")
    done = set(status.loc[status["status"] == "done", "cell_id"])

    area = _concat(os.path.join(shard_dir, "shard_*_area_stats.csv"))
    area = (area[area["cell_id"].isin(done)]
            .drop_duplicates(["cell_id", "worldcover_class"], keep="last")
            .sort_values(["cell_id", "worldcover_class"]))

    os.makedirs(out_dir, exist_ok=True)
    area.to_csv(os.path.join(out_dir, "global_area_stats.csv"), index=False)
    status.to_csv(os.path.join(out_dir, "global_status.csv"), index=False)
    for name in ("provenance", "runtime"):
        df = _concat(os.path.join(shard_dir, f"shard_*_{name}.csv"))
        if not df.empty:
            df = df[df["cell_id"].isin(done)].drop_duplicates("cell_id", keep="last").sort_values("cell_id")
            df.to_csv(os.path.join(out_dir, f"global_{name}.csv"), index=False)

    print(status["status"].value_counts().to_string())
    print(f"{len(area)} area rows across {area['cell_id'].nunique()} cells -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-dir", default=os.path.join(paths.GLOBAL_WORK, "shards"))
    ap.add_argument("--out-dir", default=paths.GLOBAL_WORK)
    a = ap.parse_args()
    aggregate(a.shard_dir, a.out_dir)
