"""
Step 2 -- planar and surface area per land-cover class for the cells of the
manifest, or for one shard of them.

For each cell: fetch WorldCover v200 and Copernicus DEM GLO-30 over the
buffered cell, grid-match WorldCover onto the DEM grid, and tally planar and
Jenness surface area per class over the pixels whose centre lies in the CEA
cell (terrain_inflation/area_tally.py). Cells run concurrently in a thread
pool (the work is dominated by network reads). Per-cell rasters are deleted
after each cell.

The same command runs everything on one machine (default: a single shard) or
one slice of the cells anywhere else: --shard i --n-shards n processes the
cells k with k mod n == i (k = row position in the manifest). Any scheduler
can fan the shards out; deployment/azure/ does so on Azure ML. The published
run used --n-shards 20.

Settled outcomes per cell:
  done                   areas written
  skipped_no_coverage    cell lies outside WorldCover's extent (south of ~60 S)
  skipped_known_dem_gap  every DEM tile the cell needs is missing from the
                         collection (see fetch_pc_data.diagnose_dem_gap)
Other failures are retried up to --max-retries times and otherwise left as
"failed". Re-running a shard into the same --out-dir skips settled cells.

Outputs (in --out-dir):
  shard_<i>_area_stats.csv           per cell x class planar / surface area
  shard_<i>_status.csv               per cell outcome (append-only)
  shard_<i>_provenance.csv           STAC item IDs read per cell
  shard_<i>_runtime.csv              timing per cell
  shard_<i>_failure_diagnostics.csv  every failed attempt

Usage:
  python -m terrain_inflation.global_inflation.run                          # all cells
  python -m terrain_inflation.global_inflation.run --shard 3 --n-shards 20  # one shard
"""

import argparse
import csv
import os
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from .. import area_tally, cea_grid, fetch_pc_data, paths

RETRY_BACKOFF_SECONDS = 10

SETTLED_STATUSES = ("done", "skipped_no_coverage", "skipped_known_dem_gap")

DIAGNOSTIC_FIELDS = ["cell_id", "attempt_label", "stage", "error",
                     "dem_reconfirmed_empty", "dem_expected_tiles", "dem_missing_tiles",
                     "dem_padded_items_found", "dem_is_regional_gap"]


def output_files(shard):
    """{key: (file name, columns)} of one shard."""
    return {
        "area": (f"shard_{shard}_area_stats.csv",
                 ["cell_id", "centroid_lon", "centroid_lat", "worldcover_class", "class_name",
                  "planar_area_ha", "surface_area_ha", "inflation_ratio"]),
        "status": (f"shard_{shard}_status.csv", ["cell_id", "status", "error"]),
        "provenance": (f"shard_{shard}_provenance.csv",
                       ["cell_id", "worldcover_items", "dem_items"]),
        "runtime": (f"shard_{shard}_runtime.csv",
                    ["cell_id", "fetch_seconds", "gridmatch_seconds", "compute_seconds",
                     "total_seconds", "data_volume_mb"]),
        "diagnostics": (f"shard_{shard}_failure_diagnostics.csv",
                        DIAGNOSTIC_FIELDS),
    }


class _StageError(Exception):
    def __init__(self, stage, original):
        super().__init__(str(original))
        self.stage = stage
        self.original = original


def run_one_cell(row, classes, scratch_dir):
    """Returns (cell_id, status, error, outputs) where outputs maps an output_files()
    key to the rows to write for this cell."""
    cell = cea_grid.get_cell(int(row["row"]), int(row["col"]))
    cell_dir = os.path.join(scratch_dir, cell.cell_id)
    os.makedirs(cell_dir, exist_ok=True)
    wc_raw = os.path.join(cell_dir, "worldcover_raw.tif")
    dem_raw = os.path.join(cell_dir, "cop_dem_raw.tif")
    wc_matched = os.path.join(cell_dir, "worldcover_matched_dem_grid.tif")

    try:
        t0 = time.time()
        try:
            wc_items = fetch_pc_data.fetch_worldcover(cell.wgs84_bbox_buffered, wc_raw)
        except Exception as e:
            raise _StageError("worldcover_fetch", e) from e
        try:
            dem_items = fetch_pc_data.fetch_copernicus_dem(cell.wgs84_bbox_buffered, dem_raw)
        except Exception as e:
            raise _StageError("dem_fetch", e) from e
        t_fetch = time.time() - t0

        t1 = time.time()
        area_tally.grid_match_to_dem(wc_raw, dem_raw, wc_matched,
                                     crosses_antimeridian=cell.crosses_antimeridian)
        t_grid = time.time() - t1

        t2 = time.time()
        planar, surface = area_tally.compute_area_by_class(wc_matched, dem_raw, cell, classes)
        t_compute = time.time() - t2

        volume = sum(os.path.getsize(p) for p in (wc_raw, dem_raw, wc_matched) if os.path.exists(p))
        area_rows = []
        for code, name in classes.items():
            p_ha, s_ha = planar[code] / 1e4, surface[code] / 1e4
            area_rows.append({
                "cell_id": cell.cell_id,
                "centroid_lon": cell.centroid_lonlat[0], "centroid_lat": cell.centroid_lonlat[1],
                "worldcover_class": code, "class_name": name,
                "planar_area_ha": p_ha, "surface_area_ha": s_ha,
                "inflation_ratio": s_ha / p_ha if p_ha > 0 else float("nan"),
            })
        outputs = {
            "area": area_rows,
            "provenance": [{"cell_id": cell.cell_id, "worldcover_items": ";".join(wc_items),
                            "dem_items": ";".join(dem_items)}],
            "runtime": [{"cell_id": cell.cell_id, "fetch_seconds": round(t_fetch, 1),
                         "gridmatch_seconds": round(t_grid, 1), "compute_seconds": round(t_compute, 1),
                         "total_seconds": round(t_fetch + t_grid + t_compute, 1),
                         "data_volume_mb": round(volume / 1e6, 1)}],
        }
        return cell.cell_id, "done", None, outputs

    except _StageError as e:
        error = f"{type(e.original).__name__}: {e.original}\n{traceback.format_exc(limit=3)}"
        diag = {"stage": e.stage}
        status = "failed"
        if e.stage == "worldcover_fetch" and cell.crosses_antimeridian:
            status = "skipped_no_coverage"  # open-ocean sliver on the seam
        elif e.stage == "dem_fetch":
            try:
                d = fetch_pc_data.diagnose_dem_gap(cell.wgs84_bbox_buffered)
                diag.update({
                    "dem_reconfirmed_empty": d["reconfirmed_empty"],
                    "dem_expected_tiles": ";".join(d["expected_tiles"]),
                    "dem_missing_tiles": ";".join(d["missing_tiles"]),
                    "dem_padded_items_found": d["padded_items_found"],
                    "dem_is_regional_gap": d["is_regional_gap"],
                })
                if d["is_regional_gap"]:
                    status = "skipped_known_dem_gap"
            except Exception:  # noqa: BLE001 -- diagnosis failure leaves a retryable "failed"
                pass
        return cell.cell_id, status, error, {"diagnostics": [diag]}
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
        return cell.cell_id, "failed", error, {"diagnostics": [{"stage": "other"}]}
    finally:
        shutil.rmtree(cell_dir, ignore_errors=True)


def _settled_cells(status_path):
    if not os.path.exists(status_path):
        return set()
    with open(status_path, newline="") as f:
        return {r["cell_id"] for r in csv.DictReader(f) if r["status"] in SETTLED_STATUSES}


def _run_pass(rows, classes, writers, flush, label, workers, scratch_dir):
    by_id = {r["cell_id"]: r for r in rows}
    n_done, failed = 0, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one_cell, r, classes, scratch_dir) for r in rows]
        for i, fut in enumerate(as_completed(futures), start=1):
            cell_id, status, error, outputs = fut.result()
            if status == "done":
                n_done += 1
                for key in ("area", "provenance", "runtime"):
                    writers[key].writerows(outputs[key])
            else:
                if status == "failed":
                    failed.append(by_id[cell_id])
                print(f"  [{status}] {cell_id}{label}: {error}")
                for d in outputs.get("diagnostics", []):
                    writers["diagnostics"].writerow({
                        "cell_id": cell_id, "attempt_label": label.strip() or "main", "error": error or "",
                        **{k: d.get(k, "") for k in DIAGNOSTIC_FIELDS
                           if k not in ("cell_id", "attempt_label", "error")},
                    })
            writers["status"].writerow({"cell_id": cell_id, "status": status, "error": error or ""})
            flush()
            if i % 50 == 0:
                print(f"  ... {i}/{len(rows)}{label}: {n_done} done, {len(failed)} failed")
    return n_done, failed


def main(manifest, shard, n_shards, out_dir, scratch_dir, workers, max_retries, classes_yaml):
    if not 0 <= shard < n_shards:
        raise SystemExit(f"--shard must be in 0..{n_shards - 1}")
    scratch_dir = scratch_dir or os.path.join(paths.CACHE_DIR, f"shard_{shard}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)
    classes = area_tally.load_enabled_classes(classes_yaml)
    files = output_files(shard)

    with open(manifest, newline="") as f:
        shard_cells = [r for k, r in enumerate(csv.DictReader(f)) if k % n_shards == shard]

    handles, writers = {}, {}
    for key, (name, fields) in files.items():
        path = os.path.join(out_dir, name)
        new = not os.path.exists(path)
        handles[key] = open(path, "a", newline="")
        writers[key] = csv.DictWriter(handles[key], fieldnames=fields)
        if new:
            writers[key].writeheader()

    def flush():
        for h in handles.values():
            h.flush()

    settled = _settled_cells(os.path.join(out_dir, files["status"][0]))
    pending = [r for r in shard_cells if r["cell_id"] not in settled]
    runnable = []
    for r in pending:
        bbox = tuple(float(r[k]) for k in ("lon_min_buf", "lat_min_buf", "lon_max_buf", "lat_max_buf"))
        if fetch_pc_data.bbox_has_worldcover_coverage(bbox):
            runnable.append(r)
        else:
            writers["status"].writerow({"cell_id": r["cell_id"], "status": "skipped_no_coverage",
                                        "error": "outside ESA WorldCover extent"})
    flush()
    print(f"Shard {shard}/{n_shards}: {len(shard_cells)} cells, {len(settled)} already settled, "
          f"{len(pending) - len(runnable)} outside WorldCover, {len(runnable)} to process")

    n_done, failed = _run_pass(runnable, classes, writers, flush, "", workers, scratch_dir)
    for attempt in range(1, max_retries + 1):
        if not failed:
            break
        print(f"Retrying {len(failed)} failed cells (attempt {attempt}/{max_retries})")
        time.sleep(RETRY_BACKOFF_SECONDS)
        n, failed = _run_pass(failed, classes, writers, flush, f" (retry {attempt})", workers, scratch_dir)
        n_done += n

    for h in handles.values():
        h.close()
    print(f"Shard {shard}/{n_shards} finished: {n_done} done, {len(failed)} still failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=paths.PUBLISHED_MANIFEST,
                    help="cell manifest (default: the published 17,505 cells)")
    ap.add_argument("--shard", type=int, default=0, help="shard to process (default 0)")
    ap.add_argument("--n-shards", type=int, default=1, help="number of shards (default 1 = all cells)")
    ap.add_argument("--out-dir", default=os.path.join(paths.GLOBAL_WORK, "shards"))
    ap.add_argument("--scratch-dir", default=None, help="per-cell rasters (default work/cache/shard_<i>)")
    ap.add_argument("--workers", type=int, default=4, help="cells processed concurrently (default 4)")
    ap.add_argument("--max-retries", type=int, default=2, help="retry passes for failed cells (default 2)")
    ap.add_argument("--classes", default=paths.WORLDCOVER_CLASSES_YAML, help="WorldCover classes YAML")
    a = ap.parse_args()
    main(a.manifest, a.shard, a.n_shards, a.out_dir, a.scratch_dir, a.workers, a.max_retries, a.classes)
