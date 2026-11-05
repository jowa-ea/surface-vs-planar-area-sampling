"""
Step 1 -- population, strata, allocation and both samples of one study block
(paper, Section "Sample-based area and accuracy estimation").

  1. Fetch WorldCover v200 and Copernicus DEM GLO-30 for the block's CEA
     cell and grid-match WorldCover onto the DEM grid (as in the global
     assessment, area_tally.py).
  2. Per-pixel planar area a_p and surface area s_p; two strata (target /
     other); stratum totals A_c (planar) and S_c (surface).
  3. Stratum map reprojected to the block's LAEA grid: N_c pixels and
     planar area A_c^EA (srs_equal_area.py).
  4. Proportional allocation of n_total units per experiment:
       Experiment A: W_c = A_c^EA / sum A^EA
       Experiment B: W_c = S_c / sum S
  5. Experiment A: equal-probability draw without replacement on the LAEA
     grid (seed_srs). Experiment B: PPS draw with replacement on the DEM
     grid, size measure s_p (pps_sampling.py, seed_pps).
  6. Both samples are pooled and shuffled (seed_ids) into the ids of the
     blind labelling sheet, so that interpreters could not tell which design
     selected a unit. The ids link each sample unit to its reference label
     in inputs/sampling_comparison/reference_labels/<block>.csv.

Inputs:  inputs/sampling_comparison/sampling_design.yaml
Results: results/sampling_comparison/<block>/
  stratum_totals.csv  per stratum: DEM-grid pixels, A_c, S_c, inflation, LAEA
                      pixels N_c, A_c^EA, and W_c and n_c of both experiments
  srs_sample.csv      Experiment A sample units
  pps_sample.csv      Experiment B sample units
  laea_grid.txt       LAEA grid of Experiment A
  provenance.csv      STAC item IDs read
Rasters go to work/sampling_comparison/<block>/.

Usage: python -m terrain_inflation.sampling.prepare_block klamath_forest
"""

import argparse
import csv
import os

import numpy as np
import yaml
from osgeo import gdal

from .. import area_tally, cea_grid, fetch_pc_data, paths
from . import pps_sampling as pps
from . import srs_equal_area as srs


gdal.UseExceptions()

with open(os.path.join(paths.SAMPLING_INPUTS, "sampling_design.yaml")) as _f:
    DESIGN = yaml.safe_load(_f)

SRS_FIELDS = ["id", "stratum", "row", "col", "x_laea", "y_laea", "lon", "lat"]
PPS_FIELDS = ["id", "stratum", "row", "col", "lon", "lat", "surface_area_m2"]


def stratum_array(classes, valid, strata):
    """Stratum code per pixel; 0 = outside the population (void in the 3x3
    elevation neighbourhood, or a map class in no stratum).
    strata: {stratum_code: [WorldCover classes]}."""
    lut = np.zeros(256, dtype=np.uint8)
    for code, values in strata.items():
        lut[list(values)] = code
    out = lut[classes]
    out[~valid] = 0
    return out


def write_geotiff(arr, geotransform, path):
    ds = gdal.GetDriverByName("GTiff").Create(path, arr.shape[1], arr.shape[0], 1, gdal.GDT_Byte,
                                              ["COMPRESS=DEFLATE"])
    ds.SetGeoTransform(geotransform)
    ds.SetProjection("EPSG:4326")
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.WriteArray(arr)
    ds = None
    return path


def assign_ids(srs_sample, pps_sample, seed):
    """Pool both samples (Experiment A first, each in draw order), shuffle
    them with `seed` and number them 1..n in shuffled order."""
    pooled = srs_sample + pps_sample
    for i, k in enumerate(np.random.default_rng(seed).permutation(len(pooled)), start=1):
        pooled[k]["id"] = i


def write_sample(sample, fields, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(sample, key=lambda s: s["id"]))
    print(f"Wrote {path} ({len(sample)} units)")


def main(block_name):
    block = DESIGN["blocks"][block_name]
    ss = DESIGN["sample_size"]
    if ss["allocation"] != "proportional":
        raise NotImplementedError("only proportional allocation is implemented")
    strata_def = {int(k): v for k, v in block["strata"].items()}
    strata = sorted(strata_def)
    cell = cea_grid.get_cell_by_id(block["cell_id"])

    work = os.path.join(paths.SAMPLING_WORK, block_name)
    out = os.path.join(paths.SAMPLING_RESULTS, block_name)
    os.makedirs(work, exist_ok=True)
    os.makedirs(out, exist_ok=True)

    # 1. data, grid-matched on the DEM grid
    wc_raw, dem = os.path.join(work, "worldcover_raw.tif"), os.path.join(work, "cop_dem.tif")
    wc = os.path.join(work, "worldcover_matched_dem_grid.tif")
    wc_items = fetch_pc_data.fetch_worldcover(cell.wgs84_bbox_buffered, wc_raw)
    dem_items = fetch_pc_data.fetch_copernicus_dem(cell.wgs84_bbox_buffered, dem)
    area_tally.grid_match_to_dem(wc_raw, dem, wc, crosses_antimeridian=cell.crosses_antimeridian)
    with open(os.path.join(out, "provenance.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "worldcover_items", "dem_items"])
        w.writerow([cell.cell_id, ";".join(wc_items), ";".join(dem_items)])

    # 2. per-pixel areas and stratum totals over the block window
    classes, a_p, s_p, valid, (r0, _, c0, _) = area_tally.per_pixel_areas(wc, dem, cell.wgs84_bbox)
    gt = gdal.Open(dem).GetGeoTransform()
    win_gt = (gt[0] + c0 * gt[1], gt[1], 0.0, gt[3] + r0 * gt[5], 0.0, gt[5])
    stratum = stratum_array(classes, valid, {c: v["worldcover"] for c, v in strata_def.items()})
    n_dem = {c: int(np.sum(stratum == c)) for c in strata}
    A_c = {c: float(np.sum(a_p[stratum == c])) for c in strata}
    S_c = {c: float(np.sum(s_p[stratum == c])) for c in strata}

    # 3. LAEA grid of Experiment A
    stratum_tif = write_geotiff(stratum, win_gt, os.path.join(work, "strata_dem_grid.tif"))
    px = DESIGN["experiment_a_srs"]["pixel_size_m"]
    laea_tif, proj4 = srs.reproject_to_laea(stratum_tif, cell, px, os.path.join(work, f"strata_laea_{px}m.tif"))
    laea, laea_gt = srs.read_laea(laea_tif)
    N_c = srs.stratum_counts(laea, strata)
    A_ea = {c: N_c[c] * px * px for c in strata}

    # 4. allocation
    n_srs = pps.proportional_allocation(A_ea, ss["n_total"])
    n_pps = pps.proportional_allocation(S_c, ss["n_total"])

    # 5. draws (row / col: Experiment A on the LAEA grid, Experiment B on the
    #    DEM grid within the block window)
    def lonlat_of(r, c):
        return win_gt[0] + (c + 0.5) * win_gt[1], win_gt[3] + (r + 0.5) * win_gt[5]

    srs_sample = srs.draw_srs_sample(laea, laea_gt, proj4, n_srs, block["seed_srs"])
    pps_sample = pps.draw_pps_sample(stratum, s_p, n_pps, block["seed_pps"], lonlat_of,
                                     DESIGN["experiment_b_pps"]["block_rows"])

    # 6. blind labelling ids
    assign_ids(srs_sample, pps_sample, block["seed_ids"])
    write_sample(srs_sample, SRS_FIELDS, os.path.join(out, "srs_sample.csv"))
    write_sample(pps_sample, PPS_FIELDS, os.path.join(out, "pps_sample.csv"))

    path = os.path.join(out, "stratum_totals.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stratum", "name", "worldcover_classes", "n_pixels_dem", "planar_area_m2",
                    "surface_area_m2", "inflation_pct", "n_pixels_laea", "planar_area_ea_m2",
                    "W_srs", "n_srs", "W_pps", "n_pps"])
        for c in strata:
            w.writerow([c, strata_def[c]["name"], ";".join(map(str, strata_def[c]["worldcover"])),
                        n_dem[c], f"{A_c[c]:.3f}", f"{S_c[c]:.3f}", f"{(S_c[c] / A_c[c] - 1) * 100:.4f}",
                        N_c[c], f"{A_ea[c]:.3f}",
                        f"{A_ea[c] / sum(A_ea.values()):.6f}", n_srs[c],
                        f"{S_c[c] / sum(S_c.values()):.6f}", n_pps[c]])
    print(f"Wrote {path}")
    with open(os.path.join(out, "laea_grid.txt"), "w") as f:
        f.write(f"{proj4}\npixel_size_m={px}\ngeotransform={laea_gt}\n")
    print(f"{block_name} ({cell.cell_id}): Experiment A n_c={n_srs}, Experiment B n_c={n_pps}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("block", choices=list(DESIGN["blocks"]))
    main(ap.parse_args().block)
