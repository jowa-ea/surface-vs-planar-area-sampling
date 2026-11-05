#!/usr/bin/env python
# coding: utf-8
"""
Experiment A (paper sec:methods-srs) -- conventional stratified random
sampling on an equal-area grid.

1. The stratum map of the block (grid-matched WorldCover on the DEM grid,
   with pixels outside the population set to 0; prepare_block.stratum_array)
   is reprojected by nearest neighbour to a Lambert azimuthal equal-area
   (LAEA) projection on the WGS84 ellipsoid, centred on the block centroid,
   with square pixels of side `pixel_size_m` aligned to multiples of the
   pixel size (gdalwarp -tap). Because only in-block pixels carry a stratum,
   the population is the set of LAEA pixels whose nearest DEM-grid pixel lies
   in the block.
2. Stratum c has N_c LAEA pixels of area a_0 = pixel_size_m^2 and planar area
   A_c^EA = N_c a_0.
3. Within each stratum, n_c pixels are drawn with equal probability, without
   replacement.
4. Estimators of Olofsson et al. (2014) with stratum weights
   W_c = A_c^EA / sum A^EA, including the n_c - 1 variance denominators.

"""

import subprocess

import numpy as np
import pyproj
from osgeo import gdal

gdal.UseExceptions()

Z_95 = 1.959963984540054
SAMPLE_FIELDS = ["sample_id", "stratum", "row", "col", "x_laea", "y_laea", "lon", "lat"]


def laea_proj4(lon0, lat0):
    return f"+proj=laea +lat_0={lat0} +lon_0={lon0} +ellps=WGS84 +units=m +no_defs"


def reproject_to_laea(stratum_tif, cell, pixel_size_m, out_path):
    """Nearest-neighbour reprojection of the stratum map to the block's LAEA
    grid. Returns (out_path, proj4)."""
    lon0, lat0 = cell.centroid_lonlat
    proj4 = laea_proj4(lon0, lat0)
    to_laea = pyproj.Transformer.from_crs("EPSG:4326", proj4, always_xy=True)
    xmin, ymin, xmax, ymax = to_laea.transform_bounds(*cell.wgs84_bbox, densify_pts=101)
    cmd = [
        "gdalwarp", "-t_srs", proj4,
        "-te", str(xmin), str(ymin), str(xmax), str(ymax),
        "-tr", str(pixel_size_m), str(pixel_size_m), "-tap",
        "-r", "near", "-ot", "Byte", "-srcnodata", "0", "-dstnodata", "0",
        "-co", "COMPRESS=DEFLATE", "-overwrite", stratum_tif, out_path,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_path, proj4


def read_laea(path):
    ds = gdal.Open(path)
    arr, gt = ds.GetRasterBand(1).ReadAsArray(), ds.GetGeoTransform()
    ds = None
    return arr, gt


def stratum_counts(arr, strata):
    """{stratum: N_c} on the LAEA grid."""
    return {c: int(np.sum(arr == c)) for c in strata}


def draw_srs_sample(arr, gt, proj4, n_per_stratum, seed):
    """n_per_stratum[c] LAEA pixels per stratum, equal probability, without
    replacement (capped at N_c)."""
    rng = np.random.default_rng(seed)
    to_wgs84 = pyproj.Transformer.from_crs(proj4, "EPSG:4326", always_xy=True)
    flat = arr.ravel()
    ncols = arr.shape[1]
    sample = []
    for c in sorted(n_per_stratum):
        idx = np.flatnonzero(flat == c)
        n_c = min(n_per_stratum[c], idx.size)
        if n_c <= 0:
            continue
        for k, i in enumerate(rng.choice(idx, size=n_c, replace=False)):
            r, col = divmod(int(i), ncols)
            x = gt[0] + (col + 0.5) * gt[1]
            y = gt[3] + (r + 0.5) * gt[5]
            lon, lat = to_wgs84.transform(x, y)
            sample.append({"sample_id": f"{c}_{k}", "stratum": c, "row": r, "col": col,
                           "x_laea": x, "y_laea": y, "lon": lon, "lat": lat})
    return sample


def _ci95(est, var):
    h = Z_95 * var ** 0.5
    return est - h, est + h


def estimate_area_and_accuracy(A_ea, sample):
    """Olofsson et al. (2014) stratified estimators.

    A_ea : {stratum: A_c^EA (m^2)}.
    sample : rows with integer "stratum" and "reference_label" (map class codes).
    """
    nan = float("nan")
    strata = list(A_ea)
    ref = sorted({r["reference_label"] for r in sample} | set(strata))
    A = sum(A_ea.values())
    W = {c: A_ea[c] / A for c in strata}
    n = {c: sum(1 for r in sample if r["stratum"] == c) for c in strata}
    p = {(c, j): (sum(1 for r in sample if r["stratum"] == c and r["reference_label"] == j) / n[c]
                  if n[c] else 0.0) for c in strata for j in ref}

    def v(c, x):  # x(1-x)/(n_c-1), Olofsson et al. (2014)
        return x * (1 - x) / (n[c] - 1) if n[c] > 1 else nan

    area = {}
    for j in ref:
        est = sum(A_ea[c] * p[(c, j)] for c in strata)
        var = sum(A_ea[c] ** 2 * v(c, p[(c, j)]) for c in strata if n[c])
        area[j] = {"estimate_m2": est, "variance_m2": var, "ci95_m2": _ci95(est, var)}

    oa = sum(W[c] * p[(c, c)] for c in strata)
    var_oa = sum(W[c] ** 2 * v(c, p[(c, c)]) for c in strata if n[c])

    ua = {}
    for c in strata:
        if not n[c]:
            ua[c] = {"estimate": nan, "variance": nan, "ci95": (nan, nan), "n": 0}
            continue
        ua[c] = {"estimate": p[(c, c)], "variance": v(c, p[(c, c)]),
                 "ci95": _ci95(p[(c, c)], v(c, p[(c, c)])), "n": n[c]}

    # Producer's accuracy, Olofsson et al. (2014, Eq. 7), stratum sizes as areas.
    pa = {}
    for j in ref:
        N_j = sum(A_ea[c] * p[(c, j)] for c in strata)  # estimated area of reference class j
        if j not in strata or not n.get(j) or N_j <= 0:
            pa[j] = {"estimate": nan, "variance": nan, "ci95": (nan, nan)}
            continue
        PA = A_ea[j] * p[(j, j)] / N_j
        term_j = A_ea[j] ** 2 * (1 - PA) ** 2 * v(j, p[(j, j)])
        term_other = PA ** 2 * sum(A_ea[c] ** 2 * v(c, p[(c, j)]) for c in strata if c != j and n[c])
        var = (term_j + term_other) / N_j ** 2
        pa[j] = {"estimate": PA, "variance": var, "ci95": _ci95(PA, var)}

    return {
        "area": area,
        "overall_accuracy": {"estimate": oa, "variance": var_oa, "ci95": _ci95(oa, var_oa)},
        "users_accuracy": ua,
        "producers_accuracy": pa,
        "error_matrix": {(c, j): W[c] * p[(c, j)] for c in strata for j in ref},
    }
