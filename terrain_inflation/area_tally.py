"""
Grid matching and per-class planar / surface area tally over one CEA cell.

Used by both parts of the repository: the global assessment tallies every
land cell, and the sampling experiments use the same functions to obtain the
wall-to-wall stratum totals of their study blocks.
"""

import os
import subprocess

import numpy as np
import yaml
from osgeo import gdal

from . import fetch_pc_data, geodesy

gdal.UseExceptions()

def load_enabled_classes(classes_yaml):
    """{WorldCover code: class name} for the classes enabled in
    classes_yaml (inputs/global_inflation/worldcover_classes.yaml: all
    except permanent water)."""
    with open(classes_yaml) as f:
        cfg = yaml.safe_load(f)
    return {code: v["name"] for code, v in cfg["classes"].items() if v["enabled"]}


def grid_match_to_dem(map_path, dem_path, out_path, crosses_antimeridian=False,
                      nodata=fetch_pc_data.WORLDCOVER_NODATA):
    """Resample a categorical map onto the exact pixel grid of the DEM
    (same origin, pixel size and dimensions), nearest neighbour. This is the
    only resampling step of the pipeline."""
    ds = gdal.Open(dem_path)
    gt, xsize, ysize = ds.GetGeoTransform(), ds.RasterXSize, ds.RasterYSize
    ds = None
    xmin, ymax = gt[0], gt[3]
    xmax, ymin = xmin + xsize * gt[1], ymax + ysize * gt[5]
    t_srs = fetch_pc_data.LON_WRAP_180_SRS if crosses_antimeridian else "EPSG:4326"
    cmd = [
        "gdalwarp", "-t_srs", t_srs,
        "-te", str(xmin), str(ymin), str(xmax), str(ymax),
        "-ts", str(xsize), str(ysize),
        "-r", "near", "-ot", "Byte", "-dstnodata", str(nodata),
        "-co", "COMPRESS=DEFLATE", "-co", "BIGTIFF=YES", "-overwrite",
    ]
    if crosses_antimeridian:
        cmd += ["--config", "CENTER_LONG", "180"]
    cmd += [map_path, out_path]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_path


def in_cell_window(gt, xsize, ysize, bbox):
    """(row0, row1, col0, col1) -- the half-open pixel index ranges whose
    centres fall within the half-open lon/lat rectangle `bbox`."""
    lon_min, lat_min, lon_max, lat_max = bbox
    lon_centers = gt[0] + (np.arange(xsize) + 0.5) * gt[1]
    lat_centers = gt[3] + (np.arange(ysize) + 0.5) * gt[5]
    cols = np.flatnonzero((lon_centers >= lon_min) & (lon_centers < lon_max))
    rows = np.flatnonzero((lat_centers >= lat_min) & (lat_centers < lat_max))
    if cols.size == 0 or rows.size == 0:
        return 0, 0, 0, 0
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def per_pixel_areas(map_path, dem_path, bbox):
    """Per-pixel class value, planar area a_p and surface area s_p (m^2)
    for the pixels whose centre lies inside `bbox` (half-open) and whose 3x3
    elevation neighbourhood contains no void.

    Returns (classes, a_p, s_p, valid, window) where the arrays cover the
    window (row0, row1, col0, col1) of the DEM grid.
    """
    map_ds, dem_ds = gdal.Open(map_path), gdal.Open(dem_path)
    xsize, ysize = dem_ds.RasterXSize, dem_ds.RasterYSize
    gt = dem_ds.GetGeoTransform()
    classes = map_ds.GetRasterBand(1).ReadAsArray()
    z = dem_ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    map_ds = dem_ds = None

    z_pad = np.pad(z, 1, mode="edge")
    lat_centers = gt[3] + (np.arange(ysize) + 0.5) * gt[5]
    dx, dy = geodesy.ground_spacing_m(lat_centers, abs(gt[1]), abs(gt[5]))
    a_p = geodesy.pixel_area_m2(dx, dy, ysize, xsize)
    s_p = geodesy.jenness_surface_area(z_pad, dx, dy, ysize, xsize)
    void = geodesy.void_in_neighbourhood(z_pad, ysize, xsize)

    r0, r1, c0, c1 = in_cell_window(gt, xsize, ysize, bbox)
    win = (slice(r0, r1), slice(c0, c1))
    return classes[win], a_p[win], s_p[win], ~void[win], (r0, r1, c0, c1)


def compute_area_by_class(map_path, dem_path, cell, class_codes):
    """Planar and surface area (m^2) per class over the CEA cell.
    Returns (planar_sums, surface_sums), each {class_code: m^2}."""
    classes, a_p, s_p, valid, _ = per_pixel_areas(map_path, dem_path, cell.wgs84_bbox)
    planar, surface = {}, {}
    for c in class_codes:
        mask = valid & (classes == c)
        planar[c] = float(np.sum(a_p[mask]))
        surface[c] = float(np.sum(s_p[mask]))
    return planar, surface
