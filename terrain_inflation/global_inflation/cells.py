"""
Step 1 -- enumerate the land cells of the global 100 km CEA grid.

A cell is kept if it intersects the Natural Earth 1:50m land polygons. Cells
straddling +-180 deg are tested as two half-boxes on either side of the
meridian. Cells are listed row by row, west to east; this order defines the
shards of step 2 (cell k belongs to shard k mod n_shards).

Output: --out (default work/global_inflation/global_land_cells_manifest.csv;
the published manifest is results/global_inflation/global_land_cells_manifest.csv)

Usage: python -m terrain_inflation.global_inflation.cells [--out FILE]
"""

import argparse
import csv
import math
import os

import geopandas as gpd
from shapely.geometry import box
from shapely.strtree import STRtree

from .. import cea_grid, paths

FIELDNAMES = ["cell_id", "row", "col", "centroid_lon", "centroid_lat",
              "lon_min", "lat_min", "lon_max", "lat_max",
              "lon_min_buf", "lat_min_buf", "lon_max_buf", "lat_max_buf"]


def _row_col_range():
    x_min, y_min = cea_grid.to_cea.transform(-180.0, -90.0)
    x_max, y_max = cea_grid.to_cea.transform(180.0, 90.0)
    s = cea_grid.CELL_SIZE_M
    return int(y_min // s), int(y_max // s), int(x_min // s), int(x_max // s)


def enumerate_land_cells(out):
    land = gpd.read_file(f"zip://{paths.ensure_land_polygons()}")
    geoms = list(land.geometry)
    tree = STRtree(geoms)
    s = cea_grid.CELL_SIZE_M

    row_min, row_max, col_min, col_max = _row_col_range()
    cells = []
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            lon_min, lat_min = cea_grid.to_wgs84.transform(col * s, row * s)
            lon_max, lat_max = cea_grid.to_wgs84.transform((col + 1) * s, (row + 1) * s)
            # The outermost rows exceed the CEA domain at the poles.
            if not all(math.isfinite(v) for v in (lon_min, lat_min, lon_max, lat_max)):
                continue
            if lon_max < lon_min:  # straddles +-180
                test_boxes = [box(lon_min, lat_min, 180.0, lat_max), box(-180.0, lat_min, lon_max, lat_max)]
            else:
                test_boxes = [box(lon_min, lat_min, lon_max, lat_max)]
            if not any(geoms[i].intersects(tb) for tb in test_boxes for i in tree.query(tb)):
                continue

            c_lon, c_lat = cea_grid.to_wgs84.transform((col + 0.5) * s, (row + 0.5) * s)
            b = cea_grid.DEM_BUFFER_DEG
            cells.append({
                "cell_id": f"row{row}_col{col}", "row": row, "col": col,
                "centroid_lon": c_lon, "centroid_lat": c_lat,
                "lon_min": lon_min, "lat_min": lat_min, "lon_max": lon_max, "lat_max": lat_max,
                "lon_min_buf": lon_min - b, "lat_min_buf": lat_min - b,
                "lon_max_buf": lon_max + b, "lat_max_buf": lat_max + b,
            })

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(cells)
    print(f"Wrote {out}: {len(cells)} land cells")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(paths.GLOBAL_WORK, "global_land_cells_manifest.csv"))
    enumerate_land_cells(ap.parse_args().out)
