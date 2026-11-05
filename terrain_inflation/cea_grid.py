"""
Global 100 km x 100 km cylindrical equal-area (CEA) grid.

Cells are 10,000 km^2 squares in a single global CEA projection (WGS84,
true-scale latitude 30 deg, central meridian 0). Because the projection is
cylindrical with lon_0 = 0, projected x depends only on longitude and y only on
latitude, so every cell is an exact longitude/latitude rectangle. The cells
partition the land surface, and the cell itself is the footprint over which
areas are tallied: a pixel belongs to a cell if its centre lies in the cell's
half-open [min, max) lon/lat bounds. No raster is reprojected to CEA.

Cells straddling +-180 deg are expressed in a continuous longitude frame
(lon_max += 360) and flagged `crosses_antimeridian`.
"""

from dataclasses import dataclass

import pyproj

CEA_PROJ4 = "+proj=cea +lat_ts=30 +lon_0=0 +ellps=WGS84 +units=m +no_defs"
CELL_SIZE_M = 100_000.0

# Lon/lat buffer read around each cell so that edge pixels have a complete
# 3x3 elevation neighbourhood. Buffer pixels are never tallied.
DEM_BUFFER_DEG = 0.01

to_cea = pyproj.Transformer.from_crs("EPSG:4326", CEA_PROJ4, always_xy=True)
to_wgs84 = pyproj.Transformer.from_crs(CEA_PROJ4, "EPSG:4326", always_xy=True)


@dataclass
class Cell:
    cell_id: str
    row: int
    col: int
    cea_bounds: tuple           # (xmin, ymin, xmax, ymax), m
    centroid_lonlat: tuple      # (lon, lat)
    wgs84_bbox: tuple           # (lon_min, lat_min, lon_max, lat_max), exact cell
    wgs84_bbox_buffered: tuple  # same, expanded by DEM_BUFFER_DEG (data fetch only)
    crosses_antimeridian: bool = False


def get_cell(row, col):
    """Cell at grid indices (row, col)."""
    xmin, ymin = col * CELL_SIZE_M, row * CELL_SIZE_M
    xmax, ymax = xmin + CELL_SIZE_M, ymin + CELL_SIZE_M
    lon_min, lat_min = to_wgs84.transform(xmin, ymin)
    lon_max, lat_max = to_wgs84.transform(xmax, ymax)

    crosses = lon_max < lon_min
    if crosses:
        lon_max += 360.0

    c_lon, c_lat = to_wgs84.transform(xmin + CELL_SIZE_M / 2.0, ymin + CELL_SIZE_M / 2.0)
    return Cell(
        cell_id=f"row{row}_col{col}",
        row=row,
        col=col,
        cea_bounds=(xmin, ymin, xmax, ymax),
        centroid_lonlat=(c_lon, c_lat),
        wgs84_bbox=(lon_min, lat_min, lon_max, lat_max),
        wgs84_bbox_buffered=(lon_min - DEM_BUFFER_DEG, lat_min - DEM_BUFFER_DEG,
                             lon_max + DEM_BUFFER_DEG, lat_max + DEM_BUFFER_DEG),
        crosses_antimeridian=crosses,
    )


def get_cell_for_point(lon, lat):
    """Cell containing (lon, lat)."""
    x, y = to_cea.transform(lon, lat)
    return get_cell(int(y // CELL_SIZE_M), int(x // CELL_SIZE_M))


def get_cell_by_id(cell_id):
    """Cell from an id of the form 'row<r>_col<c>'."""
    r, c = cell_id.replace("row", "").split("_col")
    return get_cell(int(r), int(c))
