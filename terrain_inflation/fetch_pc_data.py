"""
Windowed reads of ESA WorldCover v200 and Copernicus DEM GLO-30 from the
Microsoft Planetary Computer (STAC search + signed URLs + GDAL /vsicurl/).

Only the requested window is read; no full tiles are downloaded. Each fetch
returns the STAC item IDs it used so that runs can record their exact inputs.
Requires network access to https://planetarycomputer.microsoft.com.
"""

import math
import random
import subprocess
import time

import planetary_computer
import pystac_client

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
WORLDCOVER_COLLECTION = "esa-worldcover"
WORLDCOVER_VERSION = "v200"  # 2021 map
COP_DEM_COLLECTION = "cop-dem-glo-30"
WORLDCOVER_NODATA = 0

# Target SRS for windows crossing +-180 deg: a 0..360 longitude branch, so
# gdalwarp mosaics source tiles from both sides of the seam into one raster.
LON_WRAP_180_SRS = "+proj=longlat +datum=WGS84 +no_defs +lon_wrap=180"

# Planetary Computer rate-limits STAC requests (HTTP 429) under heavy
# parallel load; such calls are retried with exponential backoff + jitter.
RATE_LIMIT_MAX_RETRIES = 6
RATE_LIMIT_BASE_SLEEP_S = 4.0

# How far around a cell to look for neighbouring DEM tiles when diagnosing a
# DEM fetch that returned no items.
DEM_GAP_DIAGNOSTIC_PAD_DEG = 1.5

_catalog_client = None
_worldcover_extent = None


def _is_rate_limit(exc):
    return isinstance(exc, pystac_client.exceptions.APIError) and "rate limit" in str(exc).lower()


def _retry_rate_limited(fn, *args, _what="STAC call", **kwargs):
    for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 -- re-raised unless it is a rate limit
            if not _is_rate_limit(e) or attempt == RATE_LIMIT_MAX_RETRIES:
                raise
            sleep_s = RATE_LIMIT_BASE_SLEEP_S * (2 ** attempt) + random.uniform(0.0, 3.0)
            print(f"[rate-limit] {_what}: retry {attempt + 1}/{RATE_LIMIT_MAX_RETRIES} after {sleep_s:.1f}s")
            time.sleep(sleep_s)


def _catalog():
    global _catalog_client
    if _catalog_client is None:
        _catalog_client = _retry_rate_limited(
            pystac_client.Client.open, STAC_URL,
            modifier=planetary_computer.sign_inplace, _what="catalog open",
        )
    return _catalog_client


def _search_items(**search_kwargs):
    return _retry_rate_limited(
        lambda: list(_catalog().search(**search_kwargs).items()),
        _what=f"search {search_kwargs.get('collections')}",
    )


def _crosses_antimeridian(bbox):
    lon_min, _, lon_max, _ = bbox
    return lon_min > lon_max or lon_max > 180.0


def _search_bboxes(bbox):
    """One search bbox, or two real [-180, 180] halves for a window crossing
    the antimeridian (given in the continuous frame, lon_max > 180)."""
    if not _crosses_antimeridian(bbox):
        return [list(bbox)]
    lon_min, lat_min, lon_max, lat_max = bbox
    west_hi = lon_max - 360.0 if lon_max > 180.0 else lon_max
    return [[lon_min, lat_min, 180.0, lat_max], [-180.0, lat_min, west_hi, lat_max]]


def get_worldcover_spatial_extent():
    """WorldCover's spatial extent, read from its STAC collection (no data
    south of ~60 S)."""
    global _worldcover_extent
    if _worldcover_extent is None:
        col = _retry_rate_limited(_catalog().get_collection, WORLDCOVER_COLLECTION,
                                  _what="get_collection esa-worldcover")
        bb = col.extent.spatial.bboxes
        _worldcover_extent = (min(b[0] for b in bb), min(b[1] for b in bb),
                              max(b[2] for b in bb), max(b[3] for b in bb))
    return _worldcover_extent


def bbox_has_worldcover_coverage(bbox):
    _, ext_lat_min, _, ext_lat_max = get_worldcover_spatial_extent()
    _, lat_min, _, lat_max = bbox
    return not (lat_max < ext_lat_min or lat_min > ext_lat_max)


def _run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _warp_mosaic(urls, bbox, out_path, extra_args, crosses_antimeridian):
    """Mosaic and clip the source rasters to `bbox` on their native grid
    (nearest neighbour, so no values are interpolated)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    cmd = [
        "gdalwarp",
        "-t_srs", LON_WRAP_180_SRS if crosses_antimeridian else "EPSG:4326",
        "-te", str(lon_min), str(lat_min), str(lon_max), str(lat_max),
        "-r", "near",
        "-co", "COMPRESS=DEFLATE", "-co", "BIGTIFF=YES", "-overwrite",
    ]
    if crosses_antimeridian:
        cmd += ["--config", "CENTER_LONG", "180"]
    cmd += extra_args
    cmd += [f"/vsicurl/{u}" for u in urls]
    cmd.append(out_path)
    _run(cmd)


def fetch_worldcover(bbox, out_path):
    """Write the WorldCover v200 window `bbox` (lon_min, lat_min, lon_max,
    lat_max) to out_path. Returns the sorted list of STAC item IDs used."""
    items = {}
    for sb in _search_bboxes(bbox):
        for it in _search_items(collections=[WORLDCOVER_COLLECTION], bbox=sb):
            if WORLDCOVER_VERSION in it.id:
                items[it.id] = it
    if not items:
        raise RuntimeError(f"No ESA WorldCover {WORLDCOVER_VERSION} items found for bbox {bbox}")
    _warp_mosaic([it.assets["map"].href for it in items.values()], bbox, out_path,
                 ["-ot", "Byte", "-dstnodata", str(WORLDCOVER_NODATA)],
                 _crosses_antimeridian(bbox))
    return sorted(items)


def fetch_copernicus_dem(bbox, out_path):
    """Write the Copernicus DEM GLO-30 window `bbox` to out_path. Returns the
    sorted list of STAC item IDs used."""
    items = {}
    for sb in _search_bboxes(bbox):
        for it in _search_items(collections=[COP_DEM_COLLECTION], bbox=sb):
            items[it.id] = it
    if not items:
        raise RuntimeError(f"No Copernicus DEM GLO-30 items found for bbox {bbox}")
    _warp_mosaic([it.assets["data"].href for it in items.values()], bbox, out_path,
                 ["-ot", "Float32"], _crosses_antimeridian(bbox))
    return sorted(items)


def _dem_tiles_covering_bbox(bbox):
    """Expected 1x1 deg Copernicus DEM tile IDs overlapping `bbox`."""
    lon_min, lat_min, lon_max, lat_max = bbox
    tiles = []
    for la in range(math.floor(lat_min), math.ceil(lat_max)):
        for lo_raw in range(math.floor(lon_min), math.ceil(lon_max)):
            lo = ((lo_raw + 180) % 360) - 180
            ns = f"N{la:02d}" if la >= 0 else f"S{abs(la):02d}"
            ew = f"E{lo:03d}" if lo >= 0 else f"W{abs(lo):03d}"
            tiles.append(f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM")
    return tiles


def diagnose_dem_gap(bbox):
    """Distinguish a permanent hole in the DEM collection from a transient
    empty search. `is_regional_gap` is True when every DEM tile the window
    needs is absent while neighbouring tiles exist."""
    exact = [it for sb in _search_bboxes(bbox)
             for it in _search_items(collections=[COP_DEM_COLLECTION], bbox=sb)]
    pad = DEM_GAP_DIAGNOSTIC_PAD_DEG
    lon_min, lat_min, lon_max, lat_max = bbox
    padded = (lon_min - pad, lat_min - pad, lon_max + pad, lat_max + pad)
    padded_items = [it for sb in _search_bboxes(padded)
                    for it in _search_items(collections=[COP_DEM_COLLECTION], bbox=sb)]
    padded_ids = {it.id for it in padded_items}
    expected = _dem_tiles_covering_bbox(bbox)
    missing = [t for t in expected if t not in padded_ids]
    return {
        "reconfirmed_empty": len(exact) == 0,
        "expected_tiles": expected,
        "missing_tiles": missing,
        "padded_items_found": len(padded_items),
        "is_regional_gap": len(exact) == 0 and len(missing) == len(expected) and len(padded_items) > 0,
    }
