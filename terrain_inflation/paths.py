"""
Repository paths and the pinned land-mask download.

  inputs/<part>/   inputs shipped with the repository (configuration, reference labels)
  results/<part>/  outputs of the published run
  work/            downloads, rasters and other intermediate files (not tracked);
                   relocate with the TERRAIN_WORK_DIR environment variable
with <part> = global_inflation or sampling_comparison. Every script also
takes explicit paths on the command line, so none of these defaults is
required.
"""

import hashlib
import os

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = os.path.join(REPO_ROOT, "inputs")
RESULTS = os.path.join(REPO_ROOT, "results")
WORK_DIR = os.environ.get("TERRAIN_WORK_DIR") or os.path.join(REPO_ROOT, "work")
CACHE_DIR = os.path.join(WORK_DIR, "cache")

# Global surface-area underestimation
GLOBAL_INPUTS = os.path.join(INPUTS, "global_inflation")
GLOBAL_RESULTS = os.path.join(RESULTS, "global_inflation")
GLOBAL_WORK = os.path.join(WORK_DIR, "global_inflation")
WORLDCOVER_CLASSES_YAML = os.path.join(GLOBAL_INPUTS, "worldcover_classes.yaml")
PUBLISHED_MANIFEST = os.path.join(GLOBAL_RESULTS, "global_land_cells_manifest.csv")

# Sampling comparison
SAMPLING_INPUTS = os.path.join(INPUTS, "sampling_comparison")
SAMPLING_RESULTS = os.path.join(RESULTS, "sampling_comparison")
SAMPLING_WORK = os.path.join(WORK_DIR, "sampling_comparison")

# Natural Earth 1:50m land polygons, v4.1.0 -- the version used for the
# published 17,505-cell manifest.
LAND_URL = "https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip"
LAND_SHA256 = "0b8e670cf80dce9cbebe2a193bc44ba5602758c22e1fa603980553646d7ff162"
LAND_ZIP = os.path.join(CACHE_DIR, "ne_50m_land.zip")


def ensure_land_polygons():
    """Download the land polygons once and verify their checksum. A
    mismatch means Natural Earth has published a newer release; the cell
    count may then differ from the published manifest."""
    if not os.path.exists(LAND_ZIP):
        print(f"Downloading {LAND_URL} ...")
        r = requests.get(LAND_URL, timeout=120)
        r.raise_for_status()
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(LAND_ZIP, "wb") as f:
            f.write(r.content)
    with open(LAND_ZIP, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != LAND_SHA256:
        print(f"[WARN] {LAND_ZIP} has SHA-256 {digest}, expected {LAND_SHA256} "
              f"(Natural Earth v4.1.0). Results may differ from the published run.")
    return LAND_ZIP
