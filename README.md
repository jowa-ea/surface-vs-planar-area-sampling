# Accounting for relief in land-cover area estimation

Code and data for:

> Wagner, J., Skakun, S., Sawahn, R., Becker-Reshef, I. **Accounting for
> relief in land-cover area estimation: a global assessment and a
> surface-area sampling design.** [Journal] ([year]). DOI: [tbd]

Planar area in an equal-area projection ignores terrain relief. The
repository covers the two parts of the paper:

1. **Global surface-area underestimation**: for every 100 km land cell
   worldwide and each ESA WorldCover class, the planar area, the true
   (triangulated) surface area, and how much planar area underestimates
   surface area.
2. **Sampling comparison**: stratified random sampling on an equal-area grid
   (planar area) against stratified sampling with probability proportional
   to pixel surface area (PPS), in two 10,000 km² cells: sampling procedure,
   reference labels, and area and accuracy estimation.

## Repository layout

The code is split into a core package that does all the computation and
runs on any machine, and a thin deployment layer that fans the expensive
global step out over a compute cluster. Inputs and results are kept apart.

```
terrain_inflation/              core package: all computation, no cluster or machine assumptions
  cea_grid.py                   global 100 km cylindrical equal-area grid
  fetch_pc_data.py              WorldCover v200 / Copernicus DEM GLO-30 reads (Planetary Computer)
  geodesy.py                    ellipsoidal ground spacing, planar area, Jenness surface area
  area_tally.py                 grid matching, per-pixel and per-class area over a cell
  paths.py                      default locations of inputs/, results/, work/
  global_inflation/
    cells.py                    step 1: land cells -> manifest
    run.py                      step 2: planar + surface area per cell and class (all cells or one shard)
    aggregate.py                step 3: merge shard outputs
    summarize.py                step 4: per-class and per-cell underestimation
  sampling/
    prepare_block.py            step 1: strata, stratum totals, allocation, both samples
    estimate_block.py           step 2: area and accuracy estimates, design comparison
    srs_equal_area.py           Experiment A: stratified random sampling on an LAEA grid
    pps_sampling.py             Experiment B: PPS selection by surface area, estimators

deployment/azure/               orchestration of global_inflation.run on Azure ML
  config.example.yaml           all cloud specifics: workspace, VM size, nodes, image, fan-out
  orchestrate.py                setup / submit / status / collect, driven by the config
  README.md

inputs/                         shipped inputs
  global_inflation/worldcover_classes.yaml
  sampling_comparison/sampling_design.yaml
  sampling_comparison/reference_labels/<block>.csv

results/                        outputs of the published run (tables only)
  global_inflation/
  sampling_comparison/<block>/

work/                           downloads, rasters, intermediate files (created on demand, not tracked)
```

Every command takes explicit paths; the defaults point to the folders above.
`work/` can be moved with the `TERRAIN_WORK_DIR` environment variable.

## Setup

GDAL, including the `gdalwarp` command-line tool, is required. The simplest
way to get it is conda:

```bash
conda env create -f environment.yml
conda activate terrain-inflation
```

`requirements.txt` lists the same pinned versions for pip, if GDAL is
already installed. Python 3.13 was used. Run the commands below from the
repository root, or install the package with `pip install -e .` to run them
from anywhere. Inputs are read from the Microsoft Planetary Computer:
internet access is needed, no account or key.

Quick check (seconds, no downloads): recompute the published summaries and
estimates from the shipped tables; `git diff` should show no change.

```bash
python -m terrain_inflation.global_inflation.summarize
python -m terrain_inflation.sampling.estimate_block klamath_forest
python -m terrain_inflation.sampling.estimate_block ukraine_cropland
git diff --stat results/
```

## Part 1 -- Global surface-area underestimation

### Run

```bash
python -m terrain_inflation.global_inflation.cells                 # -> work/global_inflation/global_land_cells_manifest.csv
python -m terrain_inflation.global_inflation.run --workers 4       # all cells, one machine -> work/global_inflation/shards/
python -m terrain_inflation.global_inflation.aggregate             # -> work/global_inflation/
python -m terrain_inflation.global_inflation.summarize --area-stats work/global_inflation/global_area_stats.csv
```

`run` processes the published manifest unless `--manifest` is given. A full
run takes about 58 thread-hours. To spread it over several machines or a
scheduler of your choice, start one process per shard,
`run --shard i --n-shards n` for i = 0..n-1 (cell k belongs to shard k mod n;
the paper used n = 20); `aggregate` merges the shard outputs. A re-run into
the same output folder skips cells already done. `deployment/azure/` runs the
shards on an Azure ML compute cluster; see its README.

### Input

| Input | Source |
|---|---|
| `inputs/global_inflation/worldcover_classes.yaml` | WorldCover classes tallied (all except 80, permanent water) |
| ESA WorldCover 10 m 2021 v200 | Planetary Computer, collection `esa-worldcover` |
| Copernicus DEM GLO-30 | Planetary Computer, collection `cop-dem-glo-30` |
| Natural Earth 1:50m land polygons v4.1.0 | downloaded once, SHA-256 checked (`terrain_inflation/paths.py`); only used to list the land cells |

### Method

For each cell, WorldCover is resampled (nearest neighbour) onto the exact
pixel grid of the Copernicus DEM (EPSG:4326). On that grid, each pixel's
planar area follows from the WGS84 ellipsoidal ground spacing of its row,
and its surface area from the Jenness (2004) 8-triangle method with the same
spacing. Both are summed per class over the pixels whose centre lies in the
CEA cell and whose 3x3 elevation neighbourhood has no void. `summarize`
drops cell x class records below 100 ha planar area, then computes
the area-weighted relative underestimation of surface area by planar area,
sum(surface) / sum(planar) - 1, in % of planar area.

### Output (`results/global_inflation/`)

| File | Content |
|---|---|
| `global_land_cells_manifest.csv` | the 17,505 land cells: id, CEA row / column, centroid, lon / lat bounds (plain and buffered) |
| `global_status.csv` | outcome per cell: 16,088 `done`; 1,391 outside WorldCover; 7 in a Copernicus DEM tile gap; 19 small islands without a WorldCover tile |
| `global_area_stats.csv` | one row per cell x class: `planar_area_ha`, `surface_area_ha`, `inflation_ratio` (surface / planar), before the 100 ha floor |
| `global_class_summary.csv` | per class and all classes: planar and surface area (Mha), absolute (Mha) and relative (%) surface-area underestimation, number of cells (paper table) |
| `global_cell_summary.csv` | per cell, all classes combined: planar and surface area (ha), relative surface-area underestimation (%) (paper map) |

Published run: 16,059 cells after the 100 ha floor, 13,119.7 Mha planar,
13,427.8 Mha surface, +308.2 Mha (+2.35%).

## Part 2 -- Sampling comparison

### Run

```bash
python -m terrain_inflation.sampling.prepare_block klamath_forest    # -> results/sampling_comparison/klamath_forest/
python -m terrain_inflation.sampling.estimate_block klamath_forest   # -> estimates_*.csv, comparison.csv
```

and the same for `ukraine_cropland`. `prepare_block` downloads the map and
DEM of one cell (a few minutes; rasters in `work/sampling_comparison/`)
and, with the seeds in `sampling_design.yaml`, reproduces the published
samples exactly. `estimate_block` needs no download.

### Input

| Input | Content |
|---|---|
| `inputs/sampling_comparison/sampling_design.yaml` | blocks (CEA cell, strata as WorldCover classes, seeds), 400 units per experiment and block, proportional allocation, 30 m LAEA pixel size |
| `inputs/sampling_comparison/reference_labels/<block>.csv` | reference label of each of the 800 sample units of a block (both experiments) |
| WorldCover, Copernicus DEM | as in Part 1 |

| Block | Cell | Location | Target class | Share of block | Surface-area underestimation of target class |
|---|---|---|---|---|---|
| `klamath_forest` | `row48_col-119` | Klamath Mountains, California, USA | Tree cover | 75.5% | 10.4% |
| `ukraine_cropland` | `row54_col30` | Kirovohrad oblast, Ukraine | Cropland | 79.5% | 0.09% |

Reference label files:

| Column | Meaning |
|---|---|
| `id` | sample unit id, 1-800, shared by both experiments of a block |
| `lat`, `lon` | sample unit centre, WGS84 degrees (7 decimals) |
| `reference_label` | land cover in 2021: 1 = target class (tree cover / cropland), 2 = other |

The 800 units of both experiments were pooled and shuffled before labelling,
so interpreters did not know which design selected a unit or its map class.
Labels were assigned in STACNotator (https://github.com/RAAPID-ORG/stacnotator)
on Planet monthly global 3 m imagery of 2021.

### Output (`results/sampling_comparison/<block>/`)

| File | Content |
|---|---|
| `stratum_totals.csv` | per stratum: DEM-grid pixels, planar area A_c and surface area S_c (m²), relative surface-area underestimation (%), LAEA pixels N_c, planar area A_c^EA (m²), weights W_c and sample sizes n_c of both experiments |
| `srs_sample.csv` | Experiment A units: `id`, map `stratum`, LAEA `row` / `col`, `x_laea` / `y_laea`, `lon` / `lat` |
| `pps_sample.csv` | Experiment B units: `id`, map `stratum`, DEM-grid `row` / `col` within the block, `lon` / `lat`, pixel `surface_area_m2`. Drawn with replacement: a pixel can appear more than once, under different ids |
| `laea_grid.txt` | PROJ definition and geotransform of the Experiment A grid |
| `provenance.csv` | STAC item IDs read |
| `estimates_srs.csv`, `estimates_pps.csv` | per experiment: area of each reference class (ha), overall, user's and producer's accuracy, with SE and 95% CI; error matrix (`error_matrix_<map stratum>`, reference class in `class`) in proportion of block area |
| `comparison.csv` | target class: map totals, both estimates with SE, their difference (ha and % of the planar estimate) with SE, z and two-sided p |

Strata and reference labels share the codes 1 = target class, 2 = other.

## Data and licenses

The repository contains no raster data. Inputs read by the code:

- ESA WorldCover 10 m 2021 v200 (Zanaga et al., 2022), CC BY 4.0
- Copernicus DEM GLO-30, (c) DLR e.V. 2010-2014 and (c) Airbus Defence and
  Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA
- Natural Earth 1:50m land polygons v4.1.0, public domain
- WorldCover and the DEM are accessed through the Microsoft Planetary
  Computer STAC API

The reference labels in `inputs/sampling_comparison/reference_labels/` were
produced for this study by photo-interpretation of Planet imagery; the
imagery itself is not redistributed.

## Citation

[Citation tbd]

## License

[tbd]
