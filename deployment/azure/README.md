# Running the global surface-area step on Azure Machine Learning

The global run processes 17,505 land cells. Each cell takes about 13 s
(median 12.7 s, 40 MB read from the Planetary Computer), about 58
thread-hours in total: roughly 15 hours on one machine with 4 workers, or
under 1 hour fanned out over 20 cloud jobs. The work is dominated by network
reads, not CPU, so many small nodes beat a few large ones.

Azure is optional. The core command runs anywhere (see the main README):

```bash
python -m terrain_inflation.global_inflation.run --shard 3 --n-shards 20 --workers 4
```

This folder only fans that command out over an Azure ML compute cluster.
All cloud specifics (workspace, machine size, node count, image, fan-out,
wave size) are in a config file, not in the code:

| File | Role |
|---|---|
| `config.example.yaml` | template of the config; copy to `config.yaml` (git-ignored) |
| `orchestrate.py` | `setup`, `submit`, `collect`, `status`; builds one command job per shard from the config and calls the `az ml` CLI |

## Recommended settings

| Config key | Recommendation | Why |
|---|---|---|
| region (of the workspace) | West Europe | Planetary Computer data are hosted in West Europe; reads in the same region are fastest and free of egress |
| `compute.vm_size` | `Standard_D8ds_v5` (8 vCPU, 32 GB) | A cell holds 10-17 M DEM pixels and several float64 arrays of that size, and `workers_per_job` cells are processed at once. The published run used `Standard_D16d_v5` (16 vCPU, 64 GB); peak memory was not measured |
| `compute.min_instances` / `max_instances` | 0 / 20 | scales to zero when idle; one node per running shard |
| `compute.tier` | `dedicated`, or `low_priority` if you accept re-submitting pre-empted shards | a shard takes under an hour |
| `run.n_shards`, `run.workers_per_job` | 20, 4 | the published run |
| `run.wave_size` | 4 | 20 jobs starting at once trigger HTTP 429 rate limits on the Planetary Computer STAC API |

## Usage

Requires the Azure CLI with the `ml` extension (`az extension add -n ml`),
`az login`, and an existing Azure ML workspace. From the repository root:

```bash
cp deployment/azure/config.example.yaml deployment/azure/config.yaml   # then edit
python deployment/azure/orchestrate.py setup                    # once: cluster, environment, manifest asset
python deployment/azure/orchestrate.py submit --run-tag myrun   # all shards, in waves; waits
python deployment/azure/orchestrate.py status --run-tag myrun
python deployment/azure/orchestrate.py collect --run-tag myrun  # -> work/global_inflation/
python -m terrain_inflation.global_inflation.summarize --area-stats work/global_inflation/global_area_stats.csv
```

`--dry-run` (before the action) prints the `az ml` commands and writes the
job specifications to `work/azure/<run-tag>/` without submitting anything.
`setup` registers the environment from the repository's `environment.yml`
and the published manifest as the data asset (both paths are set in the
config).

The job's code snapshot is the repository root without the folders listed in
`.amlignore` (results, inputs of the sampling comparison, `work/`,
`deployment/`), so only `terrain_inflation/` and
`inputs/global_inflation/` are uploaded.

## Failed or incomplete shards

Re-submit the shard, e.g. `submit --run-tag myrun --shards 7`. Each Azure ML
job writes to a fresh output folder, so the new job processes the whole shard
again (under an hour); `jobs.csv` then points to the new job and `collect`
downloads only that one. Resuming within a shard (skipping cells already
settled) applies when the core command is re-run into the same output
directory, e.g. on a local machine.
