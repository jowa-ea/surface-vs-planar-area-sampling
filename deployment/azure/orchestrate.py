"""
Run the global inflation step on an Azure ML compute cluster.

This is orchestration only: every job runs the hardware-independent core
command

    python -m terrain_inflation.global_inflation.run --shard i --n-shards n ...

on one slice of the cells. Workspace, cluster, machine size, image and
fan-out all come from the config file (config.example.yaml); nothing about
the cloud setup is in the code.

  setup    create the compute cluster, the environment and the manifest data
           asset described in the config (once per workspace)
  submit   submit one job per shard, in waves of run.wave_size jobs, and wait;
           job names go to work/azure/<run-tag>/jobs.csv
  collect  download the outputs of the completed jobs and merge them with
           terrain_inflation.global_inflation.aggregate into work/global_inflation/
  status   status of every job of a run

Usage (from the repository root, Azure CLI with the `ml` extension logged in):
  python deployment/azure/orchestrate.py --config deployment/azure/config.yaml setup
  python deployment/azure/orchestrate.py --config deployment/azure/config.yaml submit --run-tag myrun
  python deployment/azure/orchestrate.py --config deployment/azure/config.yaml collect --run-tag myrun
Add --dry-run to print the az commands (and write the job specs) without
running them. Re-submit failed shards with e.g. `submit --run-tag myrun --shards 7`.
"""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from terrain_inflation import paths  # noqa: E402
from terrain_inflation.global_inflation import aggregate  # noqa: E402

FINAL_STATES = ("Completed", "Failed", "Canceled")


class Azure:
    """Thin wrapper around `az ml` with the workspace arguments of the config."""

    def __init__(self, cfg, dry_run=False):
        ws = cfg["workspace"]
        self.ws_args = ["--resource-group", ws["resource_group"], "--workspace-name", ws["name"]]
        if ws.get("subscription"):
            self.ws_args += ["--subscription", ws["subscription"]]
        self.dry_run = dry_run
        self.az = shutil.which("az") or "az"

    def __call__(self, *args, query=None, check=True):
        cmd = [self.az, "ml", *args, *self.ws_args]
        if query:
            cmd += ["--query", query, "-o", "tsv"]
        if self.dry_run:
            print("[dry-run] az ml " + " ".join(args))
            return "dry-run"
        r = subprocess.run(cmd, capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"az ml {' '.join(args)} failed:\n{r.stderr}")
        return r.stdout.strip()


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(path))
    cfg["environment"]["conda_file"] = os.path.normpath(os.path.join(base, cfg["environment"]["conda_file"]))
    cfg["manifest"]["path"] = os.path.normpath(os.path.join(base, cfg["manifest"]["path"]))
    return cfg


def run_dir(tag):
    d = os.path.join(paths.WORK_DIR, "azure", tag)
    os.makedirs(d, exist_ok=True)
    return d


def parse_shards(spec, n_shards):
    """'0-19', '3', '1,4,7' -> sorted shard indices."""
    if not spec:
        return list(range(n_shards))
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    bad = [i for i in out if not 0 <= i < n_shards]
    if bad:
        raise SystemExit(f"shards {bad} outside 0..{n_shards - 1}")
    return sorted(out)


def job_spec(cfg, shard, tag):
    r = cfg["run"]
    command = (
        "python -m terrain_inflation.global_inflation.run"
        " --manifest ${{inputs.manifest}} --out-dir ${{outputs.results}}"
        f" --shard {shard} --n-shards {r['n_shards']}"
        f" --workers {r['workers_per_job']} --max-retries {r['max_retries']}"
        f" --scratch-dir {r['scratch_dir']}"
    )
    return {
        "$schema": "https://azuremlschemas.azureedge.net/latest/commandJob.schema.json",
        "type": "command",
        "code": REPO,  # uploaded without the folders listed in .amlignore
        "command": command,
        "environment": f"azureml:{cfg['environment']['name']}@latest",
        "compute": f"azureml:{cfg['compute']['name']}",
        "experiment_name": f"{r['experiment']}-{tag}",
        "display_name": f"{r['experiment']}-{tag}-shard-{shard}",
        "inputs": {"manifest": {"type": "uri_file", "mode": "ro_mount",
                                "path": f"azureml:{cfg['manifest']['asset_name']}@latest"}},
        "outputs": {"results": {"type": "uri_folder", "mode": "rw_mount"}},
    }


def read_jobs(tag):
    path = os.path.join(run_dir(tag), "jobs.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {int(r["shard"]): r["job"] for r in csv.DictReader(f)}


def write_jobs(tag, jobs):
    with open(os.path.join(run_dir(tag), "jobs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shard", "job"])
        w.writerows(sorted(jobs.items()))


def setup(cfg, az):
    c, e, m = cfg["compute"], cfg["environment"], cfg["manifest"]
    az("compute", "create", "--name", c["name"], "--type", "AmlCompute", "--size", c["vm_size"],
       "--min-instances", str(c["min_instances"]), "--max-instances", str(c["max_instances"]),
       "--tier", c["tier"])
    az("environment", "create", "--name", e["name"], "--image", e["image"], "--conda-file", e["conda_file"])
    az("data", "create", "--name", m["asset_name"], "--type", "uri_file", "--path", m["path"])
    print("setup done")


def submit(cfg, az, tag, shards):
    r = cfg["run"]
    jobs = read_jobs(tag)
    wave = []
    for k, shard in enumerate(shards, start=1):
        spec_path = os.path.join(run_dir(tag), f"job_shard_{shard}.yml")
        with open(spec_path, "w") as f:
            yaml.safe_dump(job_spec(cfg, shard, tag), f, sort_keys=False)
        name = az("job", "create", "--file", spec_path, query="name")
        print(f"shard {shard} -> {name}")
        if not az.dry_run:
            jobs[shard] = name
            write_jobs(tag, jobs)
        wave.append(name)
        if len(wave) == r["wave_size"] and k < len(shards):
            wait(az, wave, r["poll_interval_s"])
            wave = []
        elif k < len(shards):
            time.sleep(0 if az.dry_run else r["submit_delay_s"])
    wait(az, wave, r["poll_interval_s"])
    print(f"submitted {len(shards)} shards; job list {os.path.join(run_dir(tag), 'jobs.csv')}")


def wait(az, names, poll_s):
    if az.dry_run or not names:
        return
    print(f"waiting on {len(names)} jobs ...")
    while True:
        states = {n: az("job", "show", "--name", n, query="status", check=False) for n in names}
        if all(s in FINAL_STATES for s in states.values()):
            for n, s in states.items():
                print(f"  {n}: {s}")
            return
        time.sleep(poll_s)


def collect(cfg, az, tag):
    jobs = read_jobs(tag)
    if not jobs:
        raise SystemExit(f"no jobs recorded for run tag {tag}")
    d = run_dir(tag)
    shard_dir = os.path.join(d, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    for shard, name in sorted(jobs.items()):
        status = az("job", "show", "--name", name, query="status", check=False)
        if status != "Completed" and not az.dry_run:
            print(f"shard {shard} ({name}): {status} -- skipped")
            continue
        dl = os.path.join(d, "downloads", name)
        az("job", "download", "--name", name, "--output-name", "results", "--download-path", dl)
        for p in glob.glob(os.path.join(dl, "**", "shard_*_*.csv"), recursive=True):
            shutil.copy(p, shard_dir)
        print(f"shard {shard} ({name}): downloaded")
    if not az.dry_run:
        aggregate.aggregate(shard_dir, paths.GLOBAL_WORK)


def status(cfg, az, tag):
    for shard, name in sorted(read_jobs(tag).items()):
        print(f"shard {shard:>3}  {name}  {az('job', 'show', '--name', name, query='status', check=False)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("setup")
    p = sub.add_parser("submit")
    p.add_argument("--run-tag", required=True)
    p.add_argument("--shards", default=None, help="e.g. 0-19, 7 or 1,4,7 (default: all)")
    for name in ("collect", "status"):
        sub.add_parser(name).add_argument("--run-tag", required=True)
    a = ap.parse_args()

    cfg = load_config(a.config)
    az = Azure(cfg, a.dry_run)
    if a.action == "setup":
        setup(cfg, az)
    elif a.action == "submit":
        submit(cfg, az, a.run_tag, parse_shards(a.shards, cfg["run"]["n_shards"]))
    elif a.action == "collect":
        collect(cfg, az, a.run_tag)
    else:
        status(cfg, az, a.run_tag)


if __name__ == "__main__":
    main()
