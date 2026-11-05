"""
Step 2 -- area and accuracy estimates of one study block and the comparison
of the two designs (paper, Sections "Comparison of designs" and "Sample-based
area estimation").

Inputs:
  inputs/sampling_comparison/sampling_design.yaml
  inputs/sampling_comparison/reference_labels/<block>.csv  reference label per sample unit id
Results of step 1 (results/sampling_comparison/<block>/):
  stratum_totals.csv                    A_c, S_c (DEM grid), A_c^EA (LAEA grid)
  srs_sample.csv, pps_sample.csv        sample units with their map stratum
Each sample unit is matched to its reference label by id (its lat / lon must
agree to 1e-6 deg).

Results (results/sampling_comparison/<block>/):
  estimates_srs.csv  Experiment A: StRS on the equal-area grid, weights A_c^EA
  estimates_pps.csv  Experiment B: PPS by surface area, weights S_c
      quantity = area_ha (per reference class), overall_accuracy,
      users_accuracy / producers_accuracy (per class), and
      error_matrix_<map stratum> (per reference class; proportion of the
      block's total area), each with standard error and 95% CI
  comparison.csv     target class: wall-to-wall map totals, both estimates,
                     their difference S_hat - A_hat (absolute and relative to
                     A_hat) and the two-sided z-test

Usage: python -m terrain_inflation.sampling.estimate_block klamath_forest
"""

import argparse
import csv
import math
import os

import yaml

from .. import paths
from . import pps_sampling as pps
from . import srs_equal_area as srs


with open(os.path.join(paths.SAMPLING_INPUTS, "sampling_design.yaml")) as _f:
    DESIGN = yaml.safe_load(_f)

FIELDS = ["quantity", "class", "estimate", "standard_error", "ci95_low", "ci95_high"]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def labelled_sample(sample_csv, labels):
    """Sample units with integer stratum (map) and reference_label."""
    rows = []
    for r in read_csv(sample_csv):
        ref = labels[r["id"]]
        if abs(float(r["lat"]) - float(ref["lat"])) > 1e-6 or abs(float(r["lon"]) - float(ref["lon"])) > 1e-6:
            raise ValueError(f"{sample_csv}: unit {r['id']} lies elsewhere than its reference label")
        rows.append({"id": r["id"], "stratum": int(r["stratum"]),
                     "reference_label": int(ref["reference_label"])})
    return rows


def estimate_rows(res):
    """Flatten an estimator result: areas in ha, accuracies and error-matrix
    cells as fractions."""
    out = []
    for j, a in res["area"].items():
        out.append(["area_ha", j, a["estimate_m2"] / 1e4, a["variance_m2"] ** 0.5 / 1e4,
                    a["ci95_m2"][0] / 1e4, a["ci95_m2"][1] / 1e4])
    oa = res["overall_accuracy"]
    out.append(["overall_accuracy", "", oa["estimate"], oa["variance"] ** 0.5, *oa["ci95"]])
    for key in ("users_accuracy", "producers_accuracy"):
        for c, v in res[key].items():
            out.append([key, c, v["estimate"], v["variance"] ** 0.5, *v["ci95"]])
    for (c, j), p in sorted(res["error_matrix"].items()):
        out.append([f"error_matrix_{c}", j, p, "", "", ""])
    return out


def write(rows, header, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Wrote {path}")


def main(block_name):
    target = DESIGN["blocks"][block_name]["target_class"]
    d = os.path.join(paths.SAMPLING_RESULTS, block_name)
    labels = {r["id"]: r for r in read_csv(os.path.join(paths.SAMPLING_INPUTS, "reference_labels",
                                                        f"{block_name}.csv"))}
    totals = {int(r["stratum"]): r for r in read_csv(os.path.join(d, "stratum_totals.csv"))}
    S_c = {c: float(r["surface_area_m2"]) for c, r in totals.items()}
    A_ea = {c: float(r["planar_area_ea_m2"]) for c, r in totals.items()}

    est = {}
    for exp, estimator, weights in (("srs", srs.estimate_area_and_accuracy, A_ea),
                                    ("pps", pps.estimate_area_and_accuracy, S_c)):
        res = estimator(weights, labelled_sample(os.path.join(d, f"{exp}_sample.csv"), labels))
        write(estimate_rows(res), FIELDS, os.path.join(d, f"estimates_{exp}.csv"))
        est[exp] = res

    a, s = est["srs"]["area"][target], est["pps"]["area"][target]
    A_hat, S_hat = a["estimate_m2"] / 1e4, s["estimate_m2"] / 1e4
    se_A, se_S = a["variance_m2"] ** 0.5 / 1e4, s["variance_m2"] ** 0.5 / 1e4
    diff, se_diff = S_hat - A_hat, math.hypot(se_A, se_S)
    z = diff / se_diff
    t = totals[target]
    rows = [
        ["map_planar_A_c_ha", float(t["planar_area_m2"]) / 1e4, ""],
        ["map_surface_S_c_ha", float(t["surface_area_m2"]) / 1e4, ""],
        ["map_planar_laea_A_c_EA_ha", float(t["planar_area_ea_m2"]) / 1e4, ""],
        ["estimate_A_srs_planar_ha", A_hat, se_A],
        ["estimate_B_pps_surface_ha", S_hat, se_S],
        ["difference_B_minus_A_ha", diff, se_diff],
        ["difference_B_minus_A_pct_of_A", diff / A_hat * 100, se_diff / A_hat * 100],
        ["z", z, ""],
        ["p_two_sided", math.erfc(abs(z) / math.sqrt(2)), ""],
    ]
    write(rows, [f"quantity (target class {target})", "value", "standard_error"],
          os.path.join(d, "comparison.csv"))
    print(f"{block_name}: A_hat {A_hat:,.0f} +- {se_A:,.0f} ha, S_hat {S_hat:,.0f} +- {se_S:,.0f} ha, "
          f"difference {diff:,.0f} +- {se_diff:,.0f} ha ({diff / A_hat * 100:.1f} +- "
          f"{se_diff / A_hat * 100:.1f}%), z = {z:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("block", choices=list(DESIGN["blocks"]))
    main(ap.parse_args().block)
