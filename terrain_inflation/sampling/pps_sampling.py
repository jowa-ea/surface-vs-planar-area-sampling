#!/usr/bin/env python
# coding: utf-8
"""
Experiment B (paper sec:stats-pps, sec:stats-estimators) -- probability-
proportional-to-size (PPS) sampling with per-pixel surface area s_p as the
size measure, and the design-based estimators of area and accuracy.

Selection: within stratum c, n_c pixels are drawn independently, with
replacement, with probability s_p / S_c, by the cumulative-total method
(Cochran, 1977, 9A.3). The draw runs in two passes over row blocks: pass 1
sums s_p per stratum and block (giving S_c), each uniform u ~ U[0, S_c) is
assigned to the block whose cumulative total first exceeds u, and pass 2
locates the pixel within that block by binary search.

Also holds the proportional allocation used by both experiments.
"""

import numpy as np

Z_95 = 1.959963984540054  # standard normal 0.975 quantile

SAMPLE_FIELDS = ["sample_id", "stratum", "row", "col", "lon", "lat", "surface_area_m2"]


def draw_pps_sample(stratum, s_p, n_per_stratum, seed, lonlat_of, block_rows=2000):
    """Draw n_per_stratum[c] pixels per stratum, with replacement, with
    probability proportional to s_p.

    stratum, s_p : 2-D arrays over the block window.
    lonlat_of(row, col) -> (lon, lat) of a pixel centre (window indices).
    Returns a list of sample dicts (SAMPLE_FIELDS).
    """
    rows_total = stratum.shape[0]
    blocks = [(r, min(block_rows, rows_total - r)) for r in range(0, rows_total, block_rows)]
    codes = sorted(n_per_stratum)

    # Pass 1: surface area per stratum and row block.
    block_sums = {c: [float(np.sum(s_p[r0:r0 + n][stratum[r0:r0 + n] == c])) for r0, n in blocks]
                  for c in codes}

    rng = np.random.default_rng(seed)
    pending = {i: [] for i in range(len(blocks))}
    for c in codes:
        S_c, n_c = float(np.sum(block_sums[c])), n_per_stratum[c]
        if S_c <= 0 or n_c <= 0:
            continue
        cum = np.cumsum(block_sums[c])
        for k, u in enumerate(rng.uniform(0.0, S_c, size=n_c)):
            i = min(int(np.searchsorted(cum, u, side="right")), len(blocks) - 1)
            pending[i].append((c, k, u - (cum[i - 1] if i > 0 else 0.0)))

    # Pass 2: locate each draw within its block.
    sample = []
    for i, (r0, n) in enumerate(blocks):
        if not pending[i]:
            continue
        st, sa = stratum[r0:r0 + n], s_p[r0:r0 + n]
        ncols = st.shape[1]
        for c, k, residual in pending[i]:
            idx = np.flatnonzero(st.ravel() == c)
            vals = sa.ravel()[idx]
            pos = min(int(np.searchsorted(np.cumsum(vals), residual, side="right")), len(idx) - 1)
            r, col = divmod(int(idx[pos]), ncols)
            lon, lat = lonlat_of(r0 + r, col)
            sample.append({"sample_id": f"{c}_{k}", "stratum": c, "row": r0 + r, "col": col,
                           "lon": lon, "lat": lat, "surface_area_m2": float(vals[pos])})
    sample.sort(key=lambda s: (s["stratum"], int(s["sample_id"].split("_")[1])))
    return sample


def proportional_allocation(W_area, n_total):
    """n_c = n_total * W_c with W_c = W_area[c] / sum(W_area), rounded by the
    largest-remainder method so that sum(n_c) == n_total exactly."""
    total = sum(W_area.values())
    exact = {c: n_total * W_area[c] / total for c in W_area}
    n = {c: int(np.floor(v)) for c, v in exact.items()}
    for c in sorted(exact, key=lambda c: exact[c] - n[c], reverse=True)[:n_total - sum(n.values())]:
        n[c] += 1
    return n


def _ci95(est, var):
    h = Z_95 * var ** 0.5
    return est - h, est + h


def estimate_area_and_accuracy(S_c, sample):
    """Area of each reference class and overall / user's / producer's
    accuracy, with variances and 95% confidence intervals.

    S_c : {stratum: surface-area total (m^2)}.
    sample : rows with integer "stratum" and "reference_label"; reference
        labels use the stratum codes.
    """
    strata = list(S_c)
    ref = sorted({r["reference_label"] for r in sample} | set(strata))
    n = {c: sum(1 for r in sample if r["stratum"] == c) for c in strata}
    S = sum(S_c.values())
    p = {(c, j): (sum(1 for r in sample if r["stratum"] == c and r["reference_label"] == j) / n[c]
                  if n[c] else 0.0) for c in strata for j in ref}

    area = {}
    for j in ref:
        est = sum(S_c[c] * p[(c, j)] for c in strata)
        var = sum(S_c[c] ** 2 * p[(c, j)] * (1 - p[(c, j)]) / n[c] for c in strata if n[c])
        area[j] = {"estimate_m2": est, "variance_m2": var, "ci95_m2": _ci95(est, var)}

    oa = sum(S_c[c] * p[(c, c)] for c in strata) / S
    var_oa = sum((S_c[c] / S) ** 2 * p[(c, c)] * (1 - p[(c, c)]) / n[c] for c in strata if n[c])

    ua = {}
    for c in strata:
        if not n[c]:
            ua[c] = {"estimate": float("nan"), "variance": float("nan"), "ci95": (float("nan"),) * 2, "n": 0}
            continue
        v = p[(c, c)] * (1 - p[(c, c)]) / n[c]
        ua[c] = {"estimate": p[(c, c)], "variance": v, "ci95": _ci95(p[(c, c)], v), "n": n[c]}

    # Producer's accuracy: ratio of two estimates, Taylor-linearised variance.
    # The numerator is the c = j summand of the denominator, so Cov = Var(numerator).
    pa = {}
    for j in ref:
        B, var_B = area[j]["estimate_m2"], area[j]["variance_m2"]
        if j not in strata or not n.get(j) or B <= 0:
            pa[j] = {"estimate": float("nan"), "variance": float("nan"), "ci95": (float("nan"),) * 2}
            continue
        A = S_c[j] * p[(j, j)]
        var_A = S_c[j] ** 2 * p[(j, j)] * (1 - p[(j, j)]) / n[j]
        r = A / B
        v = max((var_A * (1 - 2 * r) + r ** 2 * var_B) / B ** 2, 0.0)
        pa[j] = {"estimate": r, "variance": v, "ci95": _ci95(r, v)}

    return {
        "area": area,
        "overall_accuracy": {"estimate": oa, "variance": var_oa, "ci95": _ci95(oa, var_oa)},
        "users_accuracy": ua,
        "producers_accuracy": pa,
        "error_matrix": {(c, j): S_c[c] * p[(c, j)] / S for c in strata for j in ref},
    }
