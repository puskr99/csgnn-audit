"""Revision experiment D: diagnostics of the exact degree-preserving rewiring (rewire_exact.py), one random graph per
split (rng seed 20000+k, as in extra_runs.py --stage rewire_exact). For each split: accepted-swap rate, share of original
edges left, configuration-model expectation of that share (sum_e d_out(u) d_in(v) / |E|^2), exactness check, and how
much of the information the GNNs aggregate survives: Pearson correlation, over nodes, between neighbour means of the
recipe features on the original and on the rewired graph (median and IQR over features), and between each node's
share of fraud-labelled neighbours (all classes > 0) before and after, plus that share for fraud and benign nodes."""
import json, numpy as np
from common import load
import csboost_v5 as V
from rewire_exact import rewire_exact, check
out = []
for ds in ["sichuan", "bupt"]:
    for k in range(5):
        X, y, s, d, tr, va, te = load(ds, k); n = len(y)
        s2, d2, acc = rewire_exact(s, d, n, k); ok, left = check(s, d, s2, d2, n)
        do, di = np.bincount(s, minlength=n), np.bincount(d, minlength=n)
        expect = float(np.sum(do[s] * di[d]) / len(s) ** 2)
        Xb = V.base_features(X, s, d, tr)
        A1 = V.operators(s, d, n, "sym1")[0]; A2 = V.operators(s2, d2, n, "sym1")[0]
        M1, M2 = np.asarray(A1 @ Xb), np.asarray(A2 @ Xb)
        has = np.asarray(A1.sum(1)).ravel() > 0
        cors = []
        for j in range(Xb.shape[1]):
            a, b = M1[has, j], M2[has, j]
            if a.std() > 0 and b.std() > 0: cors.append(np.corrcoef(a, b)[0, 1])
        cors = np.array(cors)
        f = (y > 0).astype(float); F1, F2 = A1 @ f, A2 @ f
        rec = dict(dataset=ds, split=k, rng_seed=20000 + k, accepted_swaps=acc, original_edges_left=left,
                   config_model_expectation=expect, exact=bool(ok),
                   neighbour_mean_corr_median=float(np.median(cors)), neighbour_mean_corr_q25=float(np.quantile(cors, .25)),
                   neighbour_mean_corr_q75=float(np.quantile(cors, .75)),
                   fraud_share_corr=float(np.corrcoef(F1[has], F2[has])[0, 1]),
                   fraud_share_fraudnodes_orig=float(F1[(y > 0) & has].mean()), fraud_share_fraudnodes_rw=float(F2[(y > 0) & has].mean()),
                   fraud_share_benign_orig=float(F1[(y == 0) & has].mean()), fraud_share_benign_rw=float(F2[(y == 0) & has].mean()))
        out.append(rec); print(json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in rec.items()}), flush=True)
json.dump(out, open("results_rev/rewire_diag.json", "w"), indent=1)
