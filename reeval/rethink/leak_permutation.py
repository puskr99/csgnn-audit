"""Label-permutation test for test-label leakage.

A leakage-free pipeline must give *identical* test predictions when the labels of test nodes are
shuffled (or replaced) before prediction, because those labels must never reach the prediction path.

  (1) CSGNN as released: train once with the published config, then score the test nodes with the
      published `cost_logits(logits, labels, ...)` using (a) true labels, (b) K random permutations of the
      test labels, (c) all test labels set to the majority class. If scores change -> leakage.
  (2) CS-Boost (ours): run the full pipeline twice, once with true test labels in `y` and once with test
      labels permuted; predicted probabilities on test nodes must be bit-identical.

Usage:  python leak_permutation.py --datasets sichuan,bupt --perms 5 --out results_gpu/leak_permutation.json
"""
import argparse, json, os, sys, numpy as np, torch
from argparse import Namespace
from common import load, metrics

p = argparse.ArgumentParser()
p.add_argument("--datasets", default="sichuan,bupt")
p.add_argument("--perms", type=int, default=5)
p.add_argument("--out", default="results_gpu/leak_permutation.json")
A = p.parse_args()
os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)

import csgnn_port as CP
import csboost_v5 as V
V.DEVICE = torch.device("cpu")   # bit-exact comparison needs deterministic CPU kernels

out = {}
for ds in A.datasets.split(","):
    rng = np.random.default_rng(0)
    # ---------------------------------------------------------------- (1) CSGNN as released
    a = Namespace(dataset=ds, max_epoch=None, seed=42, split=0, es=0, out_json=None)
    lg, Cm, y, tr, va, te, cfg, _ = CP.train_model(a)
    M = Cm.detach().numpy()

    def published_scores(labels_for_scoring):
        return torch.softmax(CP.cost_logits(lg, labels_for_scoring, M), 1).numpy()

    res = {"true_labels": metrics(y[te], published_scores(y)[te])}
    perm_rows, max_abs = [], 0.0
    P_true = published_scores(y)[te]
    for k in range(A.perms):
        yp = y.copy(); yp[te] = rng.permutation(y[te])
        Pk = published_scores(yp)[te]
        max_abs = max(max_abs, float(np.abs(Pk - P_true).max()))
        perm_rows.append(metrics(y[te], Pk))          # always evaluated against the TRUE labels
    res["permuted_test_labels_mean"] = {m: float(np.mean([r[m] for r in perm_rows])) for m in perm_rows[0]}
    yc = y.copy(); yc[te] = 0
    res["all_test_labels_set_to_0"] = metrics(y[te], published_scores(yc)[te])
    res["max_abs_change_in_test_probs_under_permutation"] = max_abs
    res["label_free_scores"] = metrics(y[te], torch.softmax(lg.double(), 1).numpy()[te])
    out[f"{ds}|csgnn_released"] = res
    print(ds, "CSGNN released: AUC true labels %.4f | permuted %.4f | all-zero %.4f | max |dp| %.4f" % (
        res["true_labels"]["auc"], res["permuted_test_labels_mean"]["auc"], res["all_test_labels_set_to_0"]["auc"], max_abs), flush=True)

    # ---------------------------------------------------------------- (2) CS-Boost: predictions must be identical
    best = json.load(open("best.json"))[ds].split()
    lam = ["--lam", "1", "2"] if ds == "sichuan" else []
    av = V.parser().parse_args(["--dataset", ds] + best + lam)
    data = load(ds)
    X, y, s, d, tr, va, te = data
    r1 = V.run(av, data, seed=0, verbose=False)
    yp = y.copy(); yp[te] = rng.permutation(y[te])
    r2 = V.run(av, (X, yp, s, d, tr, va, te), seed=0, verbose=False)
    diff = float(np.abs(r1["_P"][te] - r2["_P"][te]).max())
    same_pred = bool((r1["_pred"][te] == r2["_pred"][te]).all())
    out[f"{ds}|csboost"] = dict(max_abs_change_in_test_probs=diff, identical_test_predictions=same_pred)
    print(ds, f"CS-Boost: max |dp| under test-label permutation = {diff:.2e}, identical predictions = {same_pred}", flush=True)

json.dump(out, open(A.out, "w"), indent=1)
print("saved", A.out)
