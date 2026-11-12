"""Revision check: give the strongest recipe-matched baselines the same class-weight search as CS-Boost
(exponent in {0.5, 1} x minority multiplier lambda in {1, 1.5, 2}), on top of their per-split-selected architecture
hyperparameters; selection on each split's validation set (seed 0), then seeds 0-2."""
import json, os, sys, time, numpy as np
from argparse import Namespace
from common import load
import baselines as B
import csboost_v5 as V
ds = sys.argv[1]; models = sys.argv[2].split(","); OUT = f"results_rev/lambda_{ds}.jsonl"
CFG = {}
for f in ["results_rev/pertune.jsonl", f"results_rev/pertune_gpu_{ds}.jsonl"]:
    for l in open(f):
        r = json.loads(l)
        if r["dataset"] == ds and r["method"].endswith("_r1_pertuned"): CFG[(r["method"].split("_")[0], r["split"])] = r["cfg"]
done = {}
if os.path.exists(OUT):
    for l in open(OUT): r = json.loads(l); done[(r["model"], r["split"], r["wpow"], r["lam"], r["seed"])] = r
GRID = [(0.5, 1.0), (0.5, 1.5), (0.5, 2.0), (1.0, 1.0), (1.0, 1.5), (1.0, 2.0)]
def run(m, k, data, wp, lam, seed):
    key = (m, k, wp, lam, seed)
    if key in done: return done[key]
    C = int(data[1].max() + 1)
    B.class_weights = lambda y, tr, C_, _w: V.class_weights(y, tr, C_, wp, [1.0] + [lam] * (C_ - 1))
    t0 = time.time(); r = B.run(Namespace(dataset=ds, model=m, recipe=1, epochs=1000, patience=100, **CFG[(m, k)]), data, seed)
    rec = dict(dataset=ds, model=m, split=k, wpow=wp, lam=lam, seed=seed, val=r["val"], test=r["test"], time=time.time() - t0)
    with open(OUT, "a") as fh: fh.write(json.dumps(rec) + "\n")
    done[key] = rec; return rec
for m in models:
    for k in range(5):
        data = load(ds, k)
        best = max(GRID, key=lambda g: (lambda r: r["val"]["gmean"] + r["val"]["auc"])(run(m, k, data, *g, 0)))
        for s in [0, 1, 2]:
            r = run(m, k, data, *best, s)
        print(f"[{ds} {m} split{k}] chosen wpow={best[0]} lam={best[1]}", flush=True)
