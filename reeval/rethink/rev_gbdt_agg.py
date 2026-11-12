"""Revision experiment B: CS-Boost's cost-sensitive GBDT with and without the neighbour means of the recipe features
appended (one-hop symmetric mean, the decoupled GraphSAGE learner's neighbour channel), same splits/seeds, same GBDT
hyperparameters and class weights as CS-Boost (best.json), validation-calibrated decision."""
import json, os, time, numpy as np
from common import load, metrics, fit_bias
import csboost_v5 as V
BEST = json.load(open("best.json")); LAM = {"sichuan": ["--lam", "1", "2"], "bupt": []}
OUT = "results_rev/gbdt_agg.jsonl"
done = {(r["dataset"], r["split"], r["method"], r["seed"]) for r in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
for ds in ["sichuan", "bupt"]:
    a = V.parser().parse_args(["--dataset", ds] + BEST[ds].split() + LAM[ds])
    for k in range(5):
        X, y, s, d, tr, va, te = load(ds, k); n, C = len(y), int(y.max() + 1)
        Xb = V.base_features(X, s, d, tr); w = V.class_weights(y, tr, C, a.wpow, a.lam)
        Xa = np.hstack(V.propagate(Xb, V.operators(s, d, n, "sym1"), tr))
        for name, F in [("gbdt", Xb), ("gbdt_nbrmean", Xa)]:
            for seed in [0, 1, 2]:
                if (ds, k, name, seed) in done: continue
                t0 = time.time(); P, it = V.fit_gbdt(F, y, tr, va, w, a, seed)
                lp = np.log(P + 1e-12); b = fit_bias(y[va], lp[va])[0]; pred = (lp + b).argmax(1)
                rec = dict(dataset=ds, split=k, seed=seed, method=name, iters=int(it), val=metrics(y[va], P[va], pred[va]),
                           test=metrics(y[te], P[te], pred[te]), time=time.time() - t0)
                with open(OUT, "a") as fh: fh.write(json.dumps(rec) + "\n")
                print(f"[{ds} split{k} {name} s{seed}] g={rec['test']['gmean']:.4f} auc={rec['test']['auc']:.4f} ({rec['time']:.0f}s)", flush=True)
