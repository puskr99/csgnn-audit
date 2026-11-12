"""Revision experiment E: YelpChi/Amazon MLP with and without signed-log scaling (same split, seeds, MLP and threshold rule
as fraud_runs.py). MLP-raw uses the raw features, z-scored on the training nodes, instead of signed log1p."""
import json, os, sys, time, numpy as np
sys.path.insert(0, "audit_extended")
import fraud_baselines as FB
os.chdir("fraud_data"); OUT = "../results_rev/fraud_raw.jsonl"
done = {(r["dataset"], r["method"], r["seed"]) for r in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
for ds in ["yelp", "amazon"]:
    X, y, rels, tr, va, te = FB.load(ds)
    F = {"MLP": FB.feats(X, rels, False), "MLP-raw": X.astype(np.float32)}
    for m in ["MLP-raw", "MLP"]:
        for s in range(5):
            if (ds, m, s) in done: continue
            t0 = time.time(); pr = FB.run_mlp(F[m], y, tr, va, s)
            r = {k: float(v) for k, v in FB.scores(y[te], pr[te], FB.best_thr_f1(y[va], pr[va])).items()}
            with open(OUT, "a") as fh: fh.write(json.dumps(dict(dataset=ds, method=m, seed=s, test=r, time=time.time() - t0)) + "\n")
            print(ds, m, s, r, f"{time.time()-t0:.0f}s", flush=True)
