"""Multi-split, multi-seed evaluation on Sichuan and BUPT (addresses: single fixed split; node-level bootstrap only).

Split k=0 is the shared split used in the paper; k=1..K are new stratified 20/20/60 splits (random_state=1000+k).
For every (dataset, split, seed) we run:
  v5        CS-Boost full; we also record its two components (gbdt, gnn), each with its own val-calibrated rule
  mlp/gcn/sage/gat with recipe 0 (classic) and recipe 1 (recipe-matched)
  csgnn     label-free CSGNN (200 epochs, label-free early stopping; 1 seed per split by default, CPU-bound)
One JSON line per run is appended to --out; reruns skip finished (dataset, split, method, seed) keys.

Example:  python multisplit.py --datasets sichuan,bupt --splits 0 1 2 3 4 --seeds 0 1 2
"""
import argparse, json, os, subprocess, sys, time, numpy as np
from argparse import Namespace
from common import load, metrics, fit_bias
import csboost_v5 as V
import baselines as B

p = argparse.ArgumentParser()
p.add_argument("--datasets", default="sichuan,bupt")
p.add_argument("--splits", type=int, nargs="*", default=[0, 1, 2, 3, 4])
p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
p.add_argument("--methods", default="v5,mlp_r0,gcn_r0,sage_r0,gat_r0,mlp_r1,gcn_r1,sage_r1,gat_r1,csgnn")
p.add_argument("--csgnn_seeds", type=int, nargs="*", default=[0, 1, 2])
p.add_argument("--out", default="results_gpu/multisplit.jsonl")
A = p.parse_args()
os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)

done = set()
if os.path.exists(A.out):
    for l in open(A.out):
        r = json.loads(l); done.add((r["dataset"], r["split"], r["method"], r["seed"]))

BEST = json.load(open("best.json"))
LAM = {"sichuan": ["--lam", "1", "2"], "bupt": []}
BASE_DEFAULTS = dict(hidden=64, layers=2, dropout=0.3, lr=0.01, wd=5e-4, epochs=1000, patience=100)


def write(rec):
    with open(A.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def comp_metrics(P, y, va, te):
    lp = np.log(P + 1e-12); b = fit_bias(y[va], lp[va])[0]; pred = (lp + b).argmax(1)
    return metrics(y[va], P[va], pred[va]), metrics(y[te], P[te], pred[te])


for ds in A.datasets.split(","):
    for k in A.splits:
        data = load(ds, k)
        X, y, src, dst, tr, va, te = data
        for m in A.methods.split(","):
            seeds = A.csgnn_seeds if m == "csgnn" else A.seeds
            for s in seeds:
                if (ds, k, m, s) in done:
                    continue
                t0 = time.time()
                print(f"[{ds} split{k} {m} seed{s}]", flush=True)
                if m == "v5":
                    a = V.parser().parse_args(["--dataset", ds] + BEST[ds].split() + LAM[ds])
                    r = V.run(a, data, seed=s, verbose=False)
                    base = dict(dataset=ds, split=k, seed=s, info={kk: vv for kk, vv in r["info"].items() if kk != "bias"})
                    write(dict(base, method="v5", val=r["val"], test=r["test"], time=time.time() - t0))
                    for c in ["gbdt", "gnn"]:
                        mv, mt = comp_metrics(r["_comp"][c], y, va, te)
                        write(dict(base, method=f"v5_{c}_only", val=mv, test=mt, time=None))
                        done.add((ds, k, f"v5_{c}_only", s))
                elif m == "csgnn":
                    tmp = A.out + ".csgnn_tmp"
                    if os.path.exists(tmp): os.remove(tmp)
                    subprocess.run([sys.executable, "csgnn_port.py", "--dataset", ds, "--split", str(k), "--seed", str(s),
                                    "--max_epoch", "200", "--es", "1", "--out_json", tmp],
                                   check=True, stdout=subprocess.DEVNULL)
                    rr = json.loads(open(tmp).read().strip().splitlines()[-1]); os.remove(tmp)
                    # label-free variant chosen on validation (G-Mean + AUC, the protocol's selection score)
                    best = max(["honest", "fix_pred", "fix_exp"], key=lambda v: rr["val"][v]["gmean"] + rr["val"][v]["auc"])
                    write(dict(dataset=ds, split=k, seed=s, method="csgnn", variant=best, val=rr["val"][best], test=rr["test"][best],
                               test_variants=rr["test"], val_variants=rr["val"], time=time.time() - t0))
                else:
                    model, rec = m.split("_r")
                    a = Namespace(dataset=ds, model=model, recipe=int(rec), **BASE_DEFAULTS)
                    r = B.run(a, data, s)
                    write(dict(dataset=ds, split=k, seed=s, method=m, val=r["val"], test=r["test"], time=r["time"]))
                done.add((ds, k, m, s))
                print(f"   done in {time.time() - t0:.1f}s", flush=True)
print("finished; results in", A.out)
