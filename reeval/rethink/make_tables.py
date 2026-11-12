"""Aggregate all results into tables + paired bootstrap over test nodes."""
import json, glob, numpy as np
from sklearn.metrics import roc_auc_score
from common import load, metrics

import os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "analysis_outputs")
out = {}


def agg(rows):
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in rows[0]}


for ds in ["sichuan", "bupt"]:
    X, y, s, d, tr, va, te = load(ds)
    T = {}
    # v5 variants
    for l in open(f"sweeps/final_{ds}.jsonl"):
        r = json.loads(l); T[f"v5:{r['variant']}"] = agg([x["test"] for x in r["runs"]])
    # end-to-end baselines
    for l in open("sweeps/baselines.jsonl"):
        r = json.loads(l)
        if r["args"]["dataset"] != ds: continue
        T[f"{r['args']['model']}:r{r['args']['recipe']}"] = agg([x["test"] for x in r["runs"]])
    # reproduced published baselines (saved test probabilities, their own protocol: argmax)
    for m in ["gatcobo", "pcgnn"]:
        dd = np.load(f"{R}/{ds}_probs_{m}.npy", allow_pickle=True).item()
        T[f"{m}:repro"] = {k: (v, 0.0) for k, v in metrics(dd["labels"], dd["probs"].astype(float)).items()}
    out[ds] = T
    print(f"\n===== {ds} =====")
    for k, v in T.items():
        print(f"{k:16s} " + " ".join(f"{m}={a:.4f}±{b:.4f}" for m, (a, b) in v.items()))

json.dump(out, open("sweeps/tables.json", "w"), indent=1)


# ------------------------------------------------ paired bootstrap (seed-0 predictions)
def boot(y, preds, probs, C, B=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(y)
    res = {k: {"gmean": [], "f1": [], "auc": []} for k in preds}
    for _ in range(B):
        i = rng.integers(0, n, n); yb = y[i]
        cnt = np.bincount(yb, minlength=C)
        for k in preds:
            pb = preds[k][i]
            tp = np.bincount(yb[pb == yb], minlength=C)
            pc = np.bincount(pb, minlength=C)
            rec = tp / np.maximum(cnt, 1); prec = tp / np.maximum(pc, 1)
            f1 = np.where(rec + prec > 0, 2 * rec * prec / np.maximum(rec + prec, 1e-12), 0)
            res[k]["gmean"].append(np.prod(np.maximum(rec, 1e-6)) ** (1 / C)); res[k]["f1"].append(f1.mean())
            pr = probs[k][i]
            res[k]["auc"].append(roc_auc_score(yb, pr[:, 1]) if C == 2 else roc_auc_score(yb, pr, multi_class="ovo"))
    return {k: {m: np.array(v) for m, v in d.items()} for k, d in res.items()}


BS = {}
for ds in ["sichuan", "bupt"]:
    X, y, s, d, tr, va, te = load(ds); C = y.max() + 1
    order = np.argsort(te)                     # saved published-baseline probs are in sorted-index order
    yt = y[te][order]
    preds, probs = {}, {}
    z = np.load(f"cache/v5_{ds}_full_s0.npz"); preds["v5"], probs["v5"] = z["pred"][order], z["P"][order]
    z = np.load(f"cache/v5_{ds}_gbdt_only_s0.npz"); preds["gbdt"], probs["gbdt"] = z["pred"][order], z["P"][order]
    for m in ["gatcobo", "pcgnn"]:
        dd = np.load(f"{R}/{ds}_probs_{m}.npy", allow_pickle=True).item()
        assert (dd["labels"] == yt).all()
        p = dd["probs"].astype(float); preds[m], probs[m] = p.argmax(1), p
    try:
        p = np.load(f"cache/probs_{ds}_sage_r1_s0.npy").astype(float)[order]
        preds["sage_r1"], probs["sage_r1"] = np.load(f"cache/pred_{ds}_sage_r1_s0.npy")[order], p
    except FileNotFoundError: pass
    for f in glob.glob(f"cache/csgnn_{ds}_e200_es1_s42.npy"):
        dd = np.load(f, allow_pickle=True).item(); p = dd["fix_pred"]
        preds["csgnn"], probs["csgnn"] = p[order].argmax(1), p[order]
    b = boot(yt, preds, probs, C)
    BS[ds] = {}
    print(f"\n===== bootstrap {ds} (v5 minus other; mean [95% CI]) =====")
    for k in preds:
        if k == "v5": continue
        BS[ds][k] = {}
        for m in ["gmean", "f1", "auc"]:
            dlt = b["v5"][m] - b[k][m]
            lo, hi = np.percentile(dlt, [2.5, 97.5])
            BS[ds][k][m] = [float(dlt.mean()), float(lo), float(hi)]
            print(f"  vs {k:8s} {m:6s} {dlt.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
json.dump(BS, open("sweeps/bootstrap.json", "w"), indent=1)
