"""Review-round-3 experiments on Sichuan / BUPT (same splits, seeds and metrics as multisplit.py).

Stages (--stage):
  tune    Equal-budget random search for the recipe-matched baselines (MLP, GCN, GraphSAGE, GAT, BWGNN).
          CS-Boost's learners were tuned over 108 + 12 + 6 configurations on the reference split's validation set;
          each baseline gets --budget configurations (default 60) drawn from its grid, scored on split 0 / seed 0
          by validation G-Mean + AUC (the protocol's selection score). The selected configuration is then run on all
          splits x seeds, exactly like CS-Boost (selected once, kept fixed). Methods: <model>_r1_tuned.
  bwgnn   BWGNN (Tang et al., ICML 2022; faithful port in bwgnn_port.py) on the telecom graphs, in the classic
          setting (raw z-scored features, the official class weighting generalised to C classes, epoch chosen on
          validation macro-F1, official threshold rule for binary / arg-max for 3 classes) and recipe-matched
          (log features + degree, sqrt class weights, validation-calibrated decision). Methods: bwgnn_r0, bwgnn_r1.
  rewire  Degree-preserving rewiring: every split's edge list is replaced by a directed configuration-model sample
          (destination endpoints permuted; out- and in-degree sequences preserved; self loops/duplicates dropped).
          If a model's score is unchanged, the graph contributes nothing beyond degree. Methods: <method>_rw for
          v5 (CS-Boost) and its two components, sage_r1, gcn_r1, bwgnn_r1 (+ mlp_r1 as a graph-free control).

Every run appends one JSON line to --out (default results_gpu/extra_<stage>.jsonl); reruns skip finished keys.
Example:  python extra_runs.py --stage tune --datasets sichuan,bupt
"""
import argparse, copy, itertools, json, os, random, time, numpy as np, scipy.sparse as sp, torch, torch.nn.functional as F
from argparse import Namespace
from sklearn.metrics import f1_score
from common import load, metrics, fit_bias
from csboost_v5 import base_features, class_weights
import csboost_v5 as V
import baselines as B
import bwgnn_port as BW

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DEFAULTS = dict(hidden=64, layers=2, dropout=0.3, lr=0.01, wd=5e-4, epochs=1000, patience=100)   # = multisplit.py
GRID = {
    "gnn": dict(hidden=[64, 128, 256], layers=[2, 3], dropout=[0.1, 0.3, 0.5], lr=[1e-3, 3e-3, 1e-2], wd=[0.0, 5e-4, 1e-3]),
    "bwgnn": dict(hidden=[32, 64, 128], d=[2, 3], lr=[1e-3, 3e-3, 1e-2], wd=[0.0, 5e-4], epochs=[100, 200, 400]),
}
BW_DEFAULTS = dict(hidden=64, d=2, lr=0.01, wd=0.0, epochs=100)   # official BWGNN defaults


# --------------------------------------------------------------------------- BWGNN on the telecom graphs
def run_bwgnn(data, seed, recipe, hidden=64, d=2, lr=0.01, wd=0.0, epochs=100):
    X, y, src, dst, tr, va, te = data
    N, C = len(y), int(y.max() + 1)
    torch.manual_seed(seed); np.random.seed(seed)
    if recipe:
        Xb = base_features(X, src, dst, tr, True, True); w = class_weights(y, tr, C, 0.5)
    else:
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xb = np.clip((X - mu) / sd, -15, 15).astype(np.float32)
        cnt = np.bincount(y[tr], minlength=C).astype(float)
        w = cnt.max() / cnt                                   # official: weight(neg)=1, weight(pos)=#neg/#pos
    A = sp.csr_matrix((np.ones(2 * len(src), np.float32), (np.r_[src, dst], np.r_[dst, src])), shape=(N, N))
    g = BW.build([A], homo=True, device=DEV)
    model = BW.BWGNN(Xb.shape[1], hidden, C, g, d=d).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    x = torch.tensor(Xb, dtype=torch.float32, device=DEV); yt = torch.tensor(y, device=DEV); trt = torch.tensor(tr, device=DEV)
    wt = torch.tensor(w, dtype=torch.float32, device=DEV)
    best, bestP, thr, t0 = -1, None, 0.5, time.time()
    for ep in range(epochs):
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits[trt], yt[trt], weight=wt)
        opt.zero_grad(); loss.backward(); opt.step()
        P = logits.softmax(1).detach().cpu().numpy()       # official: probabilities from this epoch's forward pass
        if recipe:                                          # protocol selection score (as baselines.py)
            mv = metrics(y[va], P[va]); s = mv["gmean"] + mv["auc"]; t = None
        elif C == 2:                                        # official: best val macro-F1 over thresholds
            s, t = BW.get_best_f1(y[va], P[va])
        else:                                               # 3 classes: official rule has no threshold -> arg-max
            s, t = f1_score(y[va], P[va].argmax(1), average="macro"), None
        if s > best: best, bestP, thr = s, P, t
    P = bestP; lp = np.log(P + 1e-12)
    if recipe:
        pred = (lp + fit_bias(y[va], lp[va])[0]).argmax(1)
    elif C == 2:
        pred = (P[:, 1] > thr).astype(int)
    else:
        pred = P.argmax(1)
    return dict(val=metrics(y[va], P[va], pred[va]), test=metrics(y[te], P[te], pred[te]), time=time.time() - t0)


# --------------------------------------------------------------------------- generic runners
def run_method(m, data, seed, hp=None, ds=None):
    """m in {v5, mlp_r0/1, gcn_r0/1, sage_r0/1, gat_r0/1, bwgnn_r0/1}; hp overrides the default hyperparameters."""
    if m == "v5":
        BEST = json.load(open("best.json")); LAM = {"sichuan": ["--lam", "1", "2"], "bupt": []}
        a = V.parser().parse_args(["--dataset", ds] + BEST[ds].split() + LAM[ds])
        r = V.run(a, data, seed=seed, verbose=False)
        X, y, src, dst, tr, va, te = data
        out = [("v5", dict(val=r["val"], test=r["test"], time=None))]
        for c in ["gbdt", "gnn"]:
            P = r["_comp"][c]; lp = np.log(P + 1e-12); pred = (lp + fit_bias(y[va], lp[va])[0]).argmax(1)
            out.append((f"v5_{c}_only", dict(val=metrics(y[va], P[va], pred[va]), test=metrics(y[te], P[te], pred[te]), time=None)))
        return out
    model, rec = m.split("_r"); rec = int(rec)
    if model == "bwgnn":
        return [(m, run_bwgnn(data, seed, rec, **{**BW_DEFAULTS, **(hp or {})}))]
    a = Namespace(dataset=ds, model=model, recipe=rec, **{**BASE_DEFAULTS, **(hp or {})})
    r = B.run(a, data, seed)
    return [(m, dict(val=r["val"], test=r["test"], time=r["time"]))]


def rewire(data, seed):
    """Directed configuration model: permute destination endpoints (out-/in-degree sequences preserved)."""
    X, y, src, dst, tr, va, te = data
    rng = np.random.default_rng(10_000 + seed)
    d2 = dst[rng.permutation(len(dst))]
    keep = src != d2
    key = np.unique(src[keep] * len(y) + d2[keep])
    s2, d2 = key // len(y), key % len(y)
    return (X, y, s2, d2, tr, va, te), len(key) / len(src)


def score(r):
    return r["val"]["gmean"] + r["val"]["auc"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["tune", "bwgnn", "rewire", "rewire_exact"])
    p.add_argument("--datasets", default="sichuan,bupt")
    p.add_argument("--splits", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--models", default="mlp,gcn,sage,gat,bwgnn", help="tune stage: models to tune (recipe-matched)")
    p.add_argument("--budget", type=int, default=60)
    p.add_argument("--rw_methods", default="v5,sage_r1,gcn_r1,bwgnn_r1,mlp_r1")
    p.add_argument("--out", default=None)
    A = p.parse_args()
    out = A.out or f"results_gpu/extra_{A.stage}.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    rows = [json.loads(l) for l in open(out)] if os.path.exists(out) else []
    done = {(r["dataset"], r["split"], r["method"], r["seed"]) for r in rows if "trial" not in r}
    trials = {(r["dataset"], r["method"], r["trial"]): r for r in rows if "trial" in r}
    write = lambda rec: open(out, "a").write(json.dumps(rec) + "\n")
    print("device:", DEV, flush=True)

    for ds in A.datasets.split(","):
        if A.stage == "tune":
            data0 = load(ds, 0)
            for model in A.models.split(","):
                key = "bwgnn" if model == "bwgnn" else "gnn"
                grid = [dict(zip(GRID[key], v)) for v in itertools.product(*GRID[key].values())]
                cand = random.Random(12345).sample(grid, min(A.budget, len(grid)))
                m = f"{model}_r1"
                for t, hp in enumerate(cand):                     # search on split 0, seed 0, validation only
                    if (ds, m, t) in trials: continue
                    t0 = time.time(); r = run_method(m, data0, 0, hp, ds)[0][1]
                    rec = dict(dataset=ds, method=m, trial=t, hp=hp, val=r["val"], time=time.time() - t0)
                    write(rec); trials[(ds, m, t)] = rec
                    print(f"[tune {ds} {m} {t}/{len(cand)}] val G+AUC={score(rec):.4f} {hp}", flush=True)
                best = max((trials[(ds, m, t)] for t in range(len(cand))), key=score)
                print(f"[tune {ds} {m}] selected {best['hp']} (val G+AUC {score(best):.4f})", flush=True)
                for k in A.splits:
                    data = load(ds, k)
                    for s in A.seeds:
                        if (ds, k, m + "_tuned", s) in done: continue
                        r = run_method(m, data, s, best["hp"], ds)[0][1]
                        write(dict(dataset=ds, split=k, seed=s, method=m + "_tuned", hp=best["hp"], **r))
                        print(f"  {ds} split{k} seed{s} {m}_tuned test G={r['test']['gmean']:.4f} AUC={r['test']['auc']:.4f}", flush=True)
        else:
            methods = ["bwgnn_r0", "bwgnn_r1"] if A.stage == "bwgnn" else A.rw_methods.split(",")
            suf = {"rewire": "_rw", "rewire_exact": "_rwx"}.get(A.stage, "")
            for k in A.splits:
                data = load(ds, k); frac = None
                if A.stage == "rewire":
                    data, frac = rewire(data, k)
                elif A.stage == "rewire_exact":               # exact degree- and reciprocity-preserving swaps (rewire_exact.py)
                    from rewire_exact import rewire_exact, check
                    X_, y_, s_, d_, tr_, va_, te_ = data
                    s2, d2, acc = rewire_exact(s_, d_, len(y_), k)
                    exact, frac = check(s_, d_, s2, d2, len(y_))   # frac = share of original edges left
                    assert exact, "rewiring did not preserve the degree sequences"
                    print(f"[rewire_exact {ds} split{k}] accepted swaps {acc:.2f}, original edges left {frac:.3f}", flush=True)
                    data = (X_, y_, s2, d2, tr_, va_, te_)
                for m in methods:
                    for s in A.seeds:
                        tag = m + suf
                        if (ds, k, tag, s) in done: continue
                        t0 = time.time()
                        for name, r in run_method(m, data, s, None, ds):
                            name = name + suf
                            write(dict(dataset=ds, split=k, seed=s, method=name, **({"original_edges_left": frac, "rewire_version": 2} if A.stage == "rewire_exact" else {"edges_kept": frac}), **r))
                            print(f"[{A.stage} {ds} split{k} seed{s} {name}] test G={r['test']['gmean']:.4f} AUC={r['test']['auc']:.4f} "
                                  f"({time.time() - t0:.0f}s)", flush=True)
