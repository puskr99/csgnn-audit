"""Revision experiment A: hyperparameters selected on EACH split's own validation set (instead of split 0 only).

CS-Boost: the same staged search as tune.py (decoupled GraphSAGE 108 -> GBDT 12 -> class weights 6), seed 0, score =
val G-Mean + val AUC, run separately on every split; the selected configuration is then run with seeds 0,1,2.
Baselines (recipe-matched): the same 60 random configurations as the split-0 search (read from extra_tune.jsonl),
scored on each split's validation set with seed 0; the best is run with seeds 0,1,2.
Also runs CS-Boost's fixed split-0 configuration (best.json) on the same hardware for a like-for-like comparison.
Everything is resumable: one JSON line per finished run in --out (and per-trial lines in --trials).

Usage: python rev_pertune.py --dataset sichuan --splits 0 1 2 3 4 --methods v5,v5_fixed,mlp,sage,gcn,gat
"""
import argparse, itertools, json, os, time, numpy as np
from argparse import Namespace
from common import load
import csboost_v5 as V
import baselines as B
import extra_runs as ER   # run_bwgnn: BWGNN on the telecom graphs (recipe-matched when recipe=1)

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--splits", type=int, nargs="*", default=[0, 1, 2, 3, 4])
p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
p.add_argument("--methods", default="v5,v5_fixed,mlp")
p.add_argument("--out", default="results_rev/pertune.jsonl")
p.add_argument("--trials", default="results_rev/pertune_trials.jsonl")
A = p.parse_args()
os.makedirs("results_rev", exist_ok=True)
ds = A.dataset


def readj(f):
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else []


def append(f, rec):
    with open(f, "a") as fh: fh.write(json.dumps(rec) + "\n")


TR = {}
for r in readj(A.trials):
    TR[(r["dataset"], r["split"], r["method"], r["stage"], json.dumps(r["cfg"], sort_keys=True))] = r
for st in ["gnn", "gbdt"]:   # split-0 search already done with tune.py (same grid, seed 0): reuse it
    for r in readj(f"sweeps/tune_{ds}_{st}.jsonl"):
        TR.setdefault((ds, 0, "v5", st, json.dumps(r["cfg"], sort_keys=True)), dict(score=r["score"]))
for r in readj("results_gpu/extra_tune.jsonl"):   # split-0 baseline search already done (same 60 configs, seed 0)
    if "trial" in r and r["dataset"] == ds:
        TR.setdefault((ds, 0, r["method"].replace("_r1", ""), "base", json.dumps(r["hp"], sort_keys=True)),
                      dict(score=r["val"]["gmean"] + r["val"]["auc"]))
DONE = {(r["dataset"], r["split"], r["method"], r["seed"]) for r in readj(A.out)}
GRIDS = {"gnn": dict(ops=["sym1", "dir1", "sym2"], hidden=[64, 128, 256], dropout=[0.1, 0.3, 0.5], lr=[1e-3, 3e-3], wd=[1e-4, 1e-3]),
         "gbdt": dict(gb_leaves=[15, 31, 63], gb_mcs=[20, 100], gb_col=[0.5, 0.8]),
         "cost": dict(wpow=[0.5, 1.0], lam=[None, "1.5", "2.0"])}
BEST = json.load(open("best.json"))
LAM = {"sichuan": ["--lam", "1", "2"], "bupt": []}
BASE_DEFAULTS = dict(epochs=1000, patience=100)
TRIALS = {}
for r in readj("results_gpu/extra_tune.jsonl"):
    if "trial" in r and r["dataset"] == ds:
        TRIALS.setdefault(r["method"].replace("_r1", ""), []).append((r["trial"], r["hp"]))
for m in TRIALS: TRIALS[m] = [hp for _, hp in sorted(TRIALS[m], key=lambda t: t[0])]


def cfg_argv(cfg, C):
    out = []
    for k, v in cfg.items():
        if k == "lam":
            if v is not None: out += ["--lam", "1"] + [v] * (C - 1)
        else: out += [f"--{k}", str(v)]
    return out


def v5_trial(data, k, stage, cfg, fixed):
    key = (ds, k, "v5", stage, json.dumps(cfg, sort_keys=True))
    if key in TR: return TR[key]["score"]
    C = int(data[1].max() + 1)
    base = {"gnn": ["--gbdt", "0"], "gbdt": ["--gnn", "0"], "cost": []}[stage]
    a = V.parser().parse_args(["--dataset", ds] + base + fixed + cfg_argv(cfg, C))
    t0 = time.time(); r = V.run(a, data, seed=0, verbose=False)
    sc = r["val"]["gmean"] + r["val"]["auc"]
    rec = dict(dataset=ds, split=k, method="v5", stage=stage, cfg=cfg, score=sc, val=r["val"], time=time.time() - t0)
    append(A.trials, rec); TR[key] = rec
    print(f"  [{ds} s{k} v5 {stage}] {cfg} score={sc:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return sc


def v5_select(data, k):
    C = int(data[1].max() + 1); chosen = {}
    for stage in ["gnn", "gbdt", "cost"]:
        g = GRIDS[stage]
        fixed = cfg_argv(chosen, C) if stage == "cost" else []
        cands = [dict(zip(g, v)) for v in itertools.product(*g.values())]
        best = max(cands, key=lambda c: v5_trial(data, k, stage, c, fixed))
        chosen.update(best)
    return chosen


def base_trial(data, k, m, hp):
    key = (ds, k, m, "base", json.dumps(hp, sort_keys=True))
    if key in TR: return TR[key]["score"]
    t0 = time.time()
    r = ER.run_bwgnn(data, 0, 1, **hp) if m == "bwgnn" else B.run(Namespace(dataset=ds, model=m, recipe=1, **hp, **BASE_DEFAULTS), data, 0)
    sc = r["val"]["gmean"] + r["val"]["auc"]
    rec = dict(dataset=ds, split=k, method=m, stage="base", cfg=hp, score=sc, val=r["val"], time=time.time() - t0)
    append(A.trials, rec); TR[key] = rec
    print(f"  [{ds} s{k} {m}] {hp} score={sc:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return sc


for k in A.splits:
    data = load(ds, k); C = int(data[1].max() + 1)
    for m in A.methods.split(","):
        name = {"v5": "v5_pertuned", "v5_fixed": "v5_fixed"}.get(m, f"{m}_r1_pertuned")
        if all((ds, k, name, s) in DONE for s in A.seeds): continue
        if m == "v5":
            cfg = v5_select(data, k); argv = ["--dataset", ds] + cfg_argv(cfg, C)
        elif m == "v5_fixed":
            cfg = BEST[ds]; argv = ["--dataset", ds] + BEST[ds].split() + LAM[ds]
        else:
            cfg = max(TRIALS[m], key=lambda hp: base_trial(data, k, m, hp))
        for s in A.seeds:
            if (ds, k, name, s) in DONE: continue
            t0 = time.time()
            if m.startswith("v5"):
                r = V.run(V.parser().parse_args(argv), data, seed=s, verbose=False)
            else:
                r = ER.run_bwgnn(data, s, 1, **cfg) if m == "bwgnn" else B.run(Namespace(dataset=ds, model=m, recipe=1, **cfg, **BASE_DEFAULTS), data, s)
            append(A.out, dict(dataset=ds, split=k, seed=s, method=name, cfg=cfg, val=r["val"], test=r["test"],
                               time=time.time() - t0))
            DONE.add((ds, k, name, s))
            print(f"[{ds} split{k} {name} seed{s}] test g={r['test']['gmean']:.4f} auc={r['test']['auc']:.4f}", flush=True)
print("finished", ds)
