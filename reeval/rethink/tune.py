"""Validation-only grid search for v5 components. Score = val G-Mean (calibrated on val) + val AUC."""
import itertools, json, sys, numpy as np
import csboost_v5 as V
from common import load

ds, stage, out = sys.argv[1], sys.argv[2], sys.argv[3]
fixed = sys.argv[4].split() if len(sys.argv) > 4 else []
grids = {
    "gnn": dict(ops=["sym1", "dir1", "sym2"], hidden=[64, 128, 256], dropout=[0.1, 0.3, 0.5], lr=[1e-3, 3e-3], wd=[1e-4, 1e-3]),
    "gbdt": dict(gb_leaves=[15, 31, 63], gb_mcs=[20, 100], gb_col=[0.5, 0.8]),
    "cost": dict(wpow=[0.5, 1.0], lam=[None, "1.5", "2.0"]),
}
base = {"gnn": ["--gbdt", "0"], "gbdt": ["--gnn", "0"], "cost": []}[stage]
data = load(ds); C = int(data[1].max() + 1)
done = set()
try:
    for l in open(out): done.add(json.dumps(json.loads(l)["cfg"], sort_keys=True))
except FileNotFoundError: pass
g = grids[stage]
for vals in itertools.product(*g.values()):
    cfg = dict(zip(g, vals))
    if json.dumps(cfg, sort_keys=True) in done: continue
    argv = ["--dataset", ds] + base + fixed
    for k, v in cfg.items():
        if k == "lam":
            if v is not None: argv += ["--lam", "1"] + [v] * (C - 1)
        else: argv += [f"--{k}", str(v)]
    a = V.parser().parse_args(argv)
    r = V.run(a, data, seed=0, verbose=False); r.pop("_P"); r.pop("_pred"); r.pop("_comp")
    score = r["val"]["gmean"] + r["val"]["auc"]
    with open(out, "a") as fh:
        fh.write(json.dumps(dict(cfg=cfg, score=score, val=r["val"], test=r["test"], info=r["info"])) + "\n")
    print(cfg, f"score={score:.4f} val_g={r['val']['gmean']:.4f} val_auc={r['val']['auc']:.4f}", flush=True)
