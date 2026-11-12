import sys, json, itertools, argparse, numpy as np
from sign import parser, make_feats, run
from common import load

ap = argparse.ArgumentParser()
ap.add_argument("--dataset"); ap.add_argument("--grid"); ap.add_argument("--out"); ap.add_argument("--base", default="")
A = ap.parse_args()
grid = json.loads(A.grid)
data = load(A.dataset)
keys = list(grid)
featkeys = ["ops", "degfeat", "logx", "maxagg"]
cache = {}
with open(A.out, "a") as fh:
    for vals in itertools.product(*grid.values()):
        argv = ["--dataset", A.dataset] + A.base.split() + sum([[f"--{k}"] + (v if isinstance(v, list) else [v]) for k, v in zip(keys, map(lambda v: [str(x) for x in v] if isinstance(v, list) else str(v), vals))], [])
        a = parser().parse_args(argv)
        fk = tuple(getattr(a, k) for k in featkeys)
        if fk not in cache:
            cache.clear(); cache[fk] = make_feats(a, *data[:4], data[4])
        r = run(a, data=data, feats=cache[fk], verbose=False)
        r.pop("_lp")
        rec = dict(cfg=dict(zip(keys, vals)), **r)
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        print(dict(zip(keys, vals)), f"ep{r['best_ep']} vcal_g={r['val_cal']['gmean']:.4f} vauc={r['val_raw']['auc']:.4f} | tcal_g={r['test_cal']['gmean']:.4f} tauc={r['test_raw']['auc']:.4f} tf1raw={r['test_raw']['f1']:.4f}", flush=True)
