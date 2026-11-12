"""Final multi-seed runs + component ablation for v5. Saves per-run test probabilities/predictions."""
import json, sys, numpy as np, csboost_v5 as V
from common import load
best = json.load(open("best.json"))
LAM = {"sichuan": "--lam 1 2", "bupt": ""}
VARIANTS = {
    "full": "", "gbdt_only": "--gnn 0", "gnn_only": "--gbdt 0", "gnn_noagg": "--ops none",
    "no_calib": "--calib 0", "no_cost": "--cost 0", "no_logx": "--logx 0", "no_deg": "--degfeat 0",
}
ds = sys.argv[1]; variants = sys.argv[2].split(",") if len(sys.argv) > 2 else list(VARIANTS)
nseeds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
data = load(ds); te = data[6]
for v in variants:
    seeds = range(10 if v == "full" else nseeds)
    argv = ["--dataset", ds] + best[ds].split() + LAM[ds].split() + VARIANTS[v].split()
    a = V.parser().parse_args(argv); rows = []
    for s in seeds:
        r = V.run(a, data, seed=s, verbose=False)
        np.savez_compressed(f"cache/v5_{ds}_{v}_s{s}.npz", P=r["_P"][te].astype(np.float32), pred=r["_pred"][te],
                            **{f"comp_{k}": p[te].astype(np.float32) for k, p in r["_comp"].items()})
        for k in ["_P", "_pred", "_comp"]: r.pop(k)
        rows.append(r)
        print(ds, v, s, " ".join(f"{k}={x:.4f}" for k, x in r["test"].items()), f"beta={r['info']['beta']} t={r['info']['time_total']:.1f}", flush=True)
    with open(f"sweeps/final_{ds}.jsonl", "a") as fh:
        fh.write(json.dumps(dict(variant=v, argv=argv, runs=rows)) + "\n")
    print(ds, v, "MEAN", " ".join(f"{k}={np.mean([r['test'][k] for r in rows]):.4f}±{np.std([r['test'][k] for r in rows]):.4f}" for k in rows[0]["test"]), flush=True)
