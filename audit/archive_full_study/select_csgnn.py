"""Validation-only selection for label-free CSGNN.

Input : results_gpu/csgnn_valsel.jsonl   (one line per dataset x split x seed; each line already carries the
         label-free variant chosen on validation, see multisplit.py)
Output: results_gpu/csgnn_selected.jsonl (one line per dataset x split: the seed whose run has the best validation
         G-Mean + AUC, i.e. seed treated as a hyperparameter, selected on validation only)
        results_gpu/csgnn_stability.json  (collapse statistics: runs whose validation G-Mean < 0.25)

CSGNN's BUPT training is strongly seed-dependent (several seeds converge to near-majority predictions); selecting
the restart on validation is what a practitioner would do and is favourable to CSGNN.
"""
import json, os, numpy as np
from collections import defaultdict

R = "results_gpu"
rows = [json.loads(l) for l in open(os.path.join(R, "csgnn_valsel.jsonl"))]
by = defaultdict(list)
for r in rows:
    by[(r["dataset"], r["split"])].append(r)

score = lambda r: r["val"]["gmean"] + r["val"]["auc"]
sel, stab = [], {}
for (ds, k), rs in sorted(by.items()):
    best = max(rs, key=score)
    sel.append(dict(dataset=ds, split=k, seed=f"sel{best['seed']}", method="csgnn", variant=best["variant"],
                    seeds_tried=sorted(r["seed"] for r in rs), val=best["val"], test=best["test"]))
for ds in sorted({r["dataset"] for r in rows}):
    rs = [r for r in rows if r["dataset"] == ds]
    coll = [r for r in rs if r["val"]["gmean"] < 0.25]
    stab[ds] = dict(runs=len(rs), collapsed=len(coll),
                    collapsed_seeds_0_2=sum(1 for r in coll if r["seed"] in (0, 1, 2)),
                    runs_seeds_0_2=sum(1 for r in rs if r["seed"] in (0, 1, 2)),
                    mean_test_gmean_all_runs=float(np.mean([r["test"]["gmean"] for r in rs])),
                    mean_test_gmean_selected=float(np.mean([s["test"]["gmean"] for s in sel if s["dataset"] == ds])),
                    selected_seeds=[s["seed"] for s in sel if s["dataset"] == ds])
with open(os.path.join(R, "csgnn_selected.jsonl"), "w") as fh:
    for s in sel: fh.write(json.dumps(s) + "\n")
json.dump(stab, open(os.path.join(R, "csgnn_stability.json"), "w"), indent=1)
print(json.dumps(stab, indent=1))
