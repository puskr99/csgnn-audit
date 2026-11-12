"""Aggregate revision experiments A and C into the LaTeX rows used in the paper (lead vs per-split-tuned CS-Boost)."""
import json, collections, glob, numpy as np
from scipy import stats
RATIO = 3.0  # n_test / n_train for 20/20/60 splits (Nadeau-Bengio correction)
R = []
for f in ["results_rev/pertune.jsonl", "results_rev/pertune_gpu_bupt.jsonl", "results_rev/pertune_gpu_sichuan.jsonl", "results_rev/pertune_gpu_bwgnn_sichuan.jsonl", "results_rev/pertune_gpu_bwgnn_bupt.jsonl"]:
    R += [json.loads(l) for l in open(f)]
for f in ["results_rev/logpub.jsonl", "results_rev/logpub_csgnn_recipe.jsonl", "results_rev/logpub_csgnn_orig_sichuan.jsonl", "results_rev/logpub_gpu.jsonl"]:
    try:
        for l in open(f):
            r = json.loads(l)
            if f == "results_rev/logpub.jsonl" and r["method"] == "csgnn": continue
            r["method"] = f"{r['method']}_{r['feat']}"; R.append(r)
    except FileNotFoundError: pass
by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in R: by[(r["dataset"], r["method"])][r["split"]].append(r)
# CSGNN: as in the main table, one run per split = the restart (seed 0, 1 or 2) with the best validation G-Mean + AUC
for key in [k for k in by if k[1].startswith("csgnn")]:
    for sp_, v in by[key].items():
        by[key][sp_] = [max(v, key=lambda x: x["val"]["gmean"] + x["val"]["auc"])]
def sm(d, m, key="test"): return {k: np.mean([x[key][m] for x in v]) for k, v in d.items()}
out = {}
for ds in ["sichuan", "bupt"]:
    ref = by[(ds, "v5_pertuned")]
    print(f"== {ds}")
    for meth in ["v5_pertuned", "v5_fixed", "mlp_r1_pertuned", "sage_r1_pertuned", "gcn_r1_pertuned", "gat_r1_pertuned", "bwgnn_r1_pertuned",
                 "pcgnn_orig", "pcgnn_recipe", "csgnn_orig", "csgnn_recipe"]:
        d = by.get((ds, meth))
        if not d: print(meth, "missing"); continue
        n = sum(len(v) for v in d.values())
        allr = [x for v in d.values() for x in v]
        key = "test_cal" if meth.startswith("pcgnn") else "test"
        mean = {m: np.mean([x[key][m] for x in allr]) for m in ["f1", "auc", "gmean"]}
        sd = {m: np.std([x[key][m] for x in allr]) for m in ["f1", "auc", "gmean"]}
        common = sorted(set(d) & set(ref))
        rg, og = sm(ref, "gmean"), sm(d, "gmean", key); ra, oa = sm(ref, "auc"), sm(d, "auc", key)
        diffs = np.array([rg[k] - og[k] for k in common])
        won = sum((rg[k] > og[k]) and (ra[k] > oa[k]) for k in common)
        if meth != "v5_pertuned" and len(diffs) > 1:
            tq = stats.t.ppf(0.975, len(diffs) - 1); se = diffs.std(ddof=1) * np.sqrt(1 / len(diffs) + RATIO)
            lead = f"{diffs.mean():+.4f} [{diffs.mean()-tq*se:+.4f}, {diffs.mean()+tq*se:+.4f}]"
        else: lead = "--"
        print(f"{meth:18s} n={n:2d} f1={mean['f1']:.4f} auc={mean['auc']:.4f}±{sd['auc']:.4f} g={mean['gmean']:.4f}±{sd['gmean']:.4f} lead={lead} won={won}/{len(common)}")
        out[f"{ds}|{meth}"] = dict(n=n, mean=mean, sd=sd, lead=lead, won=f"{won}/{len(common)}")
json.dump(out, open("results_rev/rev_summary.json", "w"), indent=1)
