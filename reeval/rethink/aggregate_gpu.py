"""Aggregate GPU-run results into tables + run-level statistics.
Reads results_gpu/{multisplit,gatcobo_multisplit,pcgnn_multisplit,fraud_runs}.jsonl and leak_permutation.json if present.
Writes results_gpu/summary.md, summary.json and per_run.csv.

Statistics (replacing the node-level bootstrap as the primary evidence):
  * mean +- s.d. over all runs (splits x seeds) and the s.d. of split-level means
  * paired comparison against CS-Boost (v5): differences computed per split (averaging seeds within a split),
    mean difference with a t-based 95% CI over splits, and a Wilcoxon signed-rank test over the
    (split, seed) pairs where both methods share a seed.
"""
import json, os, csv, numpy as np
from collections import defaultdict
from scipy import stats

R = "results_gpu"
METRICS = ["f1", "auc", "gmean"]


def read(fn):
    p = os.path.join(R, fn)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


rows = read("multisplit.jsonl") + read("gatcobo_multisplit.jsonl") + read("pcgnn_multisplit.jsonl")
_cs = read("csgnn_selected.jsonl") or read("csgnn_valsel.jsonl")   # validation-selected CSGNN supersedes the old rows
if _cs:
    rows = [r for r in rows if r["method"] != "csgnn"] + _cs
# review round 3: tuned baselines, BWGNN, rewired graphs, calibrated published baselines (trial rows excluded)
rows += [r for f in ["extra_tune.jsonl", "extra_bwgnn.jsonl", "extra_rewire.jsonl"] for r in read(f) if "trial" not in r]
rows += [r for r in read("calibrated.jsonl") if r["method"].endswith("_cal")]
TEST_TRAIN_RATIO = 3.0     # 60% test / 20% train: Nadeau-Bengio correction factor n_test / n_train
fraud = read("fraud_runs.jsonl")
out, md = {}, []

with open(os.path.join(R, "per_run.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["dataset", "split", "seed", "method"] + METRICS + ["prec", "rec"])
    for r in rows:
        w.writerow([r["dataset"], r["split"], r["seed"], r["method"]] + [round(r["test"].get(m, np.nan), 6) for m in METRICS + ["prec", "rec"]])
    for r in fraud:
        w.writerow([r["dataset"], 0, r["seed"], r["method"]] + [round(r["test"].get(m, np.nan), 6) for m in METRICS] + ["", ""])

# ------------------------------------------------------------------ telecom multi-split
by = defaultdict(list)
for r in rows:
    by[(r["dataset"], r["method"])].append(r)
for ds in sorted({r["dataset"] for r in rows}):
    md.append(f"\n## {ds}: all splits x seeds\n")
    md.append("| method | n runs | splits | " + " | ".join(f"{m} mean±sd (split-sd)" for m in METRICS) + " |")
    md.append("|---|---|---|" + "---|" * len(METRICS))
    ref = by.get((ds, "v5"), [])
    ref_split = defaultdict(list); ref_pair = {}
    for r in ref:
        ref_split[r["split"]].append(r); ref_pair[(r["split"], r["seed"])] = r
    for (d, meth), rs in sorted(by.items()):
        if d != ds: continue
        cells, rec = [], {}
        for m in METRICS:
            v = np.array([r["test"][m] for r in rs])
            sm = defaultdict(list)
            for r in rs: sm[r["split"]].append(r["test"][m])
            split_means = np.array([np.mean(x) for x in sm.values()])
            cells.append(f"{v.mean():.4f}±{v.std():.4f} ({split_means.std():.4f})")
            rec[m] = dict(mean=float(v.mean()), sd=float(v.std()), split_sd=float(split_means.std()), n=int(len(v)))
        md.append(f"| {meth} | {len(rs)} | {len(set(r['split'] for r in rs))} | " + " | ".join(cells) + " |")
        out[f"{ds}|{meth}"] = rec
    # paired vs v5
    md.append(f"\n### {ds}: paired difference CS-Boost (v5) minus method\n")
    md.append("| method | metric | mean diff over splits | 95% CI (t, over splits) | 95% CI (Nadeau-Bengio corrected) | splits won | n splits | Wilcoxon p (paired runs) | n pairs |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for (d, meth), rs in sorted(by.items()):
        if d != ds or meth == "v5" or not ref: continue
        oth_split = defaultdict(list)
        for r in rs: oth_split[r["split"]].append(r)
        common = sorted(set(oth_split) & set(ref_split))
        for m in METRICS:
            diffs = np.array([np.mean([x["test"][m] for x in ref_split[k]]) - np.mean([x["test"][m] for x in oth_split[k]]) for k in common])
            if len(diffs) >= 2:
                se = diffs.std(ddof=1) / np.sqrt(len(diffs)); tq = stats.t.ppf(0.975, len(diffs) - 1)
                ci = f"[{diffs.mean() - tq * se:+.4f}, {diffs.mean() + tq * se:+.4f}]"
                # Nadeau & Bengio (2003): overlapping random splits -> var * (1/J + n_test/n_train)
                se_nb = diffs.std(ddof=1) * np.sqrt(1 / len(diffs) + TEST_TRAIN_RATIO)
                ci_nb = f"[{diffs.mean() - tq * se_nb:+.4f}, {diffs.mean() + tq * se_nb:+.4f}]"
                nb_excl = bool(diffs.mean() - tq * se_nb > 0 or diffs.mean() + tq * se_nb < 0)
            else:
                ci, ci_nb, nb_excl = "n/a", "n/a", False
            wins = int((diffs > 0).sum())
            pairs = [(ref_pair[(r["split"], r["seed"])]["test"][m], r["test"][m]) for r in rs if (r["split"], r["seed"]) in ref_pair]
            if len(pairs) >= 5 and any(abs(a - b) > 0 for a, b in pairs):
                pw = stats.wilcoxon([a for a, _ in pairs], [b for _, b in pairs]).pvalue; pw_s = f"{pw:.2g}"
            else:
                pw_s = "n/a"
            md.append(f"| {meth} | {m} | {diffs.mean():+.4f} | {ci} | {ci_nb} | {wins}/{len(common)} | {len(common)} | {pw_s} | {len(pairs)} |")
            out[f"{ds}|paired|{meth}|{m}"] = dict(mean_diff=float(diffs.mean()) if len(diffs) else None, ci=ci, ci_nb=ci_nb,
                                                  nb_excludes_zero=nb_excl, wins=wins, wilcoxon=pw_s, n_splits=len(common), n_pairs=len(pairs))

# ------------------------------------------------------------------ fraud head-to-head
if fraud:
    fb = defaultdict(list)
    for r in fraud: fb[(r["dataset"], r["method"])].append(r)
    for ds in sorted({r["dataset"] for r in fraud}):
        md.append(f"\n## {ds}: controlled head-to-head (same split, same seeds)\n")
        md.append("| method | n | AUC | AP | Macro-F1 | G-Mean |")
        md.append("|---|---|---|---|---|---|")
        for (d, meth), rs in sorted(fb.items()):
            if d != ds: continue
            cell = lambda m: (f"{np.mean([r['test'][m] for r in rs]):.4f}±{np.std([r['test'][m] for r in rs]):.4f}" if m in rs[0]["test"] else "–")
            md.append(f"| {meth} | {len(rs)} | {cell('auc')} | {cell('ap')} | {cell('f1')} | {cell('gmean')} |")
            out[f"{ds}|{meth}"] = {m: [float(np.mean([r['test'][m] for r in rs])), float(np.std([r['test'][m] for r in rs]))] for m in rs[0]["test"]}
        ref = {r["seed"]: r for r in fb.get((ds, "GBDT+agg"), [])}
        md.append(f"\nPaired over seeds, GBDT+agg minus method (AUC): " + "; ".join(
            f"{meth}: {np.mean([ref[r['seed']]['test']['auc'] - r['test']['auc'] for r in rs if r['seed'] in ref]):+.4f}"
            for (d, meth), rs in sorted(fb.items()) if d == ds and meth != "GBDT+agg" and ref))

# ------------------------------------------------------------------ leakage test
lp = os.path.join(R, "leak_permutation.json")
if os.path.exists(lp):
    L = json.load(open(lp)); out["leak_permutation"] = L
    md.append("\n## Label-permutation leakage test\n")
    for k, v in L.items():
        if "csgnn" in k:
            md.append(f"- {k}: AUC with true test labels {v['true_labels']['auc']:.4f}, permuted {v['permuted_test_labels_mean']['auc']:.4f}, "
                      f"all-zero {v['all_test_labels_set_to_0']['auc']:.4f}, label-free {v['label_free_scores']['auc']:.4f}; "
                      f"max |Δp| under permutation {v['max_abs_change_in_test_probs_under_permutation']:.4f}")
        else:
            md.append(f"- {k}: max |Δp| = {v['max_abs_change_in_test_probs']:.2e}, identical predictions = {v['identical_test_predictions']}")

open(os.path.join(R, "summary.md"), "w").write("# GPU-run summary\n" + "\n".join(md) + "\n")
json.dump(out, open(os.path.join(R, "summary.json"), "w"), indent=1)
print("\n".join(md))
