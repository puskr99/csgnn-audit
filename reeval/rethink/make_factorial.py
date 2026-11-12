"""Summarise results_gpu/factorial.jsonl (2^4 factorial over recipe ingredients).

Main effect of ingredient X = mean over the 8 settings of the other three ingredients of
[metric(X on) - metric(X off)], computed per split (seeds averaged), then mean and t-based 95% CI over splits.
Writes results_gpu/factorial_summary.md, results_gpu/factorial_summary.json and paper/table_factorial.tex.
"""
import json, os, itertools, numpy as np
from collections import defaultdict
from scipy import stats

R = "results_gpu"
rows = [json.loads(l) for l in open(os.path.join(R, "factorial.jsonl"))]
ING = [("L", "log-scaling"), ("D", "degree features"), ("W", "class weights"), ("K", "calibrated decision")]
METRICS = ["gmean", "auc", "f1"]

cell = defaultdict(list)                    # (ds, model, split, L,D,W,K) -> list over seeds of test dicts
for r in rows:
    cell[(r["dataset"], r["model"], r["split"], r["L"], r["D"], r["W"], r["K"])].append(r["test"])


def cmean(ds, mdl, k, combo, m):
    v = cell.get((ds, mdl, k) + tuple(combo))
    return None if not v else float(np.mean([t[m] for t in v]))


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2: return float(x.mean()), float("nan"), float("nan")
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


out, md = {}, ["# Recipe factorial (2^4): log-scaling L, degree features D, class weights W, calibrated decision K\n"]
combos = list(itertools.product([0, 1], repeat=4))
for ds in sorted({r["dataset"] for r in rows}):
    for mdl in sorted({r["model"] for r in rows if r["dataset"] == ds}):
        splits = sorted({r["split"] for r in rows if r["dataset"] == ds and r["model"] == mdl})
        md.append(f"\n## {ds} / {mdl} ({len(splits)} splits)\n")
        md.append("| L D W K | " + " | ".join(METRICS) + " |\n|---|" + "---|" * len(METRICS))
        for c in combos:
            vals = [np.mean([v for v in (cmean(ds, mdl, k, c, m) for k in splits) if v is not None] or [np.nan]) for m in METRICS]
            md.append(f"| {' '.join(map(str, c))} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
            out[f"{ds}|{mdl}|cell|{''.join(map(str, c))}"] = dict(zip(METRICS, map(float, vals)))
        md.append("\n| ingredient | " + " | ".join(f"main effect on {m} (95% CI over splits)" for m in METRICS) + " |\n|---|" + "---|" * len(METRICS))
        for i, (code, name) in enumerate(ING):
            cells = []
            for m in METRICS:
                per_split = []
                for k in splits:
                    diffs = []
                    for c in combos:
                        if c[i] == 1: continue
                        on = list(c); on[i] = 1
                        a, b = cmean(ds, mdl, k, on, m), cmean(ds, mdl, k, c, m)
                        if a is not None and b is not None: diffs.append(a - b)
                    if diffs: per_split.append(np.mean(diffs))
                mu, lo, hi = ci(per_split)
                out[f"{ds}|{mdl}|main|{code}|{m}"] = dict(mean=mu, lo=lo, hi=hi, n=len(per_split))
                cells.append(f"{mu:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            md.append(f"| {name} ({code}) | " + " | ".join(cells) + " |")

os.makedirs(R, exist_ok=True)
open(os.path.join(R, "factorial_summary.md"), "w").write("\n".join(md) + "\n")
json.dump(out, open(os.path.join(R, "factorial_summary.json"), "w"), indent=1)

# ------------------------------------------------------------------ LaTeX table for the paper
models = [("mlp", "MLP"), ("sage", "GraphSAGE")]
dsets = [("sichuan", "Sichuan"), ("bupt", "BUPT")]
have = lambda ds, mdl: f"{ds}|{mdl}|cell|0000" in out


def eff(ds, mdl, code, m):
    e = out.get(f"{ds}|{mdl}|main|{code}|{m}")
    if not e: return "--"
    sig = (e["lo"] > 0 or e["hi"] < 0) if np.isfinite(e["lo"]) else False
    s = f"{e['mean']:+.3f}"
    return f"\\textbf{{{s}}}" if sig else s


cols = [(ds, mdl) for ds, _ in dsets for mdl, _ in models if have(ds, mdl)]
lines = []
for m, mname in [("gmean", "G-Mean"), ("auc", "AUC")]:
    lines.append(f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\textit{{Main effect on {mname}}}}} \\\\")
    for code, name in ING:
        lines.append(f"{name} & " + " & ".join(eff(ds, mdl, code, m) for ds, mdl in cols) + " \\\\")
for lab, c in [("All off (classic)", "0000"), ("All on (recipe)", "1111")]:
    lines.append(f"{lab}, G-Mean & " + " & ".join(f"{out[f'{ds}|{mdl}|cell|{c}']['gmean']:.3f}" for ds, mdl in cols) + " \\\\")
head = (" & ".join(f"\\multicolumn{{{sum(have(ds, m) for m, _ in models)}}}{{c}}{{{dn}}}" for ds, dn in dsets)
        + " \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n & " + " & ".join(mn for ds, _ in dsets for mdl, mn in models if have(ds, mdl)))
tex = ("\\begin{table}[!t]\n\\caption{Recipe ingredients switched on and off independently ($2^4$ factorial, five splits "
       "$\\times$ three seeds). Main effect = average change when the ingredient is switched on, over all settings of the "
       "other three; bold = 95\\% CI over splits excludes zero. Runs are separate from those in Table~\\ref{tab:main}; the all-off and all-on cells match its classic and recipe-matched rows to within 0.003.}\n\\label{tab:factorial}\n\\centering\\scriptsize\n"
       f"\\begin{{tabular}}{{l{'c' * len(cols)}}}\n\\toprule\n & {head} \\\\\n\\midrule\n" + "\n".join(lines) +
       "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
os.makedirs("paper", exist_ok=True)
open("paper/table_factorial.tex", "w").write(tex)
print("\n".join(md))
