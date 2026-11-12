"""Fraud-vs-rest ranking quality: AP and precision/recall at review budgets; PR curves (PDF)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve
from common import load
import os
R = os.environ.get("R", "../results/analysis_outputs")  # saved test probabilities of PC-GNN / GAT-COBO (not included)
COL = {"v5": "#2a78d6", "pcgnn": "#eb6834", "sage_r1": "#1baf7a", "gatcobo": "#4a3aa7", "csgnn": "#e34948"}
NAME = {"v5": "CS-Boost", "pcgnn": "PC-GNN", "gatcobo": "GAT-COBO", "sage_r1": "GraphSAGE + recipe", "csgnn": "CSGNN (label-free)"}
STY = {"v5": "-", "pcgnn": "--", "gatcobo": "-.", "sage_r1": ":", "csgnn": (0, (5, 1, 1, 1))}
out = {}
for ds in ["sichuan", "bupt"]:
    X, y, s, d, tr, va, te = load(ds); order = np.argsort(te); yt = y[te][order]
    S = {}
    # proposed: mean over the 10 seeds is not used; report seed-wise mean ± sd for AP, curve from seed 0
    S["v5"] = [np.load(f"cache/v5_{ds}_full_s{k}.npz")["P"][order][:, 1] for k in range(10)]
    S["sage_r1"] = [np.load(f"cache/probs_{ds}_sage_r1_s{k}.npy")[order][:, 1] for k in range(3)]
    for m in ["pcgnn", "gatcobo"]:
        S[m] = [np.load(f"{R}/{ds}_probs_{m}.npy", allow_pickle=True).item()["probs"][:, 1]]
    S["csgnn"] = [np.load(f"cache/csgnn_{ds}_e200_es1_s42.npy", allow_pickle=True).item()["fix_pred"][order][:, 1]]  # best label-free variant (predicted-class repair)
    pos = (yt == 1); n = len(yt); res = {}
    for m, runs in S.items():
        rows = []
        for sc in runs:
            idx = np.argsort(-sc); r = dict(ap=average_precision_score(pos, sc))
            for b in [0.01, 0.05]:
                k = int(round(b * n)); hit = pos[idx[:k]].sum()
                r[f"p@{b}"] = hit / k; r[f"r@{b}"] = hit / pos.sum()
            rows.append(r)
        res[m] = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in rows[0]}
        print(ds, f"{m:8s}", " ".join(f"{k}={v[0]:.4f}±{v[1]:.4f}" for k, v in res[m].items()))
    out[ds] = dict(res=res, prevalence=float(pos.mean()), n=int(n), k1=int(round(0.01 * n)), k5=int(round(0.05 * n)))
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for m in ["v5", "pcgnn", "sage_r1", "gatcobo", "csgnn"]:
        p, r, _ = precision_recall_curve(pos, S[m][0])
        ax.plot(r, p, linestyle=STY[m], color=COL[m], lw=2 if m == "v5" else 1.5, label=f"{NAME[m]} (AP {res[m]['ap'][0]:.3f})")
    ax.axhline(pos.mean(), color="#8a8984", lw=1, ls=":", label=f"Random (prevalence {pos.mean():.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(color="#e4e3df", lw=0.6); [ax.spines[k].set_visible(False) for k in ["top", "right"]]
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(f"paper/pr_{ds}.pdf"); fig.savefig(f"paper/pr_{ds}.png", dpi=130); plt.close(fig)
json.dump(out, open("sweeps/ranking.json", "w"), indent=1)
