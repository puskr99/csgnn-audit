"""Apply CSGNN's inference-time `cost_logits` (which indexes the cost matrix by the TRUE label
of every node, test nodes included) to honest model outputs and measure the metric change.
Cost matrix = CSGNN's own log1p-inverse initialisation from training labels (Eq. 21)."""
import numpy as np, sys
sys.path.insert(0, ".")
from common import load, metrics, fmt

import os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "analysis_outputs")


def csgnn_cost_matrix(ytr, C):
    n = np.bincount(ytr, minlength=C).astype(float)
    M = n[:, None] / n[None, :]          # row = c_pred, col = c_actual
    np.fill_diagonal(M, 1.0)
    return np.log1p(M)


def csgnn_cost_logits(probs, y_true, M):
    # CSGNN: logits[i,c] = cost[row=label_i, col=c] * exp(feat[i,c]); row-normalised
    z = M[y_true] * probs                # exp(log-softmax) == probs up to a per-row constant
    return z / z.sum(1, keepdims=True)


for ds in ["sichuan", "bupt"]:
    X, y, *_, tr, va, te = load(ds)
    C = y.max() + 1
    M = csgnn_cost_matrix(y[tr], C)
    print(f"\n=== {ds}  cost matrix (log1p-inverse init) ===\n{np.round(M, 3)}")
    for model in ["graphsage", "pcgnn", "proposed"]:
        d = np.load(f"{R}/{ds}_probs_{model}.npy", allow_pickle=True).item()
        p, yt = d["probs"].astype(float), d["labels"]
        print(f"  {model:9s} honest : {fmt(metrics(yt, p))}")
        print(f"  {model:9s} +leak  : {fmt(metrics(yt, csgnn_cost_logits(p, yt, M)))}")
    # what does the leak do to an uninformative model?
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(C), size=len(te))
    yt = y[te]
    print(f"  random    honest : {fmt(metrics(yt, p))}")
    print(f"  random    +leak  : {fmt(metrics(yt, csgnn_cost_logits(p, yt, M)))}")
