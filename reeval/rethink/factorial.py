"""Full 2^4 factorial over the recipe ingredients, for a graph-free MLP and GraphSAGE (same trainer as baselines.py).

Ingredients (each on/off independently):
  L = signed log1p scaling of the features           (off: z-score of raw features, same clipping)
  D = log in/out-degree features appended
  W = square-root inverse-frequency class weights    (off: unweighted cross-entropy)
  K = validation-calibrated per-class decision offsets maximising validation G-Mean (off: arg-max)
L=D=W=K=0 is the classic setting (recipe 0), L=D=W=K=1 is the recipe-matched setting (recipe 1).
Calibration only changes decisions, so each trained model is scored with and without K (one training per L,D,W).

Usage: python factorial.py --datasets sichuan,bupt --models mlp,sage --splits 0 1 2 3 4 --seeds 0 1 2
Writes one JSON line per (dataset, model, split, seed, L, D, W, K) to --out; resumable.
"""
import argparse, copy, itertools, json, os, time, numpy as np, scipy.sparse as sp, torch, torch.nn.functional as F
from common import load, metrics, fit_bias
from csboost_v5 import base_features, class_weights
import baselines as B

DEVICE = B.DEVICE if hasattr(B, "DEVICE") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
HP = dict(hidden=64, layers=2, dropout=0.3, lr=0.01, wd=5e-4, epochs=1000, patience=100)   # = multisplit BASE_DEFAULTS


def train(model_name, Xb, w, data, seed):
    X, y, src, dst, tr, va, te = data
    N, C = len(y), int(y.max() + 1)
    torch.manual_seed(seed); np.random.seed(seed)
    if model_name == "sage":
        S = sp.csr_matrix((np.ones(2 * len(src)), (np.r_[src, dst], np.r_[dst, src])), shape=(N, N)); S.data[:] = 1
        d = np.asarray(S.sum(1)).ravel(); d[d == 0] = 1; ctx = B.sp_t(sp.diags(1 / d) @ S)
        model = B.SAGE(Xb.shape[1], HP["hidden"], C, HP["layers"], HP["dropout"], ctx).to(DEVICE)
    else:
        model = B.MLP(Xb.shape[1], HP["hidden"], C, HP["layers"], HP["dropout"], None).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    x, yt, trt, vat = (torch.tensor(v, device=DEVICE) for v in (Xb, y, tr, va))
    wt = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    best, bad = -1, 0
    for ep in range(HP["epochs"]):
        model.train(); opt.zero_grad()
        F.cross_entropy(model(x)[trt], yt[trt], weight=wt).backward(); opt.step()
        if ep % 5: continue
        model.eval()
        with torch.no_grad(): pv = F.softmax(model(x)[vat], 1).cpu().numpy()
        mv = metrics(y[va], pv); s = mv["gmean"] + mv["auc"]           # same early-stopping score as baselines.py
        if s > best + 1e-6: best, state, bad = s, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 5
            if bad >= HP["patience"]: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad(): P = F.softmax(model(x), 1).cpu().numpy()
    return P


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="sichuan,bupt")
    p.add_argument("--models", default="mlp,sage")
    p.add_argument("--splits", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--out", default="results_gpu/factorial.jsonl")
    A = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    done = set()
    if os.path.exists(A.out):
        for l in open(A.out):
            r = json.loads(l); done.add((r["dataset"], r["model"], r["split"], r["seed"], r["L"], r["D"], r["W"]))
    print("device:", DEVICE, flush=True)
    for ds in A.datasets.split(","):
        for k in A.splits:
            data = load(ds, k); X, y, src, dst, tr, va, te = data; C = int(y.max() + 1)
            for mdl in A.models.split(","):
                for s in A.seeds:
                    for L, D, W in itertools.product([0, 1], repeat=3):
                        if (ds, mdl, k, s, L, D, W) in done: continue
                        t0 = time.time()
                        Xb = base_features(X, src, dst, tr, bool(L), bool(D))
                        w = class_weights(y, tr, C, 0.5) if W else np.ones(C)
                        P = train(mdl, Xb, w, data, s)
                        lp = np.log(P + 1e-12)
                        b = fit_bias(y[va], lp[va])[0]
                        with open(A.out, "a") as fh:
                            for K in (0, 1):
                                pred = (lp + (b if K else 0)).argmax(1)
                                fh.write(json.dumps(dict(dataset=ds, model=mdl, split=k, seed=s, L=L, D=D, W=W, K=K,
                                                         val=metrics(y[va], P[va], pred[va]), test=metrics(y[te], P[te], pred[te]),
                                                         time=time.time() - t0)) + "\n")
                        print(ds, k, mdl, s, f"L{L}D{D}W{W}", f"{time.time() - t0:.0f}s", flush=True)
