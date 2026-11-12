"""CS-Boost-GNN v5 — cost-sensitive boosted trees + decoupled GraphSAGE, validation-weighted ensemble,
validation-calibrated cost-sensitive decision rule.

Pipeline (every fitted quantity uses the training split; every selection uses the validation split only):
  1. Structure-aware input: signed log1p of node features + log in/out degree, standardised on train.
  2. Learner A (Boost): LightGBM, cost-sensitive sample weights w_c = lambda_c * sqrt(N / (C n_c)), early stop on val.
  3. Learner B (GNN): decoupled GraphSAGE  h = MLP([W_0 x_v || W_1 mean_{u in N(v)} x_u || ...]), same class weights.
  4. Ensemble p = beta * p_GNN + (1 - beta) * p_GBDT, beta chosen on val AUC.
  5. Decision rule: y = argmax_c (log p_c + b_c), b fitted on val to maximise G-Mean (post-hoc cost-sensitive threshold).
"""
import argparse, json, time, copy, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import scipy.sparse as sp, lightgbm as lgb
from common import load, metrics, fit_bias, gmean

# ---------------------------------------------------------------- features / operators
def rownorm(A):
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1
    return (sp.diags(1.0 / d) @ A).tocsr()


def operators(src, dst, N, ops):
    Ao = sp.csr_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(N, N))
    Ai = Ao.T.tocsr()
    S = (Ao + Ai).tocsr(); S.data[:] = 1
    As, Ao, Ai = rownorm(S), rownorm(Ao), rownorm(Ai)
    return {"none": [], "sym1": [As], "sym2": [As, As @ As], "dir1": [Ao, Ai], "dir2": [Ao, Ai, Ao @ Ao, Ai @ Ai]}[ops]


def base_features(X, src, dst, tr, logx=True, degfeat=True):
    N = len(X)
    B = [X]
    if degfeat:
        B.append(np.log1p(np.stack([np.bincount(src, minlength=N), np.bincount(dst, minlength=N)], 1)))
    B = np.hstack(B).astype(np.float64)
    if logx:
        B = np.sign(B) * np.log1p(np.abs(B))
    mu, sd = B[tr].mean(0), B[tr].std(0) + 1e-8
    return np.clip((B - mu) / sd, -15, 15).astype(np.float32)


def class_weights(y, tr, C, wpow=0.5, lam=None):
    w = (len(tr) / (C * np.bincount(y[tr], minlength=C))) ** wpow
    if lam is not None:
        w = w * np.asarray(lam)
    return w / w.mean()


# ---------------------------------------------------------------- learner A: cost-sensitive GBDT
def fit_gbdt(Xb, y, tr, va, w, a, seed):
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=a.gb_lr, num_leaves=a.gb_leaves,
                           min_child_samples=a.gb_mcs, subsample=0.8, subsample_freq=1,
                           colsample_bytree=a.gb_col, verbose=-1, n_jobs=os.cpu_count(), random_state=seed)
    m.fit(Xb[tr], y[tr], sample_weight=w[y[tr]], eval_set=[(Xb[va], y[va])],
          callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict_proba(Xb), m.best_iteration_


# ---------------------------------------------------------------- learner B: decoupled GraphSAGE
class DSAGE(nn.Module):
    def __init__(self, nb, din, h, C, L, dp):
        super().__init__()
        self.proj = nn.ModuleList(nn.Linear(din, h) for _ in range(nb))
        mods, dim = [], h * nb
        for _ in range(L):
            mods += [nn.Linear(dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dp)]; dim = h
        self.mlp = nn.Sequential(*mods, nn.Linear(dim, C)); self.dp = dp

    def forward(self, xs):
        return self.mlp(torch.cat([F.dropout(F.relu(p(x)), self.dp, self.training) for p, x in zip(self.proj, xs)], 1))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_gnn(blocks, y, tr, va, w, a, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    C = int(y.max() + 1)
    xs = [torch.tensor(b, device=DEVICE) for b in blocks]; yt = torch.tensor(y, device=DEVICE)
    model = DSAGE(len(xs), xs[0].shape[1], a.hidden, C, a.layers, a.dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    wt = torch.tensor(w, dtype=torch.float32, device=DEVICE); trn = torch.tensor(tr, device=DEVICE)
    vat = torch.tensor(va, device=DEVICE)
    best, bad, epochs = -1, 0, 0
    for ep in range(a.epochs):
        model.train(); perm = trn[torch.randperm(len(trn), device=DEVICE)]
        for i in range(0, len(perm), a.batch):
            b = perm[i:i + a.batch]
            loss = F.cross_entropy(model([x[b] for x in xs]), yt[b], weight=wt)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = F.softmax(model([x[vat] for x in xs]), 1).cpu().numpy()
        mv = metrics(y[va], pv); s = mv["gmean"] + mv["auc"]
        epochs = ep + 1
        if s > best + 1e-6: best, state, bad = s, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= a.patience: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        return F.softmax(model(xs), 1).cpu().numpy(), epochs


def propagate(Xb, ops, tr):
    out = [Xb]
    for A in ops:
        Z = np.asarray(A @ Xb)
        out.append(((Z - Z[tr].mean(0)) / (Z[tr].std(0) + 1e-6)).astype(np.float32))
    return out


# ---------------------------------------------------------------- full pipeline
def run(a, data=None, seed=0, verbose=True):
    X, y, src, dst, tr, va, te = data if data is not None else load(a.dataset)
    N, C = len(y), int(y.max() + 1)
    t0 = time.time()
    Xb = base_features(X, src, dst, tr, bool(a.logx), bool(a.degfeat))
    w = class_weights(y, tr, C, a.wpow if a.cost else 0.0, a.lam if a.cost else None)
    probs, info = {}, {}
    if a.gbdt:
        t = time.time(); probs["gbdt"], info["gbdt_iters"] = fit_gbdt(Xb, y, tr, va, w, a, seed); info["t_gbdt"] = time.time() - t
    if a.gnn:
        t = time.time(); blocks = propagate(Xb, operators(src, dst, N, a.ops), tr); info["t_prop"] = time.time() - t
        t = time.time(); probs["gnn"], info["gnn_epochs"] = fit_gnn(blocks, y, tr, va, w, a, seed); info["t_gnn"] = time.time() - t
    if len(probs) == 2:
        grid = np.linspace(0, 1, 11)
        sc = [metrics(y[va], b * probs["gnn"][va] + (1 - b) * probs["gbdt"][va])["auc"] for b in grid]
        beta = float(grid[int(np.argmax(sc))]) if a.beta < 0 else a.beta
        P = beta * probs["gnn"] + (1 - beta) * probs["gbdt"]
    else:
        beta = None; P = next(iter(probs.values()))
    lp = np.log(P + 1e-12)
    bias = fit_bias(y[va], lp[va])[0] if a.calib else np.zeros(C)
    pred = (lp + bias).argmax(1)
    info.update(beta=beta, bias=bias.tolist(), time_total=time.time() - t0)
    res = dict(val=metrics(y[va], P[va], pred[va]), test=metrics(y[te], P[te], pred[te]), info=info)
    if verbose:
        print(f"[{a.dataset} seed={seed}] " + " ".join(f"{k}={v:.4f}" for k, v in res["test"].items()) +
              f" | beta={beta} t={info['time_total']:.1f}s")
    res["_P"], res["_pred"], res["_comp"] = P, pred, probs
    return res


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="bupt")
    p.add_argument("--logx", type=int, default=1); p.add_argument("--degfeat", type=int, default=1)
    p.add_argument("--cost", type=int, default=1); p.add_argument("--wpow", type=float, default=0.5)
    p.add_argument("--lam", type=float, nargs="*", default=None)
    p.add_argument("--calib", type=int, default=1)
    p.add_argument("--gbdt", type=int, default=1); p.add_argument("--gnn", type=int, default=1)
    p.add_argument("--beta", type=float, default=-1, help="<0: choose on val AUC")
    p.add_argument("--ops", default="sym1")
    p.add_argument("--hidden", type=int, default=128); p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3); p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=1e-4); p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=2048); p.add_argument("--patience", type=int, default=30)
    p.add_argument("--gb_lr", type=float, default=0.03); p.add_argument("--gb_leaves", type=int, default=31)
    p.add_argument("--gb_mcs", type=int, default=20); p.add_argument("--gb_col", type=float, default=0.5)
    p.add_argument("--seeds", type=int, nargs="*", default=[0])
    p.add_argument("--out", default=None)
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    data = load(a.dataset); rows = []
    for s in a.seeds:
        r = run(a, data, seed=s); r.pop("_P"); r.pop("_pred"); r.pop("_comp"); rows.append(r)
    if len(rows) > 1:
        for k in rows[0]["test"]:
            v = [r["test"][k] for r in rows]; print(f"  {k:6s} {np.mean(v):.4f} ± {np.std(v):.4f}")
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(json.dumps(dict(args=vars(a), runs=rows)) + "\n")
