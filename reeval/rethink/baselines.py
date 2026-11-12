"""End-to-end full-batch GNN baselines (GCN, GraphSAGE-mean, GAT) and MLP, pure PyTorch.
--recipe 0 : z-scored raw features, unweighted CE, argmax decision        (classic setting)
--recipe 1 : log-scaled features + log degree, sqrt-inverse class weights, val-calibrated decision (same as v5)
"""
import argparse, copy, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, scipy.sparse as sp
from common import load, metrics, fit_bias
from csboost_v5 import base_features, class_weights


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sp_t(A):
    A = A.tocoo()
    return torch.sparse_coo_tensor(np.vstack([A.row, A.col]), A.data.astype(np.float32), A.shape).coalesce().to(DEVICE)


class GCN(nn.Module):
    def __init__(s, din, h, C, L, dp, A):
        super().__init__(); s.A, s.dp = A, dp
        dims = [din] + [h] * (L - 1) + [C]; s.lins = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(L))
    def forward(s, x):
        for i, l in enumerate(s.lins):
            x = torch.sparse.mm(s.A, l(F.dropout(x, s.dp, s.training)))
            if i < len(s.lins) - 1: x = F.relu(x)
        return x


class SAGE(nn.Module):
    def __init__(s, din, h, C, L, dp, A):
        super().__init__(); s.A, s.dp = A, dp
        dims = [din] + [h] * (L - 1) + [C]
        s.ls = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(L))
        s.ln = nn.ModuleList(nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(L))
    def forward(s, x):
        for i in range(len(s.ls)):
            x = F.dropout(x, s.dp, s.training)
            x = s.ls[i](x) + s.ln[i](torch.sparse.mm(s.A, x))
            if i < len(s.ls) - 1: x = F.relu(x)
        return x


class GATLayer(nn.Module):
    def __init__(s, din, dout, heads, concat=True):
        super().__init__(); s.H, s.d, s.concat = heads, dout, concat
        s.W = nn.Linear(din, heads * dout, bias=False)
        s.a_s = nn.Parameter(torch.randn(heads, dout) * 0.1); s.a_d = nn.Parameter(torch.randn(heads, dout) * 0.1)
    def forward(s, x, src, dst, N):
        h = s.W(x).view(N, s.H, s.d)
        e = F.leaky_relu((h * s.a_s).sum(-1)[src] + (h * s.a_d).sum(-1)[dst], 0.2)        # E x H
        e = e - torch.zeros(N, s.H, device=e.device).index_reduce_(0, dst, e, "amax", include_self=False)[dst]
        e = e.exp(); z = torch.zeros(N, s.H, device=e.device).index_add_(0, dst, e)[dst]
        alpha = e / (z + 1e-16)
        out = torch.zeros(N, s.H, s.d, device=h.device).index_add_(0, dst, h[src] * alpha.unsqueeze(-1))
        return out.reshape(N, -1) if s.concat else out.mean(1)


class GAT(nn.Module):
    def __init__(s, din, h, C, L, dp, edges, heads=4):
        super().__init__(); s.src, s.dst = edges; s.dp = dp
        s.l1 = GATLayer(din, h // heads, heads); s.l2 = GATLayer(h, C, 1, concat=False)
    def forward(s, x):
        N = x.shape[0]
        x = F.elu(s.l1(F.dropout(x, s.dp, s.training), s.src, s.dst, N))
        return s.l2(F.dropout(x, s.dp, s.training), s.src, s.dst, N)


class MLP(nn.Module):
    def __init__(s, din, h, C, L, dp, _):
        super().__init__()
        dims = [din] + [h] * (L - 1); mods = []
        for i in range(L - 1): mods += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dp)]
        s.net = nn.Sequential(*mods, nn.Linear(dims[-1], C))
    def forward(s, x): return s.net(x)


def run(a, data, seed):
    X, y, src, dst, tr, va, te = data
    N, C = len(y), int(y.max() + 1)
    torch.manual_seed(seed); np.random.seed(seed)
    if a.recipe:
        Xb = base_features(X, src, dst, tr, True, True); w = class_weights(y, tr, C, 0.5)
    else:
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xb = np.clip((X - mu) / sd, -15, 15).astype(np.float32); w = np.ones(C)
    S = sp.csr_matrix((np.ones(2 * len(src)), (np.r_[src, dst], np.r_[dst, src])), shape=(N, N)); S.data[:] = 1
    if a.model == "gcn":
        At = S + sp.eye(N); d = np.asarray(At.sum(1)).ravel() ** -0.5; ctx = sp_t(sp.diags(d) @ At @ sp.diags(d))
    elif a.model == "sage":
        d = np.asarray(S.sum(1)).ravel(); d[d == 0] = 1; ctx = sp_t(sp.diags(1 / d) @ S)
    elif a.model == "gat":
        Sl = (S + sp.eye(N)).tocoo(); ctx = (torch.tensor(Sl.row, dtype=torch.long, device=DEVICE), torch.tensor(Sl.col, dtype=torch.long, device=DEVICE))
    else: ctx = None
    model = {"gcn": GCN, "sage": SAGE, "gat": GAT, "mlp": MLP}[a.model](Xb.shape[1], a.hidden, C, a.layers, a.dropout, ctx).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    x, yt, trt, vat = (torch.tensor(v, device=DEVICE) for v in (Xb, y, tr, va))
    wt = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    best, bad, t0 = -1, 0, time.time()
    for ep in range(a.epochs):
        model.train(); opt.zero_grad()
        F.cross_entropy(model(x)[trt], yt[trt], weight=wt).backward(); opt.step()
        if ep % 5: continue
        model.eval()
        with torch.no_grad(): pv = F.softmax(model(x)[vat], 1).cpu().numpy()
        mv = metrics(y[va], pv); s = mv["gmean"] + mv["auc"]
        if s > best + 1e-6: best, state, bad = s, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 5
            if bad >= a.patience: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad(): P = F.softmax(model(x), 1).cpu().numpy()
    lp = np.log(P + 1e-12)
    b = fit_bias(y[va], lp[va])[0] if a.recipe else np.zeros(C)
    pred = (lp + b).argmax(1)
    return dict(val=metrics(y[va], P[va], pred[va]), test=metrics(y[te], P[te], pred[te]), time=time.time() - t0, _P=P, _pred=pred)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sichuan"); p.add_argument("--model", default="sage")
    p.add_argument("--recipe", type=int, default=1)
    p.add_argument("--hidden", type=int, default=64); p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3); p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--wd", type=float, default=5e-4); p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--seeds", type=int, nargs="*", default=[0]); p.add_argument("--out", default=None)
    a = p.parse_args(); data = load(a.dataset); rows = []
    for s in a.seeds:
        r = run(a, data, s); P = r.pop("_P"); pred = r.pop("_pred"); rows.append(r)
        np.save(f"cache/probs_{a.dataset}_{a.model}_r{a.recipe}_s{s}.npy", P[data[6]].astype(np.float32)); np.save(f"cache/pred_{a.dataset}_{a.model}_r{a.recipe}_s{s}.npy", pred[data[6]])
        print(f"[{a.dataset} {a.model} recipe={a.recipe} s={s}] " + " ".join(f"{k}={v:.4f}" for k, v in r["test"].items()) + f" t={r['time']:.0f}s", flush=True)
    if a.out:
        with open(a.out, "a") as fh: fh.write(json.dumps(dict(args=vars(a), runs=rows)) + "\n")
