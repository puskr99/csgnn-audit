"""Precomputed-propagation model (SIGN-style): features propagated along directed operators once,
then a per-operator linear projection + MLP trained with cost-sensitive CE. Full-batch / mini-batch."""
import argparse, time, copy, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from common import load, metrics, fit_bias, fmt, gmean


def rownorm(A):
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1
    return sp.diags(1.0 / d) @ A


def build_ops(src, dst, N, mode):
    Ao = rownorm(sp.csr_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(N, N)))  # mean over out-nbrs
    Ai = rownorm(sp.csr_matrix((np.ones(len(src), np.float32), (dst, src)), shape=(N, N)))  # mean over in-nbrs
    S = sp.csr_matrix((np.ones(2 * len(src), np.float32), (np.r_[src, dst], np.r_[dst, src])), shape=(N, N))
    S.data[:] = 1; As = rownorm(S)
    if mode == "none": return {}
    if mode == "sym1": return {"s": As}
    if mode == "sym2": return {"s": As, "ss": As @ As}
    if mode == "dir1": return {"o": Ao, "i": Ai}
    if mode == "dir2": return {"o": Ao, "i": Ai, "oo": Ao @ Ao, "ii": Ai @ Ai, "oi": Ao @ Ai, "io": Ai @ Ao}
    if mode == "dir2s": return {"o": Ao, "i": Ai, "s": As, "oo": Ao @ Ao, "ii": Ai @ Ai, "oi": Ao @ Ai, "io": Ai @ Ao}
    raise ValueError(mode)


def make_feats(a, X, y, src, dst, tr):
    N = len(y)
    od = np.bincount(src, minlength=N); idg = np.bincount(dst, minlength=N)
    base = [X]
    if a.degfeat:
        base.append(np.log1p(np.stack([od, idg], 1)).astype(np.float32))
    B = np.hstack(base)
    if a.logx:
        B = np.sign(B) * np.log1p(np.abs(B))
    B = StandardScaler().fit(B[tr]).transform(B).astype(np.float32)
    B = np.clip(np.nan_to_num(B), -15, 15)
    ops = build_ops(src, dst, N, a.ops)
    blocks = [B] + [np.asarray(op @ B, np.float32) for op in ops.values()]
    if a.maxagg:   # elementwise max over in/out neighbours (1 hop)
        for s_, d_ in [(src, dst), (dst, src)]:
            M = np.full_like(B, -15.0)
            np.maximum.at(M, s_, B[d_])
            M[np.bincount(s_, minlength=N) == 0] = 0
            blocks.append(M)
    # standardise each propagated block on train
    out = []
    for b in blocks:
        mu, sd = b[tr].mean(0), b[tr].std(0) + 1e-6
        out.append(((b - mu) / sd).astype(np.float32))
    return out


class Net(nn.Module):
    def __init__(self, nblocks, din, h, C, L, dp, indp):
        super().__init__()
        self.proj = nn.ModuleList(nn.Linear(din, h) for _ in range(nblocks))
        layers = []
        for i in range(L):
            layers += [nn.Linear(h * nblocks if i == 0 else h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dp)]
        self.mlp = nn.Sequential(*layers, nn.Linear(h if L else h * nblocks, C))
        self.indp, self.dp = indp, dp

    def forward(self, xs):
        hs = [F.dropout(F.relu(p(F.dropout(x, self.indp, self.training))), self.dp, self.training) for p, x in zip(self.proj, xs)]
        return self.mlp(torch.cat(hs, 1))


def run(a, data=None, feats=None, verbose=True):
    X, y, src, dst, tr, va, te = data if data is not None else load(a.dataset)
    N, C = len(y), int(y.max() + 1)
    if feats is None:
        feats = make_feats(a, X, y, src, dst, tr)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    xs = [torch.tensor(f) for f in feats]
    yt = torch.tensor(y)
    ntr = np.bincount(y[tr], minlength=C)
    w = (len(tr) / (C * ntr)) ** a.wpow
    if a.cost: w = w * np.array(a.cost)
    w = torch.tensor(w / w.mean(), dtype=torch.float32)
    model = Net(len(xs), xs[0].shape[1], a.hidden, C, a.layers, a.dropout, a.indrop)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    best, bad, t0 = -1, 0, time.time()
    trn = torch.tensor(tr)
    for ep in range(a.epochs):
        model.train()
        perm = trn[torch.randperm(len(trn))]
        bs = a.batch if a.batch > 0 else len(trn)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            out = model([x[b] for x in xs])
            loss = F.cross_entropy(out, yt[b], weight=w, label_smoothing=a.ls)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lv = F.log_softmax(model([x[va] for x in xs]), 1).numpy()
        if a.select == "gmean": s = gmean(y[va], lv.argmax(1))
        elif a.select == "auc": s = metrics(y[va], np.exp(lv))["auc"]
        else: s = gmean(y[va], lv.argmax(1)) + metrics(y[va], np.exp(lv))["auc"]
        if s > best + 1e-6:
            best, state, bad, bep = s, copy.deepcopy(model.state_dict()), 0, ep
        else:
            bad += 1
            if bad >= a.patience: break
    tt = time.time() - t0
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        lp = F.log_softmax(model(xs), 1).numpy()
    res = dict(best_ep=bep, time=tt, val_raw=metrics(y[va], np.exp(lp[va])), test_raw=metrics(y[te], np.exp(lp[te])))
    b, _ = fit_bias(y[va], lp[va]); res["bias"] = b.tolist()
    res["val_cal"] = metrics(y[va], np.exp(lp[va]), (lp[va] + b).argmax(1))
    res["test_cal"] = metrics(y[te], np.exp(lp[te]), (lp[te] + b).argmax(1))
    if verbose:
        print(f"[{a.dataset} ops={a.ops}] ep{bep} {tt:.1f}s")
        for k in ["val_raw", "test_raw", "val_cal", "test_cal"]:
            print(f"  {k:9s} {fmt(res[k])}")
    res["_lp"] = lp
    return res


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="bupt")
    p.add_argument("--ops", default="dir2")
    p.add_argument("--degfeat", type=int, default=1)
    p.add_argument("--logx", type=int, default=0)
    p.add_argument("--maxagg", type=int, default=0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--indrop", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--ls", type=float, default=0.0)
    p.add_argument("--wpow", type=float, default=0.5)
    p.add_argument("--cost", type=float, nargs="*", default=None)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--select", default="gmean")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
