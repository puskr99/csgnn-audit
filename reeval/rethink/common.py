import os, numpy as np, scipy.sparse as sp, pandas as pd
from sklearn.metrics import recall_score, roc_auc_score, f1_score, precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE, exist_ok=True)


def make_split(y, seed):
    """Stratified 20/20/60 split, same procedure as the shared split (seed 2 reproduces it exactly)."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, tmp, _, ytmp = train_test_split(idx, y, train_size=0.2, stratify=y, random_state=seed, shuffle=True)
    va, te = train_test_split(tmp, train_size=0.25, stratify=ytmp, random_state=seed, shuffle=True)
    return tr, va, te


def split_seed(k):
    """Split k=0 is the shared split (random_state=2); k>=1 are new splits (random_state=1000+k)."""
    return 2 if k == 0 else 1000 + k


def load(name, split=0):
    """Returns X (raw float32), y, directed edges (src,dst) without self loops/dups, split idx.
    split=0: shared split from data/*_split.npz; split=k>0: new stratified split with random_state=1000+k."""
    X, y, src, dst, tr, va, te = _load_base(name)
    if split == 0:
        return X, y, src, dst, tr, va, te
    tr, va, te = make_split(y, split_seed(split))
    return X, y, src, dst, tr, va, te


def _load_base(name):
    f = os.path.join(CACHE, f"{name}.npz")
    if os.path.exists(f):
        d = np.load(f)
        return d["X"], d["y"], d["src"], d["dst"], d["tr"], d["va"], d["te"]
    if name == "sichuan":
        df = pd.read_csv(f"{ROOT}/data/Sichuan/all_feat_with_label.csv")
        a = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
        X = np.nan_to_num(a[:, :-1].astype(np.float32))
        y = a[:, -1].astype(np.int64)
        A = sp.load_npz(f"{ROOT}/data/Sichuan/node_adj_sparse.npz").tocoo()
        src, dst = A.row.astype(np.int64), A.col.astype(np.int64)
        s = np.load(f"{ROOT}/data/Sichuan/sichuan_split.npz")
    else:
        F = np.loadtxt(f"{ROOT}/data/BUPT/TF.features")
        fid = F[:, 0].astype(np.int64); Xr = F[:, 1:].astype(np.float32)
        L = np.loadtxt(f"{ROOT}/data/BUPT/TF.labels").astype(np.int64)
        ld = dict(zip(L[:, 0], L[:, 1]))
        keep = np.array([i in ld for i in fid])
        fid = fid[keep]; X = np.nan_to_num(Xr[keep]); y = np.array([ld[i] for i in fid])
        mp = {n: i for i, n in enumerate(fid)}
        E = np.loadtxt(f"{ROOT}/data/BUPT/TF.edgelist", dtype=np.int64)
        ok = np.array([(u in mp) and (v in mp) for u, v in E])
        E = E[ok]
        src = np.array([mp[u] for u in E[:, 0]]); dst = np.array([mp[v] for v in E[:, 1]])
        s = np.load(f"{ROOT}/data/BUPT/bupt_split.npz")
    m = src != dst
    src, dst = src[m], dst[m]
    key = np.unique(src * len(y) + dst)
    src, dst = key // len(y), key % len(y)
    tr, va, te = s["train"], s["val"], s["test"]
    np.savez(f, X=X, y=y, src=src, dst=dst, tr=tr, va=va, te=te)
    return X, y, src, dst, tr, va, te


def gmean(y, p):
    r = recall_score(y, p, average=None, zero_division=0)
    return float(np.prod(np.maximum(r, 1e-6)) ** (1 / len(r)))


def metrics(y, prob, pred=None):
    C = prob.shape[1]
    prob = prob / np.maximum(prob.sum(1, keepdims=True), 1e-12)
    if pred is None:
        pred = prob.argmax(1)
    auc = roc_auc_score(y, prob[:, 1]) if C == 2 else roc_auc_score(y, prob, multi_class="ovo", average="macro")
    return dict(prec=precision_score(y, pred, average="macro", zero_division=0),
                rec=recall_score(y, pred, average="macro", zero_division=0),
                f1=f1_score(y, pred, average="macro", zero_division=0),
                auc=auc, gmean=gmean(y, pred))


def fit_bias(y, logp, iters=3, grid=np.linspace(-3, 3, 61)):
    """Coordinate-ascent per-class additive bias on log-probs maximizing G-Mean (fit on validation)."""
    C = logp.shape[1]; b = np.zeros(C)
    best = gmean(y, (logp + b).argmax(1))
    for _ in range(iters):
        for c in range(1, C):
            for v in grid:
                bb = b.copy(); bb[c] = v
                g = gmean(y, (logp + bb).argmax(1))
                if g > best + 1e-9:
                    best, b = g, bb
    return b, best


def fmt(m):
    return " ".join(f"{k}={v:.4f}" for k, v in m.items())
