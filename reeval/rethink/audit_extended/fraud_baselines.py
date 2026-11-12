"""Fair simple baselines on YelpChi / Amazon under the BWGNN/PC-GNN protocol
(40% train, remaining split 1:2 val/test, random_state=2; Amazon uses labelled nodes 3305+).
Threshold for F1-macro chosen on validation (0.05 grid, as in BWGNN)."""
import os, sys, json, time, numpy as np, scipy.io as sio, scipy.sparse as sp, lightgbm as lgb
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score

REL = {"yelp": ["net_rur", "net_rtr", "net_rsr"], "amazon": ["net_upu", "net_usu", "net_uvu"]}


def load(name):
    d = sio.loadmat({"yelp": "YelpChi.mat", "amazon": "Amazon.mat"}[name])
    X = d["features"].toarray().astype(np.float32) if sp.issparse(d["features"]) else d["features"].astype(np.float32)
    y = d["label"].ravel().astype(int)
    rels = [d[r].tocsr() for r in REL[name]] + [d["homo"].tocsr()]
    idx = np.arange(len(y)) if name == "yelp" else np.arange(3305, len(y))
    tr, rest = train_test_split(idx, train_size=0.4, stratify=y[idx], random_state=2, shuffle=True)
    va, te = train_test_split(rest, test_size=0.67, stratify=y[rest], random_state=2, shuffle=True)
    return X, y, rels, tr, va, te


def rownorm(A):
    A = A.copy().astype(np.float32); A.setdiag(0); A.eliminate_zeros(); A.data[:] = 1
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1
    return sp.diags(1 / d) @ A, np.asarray(A.sum(1)).ravel()


def feats(X, rels, agg):
    Xl = np.sign(X) * np.log1p(np.abs(X))
    B = [Xl]
    if agg:
        for A in rels:
            An, deg = rownorm(A)
            B += [np.asarray(An @ Xl), np.log1p(deg)[:, None]]
    return np.hstack(B).astype(np.float32)


def best_thr_f1(y, p):
    best, bt = -1, 0.5
    for t in np.linspace(0.05, 0.95, 19):
        f = f1_score(y, (p > t).astype(int), average="macro")
        if f > best: best, bt = f, t
    return bt


def scores(y, p, t):
    pred = (p > t).astype(int)
    r = recall_score(y, pred, average=None)
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                f1=f1_score(y, pred, average="macro"), gmean=float(np.sqrt(r[0] * r[1])))


def run_gbdt(Fm, y, tr, va, seed):
    w = np.sqrt(len(tr) / (2 * np.bincount(y[tr])))
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.6, verbose=-1, n_jobs=os.cpu_count(), random_state=seed)
    m.fit(Fm[tr], y[tr], sample_weight=w[y[tr]], eval_set=[(Fm[va], y[va])], callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict_proba(Fm)[:, 1]


def run_mlp(Fm, y, tr, va, seed, hidden=256, epochs=300):
    torch.manual_seed(seed)
    mu, sd = Fm[tr].mean(0), Fm[tr].std(0) + 1e-6
    x = torch.tensor(np.clip((Fm - mu) / sd, -15, 15), dtype=torch.float32); yt = torch.tensor(y)
    w = torch.tensor(np.sqrt(len(tr) / (2 * np.bincount(y[tr]))), dtype=torch.float32)
    net = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 2))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    trt = torch.tensor(tr); best, bad = -1, 0
    for ep in range(epochs):
        net.train(); perm = trt[torch.randperm(len(trt))]
        for i in range(0, len(perm), 2048):
            b = perm[i:i + 2048]
            loss = F.cross_entropy(net(x[b]), yt[b], weight=w); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): pv = F.softmax(net(x[va]), 1)[:, 1].numpy()
        s = roc_auc_score(y[va], pv)
        if s > best + 1e-5: best, state, bad = s, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 30: break
    net.load_state_dict(state); net.eval()
    with torch.no_grad(): return F.softmax(net(x), 1)[:, 1].numpy()


if __name__ == "__main__":
    out = {}
    for ds in sys.argv[1].split(","):
        X, y, rels, tr, va, te = load(ds)
        print(ds, X.shape, "pos rate", y[tr].mean().round(4), len(tr), len(va), len(te), flush=True)
        for agg in [0, 1]:
            Fm = feats(X, rels, agg)
            for name, fn in [("GBDT", run_gbdt), ("MLP", run_mlp)]:
                rows = []
                for seed in range(5):
                    t0 = time.time(); p = fn(Fm, y, tr, va, seed)
                    t = best_thr_f1(y[va], p[va]); r = scores(y[te], p[te], t); r["time"] = time.time() - t0; rows.append(r)
                key = f"{ds}|{name}{'+agg' if agg else ''}"
                out[key] = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in rows[0]}
                print(key, " ".join(f"{k}={v[0]:.4f}±{v[1]:.4f}" for k, v in out[key].items()), flush=True)
    json.dump(out, open(f"fraud_baselines_{sys.argv[1].replace(',', '_')}.json", "w"), indent=1)
