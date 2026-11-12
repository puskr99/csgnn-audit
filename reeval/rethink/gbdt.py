"""Graph-augmented cost-sensitive gradient boosting (exploration)."""
import argparse, numpy as np, lightgbm as lgb, scipy.sparse as sp, time
from common import load, metrics, fit_bias, fmt
from sign import rownorm


def graph_feats(X, y, src, dst, tr, opts):
    N, C = len(y), y.max() + 1
    Xl = np.sign(X) * np.log1p(np.abs(X))
    od = np.bincount(src, minlength=N); idg = np.bincount(dst, minlength=N)
    blocks, names = [Xl, np.log1p(np.stack([od, idg], 1))], ["x", "deg"]
    Ao = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(N, N))
    Ai = Ao.T.tocsr()
    S = (Ao + Ai); S.data[:] = 1
    if "recip" in opts:
        R = Ao.multiply(Ai); blocks.append(np.log1p(np.asarray(R.sum(1)))); names.append("recip")
    ops = {}
    if "sym" in opts: ops["s"] = rownorm(S)
    if "dir" in opts: ops["o"], ops["i"] = rownorm(Ao), rownorm(Ai)
    for k, A in ops.items():
        blocks.append(np.asarray(A @ Xl)); names.append(f"mean_{k}")
        if "deg2" in opts:
            blocks.append(np.asarray(A @ np.log1p(np.stack([od, idg], 1)))); names.append(f"nbrdeg_{k}")
    if "label" in opts:   # distribution of TRAIN labels among neighbours (never includes the node's own label)
        Y = np.zeros((N, C)); Y[tr, y[tr]] = 1
        mats = {"s": S} if "sym" in opts or "dir" not in opts else {}
        if "dir" in opts: mats.update({"o": Ao, "i": Ai})
        for k, A in mats.items():
            cnt = np.asarray(A @ Y); tot = cnt.sum(1, keepdims=True)
            blocks += [cnt / np.maximum(tot, 1), np.log1p(tot)]; names += [f"lab_{k}", f"nlab_{k}"]
    return np.hstack(blocks).astype(np.float32), names


def run(a, data=None, verbose=True):
    X, y, src, dst, tr, va, te = data if data is not None else load(a.dataset)
    C = y.max() + 1
    F, _ = graph_feats(X, y, src, dst, tr, a.opts.split(","))
    w = (len(tr) / (C * np.bincount(y[tr]))) ** a.wpow
    t0 = time.time()
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=a.lr, num_leaves=a.leaves, min_child_samples=a.mcs,
                           subsample=0.8, subsample_freq=1, colsample_bytree=a.colsample, reg_lambda=a.l2,
                           verbose=-1, n_jobs=2, random_state=a.seed)
    m.fit(F[tr], y[tr], sample_weight=w[y[tr]], eval_set=[(F[va], y[va])],
          callbacks=[lgb.early_stopping(200, verbose=False)])
    p = m.predict_proba(F); lp = np.log(p + 1e-12)
    b, _ = fit_bias(y[va], lp[va])
    r = dict(iters=m.best_iteration_, time=time.time() - t0,
             val_raw=metrics(y[va], p[va]), test_raw=metrics(y[te], p[te]),
             val_cal=metrics(y[va], p[va], (lp[va] + b).argmax(1)), test_cal=metrics(y[te], p[te], (lp[te] + b).argmax(1)))
    if verbose:
        print(f"[{a.dataset} {a.opts}] iters={r['iters']} {r['time']:.0f}s")
        for k in ["val_cal", "test_raw", "test_cal"]: print(f"  {k:9s} {fmt(r[k])}")
    r["_p"] = p
    return r


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="bupt"); p.add_argument("--opts", default="")
    p.add_argument("--lr", type=float, default=0.03); p.add_argument("--leaves", type=int, default=31)
    p.add_argument("--mcs", type=int, default=20); p.add_argument("--colsample", type=float, default=0.5)
    p.add_argument("--l2", type=float, default=0.0); p.add_argument("--wpow", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
