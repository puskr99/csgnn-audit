"""Controlled head-to-head on YelpChi / Amazon (addresses: GNN numbers copied from papers).
Same split as BWGNN / PC-GNN (40% train, rest 1:2 val/test, random_state=2; Amazon labelled users 3305+),
same seeds for every method, F1 threshold chosen on validation (0.05 grid).

Methods: MLP (log features), GBDT, GBDT + neighbour means (from audit_extended/fraud_baselines.py),
BWGNN homo and hetero (bwgnn_port.py, faithful port of the official code), and optionally the official
BWGNN via DGL (--official_bwgnn, needs dgl + a clone of the official repo in third_party/).

Data: fraud_data/YelpChi.mat and fraud_data/Amazon.mat (unzip from the CARE-GNN repo, see run_all.sh).
Usage: python fraud_runs.py --datasets yelp,amazon --seeds 0 1 2 3 4
"""
import argparse, json, os, sys, time, numpy as np, scipy.io as sio, scipy.sparse as sp, torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_extended"))
import fraud_baselines as FB
import bwgnn_port as BW

p = argparse.ArgumentParser()
p.add_argument("--datasets", default="yelp,amazon")
p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
p.add_argument("--methods", default="MLP,GBDT,GBDT+agg,BWGNN-homo,BWGNN-hetero")
p.add_argument("--data_dir", default="fraud_data")
p.add_argument("--official_bwgnn", action="store_true")
p.add_argument("--out", default="results_gpu/fraud_runs.jsonl")
A = p.parse_args()
os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REL = {"yelp": ["net_rur", "net_rtr", "net_rsr"], "amazon": ["net_upu", "net_usu", "net_uvu"]}

done = set()
if os.path.exists(A.out):
    for l in open(A.out):
        r = json.loads(l); done.add((r["dataset"], r["method"], r["seed"]))


def load(name):
    d = sio.loadmat(os.path.join(A.data_dir, {"yelp": "YelpChi.mat", "amazon": "Amazon.mat"}[name]))
    X = d["features"].toarray().astype(np.float32) if sp.issparse(d["features"]) else d["features"].astype(np.float32)
    y = d["label"].ravel().astype(int)
    rels = [d[r].tocsr() for r in REL[name]]
    idx = np.arange(len(y)) if name == "yelp" else np.arange(3305, len(y))
    tr, rest = train_test_split(idx, train_size=0.4, stratify=y[idx], random_state=2, shuffle=True)
    va, te = train_test_split(rest, test_size=0.67, stratify=y[rest], random_state=2, shuffle=True)
    return X, y, rels, d["homo"].tocsr(), tr, va, te


def write(rec):
    with open(A.out, "a") as fh: fh.write(json.dumps(rec) + "\n")


for ds in A.datasets.split(","):
    X, y, rels, homo, tr, va, te = load(ds)
    feats = {}
    for m in A.methods.split(","):
        for s in A.seeds:
            if (ds, m, s) in done: continue
            t0 = time.time(); torch.manual_seed(s); np.random.seed(s)
            if m in ("MLP", "GBDT", "GBDT+agg"):
                agg = m.endswith("+agg")
                if agg not in feats: feats[agg] = FB.feats(X, rels + [homo], agg)
                fn = FB.run_mlp if m == "MLP" else FB.run_gbdt
                pr = fn(feats[agg], y, tr, va, s)
                t = FB.best_thr_f1(y[va], pr[va]); r = FB.scores(y[te], pr[te], t)
            else:
                homo_flag = m == "BWGNN-homo"
                g = BW.build(rels, homo_flag, DEV)
                Model = BW.BWGNN if homo_flag else BW.BWGNN_Hetero
                model = Model(X.shape[1], 64, 2, g, d=2).to(DEV)
                r = BW.train(model, X, y, tr, va, te, epochs=100, device=DEV)
            r = {k: float(v) for k, v in r.items()}
            write(dict(dataset=ds, method=m, seed=s, test=r, time=time.time() - t0))
            print(ds, m, s, " ".join(f"{k}={v:.4f}" for k, v in r.items() if k != "epoch"), flush=True)

if A.official_bwgnn:
    # Official implementation (needs dgl). Runs main.train() from the cloned repo with the same split code.
    sys.path.insert(0, "third_party/Rethinking-Anomaly-Detection")
    import importlib, io, contextlib
    import dgl
    from dataset import Dataset
    from BWGNN import BWGNN as OB, BWGNN_Hetero as OBH
    import main as OM
    for ds in A.datasets.split(","):
        for homo_flag in (1, 0):
            name = f"BWGNN-{'homo' if homo_flag else 'hetero'}-official"
            for s in A.seeds:
                if (ds, name, s) in done: continue
                torch.manual_seed(s); np.random.seed(s)
                g = Dataset(ds, homo_flag).graph
                OM.dataset_name = ds
                args = argparse.Namespace(train_ratio=0.4, epoch=100)
                model = (OB if homo_flag else OBH)(g.ndata["feature"].shape[1], 64, 2, g, d=2)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    mf1, auc = OM.train(model, g, args)
                write(dict(dataset=ds, method=name, seed=s, test=dict(f1=float(mf1), auc=float(auc))))
                print(ds, name, s, f"f1={mf1:.4f} auc={auc:.4f}", flush=True)
print("finished; results in", A.out)
