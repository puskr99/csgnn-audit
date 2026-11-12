"""Run the UNMODIFIED official PC-GNN code (src/) on Sichuan.
Only two things are injected: the data loader (Sichuan graph as homo + the same graph for the 3 relations the code expects)
and the shared 20/20/60 split instead of the hard-coded YelpChi split. Everything else (model, row-normalisation, training,
sigmoid/threshold-0.5 evaluation, best-val-AUC checkpoint) is the official code.
Usage (from inside a clone of github.com/PonderLY/PC-GNN, e.g. rethink/third_party/PC-GNN):
  python ../../pcgnn_official_sichuan.py --seed 72 --scaler                     # reproduces the thesis run
  python ../../pcgnn_official_sichuan.py --split K --seed S --scaler --out ../../results_gpu/pcgnn_official_sichuan.jsonl
"""
import argparse, sys, numpy as np
from collections import defaultdict
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # rethink/ (common.py)
sys.path.insert(0, os.getcwd())                                    # PC-GNN clone (src/, main.py)
from common import load
import src.model_handler as MH

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=72); p.add_argument("--lr", type=float, default=0.003)
p.add_argument("--emb", type=int, default=64); p.add_argument("--epochs", type=int, default=51)
p.add_argument("--split", type=int, default=0); p.add_argument("--out", default="")
p.add_argument("--scaler", action="store_true", help="use the official code's commented-out StandardScaler instead of row-normalisation")
A = p.parse_args()

X, y, src, dst, tr, va, te = load("sichuan", A.split)
adj = defaultdict(set)
for i in range(len(y)): adj[i].add(i)                      # self loops, as sparse_to_adjlist
for u, v in zip(src.tolist(), dst.tolist()): adj[u].add(v); adj[v].add(u)
feats = X.astype(np.float64)
if A.scaler:
    feats = (feats - feats[tr].mean(0)) / np.where(feats[tr].std(0) > 0, feats[tr].std(0), 1)
MH.load_data = lambda data, prefix=None: ([adj, adj, adj, adj], feats, y)
if A.scaler:
    MH.normalize = lambda m: m                              # official code: StandardScaler block replaces normalize

calls = {"n": 0}
def shared_split(index, labels, **kw):                      # first call -> train/rest, second -> valid/test
    calls["n"] += 1
    if calls["n"] == 1:
        rest = np.concatenate([va, te]); return list(tr), list(rest), y[tr], y[rest]
    return list(va), list(te), y[va], y[te]
MH.train_test_split = shared_split

cfg = dict(data_name="yelp", data_dir="", train_ratio=0.4, test_ratio=0.67, save_dir="/tmp/pcgnn_models/",
           model="PCGNN", multi_relation="GNN", emb_size=A.emb, thres=0.5, rho=0.5, seed=A.seed,
           optimizer="adam", lr=A.lr, weight_decay=0.001, batch_size=1024, num_epochs=A.epochs, valid_epochs=5,
           alpha=2, no_cuda=True, cuda_id="0")
import main as M
M.set_random_seed(A.seed)
model = MH.ModelHandler(cfg)
f1, f1_1, f1_0, auc, g = model.train()
if A.out:
    import json
    with open(A.out, "a") as fh:
        fh.write(json.dumps(dict(dataset="sichuan", split=A.split, seed=A.seed, method="pcgnn_official", scaler=A.scaler,
                                 test=dict(f1=float(f1), auc=float(auc), gmean=float(g)))) + "\n")
print(f"RESULT split={A.split} seed={A.seed} scaler={A.scaler} lr={A.lr} emb={A.emb}: F1-macro={f1:.4f} AUC={auc:.4f} G-Mean={g:.4f}")
