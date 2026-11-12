"""Run the published GAT-COBO code (code/GAT-COBO/main.py, default configuration, log1p-inverse cost)
on the shared split and on the new splits used by multisplit.py, then score its saved test probabilities.
Needs DGL (see repo requirements.txt). Writes one JSON line per (dataset, split, seed) to --out.

Usage: python run_gatcobo_splits.py --splits 0 1 2 3 4 --seeds 42
"""
import argparse, json, os, subprocess, sys, numpy as np
from common import load, metrics

HERE = os.path.dirname(os.path.abspath(__file__))
GC = os.path.join(os.path.dirname(HERE), "code", "GAT-COBO")
p = argparse.ArgumentParser()
p.add_argument("--datasets", default="sichuan,bupt")
p.add_argument("--splits", type=int, nargs="*", default=[0, 1, 2, 3, 4])
p.add_argument("--seeds", type=int, nargs="*", default=[42])
p.add_argument("--out", default="results_gpu/gatcobo_multisplit.jsonl")
p.add_argument("--gpu", action="store_true",
               help="run GAT-COBO on GPU (default: CPU, the setting used for the paper's shared-split reproduction)")
A = p.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
done = set()
if os.path.exists(A.out):
    for l in open(A.out):
        r = json.loads(l); done.add((r["dataset"], r["split"], r["seed"]))

for ds in A.datasets.split(","):
    for k in A.splits:
        X, y, src, dst, tr, va, te = load(ds, k)
        sdir = os.path.join(HERE, "splits", f"gatcobo_{ds}_split{k}")
        os.makedirs(sdir, exist_ok=True)
        for name, idx in [("train", tr), ("val", va), ("test", te)]:
            np.save(os.path.join(sdir, f"{name}_idx.npy"), np.asarray(idx, dtype=np.int64))
        for s in A.seeds:
            if (ds, k, s) in done: continue
            pf = os.path.join(HERE, "results_gpu", "gatcobo_probs", f"{ds}_split{k}_seed{s}.npy")
            cmd = [sys.executable, "main.py", "--dataset", {"sichuan": "Sichuan", "bupt": "BUPT"}[ds], "--cost", "2",
                   "--seed", str(s), "--shared_split_dir", sdir, "--save_probs", pf]
            print(" ".join(cmd), flush=True)
            env = os.environ.copy()
            if not A.gpu:
                env["CUDA_VISIBLE_DEVICES"] = ""   # published code mixes CPU/GPU tensors; CPU reproduces the paper's numbers
            subprocess.run(cmd, cwd=GC, check=True, env=env)
            d = np.load(pf, allow_pickle=True).item()
            assert (np.sort(d["test_idx"]) == np.sort(te)).all(), "test nodes differ from split"
            assert (d["labels"] == y[d["test_idx"]]).all(), "node order mismatch between GAT-COBO data and our loader"
            m = metrics(d["labels"], d["probs"].astype(float))
            with open(A.out, "a") as fh:
                fh.write(json.dumps(dict(dataset=ds, split=k, seed=s, method="gatcobo", test=m)) + "\n")
            print(ds, k, s, m, flush=True)
