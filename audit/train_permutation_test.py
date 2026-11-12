"""Does any label of the evaluated (test) nodes influence TRAINING in the released CSGNN code?

The released `main.main(args)` is run UNCHANGED several times with the same seed.  Between `load_data` (which
builds the released split from the true labels, so the split itself never changes) and the start of training,
a thin wrapper optionally permutes the labels of one group of nodes:

  control_a, control_b : no change (two identical runs; checks that training is deterministic)
  permute_test         : labels of the TEST nodes randomly permuted before training
  permute_val          : labels of the VALIDATION nodes randomly permuted (positive control: the released
                         cost-matrix update, main.py:197-202, uses validation labels, so this should change training)

For every run we record, through the same recording wrapper as csgnn_audit.py, the raw network outputs `z`
(all nodes) and the learned cost matrix `M` at the final inference call (main.py:261), plus the outputs of every
training epoch.  If test labels never reach training, `permute_test` must give outputs identical to `control_a`.

Usage (repository root; see .github/README.md for the data path):
    python audit/train_permutation_test.py --config config/csgnn_bupt.yml --seed 42
"""
import argparse, copy, json, os, platform, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import csgnn_audit as H              # recording wrapper around CostMatrix.cost_logits (no behaviour change)
import main as csgnn_main            # released main.py (unchanged)

_ORIG_LOAD = csgnn_main.load_data
MODE = {"group": None, "seed": 0}


def _load_data_with_permutation(args):
    graph, num_classes, feat = _ORIG_LOAD(args)          # released loader: builds the released split
    g = MODE["group"]
    if g is not None:
        mask = graph.ndata[f"{g}_mask"].bool()
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        lab = graph.ndata["label"].clone()
        rng = np.random.default_rng(MODE["seed"])
        perm = torch.from_numpy(rng.permutation(len(idx))).to(idx.device)
        orig = lab[idx].clone()
        lab[idx] = orig[perm]
        graph.ndata["label"] = lab
        MODE["n_changed"] = int((lab[idx] != orig).sum())
        MODE["n_group"] = int(len(idx))
    return graph, num_classes, feat


csgnn_main.load_data = _load_data_with_permutation      # main.main looks load_data up in its module globals


def run_once(cfg, seed, group):
    args = argparse.Namespace(**copy.deepcopy(cfg))
    MODE["group"] = group
    H.RECORD_ALL = True
    H.CALLS.clear()
    csgnn_main.setup_seed(seed)
    t0 = time.time()
    csgnn_main.main(args)
    secs = time.time() - t0
    calls = [dict(z=c["z"].clone(), M=c["M"].clone()) for c in H.CALLS]
    return calls, secs


def max_diff(a, b):
    return max(float((x["z"] - y["z"]).abs().max()) for x, y in zip(a, b)), \
           max(float((x["M"] - y["M"]).abs().max()) for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perm-seed", type=int, default=12345)
    ap.add_argument("--out", default=os.path.join(ROOT, "audit", "results"))
    A = ap.parse_args()
    cfg = csgnn_main.get_config(A.config)
    csgnn_main.device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    MODE["seed"] = A.perm_seed
    os.makedirs(A.out, exist_ok=True)

    runs, secs, changed = {}, {}, {}
    for name, group in [("control_a", None), ("control_b", None), ("permute_test", "test"), ("permute_val", "val")]:
        print(f"\n===== {cfg['dataset']} seed {A.seed}: {name} =====", flush=True)
        MODE["n_changed"] = 0; MODE["n_group"] = 0
        runs[name], secs[name] = run_once(cfg, A.seed, group)
        changed[name] = dict(nodes_in_group=MODE["n_group"], labels_changed=MODE["n_changed"])

    ref = runs["control_a"]
    res = dict(dataset=cfg["dataset"], seed=A.seed, max_epoch=cfg["max_epoch"], config=A.config,
               n_recorded_calls=len(ref), seconds=secs, labels_changed=changed, comparisons={})
    for name in ("control_b", "permute_test", "permute_val"):
        dz, dm = max_diff(ref, runs[name])
        dz_final = float((ref[-1]["z"] - runs[name][-1]["z"]).abs().max())
        res["comparisons"][name] = dict(max_abs_diff_outputs_all_epochs=dz, max_abs_diff_cost_matrix_all_epochs=dm,
                                        max_abs_diff_final_outputs=dz_final,
                                        identical=bool(dz == 0.0 and dm == 0.0))
        print(f"[train-perm] {name:12s} vs control_a: max|dz| {dz:.3g}  max|dM| {dm:.3g}  "
              f"identical={res['comparisons'][name]['identical']}", flush=True)
    res["env"] = dict(python=platform.python_version(), torch=torch.__version__,
                      dgl=__import__("dgl").__version__, numpy=np.__version__)
    tag = f"train_permutation_{cfg['dataset'].lower()}_seed{A.seed}.json"
    with open(os.path.join(A.out, tag), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
