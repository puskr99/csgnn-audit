"""Reproduction of the released CSGNN evaluation (xxhu94/CSGNN, commit aef3136).

The released training/evaluation function `main.main(args)` is executed UNCHANGED.  The only thing this
harness does is wrap `model.CostMatrix.cost_logits` so that every call is recorded (inputs are copied, the
original function is then called and its output returned untouched).  The last recorded call is the
inference call at main.py:261, so we obtain exactly the raw GNN outputs `z` and the learned cost matrix `M`
that the released code used to produce its printed test metrics.

From those we score the *same trained model* in several ways (test nodes only):

  released       : main.py's own scoring: cost_logits(z, TRUE labels of all nodes), softmax, argmax
                   (re-computed here; matches the numbers main.py prints, see audit/logs)
  shuffled_k     : as released, but the test-node labels are randomly permuted before the call
  test_labels_0  : as released, but every test-node label is set to 0 (majority class)
  lf_none        : label-free, cost layer dropped           p = softmax(z)
  lf_pred_row    : label-free, row of the PREDICTED class   cost_logits(z, argmax z)
  lf_expected    : label-free, expected cost row            p ~ (softmax(z)^T M) * exp(z)

If test labels did not enter the prediction path, released/shuffled/test_labels_0 would give identical predictions.

Optional `--select-epoch`: additionally record the GNN outputs of every training epoch (the model has no
dropout/batch-norm, so training-mode outputs equal eval-mode outputs of the parameters at that epoch) and
choose the epoch on VALIDATION nodes only, using label-free scores (val G-Mean + val AUC, the selection rule
of our re-evaluation paper).  Test metrics are then reported for that epoch.  Use with `--max-epoch 200`.

Usage (from the repository root, after `python data_process.py`; the released code hard-codes
/code/CSGNN/data, see .github/README.md):
    python audit/csgnn_audit.py --config config/csgnn_sichuan.yml --seeds 0 1 2 42 --perms 5
    python audit/csgnn_audit.py --config config/csgnn_bupt.yml    --seeds 0 1 2 42 --perms 5
"""
import argparse, copy, json, os, sys, time, platform
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import model as csgnn_model          # released model.py (unchanged)
import main as csgnn_main            # released main.py  (unchanged)
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score
from imblearn.metrics import geometric_mean_score

# ---------------------------------------------------------------- recording wrapper (no behaviour change)
_ORIG_COST_LOGITS = csgnn_model.CostMatrix.cost_logits
CALLS = []
RECORD_ALL = False


def _recording_cost_logits(self, feat, labels, cost_table):
    rec = dict(z=feat.detach().cpu().double().clone(),
               M=self.cost_matrix.detach().cpu().double().clone(),
               cost_table=cost_table.copy())
    if RECORD_ALL or not CALLS:
        CALLS.append(rec)
    else:
        CALLS[-1] = rec                      # keep only the most recent call (memory)
    return _ORIG_COST_LOGITS(self, feat, labels, cost_table)


csgnn_model.CostMatrix.cost_logits = _recording_cost_logits


# ---------------------------------------------------------------- scoring exactly as main.py does
def released_style_scores(cost_obj, z, labels_used, cost_table):
    """main.py:261-277: logp = cost_logits(z, labels); AUC on softmax(logp); prediction argmax(logp)."""
    logp = _ORIG_COST_LOGITS(cost_obj, z.float(), labels_used, cost_table).double()
    return torch.softmax(logp, dim=1), logp.argmax(1)


def metrics(y, prob, pred, C):
    y = np.asarray(y); prob = np.asarray(prob); pred = np.asarray(pred)
    if C == 2:
        auc = roc_auc_score(y, prob[:, 1], average="macro")
    else:
        auc = roc_auc_score(y, prob, average="macro", multi_class="ovo")
    return dict(auc=float(auc),
                gmean=float(geometric_mean_score(y, pred)),
                recall=float(recall_score(y, pred, average="macro")),
                f1=float(f1_score(y, pred, average="macro")),
                precision=float(precision_score(y, pred, average="macro", zero_division=0)))


class _CostShim:
    """Minimal object so the ORIGINAL cost_logits can be called with a recorded cost matrix."""
    def __init__(self, M):
        self.cost_matrix = torch.nn.Parameter(M.float())


def score_all(rec, labels, idx, C, perms, rng, label_free_only=False):
    z, M, table = rec["z"], rec["M"], rec["cost_table"]
    shim = _CostShim(M)
    y = labels[idx].numpy()
    out = {}
    q = torch.softmax(z, 1)
    # label-free variants
    out["lf_none"] = metrics(y, q[idx], q[idx].argmax(1), C)
    pp, pr = released_style_scores(shim, z, z.argmax(1), table)
    out["lf_pred_row"] = metrics(y, pp[idx], pr[idx], C)
    w = (q @ M) * torch.exp(z)
    pe = w / w.sum(1, keepdim=True)
    out["lf_expected"] = metrics(y, pe[idx], pe[idx].argmax(1), C)
    if label_free_only:
        return out
    # released scoring (uses true labels of evaluated nodes)
    p_true, h_true = released_style_scores(shim, z, labels, table)
    out["released"] = metrics(y, p_true[idx], h_true[idx], C)
    # permutation test
    max_change, sh = 0.0, []
    for k in range(perms):
        lab = labels.clone()
        lab[idx] = lab[idx][torch.from_numpy(rng.permutation(len(idx)))]
        p_k, h_k = released_style_scores(shim, z, lab, table)
        sh.append(metrics(y, p_k[idx], h_k[idx], C))
        max_change = max(max_change, float((p_k[idx] - p_true[idx]).abs().max()))
    out["shuffled_mean"] = {m: float(np.mean([s[m] for s in sh])) for m in sh[0]}
    out["shuffled_runs"] = sh
    out["max_abs_prob_change_under_shuffle"] = max_change
    lab0 = labels.clone(); lab0[idx] = 0
    p0, h0 = released_style_scores(shim, z, lab0, table)
    out["test_labels_0"] = metrics(y, p0[idx], h0[idx], C)
    # uniformly random predictor passed through the same label-indexed transform
    zr = torch.log(torch.from_numpy(rng.dirichlet(np.ones(C), size=z.shape[0])))
    pr0 = torch.softmax(zr, 1)
    out["random_predictor"] = metrics(y, pr0[idx], pr0[idx].argmax(1), C)
    prl, hrl = released_style_scores(shim, zr, labels, table)
    out["random_predictor_released_transform"] = metrics(y, prl[idx], hrl[idx], C)
    return out


def get_split(args):
    """Rebuild the released split (main.gen_mask, random_state=2) to obtain labels and index sets."""
    graph, num_classes, _ = csgnn_main.load_data(args)
    labels = graph.ndata["label"].long().squeeze().cpu()
    idx = {k: torch.nonzero(graph.ndata[f"{k}_mask"].cpu(), as_tuple=False).squeeze(1)
           for k in ("train", "val", "test")}
    return labels, idx, num_classes


def main():
    global RECORD_ALL
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--perms", type=int, default=5)
    ap.add_argument("--max-epoch", type=int, default=None, help="override released max_epoch")
    ap.add_argument("--select-epoch", action="store_true", help="label-free validation epoch selection")
    ap.add_argument("--out", default=os.path.join(ROOT, "audit", "results"))
    A = ap.parse_args()

    cfg = csgnn_main.get_config(A.config)
    os.makedirs(A.out, exist_ok=True)
    csgnn_main.device = "cpu" if not torch.cuda.is_available() else "cuda:0"   # main.py's global
    RECORD_ALL = A.select_epoch

    for seed in A.seeds:
        args = argparse.Namespace(**copy.deepcopy(cfg))
        if A.max_epoch is not None:
            args.max_epoch = A.max_epoch
        tag = f"{args.dataset.lower()}_ep{args.max_epoch}{'_sel' if A.select_epoch else ''}_seed{seed}"
        print(f"\n===== {tag} =====", flush=True)
        CALLS.clear()
        csgnn_main.setup_seed(seed)          # released seeding function (main.py:45); released entry uses 42
        t0 = time.time()
        csgnn_main.main(args)                # released training + evaluation, unchanged; prints its metrics
        secs = time.time() - t0
        labels, idx, C = get_split(args)
        rng = np.random.default_rng(1000 + seed)
        res = dict(dataset=args.dataset, seed=seed, max_epoch=args.max_epoch, config=A.config,
                   seconds=round(secs, 1), n_test=int(len(idx["test"])),
                   final=score_all(CALLS[-1], labels, idx["test"], C, A.perms, rng))
        if A.select_epoch:
            train_calls = CALLS[:-1]
            best, best_e = -1, None
            for e, rec in enumerate(train_calls):
                v = score_all(rec, labels, idx["val"], C, 0, rng, label_free_only=True)
                for var, m in v.items():
                    s = m["gmean"] + m["auc"]
                    if s > best:
                        best, best_e, best_var = s, e, var
            sel = score_all(train_calls[best_e], labels, idx["test"], C, 0, rng, label_free_only=True)
            res["label_free_selected"] = dict(epoch=best_e, variant=best_var, val_score=best,
                                              test=sel[best_var], test_all_variants=sel)
        res["env"] = dict(python=platform.python_version(), torch=torch.__version__,
                          dgl=__import__("dgl").__version__, numpy=np.__version__,
                          sklearn=__import__("sklearn").__version__)
        with open(os.path.join(A.out, f"{tag}.json"), "w") as f:
            json.dump(res, f, indent=1)
        r = res["final"]
        print(f"[audit] released     AUC {r['released']['auc']:.4f}  G-Mean {r['released']['gmean']:.4f}")
        print(f"[audit] shuffled     AUC {r['shuffled_mean']['auc']:.4f}  G-Mean {r['shuffled_mean']['gmean']:.4f}"
              f"  max |dp| {r['max_abs_prob_change_under_shuffle']:.4f}")
        for v in ("lf_none", "lf_pred_row", "lf_expected"):
            print(f"[audit] {v:12s} AUC {r[v]['auc']:.4f}  G-Mean {r[v]['gmean']:.4f}")
        if A.select_epoch:
            s = res["label_free_selected"]
            print(f"[audit] val-selected epoch {s['epoch']} ({s['variant']}): "
                  f"AUC {s['test']['auc']:.4f}  G-Mean {s['test']['gmean']:.4f}")


if __name__ == "__main__":
    main()
