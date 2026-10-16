# CSGNN evaluation audit

This repository is an evaluation audit of the released implementation of

> X. Hu et al., "Cost Sensitive GNN-based Imbalanced Learning for Mobile Social Network Fraud Detection",
> *IEEE Transactions on Computational Social Systems*, 2023, doi:10.1109/TCSS.2023.3302651
> Code: <https://github.com/xxhu94/CSGNN>

**Finding.** At test time the released code passes the network outputs through a cost-sensitive layer that
selects, for every node, the cost-matrix row of that node's **ground-truth label**
(`main.py:261` → `model.py:203-213`). Test labels therefore reach the predictions, and the reported test AUC and
G-Mean are computed from these predictions.

## The original code is preserved unchanged

The full upstream history (9 commits, authors and dates intact) is kept in this repository:

| Ref | Points to |
|---|---|
| tag `original-aef3136` | `aef3136bd96b88ebee07cafc209fa9105a58a555` (upstream HEAD, 25 Nov 2023) |
| branch `original` | the same commit, never modified |
| branch `main` | `aef3136` + commits that only **add** files under `audit/` and `.github/` |

The upstream `README.md` and all code, config and data files at the repository root are byte-identical to `aef3136`.
Check it yourself:

```bash
bash audit/verify_original.sh          # git diff aef3136 HEAD over every original path -> must be empty
git log --format='%h %an %ad %s' original-aef3136
```

## Reproduce (CPU only, about 1 h 15 min on 2 cores)

```bash
git clone <this repository> csgnn-audit && cd csgnn-audit
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r audit/requirements.txt      # scikit-learn < 1.2 is required by the released data_process.py
bash audit/run_all.sh
```

`run_all.sh` (1) verifies the original files, (2) builds the DGL graphs with the released `data_process.py`,
(3) links `/code/CSGNN` to the checkout because the released `main.py` hard-codes that data path (no file is
edited), (4) runs the released `main.py` with both released configs, (5) runs `audit/csgnn_audit.py`, and
(6) writes `audit/RESULTS.md`.

`audit/csgnn_audit.py` imports the released `main.py` and `model.py` and calls the released `main.main(args)`
unchanged. Its only intervention is a recording wrapper around `CostMatrix.cost_logits`. The wrapper copies the
inputs, calls the original function and returns that function's output untouched. From the recorded raw outputs `z`
and the learned cost matrix `M` of the final inference call it re-scores the same trained model:

| Scoring | Test labels used? |
|---|---|
| Released (`cost_logits(z, true labels)`), re-computed and checked against what `main.py` prints | yes |
| Same, test labels randomly permuted (5 permutations) / all set to 0 | yes (wrong ones) |
| Label-free: cost layer dropped, `softmax(z)` | no |
| Label-free: cost row of the predicted class, `cost_logits(z, argmax z)` | no |
| Label-free: expected cost row, `p ∝ (softmax(z)ᵀM) ⊙ exp(z)` | no |

A leakage-free inference procedure must give identical predictions in the first two rows.

## Results (released code, released configs, test set of the released split)

See `audit/RESULTS.md` for the full tables (generated) and `audit/logs/` for the raw console output.

## Repository layout

```
(root)                     upstream CSGNN at aef3136, unchanged
.github/README.md          this file (GitHub shows it in place of the upstream README)
audit/CODE_REFERENCES.md   line-by-line references and what can / cannot be confirmed
audit/csgnn_audit.py       audit harness (runs the released main.main unchanged)
audit/summarise.py         results/*.json -> RESULTS.md
audit/run_all.sh           end-to-end reproduction
audit/verify_original.sh   proves the upstream files are unchanged
audit/requirements.txt     exact environment used
audit/results/             one JSON per run (all metrics, all scorings, environment)
audit/logs/                console logs of every run
audit/archive_full_study/  CSGNN scripts and results of the full re-evaluation study (5 splits x 4 seeds, 200 epochs)
```

## Licence

The upstream code is redistributed under its original licence (`LICENSE`, Apache-2.0). Audit files are
provided for verification of the published evaluation.
