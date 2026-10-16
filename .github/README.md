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

## Results: claimed vs. reproduced

Released code, unchanged; test set of the released split (`random_state=2`, 20/20/60). "Claimed" is from
Hu et al. (TCSS 2023). Rows starting with "Same model" re-score the model of the row above; only the scoring differs.

| Dataset | Setting | AUC | G-Mean | Macro recall |
|---|---|---|---|---|
| **BUPT** | Claimed in the paper | 0.9698 | 0.8262 | 0.8317 |
| | Reproduced: released config (1 epoch), released seed 42, released scoring | 0.9698 | 0.8259 | 0.8313 |
| | Same model, test labels shuffled (mean of 5) | 0.8423 | 0.6634 | 0.6814 |
| | Same model, label-free (expected-cost-row repair) | 0.8590 | 0.6819 | 0.7000 |
| | Released scoring, seeds 0/1/2/42 (mean ± sd) | 0.9569 ± 0.0143 | 0.4551 ± 0.3481 | 0.6385 ± 0.1376 |
| | Label-free, 200 epochs, validation-selected, 5 splits (mean ± sd)¹ | 0.8765 ± 0.0260 | 0.7359 ± 0.0324 | 0.7665 ± 0.0213 |
| **Sichuan** | Claimed in the paper | 0.9539 | 0.8864 | 0.8878 |
| | Reproduced: released config (15 epochs), seed 42, released scoring | 0.9543 | 0.8654 | 0.8707 |
| | Same model, label-free (expected-cost-row repair) | 0.9287 | 0.8649 | 0.8702 |
| | Released scoring, 200 epochs, seed 42 | 0.9558 | 0.8859 | 0.8900 |
| | Same 200-epoch model, label-free; epoch chosen on validation | 0.9390 | 0.8758 | 0.8785 |
| | Label-free, 200 epochs, validation-selected, 5 splits (mean ± sd)¹ | 0.9407 ± 0.0027 | 0.8762 ± 0.0037 | 0.8793 ± 0.0036 |

¹ From the full re-evaluation study (`audit/archive_full_study/`, PyTorch port of the released code).

**Summary**

- On BUPT the released code reproduces the published AUC exactly (0.9698), but only when every test node's score is
  re-weighted using its true label. Shuffling the test labels lowers the AUC to 0.84. Scoring the same model without
  test labels gives 0.81–0.86. With a fair training budget and validation-only selection the result is 0.88 AUC and
  0.74 G-Mean, against the published 0.97 and 0.83.
- The published BUPT G-Mean is reached only with seed 42. With seeds 0, 1 and 2 the same released configuration gives
  G-Mean 0.59, 0.00 and 0.41.
- On Sichuan the leak is smaller: +0.026 AUC with the released 15 epochs, and +0.016 AUC / +0.012 G-Mean after 200
  epochs. The published G-Mean is matched only after 200 epochs with the released scoring; without test labels the
  same model gives 0.940 AUC / 0.874 G-Mean.
- The permutation test needs no code reading: with label-free inference, shuffling test labels must leave every
  prediction unchanged. Under the released scoring, individual test probabilities change by up to 0.084 (BUPT) and
  0.143 (Sichuan, 200 epochs).

Full tables: `audit/RESULTS.md`. Raw console output: `audit/logs/`.

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
