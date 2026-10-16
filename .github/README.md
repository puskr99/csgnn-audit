# CSGNN reproducibility study

This repository documents a reproduction of the results reported in

> X. Hu et al., "Cost Sensitive GNN-based Imbalanced Learning for Mobile Social Network Fraud Detection",
> *IEEE Transactions on Computational Social Systems*, 2023, doi:10.1109/TCSS.2023.3302651
> Code: <https://github.com/xxhu94/CSGNN>

We thank the authors for releasing their code and data, which made this study possible. CSGNN is a strong
reference method on the Sichuan and BUPT telecom-fraud benchmarks, and we reproduced it as a baseline for our own
work. During that reproduction we noticed a detail of the evaluation procedure that affects the reported test
metrics. This repository records exactly what we observed, so that the observation can be checked independently
and, if confirmed, corrected.

## Observation

In the released implementation, the cost-sensitive output layer (`CostMatrix.cost_logits`, `model.py:203-213`)
weights each node's class scores with the row of the cost matrix that corresponds to the node's label. This is
natural during training. The same function is also applied after training (`main.py:261`), with the labels of all
nodes, before the test AUC and G-Mean are computed. As a result, the ground-truth labels of the test nodes
influence their predicted scores.

This is most likely an implementation detail rather than part of the method's design: the label-indexed layer is
needed for training, and reusing it at test time is an easy step to take. We make no claim about how the published
tables were produced; we only report what the released code does and what it produces.

## The original code is preserved unchanged

The full upstream history (9 commits, authors and dates intact) is kept in this repository:

| Ref | Points to |
|---|---|
| tag `original-aef3136` | `aef3136bd96b88ebee07cafc209fa9105a58a555` (upstream HEAD, 25 Nov 2023) |
| branch `original` | the same commit, never modified |
| branch `main` | `aef3136` + commits that only **add** files under `audit/` and `.github/` |

The upstream `README.md` and all code, config and data files at the repository root are byte-identical to `aef3136`.
To check:

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
(3) links `/code/CSGNN` to the checkout because the released `main.py` reads its data from that path (no file is
edited), (4) runs the released `main.py` with both released configurations, (5) runs `audit/csgnn_audit.py`, and
(6) writes `audit/RESULTS.md`.

`audit/csgnn_audit.py` imports the released `main.py` and `model.py` and calls the released `main.main(args)`
unchanged. Its only addition is a recording wrapper around `CostMatrix.cost_logits`. The wrapper copies the inputs,
calls the original function and returns that function's output untouched. From the recorded network outputs `z`
and the learned cost matrix `M` of the final call, it scores the same trained model in several ways:

| Scoring of the same trained model | Uses test labels? |
|---|---|
| As released: `cost_logits(z, labels)`; re-computed and checked against what `main.py` prints | yes |
| As released, but with the test labels randomly permuted (5 permutations) or all set to 0 | yes (permuted) |
| Without the cost layer: `softmax(z)` | no |
| Cost row of the predicted class: `cost_logits(z, argmax z)` | no |
| Expected cost row: `p ∝ (softmax(z)ᵀM) ⊙ exp(z)` | no |

The permuted-label rows are a simple check that needs no reading of the code: if test labels do not enter
the prediction path, permuting them leaves every prediction unchanged.

## Results: reported vs. reproduced

Released code, unchanged; test set of the released split (`random_state=2`, 20/20/60). "Reported" values are taken
from Hu et al. (TCSS 2023). Rows beginning "Same model" re-score the model of the row above; only the scoring differs.

| Dataset | Setting | AUC | G-Mean | Macro recall |
|---|---|---|---|---|
| **BUPT** | Reported in the paper | 0.9698 | 0.8262 | 0.8317 |
| | Released configuration (1 epoch), released seed 42, released scoring | 0.9698 | 0.8259 | 0.8313 |
| | Same model, test labels permuted (mean of 5) | 0.8423 | 0.6634 | 0.6814 |
| | Same model, scored without test labels (expected cost row) | 0.8590 | 0.6819 | 0.7000 |
| | Released scoring, seeds 0/1/2/42 (mean ± sd) | 0.9569 ± 0.0143 | 0.4551 ± 0.3481 | 0.6385 ± 0.1376 |
| | Without test labels, 200 epochs, validation-based selection, 5 splits (mean ± sd)¹ | 0.8765 ± 0.0260 | 0.7359 ± 0.0324 | 0.7665 ± 0.0213 |
| **Sichuan** | Reported in the paper | 0.9539 | 0.8864 | 0.8878 |
| | Released configuration (15 epochs), seed 42, released scoring | 0.9543 | 0.8654 | 0.8707 |
| | Same model, scored without test labels (expected cost row) | 0.9287 | 0.8649 | 0.8702 |
| | Released scoring, 200 epochs, seed 42 | 0.9558 | 0.8859 | 0.8900 |
| | Same 200-epoch model, scored without test labels; epoch chosen on validation | 0.9390 | 0.8758 | 0.8785 |
| | Without test labels, 200 epochs, validation-based selection, 5 splits (mean ± sd)¹ | 0.9407 ± 0.0027 | 0.8762 ± 0.0037 | 0.8793 ± 0.0036 |

¹ From our wider re-evaluation study (`audit/archive_full_study/`, a PyTorch port of the released code).

**Summary**

- The released code reproduces the reported BUPT AUC exactly (0.9698) and the G-Mean to within 0.0003 under the
  released configuration and seed. When the same model is scored without test labels, BUPT AUC is 0.81–0.86. With
  200 epochs and validation-based model selection, it is 0.88 AUC and 0.74 G-Mean.
- On BUPT the released configuration trains for one epoch, and results vary considerably with the random seed:
  G-Mean is 0.83, 0.59, 0.00 and 0.41 for seeds 42, 0, 1 and 2.
- On Sichuan the effect of the label-indexed scoring is small. It raises AUC by 0.026 at the released 15 epochs and
  by 0.016 after 200 epochs, and changes G-Mean by less than 0.001 at 15 epochs and by 0.012 after 200 epochs. The
  paper states 15 epochs for Sichuan, matching the released configuration, so the gap between the reported G-Mean
  (0.8864) and the reproduced one (0.8654) remains unexplained. After 200 epochs the released scoring gives 0.8859,
  but we do not suggest that this is how the reported number was obtained.
- Under the released scoring, permuting the test labels changes individual test probabilities by up to 0.084 (BUPT)
  and 0.143 (Sichuan, 200 epochs).

These results describe the released implementation only. They do not evaluate the CSGNN method itself, which can
be run without test labels at inference, as the label-free rows show. Full tables are in `audit/RESULTS.md` and raw
console output in `audit/logs/`.

## Limitations

- We have no access to the original experimental runs, so we cannot confirm that the published tables were
  produced with this exact code, configuration and software environment.
- We could not reproduce the reported Sichuan G-Mean with the 15-epoch configuration stated in the paper and the
  released code.
- The paper (Section 4.3.2, Eq. 13) defines the cost-sensitive output as `Softmax(C[v, k] · z_v[k])` and does not
  state how the cost-matrix row is chosen for a test node, whose label is unknown at test time. We report what the
  released code does; we do not infer the authors' intended test-time procedure.
- We did not examine the paper's other experiments (ablations, parameter studies, baselines).

Corrections and comments are welcome; please open an issue or contact the maintainers. We will update this
repository if any part of the analysis turns out to be wrong.

## Repository layout

```
(root)                     upstream CSGNN at aef3136, unchanged
.github/README.md          this file (GitHub shows it in place of the upstream README)
audit/CODE_REFERENCES.md   line-by-line code references and what can / cannot be confirmed
audit/csgnn_audit.py       evaluation script (runs the released main.main unchanged)
audit/summarise.py         results/*.json -> RESULTS.md
audit/run_all.sh           end-to-end reproduction
audit/verify_original.sh   checks that the upstream files are unchanged
audit/requirements.txt     exact environment used
audit/results/             one JSON per run (all metrics, all scorings, environment)
audit/logs/                console logs of every run
audit/archive_full_study/  CSGNN scripts and results of the wider study (5 splits x 4 seeds, 200 epochs)
```

## Licence

The upstream code is redistributed under its original licence (`LICENSE`, Apache-2.0).
