# Archived results from the full re-evaluation study (provenance only)

These files are copied unchanged from the study behind our re-evaluation manuscript (run on a GPU server,
Oct 2026). They are kept here so every CSGNN number in the manuscript can be traced; they are **not**
needed to reproduce the audit (use `audit/run_all.sh`, which runs the released code itself).

| File | What it is |
|---|---|
| `csgnn_port.py` | Pure-PyTorch port of the released CSGNN (only change: vectorised top-p neighbour sampling). `--eval leak` = released scoring, `--eval honest` = label-free. Used for the 5-split × 4-seed, 200-epoch runs because the released per-node Python loop is slow on BUPT. |
| `common.py` | Data loading, stratified 20/20/60 splits (`random_state=2` reproduces the released split exactly; splits 1–4 use `1000 + k`), metrics. Expects `data/{sichuan,bupt}_*.npz` exported from the released DGL graphs. |
| `leak_permutation.py` | Label-permutation test on the port (also checks our own method; that part needs code not included here). |
| `select_csgnn.py` | Validation-only selection of seed and label-free variant per split. |
| `results/csgnn_valsel.jsonl` | 40 runs (2 datasets × 5 splits × seeds 0, 1, 2, 42), 200 epochs, early stopping on label-free validation scores; validation and test metrics for released scoring (`leak`) and three label-free variants (`honest` = cost layer dropped, `fix_pred` = predicted-class row, `fix_exp` = expected cost row). |
| `results/csgnn_selected.jsonl` | One run per split chosen on validation only (variant and seed). |
| `results/csgnn_stability.json` | BUPT collapse count (11 of 20 runs end with G-Mean ≈ 0). |
| `results/leak_permutation.json` | Port, released config, seed 42: true vs. permuted test labels. |

Summary of `csgnn_selected.jsonl` (5 splits, validation-selected, test metrics, mean ± sd):

| | AUC | G-Mean | Macro recall | Macro F1 |
|---|---|---|---|---|
| BUPT | 0.8765 ± 0.0260 | 0.7359 ± 0.0324 | 0.7665 ± 0.0213 | 0.7814 ± 0.0144 |
| Sichuan | 0.9407 ± 0.0027 | 0.8762 ± 0.0037 | 0.8793 ± 0.0036 | 0.8888 ± 0.0046 |
| Published (Hu et al.) BUPT / Sichuan | 0.9698 / 0.9539 | 0.8262 / 0.8864 | 0.8317 / 0.8878 | – |
