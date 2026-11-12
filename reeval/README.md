# Re-evaluation of graph-based telecom fraud detection (`reeval/`)

Code, split indices and per-run results for the manuscript

> P. Adhikari, C. Polprasert, S. Pradhanang. *Label Leakage, Simple Baselines and What Actually Drives Performance:
> A Re-evaluation of Graph-Based Telecom Fraud Detection.* Submitted to *Machine Learning with Applications*.

The CSGNN audit (released code preserved unchanged at `aef3136`, audit scripts, logs and results) is in the
repository root and in `../audit/`. This directory holds everything else in the paper: the shared protocol, CS-Boost,
all baselines and ports, and the scripts and results of every experiment.

## Layout

```
splits/                 index files (train / val / test) of all five splits, both datasets
data/                   where the benchmark files go (see data/README.md); the split-0 index files are included
rethink/common.py       data loading, splits (split 0 = shared split; split k = stratified 20/20/60, random_state 1000+k),
                        metrics, validation-calibrated decision offsets (fit_bias)
rethink/csboost_v5.py   CS-Boost: signed-log + log-degree input, cost-sensitive LightGBM, decoupled GraphSAGE,
                        validation-weighted ensemble, calibrated decision
rethink/baselines.py    MLP, GCN, GraphSAGE, GAT (classic and recipe-matched)
rethink/bwgnn_port.py   BWGNN (port of the official code); extra_runs.py runs it on the telecom graphs
rethink/pcgnn_port.py   PC-GNN (port of the official code, three-class generalisation)
rethink/csgnn_port.py   CSGNN (PyTorch port of the released code, vectorised neighbour loop), label-free scorings
rethink/run_gatcobo_splits.py  GAT-COBO, published code (needs a clone of xxhu94/GAT-COBO and DGL)
rethink/best.json       CS-Boost hyperparameters selected on the shared split
rethink/sweeps/         validation searches on the shared split (CS-Boost staged search, baseline searches)
rethink/results_gpu/    main comparison, factorial design, rewiring, published methods, YelpChi/Amazon
rethink/results_rev/    revision experiments (per-split tuning, recipe input for published methods, raw-feature MLP,
                        GBDT + neighbour means, rewiring diagnostics, class-weight check)
rethink/logs_gpu/, logs_rev/   console logs
```

## Experiments and scripts

| Paper | Script(s) | Results |
|---|---|---|
| Label-permutation test, CS-Boost (Sec. 3) | `leak_permutation.py` | `results_gpu/leak_permutation.json` |
| Main comparison, 5 splits x 3 seeds (Table 3) | `multisplit.py`, `run_gatcobo_splits.py`, `pcgnn_port.py`, `select_csgnn.py` | `results_gpu/multisplit.jsonl`, `gatcobo_multisplit.jsonl`, `pcgnn_multisplit.jsonl`, `csgnn_valsel.jsonl` |
| Comparable-budget tuning, calibrated published methods (Table 4) | `extra_runs.py --stage tune / bwgnn` | `results_gpu/extra_tune.jsonl`, `extra_bwgnn.jsonl`, `calibrated.jsonl`, `*_calib.jsonl` |
| Per-split selection, published methods with recipe input (Table 5) | `rev_pertune.py`, `rev_logpub.py`, `rev_tables.py`, `run_rev_gpu.sh`, `run_rev_bwgnn_gpu.sh` | `results_rev/pertune*.jsonl`, `logpub*.jsonl`, `rev_summary.json` |
| Class-weight check (App. B) | `rev_lambda.py` | `results_rev/lambda_sichuan.jsonl` |
| 2^4 factorial (Table 6) | `factorial.py`, `make_factorial.py` | `results_gpu/factorial*.{jsonl,json,md}` |
| Rewiring (Table 9) and diagnostics | `rewire_exact.py`, `extra_runs.py --stage rewire_exact`, `rev_rewire_diag.py` | `results_gpu/extra_rewire_exact.jsonl`, `results_rev/rewire_diag.json` |
| GBDT + neighbour means on telecom (Sec. 6.3) | `rev_gbdt_agg.py` | `results_rev/gbdt_agg.jsonl` |
| Ranking (Table 7, Fig. 2) | `ranking.py` | `sweeps/ranking.json` |
| YelpChi / Amazon (Table 11) | `fraud_runs.py`, `audit_extended/fraud_baselines.py`, `rev_fraud_raw.py` | `results_gpu/fraud_runs.jsonl`, `results_rev/fraud_raw.jsonl` |
| Summary tables, paired intervals | `aggregate_gpu.py`, `rev_tables.py` | `results_gpu/summary.{md,json}`, `results_rev/rev_summary.json` |

Every runner appends one JSON line per finished run and skips finished runs when restarted.

## Setup and running

```bash
cd reeval/rethink
python -m venv .venv && source .venv/bin/activate
pip install torch                 # CUDA build matching your driver; CPU also works (slower)
pip install -r requirements.txt
# data: see ../data/README.md
bash run_all.sh                   # main study (stages: setup leak split gatcobo pcgnn csgnn factorial fraud aggregate)
bash run_rev_gpu.sh               # revision experiments: per-split tuning, recipe input for PC-GNN / CSGNN
bash run_rev_bwgnn_gpu.sh         # per-split tuning of BWGNN
python rev_gbdt_agg.py; python rev_rewire_diag.py; python rev_fraud_raw.py; python rev_lambda.py sichuan mlp,sage
python rev_tables.py              # rows of Table 5
```

Environments used: GPU server with one NVIDIA GeForce RTX 4090 (Python 3.10, PyTorch 2.4.0, CUDA 12.1), and a
two-vCPU machine (Intel Xeon 2.1 GHz; Python 3.11, PyTorch 2.1.2, LightGBM 4.7.0, scikit-learn 1.1.3). The
reference-split configuration of CS-Boost gives the same G-Mean on both to within 0.001.

## Protocol in brief

- Five stratified 20/20/60 splits; split 0 is the shared split used by prior work (`random_state=2`), splits 1-4 use
  `random_state=1000+k`. The files in `splits/` hold the node indices in the node order produced by
  `rethink/common.py::load`.
- Everything fitted uses the training split; every selection (early stopping, hyperparameters, ensemble weight,
  decision offsets) uses validation G-Mean + validation AUC; the test split is scored once per run.
- Comparisons are made at the level of split means, with 95% intervals corrected for overlapping training sets
  (Nadeau and Bengio, 2003).
