#!/usr/bin/env bash
# Run every experiment requested in the review round on a GPU server.
# Usage (from inside rethink/):   bash run_all.sh            # everything
#                                 bash run_all.sh leak split  # only some stages
# Stages: setup leak split gatcobo pcgnn csgnn factorial fraud aggregate
# All results land in rethink/results_gpu/. Every runner resumes: re-running skips finished runs.
set -euo pipefail
cd "$(dirname "$0")"
STAGES="${*:-setup leak split gatcobo pcgnn csgnn factorial fraud aggregate}"
mkdir -p results_gpu logs_gpu
has() { [[ " $STAGES " == *" $1 "* ]]; }

python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

if has setup; then
  # YelpChi / Amazon .mat files ship inside the CARE-GNN repository
  if [ ! -f fraud_data/YelpChi.mat ]; then
    mkdir -p third_party fraud_data
    [ -d third_party/CARE-GNN ] || git clone --depth 1 https://github.com/YingtongDou/CARE-GNN third_party/CARE-GNN
    unzip -o -q third_party/CARE-GNN/data/YelpChi.zip -d fraud_data
    unzip -o -q third_party/CARE-GNN/data/Amazon.zip -d fraud_data
  fi
  # official BWGNN (only used with --official_bwgnn, needs dgl)
  [ -d third_party/Rethinking-Anomaly-Detection ] || git clone --depth 1 https://github.com/squareRoot3/Rethinking-Anomaly-Detection third_party/Rethinking-Anomaly-Detection || true
  ls fraud_data
fi

# 1) label-permutation leakage test (CPU, ~5 min)
if has leak; then
  python leak_permutation.py --datasets sichuan,bupt --perms 5 --out results_gpu/leak_permutation.json 2>&1 | tee logs_gpu/leak.log
fi

# 2) multi-split x multi-seed on Sichuan / BUPT: CS-Boost, components, classic & recipe-matched GNNs, CSGNN label-free
if has split; then
  python multisplit.py --datasets sichuan,bupt --splits 0 1 2 3 4 --seeds 0 1 2 \
         --out results_gpu/multisplit.jsonl 2>&1 | tee logs_gpu/multisplit.log
fi

# 3) published GAT-COBO code on the same splits (needs dgl + imbalanced-learn; see ../README.md)
if has gatcobo; then
  python run_gatcobo_splits.py --datasets sichuan,bupt --splits 0 1 2 3 4 --seeds 42 \
         --out results_gpu/gatcobo_multisplit.jsonl 2>&1 | tee logs_gpu/gatcobo.log || echo "GAT-COBO stage failed (dgl missing?) - continuing"
fi

# 3b) PC-GNN (port of the official code, three-class generalisation) on the same splits/seeds (~70 min on CPU)
if has pcgnn; then
  python pcgnn_port.py --datasets sichuan,bupt --splits 0 1 2 3 4 --seeds 0 1 2 \
         --out results_gpu/pcgnn_multisplit.jsonl 2>&1 | tee logs_gpu/pcgnn.log
fi

# 3c) label-free CSGNN, 5 splits x 3 seeds, variant chosen on VALIDATION (replaces the old per-metric best-of-three)
if has csgnn; then
  python multisplit.py --datasets sichuan,bupt --splits 0 1 2 3 4 --methods csgnn --csgnn_seeds 0 1 2 \
         --out results_gpu/csgnn_valsel.jsonl 2>&1 | tee logs_gpu/csgnn_valsel.log
  # seed 42 (released default) as a fourth restart; restart chosen per split on validation
  python multisplit.py --datasets sichuan,bupt --splits 0 1 2 3 4 --methods csgnn --csgnn_seeds 42 \
         --out results_gpu/csgnn_valsel.jsonl 2>&1 | tee -a logs_gpu/csgnn_valsel.log
  python select_csgnn.py
fi

# 3d) 2^4 factorial over the recipe ingredients (log-scaling, degree features, class weights, calibration), MLP + GraphSAGE
if has factorial; then
  python factorial.py --datasets sichuan,bupt --models mlp,sage --splits 0 1 2 3 4 --seeds 0 1 2 \
         --out results_gpu/factorial.jsonl 2>&1 | tee logs_gpu/factorial.log
  python make_factorial.py | tee logs_gpu/factorial_summary.log
fi

# 4) YelpChi / Amazon controlled head-to-head (BWGNN port + our baselines, same split & seeds)
if has fraud; then
  python fraud_runs.py --datasets yelp,amazon --seeds 0 1 2 3 4 --out results_gpu/fraud_runs.jsonl 2>&1 | tee logs_gpu/fraud.log
  # optional: also the official BWGNN code via DGL (uncomment if dgl is installed)
  # python fraud_runs.py --datasets yelp,amazon --seeds 0 1 2 3 4 --methods "" --official_bwgnn --out results_gpu/fraud_runs.jsonl
fi

if has aggregate; then
  python aggregate_gpu.py | tee logs_gpu/aggregate.log
  tar czf results_gpu.tgz results_gpu logs_gpu
  echo "Done. Send back: rethink/results_gpu.tgz"
fi
