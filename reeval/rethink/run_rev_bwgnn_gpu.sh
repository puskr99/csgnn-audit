#!/usr/bin/env bash
# BWGNN per-split hyperparameter selection (same 60 configurations as the split-0 search), Sichuan and BUPT.
# Run from rethink/ in the GPU environment:  mkdir -p logs_rev && nohup bash run_rev_bwgnn_gpu.sh > logs_rev/bwgnn.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"; mkdir -p results_rev logs_rev
for ds in sichuan bupt; do
  python -u rev_pertune.py --dataset $ds --methods bwgnn \
         --out results_rev/pertune_gpu_bwgnn_$ds.jsonl --trials results_rev/pertune_gpu_bwgnn_trials_$ds.jsonl 2>&1 | tee -a logs_rev/bwgnn_$ds.log
done
tar czf results_rev_bwgnn.tgz results_rev/pertune_gpu_bwgnn_* logs_rev/bwgnn_*
echo "Done. Send back: rethink/results_rev_bwgnn.tgz"
