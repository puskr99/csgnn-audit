#!/usr/bin/env bash
# Revision experiments for the GPU server (run from inside rethink/, same environment as run_all.sh).
#   A  per-split hyperparameter selection, BUPT: CS-Boost (staged search) + all recipe-matched baselines
#   A  per-split selection, Sichuan: CS-Boost, MLP, GraphSAGE, GCN, GAT
#   C  PC-GNN port and label-free CSGNN port with the recipe input, Sichuan and BUPT
# Usage:  mkdir -p logs_rev && nohup bash run_rev_gpu.sh > logs_rev/run_rev_gpu.out 2>&1 &
# Every step is resumable: if it dies, start it again. Send back:  rethink/results_rev_gpu.tgz
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p results_rev logs_rev
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
python -u rev_pertune.py --dataset bupt --methods v5_fixed,v5,mlp,sage,gcn,gat \
       --out results_rev/pertune_gpu_bupt.jsonl --trials results_rev/pertune_gpu_trials_bupt.jsonl 2>&1 | tee -a logs_rev/A_bupt.log
python -u rev_pertune.py --dataset sichuan --methods v5_fixed,v5,mlp,sage,gcn,gat \
       --out results_rev/pertune_gpu_sichuan.jsonl --trials results_rev/pertune_gpu_trials_sichuan.jsonl 2>&1 | tee -a logs_rev/A_sichuan.log
python -u rev_logpub.py --method pcgnn --dataset sichuan --out results_rev/logpub_gpu.jsonl 2>&1 | tee -a logs_rev/C_pcgnn_sichuan.log
python -u rev_logpub.py --method csgnn --dataset sichuan --out results_rev/logpub_gpu.jsonl 2>&1 | tee -a logs_rev/C_csgnn_sichuan.log
python -u rev_logpub.py --method pcgnn --dataset bupt --out results_rev/logpub_gpu.jsonl 2>&1 | tee -a logs_rev/C_pcgnn_bupt.log
python -u rev_logpub.py --method csgnn --dataset bupt --out results_rev/logpub_gpu.jsonl 2>&1 | tee -a logs_rev/C_csgnn_bupt.log
tar czf results_rev_gpu.tgz results_rev logs_rev
echo "Done. Send back: rethink/results_rev_gpu.tgz"
