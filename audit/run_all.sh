#!/usr/bin/env bash
# Reproduce all results from a fresh clone (CPU is enough; ~3-4 min per BUPT/Sichuan run on 2 cores, ~8 GB RAM + swap).
#   git clone <this repo> csgnn-audit && cd csgnn-audit && bash audit/run_all.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 0. the released files must be byte-identical to commit aef3136
bash audit/verify_original.sh

# 1. data, exactly as the released README says
(cd data && unzip -o -q BUPT.zip && unzip -o -q Sichuan.zip)
python data_process.py

# 2. the released main.py hard-codes /code/CSGNN/data/... ; point that path at this checkout instead of editing it
if [ ! -e /code/CSGNN ]; then
  sudo mkdir -p /code && sudo ln -s "$ROOT" /code/CSGNN
fi

mkdir -p audit/logs audit/results
# 3. released code, released configs, released seed (42) -- nothing changed
python main.py --config ./config/csgnn_bupt.yml    2>&1 | tee audit/logs/original_bupt_seed42.log
python main.py --config ./config/csgnn_sichuan.yml 2>&1 | tee audit/logs/original_sichuan_seed42.log

# 4. same training code, scored with / without test labels, 4 seeds, 5 label permutations
python audit/csgnn_audit.py --config config/csgnn_bupt.yml    --seeds 42 0 1 2 --perms 5 2>&1 | tee audit/logs/audit_bupt.log
python audit/csgnn_audit.py --config config/csgnn_sichuan.yml --seeds 42 0 1 2 --perms 5 2>&1 | tee audit/logs/audit_sichuan.log

# 5. longer training: 200 epochs, epoch + label-free variant chosen on validation only (Sichuan ~45 min; BUPT ~5 h)
python audit/csgnn_audit.py --config config/csgnn_sichuan.yml --seeds 42 --max-epoch 200 --select-epoch 2>&1 | tee audit/logs/audit_sichuan_ep200.log
# python audit/csgnn_audit.py --config config/csgnn_bupt.yml --seeds 42 --max-epoch 200 --select-epoch 2>&1 | tee audit/logs/audit_bupt_ep200.log

# 6. do test labels influence training? permute them (and, as a positive control, the validation labels) before training
python audit/train_permutation_test.py --config config/csgnn_bupt.yml    --seed 42 2>&1 | tee audit/logs/train_perm_bupt.log
python audit/train_permutation_test.py --config config/csgnn_sichuan.yml --seed 42 2>&1 | tee audit/logs/train_perm_sichuan.log

python audit/summarise.py > audit/RESULTS.md
