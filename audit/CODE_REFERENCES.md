# Code references: how labels enter the released evaluation path

All references are to the unmodified files at commit
`aef3136bd96b88ebee07cafc209fa9105a58a555` (xxhu94/CSGNN, "Update README.md", 25 Nov 2023), preserved in this
repository as tag `original-aef3136` and branch `original`. View any line with
`git show original-aef3136:main.py | sed -n 255,280p`.

| File:line | Code | Meaning |
|---|---|---|
| `main.py:145` | `labels = graph.ndata['label'].long().squeeze().to(device)` | Ground-truth labels of **all** nodes, test nodes included |
| `main.py:188` | `logits_gnn=cost.cost_logits(logits_gnn,labels,cost_table)` | Training: cost layer applied to every node, indexed by its true label |
| `main.py:226-230` | `val_auc = roc_auc_score(...logits_gnn...)` | Validation AUC is computed from label-indexed scores; it drives early stopping if `early_stop: True` (`main.py:249-251`) |
| `main.py:259-261` | `logits_gnn, logits_sim = model.forward(graph, feat)` / `logits_gnn = cost.cost_logits(logits_gnn, labels, cost_table)` | **Inference**: same label-indexed layer applied to test nodes |
| `main.py:262-263` | `logp = logits_gnn` / `test_h=torch.argmax(logp[test_idx],dim=1)` | Test predictions taken from the label-indexed scores |
| `main.py:269, 272` | `roc_auc_score(labels[test_mask], torch.softmax(logp[test_mask]) ...)` | Reported test AUC |
| `main.py:276-277` | `geometric_mean_score(...)`, `recall_score(...)` | Reported test G-Mean and macro recall |
| `model.py:203` | `def cost_logits(self, feat, labels, cost_table)` | Cost-sensitive output layer |
| `model.py:208-209` | `pd.DataFrame({'row': labels.cpu(), 'column': i ...}).merge(cost_table, on=['row','column'])` | Row of the cost matrix selected by each node's **label** |
| `model.py:211, 213` | `torch.mul(cost, torch.exp(feat))`, row normalisation | `p[i,c] ∝ M[y_i, c] · exp(z[i,c])` |
| `main.py:296` | `setup_seed(42)` | Single fixed seed |
| `config/csgnn_bupt.yml:7,12` | `early_stop: False`, `max_epoch: 1` | BUPT trains for one epoch |
| `config/csgnn_sichuan.yml:7,12` | `early_stop: False`, `max_epoch: 15` | Sichuan trains for 15 epochs |

Secondary (minor) observations
- `main.py:197-202`: the learned cost matrix is updated from validation-node labels and predictions (Eq. 17–20 of
  the paper). This uses validation labels for training, which is a design choice and unrelated to the test-set observation.
- `utils.py:41-43`: the early-stopping checkpoint stores `state_dict()` only; the RL neighbour-sampling
  thresholds `layer.p` are plain attributes and are not restored. Irrelevant for the released configs
  (early stopping off).
- `data_process.py:30, 89`: features are z-scored with `StandardScaler` fitted on all nodes before the split (transductive; uses no labels).

## What we can and cannot confirm

**Confirmed from the released code (this repository, reproducible):**
1. At inference the score of every test node is re-weighted by the cost-matrix row of its ground-truth label
   (lines above).
2. Running the released code unchanged with its released configuration and seed reproduces the published BUPT
   test AUC exactly (0.9698) and the G-Mean/recall to within 0.0004 (see `audit/logs/original_bupt_seed42.log`).
3. Shuffling only the test labels before the inference call changes the reported metrics (label-permutation
   test, `audit/csgnn_audit.py`), which would not happen if test labels did not enter the prediction path.
4. Scoring the same trained model without test labels gives lower BUPT numbers (see `audit/RESULTS.md`).

**What we cannot confirm from the released code:**
- That the published tables were produced with exactly this code, configuration and software environment
  (we have no access to the authors' runs). The exact match of the BUPT AUC under the released configuration
  suggests that it is, but we cannot verify it.
- Sichuan: the released configuration (15 epochs) gives AUC 0.9543 / G-Mean 0.8654 versus the published
  0.9539 / 0.8864. Trained for 200 epochs (seed 42), the released scoring gives 0.9558 / 0.8859, much closer to the
  published values, which suggests (but does not prove) that the published Sichuan numbers came from longer
  training. The same 200-epoch model scored without test labels gives 0.9403 / 0.8741.
- Whether other reported experiments (ablations, parameter studies, other baselines in the paper) used the
  same scoring path.
