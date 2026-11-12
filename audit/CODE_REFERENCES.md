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

Orientation of the cost matrix (relevant for any label-free cost-sensitive decision rule)
- `utils.py:140-159` (`_set_cost_matrix`, called from `model.py:177-178` with training labels, `main.py:167`):
  the matrix is built row by row over `c_pred` (line 147) with entries `n[c_pred] / n[c_actual]` (line 149),
  diagonal set to 1 (line 152), then `log1p` (line 159). Rows = predicted class, columns = true class.
- `utils.py:123-133` (`misclassification_cost`): looks up `row = y_pred`, `column = y_true`, the same convention.
- `model.py:208-209` (`cost_logits`): looks up `row = label`, i.e. the row is indexed by the node's *true* class.
- Paper, Section 3: `C_ij` is the cost of classifying a class-`i` sample as class `j` (rows = true class).
- The diagonal is `log1p(1) ≈ 0.69`, not 0, and the matrix is updated during training from validation labels (`main.py:197-207`, step at 237-240).
- The update target `T = cost_weight · H · S · R` (`main.py:197-203`) uses `R = confusion_matrix(y_true, y_pred)`,
  whose rows are true classes. The matrix is therefore initialised with rows = predicted class but fitted to a
  target with rows = true class, so the orientation of the learned matrix is ambiguous.

## What we can and cannot confirm

**Confirmed from the released code (this repository, reproducible):**
1. At inference the score of every test node is re-weighted by the cost-matrix row of its ground-truth label
   (lines above).
2. Running the released code unchanged with its released configuration and seed reproduces the published BUPT
   test AUC exactly (0.9698) and the G-Mean/recall to within 0.0004 (see `audit/logs/original_bupt_seed42.log`).
3. Shuffling only the test labels before the inference call changes the reported metrics (label-permutation
   test, `audit/csgnn_audit.py`), which would not happen if test labels did not enter the prediction path.
4. Scoring the same trained model without test labels gives lower BUPT numbers (see `audit/RESULTS.md`).
5. Test labels do not influence training: permuting them before training leaves the network outputs of every
   epoch and the learned cost matrix bit-identical (`audit/train_permutation_test.py`, both datasets); permuting
   validation labels instead changes them.

**What we cannot confirm from the released code:**
- That the published tables were produced with exactly this code, configuration and software environment
  (we have no access to the authors' runs). The exact match of the BUPT AUC under the released configuration
  suggests that it is, but we cannot verify it.
- Sichuan: the released configuration (15 epochs, as also stated in the paper) gives AUC 0.9543 / G-Mean 0.8654
  versus the published 0.9539 / 0.8864; we could not explain the G-Mean gap. (After 200 epochs the released scoring
  gives 0.9558 / 0.8859 and label-free scoring 0.9403 / 0.8741, but the paper states 15 epochs.)
- The authors' intended test-time procedure. The paper (Section 4.3.2, Eq. 13: `p_k(v) = Softmax(C[v, k] · z_v[k])
  = C_{v,k} exp(z_v[k]) / sum_k' C_{v,k'} exp(z_v[k'])`; the written-out form is exactly what `cost_logits` computes)
  indexes the cost matrix by node `v` and does not say how the row is chosen when the node's label is unknown; it
  does not describe inference separately. Section 3 defines `C_ij` as the cost of classifying a class-`i` sample as
  class `j`, which suggests the row is the node's true class, as in the released code.
- Whether other reported experiments (ablations, parameter studies, other baselines in the paper) used the
  same scoring path.

## Permalinks (commit aef3136)

- Labels of all nodes: <https://github.com/xxhu94/CSGNN/blob/aef3136bd96b88ebee07cafc209fa9105a58a555/main.py#L145>
- Cost layer in training: <https://github.com/xxhu94/CSGNN/blob/aef3136bd96b88ebee07cafc209fa9105a58a555/main.py#L188>
- Cost-matrix update from validation labels: <https://github.com/xxhu94/CSGNN/blob/aef3136bd96b88ebee07cafc209fa9105a58a555/main.py#L197-L207>
- Cost layer at test time: <https://github.com/xxhu94/CSGNN/blob/aef3136bd96b88ebee07cafc209fa9105a58a555/main.py#L261>
- Test predictions and metrics: <https://github.com/xxhu94/CSGNN/blob/aef3136bd96b88ebee07cafc209fa9105a58a555/main.py#L262-L277>
- `cost_logits`: <https://github.com/xxhu94/CSGNN/blob/aef3136bd96b88ebee07cafc209fa9105a58a555/model.py#L203-L213>
- `_set_cost_matrix`: <https://github.com/xxhu94/CSGNN/blob/aef3136bd96b88ebee07cafc209fa9105a58a555/utils.py#L140-L159>
