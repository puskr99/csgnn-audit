# Data

The benchmarks are not redistributed here. Both are public; we used the copies distributed with the GAT-COBO code
(<https://github.com/xxhu94/GAT-COBO>). Place the files as follows (paths used by `rethink/common.py`):

```
data/Sichuan/all_feat_with_label.csv
data/Sichuan/node_adj_sparse.npz
data/Sichuan/sichuan_split.npz      (included: shared split)
data/BUPT/TF.features
data/BUPT/TF.labels
data/BUPT/TF.edgelist
data/BUPT/bupt_split.npz            (included: shared split)
```

- Sichuan: 2020 Sichuan Big Data Competition (China Mobile Sichuan), prepared as a graph by Hu et al. (2022).
- BUPT: released by Liu et al. (2019).
- YelpChi and Amazon (Sec. 7) come from the CARE-GNN repository (<https://github.com/YingtongDou/CARE-GNN>);
  `rethink/run_all.sh setup` downloads them into `rethink/fraud_data/`.

The index files of all five splits are in `../splits/`.
