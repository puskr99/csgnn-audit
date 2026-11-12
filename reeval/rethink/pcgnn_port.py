"""PC-GNN (Liu et al., WWW 2021, "Pick and Choose") for the telecom graphs, ported from the official code
(github.com/PonderLY/PC-GNN: src/layers.py, src/model.py, src/model_handler.py, src/utils.py).

Kept from the official implementation (same defaults as config/pcgnn_*.yml):
  * PICK  : label-balanced node sampler, P(v) ~ degree(v) / LF(class(v)), |sample| per epoch = C x mean minority count
            (= 2 x #positives in the binary original)
  * CHOOSE: per-relation neighbour filtering by L1 distance of label-aware scores, keep ceil(0.5 x deg) neighbours;
            in training, minority-class centres additionally get the rho x k nearest training nodes of their own class
  * AGGREGATE: IntraAgg = relu([h_v ; mean(h_N(v))] W), InterAgg = relu([h_v ; intra_1..R] W)
  * loss = CE(GNN logits) + alpha x CE(label-aware logits); Adam, lr, weight decay 1e-3, batch 1024,
            validation AUC every 5 epochs, best-validation-AUC checkpoint is tested.
Generalisations needed for these datasets (the original is binary and has three relations):
  * one relation (the telecom graphs are homogeneous), self-loops added as in `sparse_to_adjlist`
  * C-class heads; CHOOSE distance = L1 over the C label-aware scores (the original uses score[:,0] of a 2-way head);
    "minority" = every class except the training-majority class, oversampling draws from the centre's own class
  * probabilities = softmax of the GNN logits, decision = arg-max (the protocol used for the shared-split numbers)
"""
import argparse, json, math, os, random, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn import init
from common import load, metrics

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ CHOOSE step (src/layers.py)
def choose_train(center_scores, center_labels, neigh_scores, neighs_list, minor_scores, minor_lists, sample_list, rho, majority):
    samp = []
    for i in range(len(neighs_list)):
        cs = center_scores[i]
        d = (neigh_scores[i] - cs).abs().sum(1)
        order = torch.argsort(d).tolist()
        k = sample_list[i]
        neighs = neighs_list[i]
        sel = [neighs[j] for j in order[:k]] if len(neighs) > k + 1 else list(neighs)
        c = int(center_labels[i])
        if c != majority and len(minor_lists[c]):
            n_over = int(k * rho)
            if n_over > 0:
                dm = (minor_scores[c] - cs).abs().sum(1)
                om = torch.argsort(dm)[:n_over].tolist()
                sel.extend(minor_lists[c][j] for j in om)
        samp.append(set(sel))
    return samp


def choose_test(center_scores, neigh_scores, neighs_list, sample_list):
    samp = []
    for i in range(len(neighs_list)):
        d = (neigh_scores[i] - center_scores[i]).abs().sum(1)
        order = torch.argsort(d).tolist()
        k = sample_list[i]; neighs = neighs_list[i]
        samp.append(set([neighs[j] for j in order[:k]] if len(neighs) > k + 1 else neighs))
    return samp


class IntraAgg(nn.Module):
    def __init__(self, feats, feat_dim, emb, rho):
        super().__init__()
        self.feats, self.rho = feats, rho
        self.weight = nn.Parameter(torch.empty(2 * feat_dim, emb)); init.xavier_uniform_(self.weight)

    def forward(self, nodes, labels, neighs_list, center_scores, neigh_scores, minor_scores, minor_lists, sample_list, majority, train):
        if train:
            samp = choose_train(center_scores, labels, neigh_scores, neighs_list, minor_scores, minor_lists, sample_list, self.rho, majority)
        else:
            samp = choose_test(center_scores, neigh_scores, neighs_list, sample_list)
        uniq = list(set.union(*samp)); pos = {n: j for j, n in enumerate(uniq)}
        rows = [i for i, s in enumerate(samp) for _ in s]; cols = [pos[n] for s in samp for n in s]
        M = torch.zeros(len(samp), len(uniq), device=DEV); M[rows, cols] = 1.0
        M = M / M.sum(1, keepdim=True)
        agg = M @ self.feats[torch.as_tensor(uniq, device=DEV)]
        self_f = self.feats[torch.as_tensor(nodes, device=DEV)]
        return F.relu(torch.cat([self_f, agg], 1) @ self.weight)


class PCGNN(nn.Module):
    def __init__(self, feats, adj_lists, n_cls, emb, rho, alpha, minor_lists, majority, thr=0.5):
        super().__init__()
        self.feats, self.adj_lists, self.C, self.alpha = feats, adj_lists, n_cls, alpha
        self.minor_lists, self.majority, self.thr = minor_lists, majority, thr
        d = feats.shape[1]
        self.label_clf = nn.Linear(d, n_cls)
        self.intra = nn.ModuleList([IntraAgg(feats, d, emb, rho) for _ in adj_lists])
        self.w_inter = nn.Parameter(torch.empty(emb * len(adj_lists) + d, emb)); init.xavier_uniform_(self.w_inter)
        self.w_out = nn.Parameter(torch.empty(n_cls, emb)); init.xavier_uniform_(self.w_out)

    def forward(self, nodes, labels, train):
        to_neighs = [[list(adj[n]) for n in nodes] for adj in self.adj_lists]
        uniq = set(nodes)
        for tn in to_neighs:
            for s in tn: uniq.update(s)
        uniq = list(uniq); pos = {n: j for j, n in enumerate(uniq)}
        bs = self.label_clf(self.feats[torch.as_tensor(uniq, device=DEV)])
        center = bs[torch.as_tensor([pos[n] for n in nodes], device=DEV)]
        minor_scores = {c: self.label_clf(self.feats[torch.as_tensor(l, device=DEV)]) for c, l in self.minor_lists.items() if len(l)}
        outs = []
        for r, tn in enumerate(to_neighs):
            nsc = [bs[torch.as_tensor([pos[x] for x in s], device=DEV)] for s in tn]
            kl = [math.ceil(len(s) * self.thr) for s in tn]
            outs.append(self.intra[r](nodes, labels, tn, center, nsc, minor_scores, self.minor_lists, kl, self.majority, train))
        self_f = self.feats[torch.as_tensor(nodes, device=DEV)]
        h = F.relu(torch.cat([self_f] + outs, 1) @ self.w_inter)
        return h @ self.w_out.t(), center

    def loss(self, nodes, labels):
        logits, lab = self.forward(nodes, labels, True)
        y = torch.as_tensor(labels, device=DEV)
        return F.cross_entropy(logits, y) + self.alpha * F.cross_entropy(lab, y)

    @torch.no_grad()
    def predict(self, nodes, bs=1024):
        out = []
        for i in range(0, len(nodes), bs):
            b = list(nodes[i:i + bs])
            out.append(torch.softmax(self.forward(b, None, False)[0], 1).cpu().numpy())
        return np.concatenate(out)


def pick_step(idx_train, y_train, deg, size, rng):
    lf = np.bincount(y_train)[y_train].astype(float)        # label frequency of each node's class
    p = deg[idx_train] / lf
    return rng.choices(list(idx_train), weights=p.tolist(), k=size)


def preprocess(X, mode, tr=None):
    if mode == "zscore_train":                               # official code's (commented-out) StandardScaler fit on training nodes
        mu, sd = X[tr].mean(0), X[tr].std(0)
        return (X - mu) / np.where(sd > 0, sd, 1.0)
    if mode == "rownorm":                                    # official: utils.normalize (row sum + 0.01)
        return X / (X.sum(1, keepdims=True) + 0.01)
    if mode == "zscore":
        return (X - X.mean(0)) / (X.std(0) + 1e-8)
    if mode == "log":
        Z = np.sign(X) * np.log1p(np.abs(X)); return (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    raise ValueError(mode)


def run(ds, split, seed, emb, lr, feat="zscore_train", epochs=51, valid_every=5, wd=1e-3, bs=1024, rho=0.5, alpha=2.0, verbose=False):
    X, y, src, dst, tr, va, te = load(ds, split)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    n, C = len(y), int(y.max()) + 1
    adj = [set([i]) for i in range(n)]                        # self loops + symmetric, as sparse_to_adjlist
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[u].add(v); adj[v].add(u)
    deg = np.array([len(a) for a in adj], dtype=float)
    feats = torch.tensor(preprocess(X.astype(np.float64), feat, tr), dtype=torch.float32, device=DEV)
    ytr = y[tr]; cnt = np.bincount(ytr, minlength=C); majority = int(cnt.argmax())
    minor_lists = {c: [int(i) for i in tr[ytr == c]] for c in range(C) if c != majority}
    size = int(round(C * np.mean([cnt[c] for c in range(C) if c != majority])))
    model = PCGNN(feats, [adj], C, emb, rho, alpha, minor_lists, majority).to(DEV)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    best, best_state, best_ep, t0 = -1, None, -1, time.time()
    for ep in range(epochs):
        model.train()
        samp = pick_step(tr, ytr, deg, size, rng); rng.shuffle(samp)
        for i in range(0, len(samp), bs):
            b = samp[i:i + bs]
            if not b: continue
            opt.zero_grad(); l = model.loss(b, y[np.array(b)]); l.backward(); opt.step()
        if ep % valid_every == 0:
            model.eval(); pv = model.predict(va)
            m = metrics(y[va], pv)
            if verbose: print(f"  ep {ep} loss {l.item():.4f} val auc {m['auc']:.4f} g {m['gmean']:.4f}", flush=True)
            if m["auc"] > best:
                best, best_ep = m["auc"], ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    pt = model.predict(te)
    return dict(test=metrics(y[te], pt), best_epoch=best_ep, val_auc=best, time=time.time() - t0), pt, te


HP = {"sichuan": dict(emb=64, lr=0.003), "bupt": dict(emb=128, lr=0.003)}   # shared-split settings (pcgnn_main_results.csv)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="sichuan,bupt")
    p.add_argument("--splits", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--feat", default="zscore_train", choices=["zscore_train", "rownorm", "zscore", "log"])
    p.add_argument("--epochs", type=int, default=51)
    p.add_argument("--out", default="results_gpu/pcgnn_multisplit.jsonl")
    p.add_argument("--save_probs", default="", help="directory for test probabilities (optional)")
    p.add_argument("--verbose", action="store_true")
    A = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    done = set()
    if os.path.exists(A.out):
        for l in open(A.out):
            r = json.loads(l); done.add((r["dataset"], r["split"], r["seed"]))
    print("device:", DEV, flush=True)
    for ds in A.datasets.split(","):
        for k in A.splits:
            for s in A.seeds:
                if (ds, k, s) in done: continue
                r, pt, te = run(ds, k, s, feat=A.feat, epochs=A.epochs, verbose=A.verbose, **HP[ds])
                rec = dict(dataset=ds, split=k, seed=s, method="pcgnn", feat=A.feat, **r)
                with open(A.out, "a") as fh: fh.write(json.dumps(rec) + "\n")
                if A.save_probs:
                    os.makedirs(A.save_probs, exist_ok=True)
                    np.save(os.path.join(A.save_probs, f"{ds}_split{k}_seed{s}.npy"), dict(test_idx=te, probs=pt))
                print(ds, k, s, {m: round(v, 4) for m, v in r["test"].items()}, f"ep={r['best_epoch']} t={r['time']:.0f}s", flush=True)
