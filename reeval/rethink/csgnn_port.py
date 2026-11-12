"""Faithful pure-PyTorch port of CSGNN (Hu et al., IEEE TCSS 2023; github.com/xxhu94/CSGNN, main.py/model.py/utils.py).
Only change vs. original: top-p neighbour sampling is vectorised (same semantics) so BUPT runs in reasonable time.

--eval leak   : published protocol. Test logits pass through CostMatrix.cost_logits(feat, labels, ...), which
                multiplies node i's scores by row `labels[i]` of the cost matrix, i.e. uses the TRUE label of
                every test node at inference time.
--eval honest : identical trained model, test scores = softmax of the GNN output (no ground-truth labels used).
"""
import os, argparse, time, copy, numpy as np, torch, torch.nn as nn, scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from common import load, metrics, fmt

CFG = {"sichuan": dict(hid_dim=256, num_layers=3, max_epoch=15),
       "bupt": dict(hid_dim=64, num_layers=5, max_epoch=1)}
COMMON = dict(lr=0.01, wd=1e-3, lr_cost=0.01, wd_cost=1e-3, sim_weight=2, step_size=0.02, cost_weight=1000)


def build_graph(ds, split=0):
    X, y, src, dst, tr, va, te = load(ds, split)
    N = len(y)
    if ds == "sichuan":   # adj + adj.T (symmetric) + I
        s, d = np.r_[src, dst], np.r_[dst, src]
    else:                 # BUPT_process: keep edges with u<v, mirror them, + self loops
        m = src < dst; s, d = np.r_[src[m], dst[m]], np.r_[dst[m], src[m]]
    k = np.unique(np.r_[s * N + d, np.arange(N) * N + np.arange(N)])
    s, d = k // N, k % N
    Xs = StandardScaler().fit_transform(X).astype(np.float32)   # original z-scores on all nodes
    return Xs, y, s, d, tr, va, te


class Layer(nn.Module):
    def __init__(self, din, dout, C, step):
        super().__init__()
        self.linear, self.MLP = nn.Linear(din, dout), nn.Linear(din, C)
        self.p, self.last, self.f, self.cvg, self.step = 0.5, 0.0, [], False, step

    def forward(self, h, s, d, N):
        z = torch.tanh(self.MLP(h))           # identical to MLP on src/dst copies, memory-light
        dist = (z[s] - z[d]).abs().sum(1)
        self.dist = dist
        # top-p sampling of in-edges of each dst by smallest distance (vectorised)
        dd = dist.detach().numpy()
        order = np.lexsort((dd, d))
        deg = np.bincount(d, minlength=N)
        start = np.r_[0, np.cumsum(deg)[:-1]]
        rank = np.arange(len(d)) - start[d[order]]
        k = np.ceil(deg * self.p).astype(int)
        keep = order[rank < k[d[order]]]
        ks, kd = s[keep], d[keep]
        cnt = np.maximum(np.bincount(kd, minlength=N), 1)
        A = torch.sparse_coo_tensor(np.vstack([kd, ks]), (1.0 / cnt[kd]).astype(np.float32), (N, N)).coalesce()
        hr = torch.tanh(torch.sparse.mm(A, h))   # mean over sampled in-neighbours
        return self.linear(torch.tanh(self.p * hr + h))


class CSGNN(nn.Module):
    def __init__(self, din, C, hid, L, step):
        super().__init__()
        dims = [din] + [hid] * (L - 1) + [C]
        self.layers = nn.ModuleList(Layer(dims[i], dims[i + 1], C, step) for i in range(L))
        self.step = step

    def forward(self, x, s, d, N):
        sim = torch.tanh(self.layers[0].MLP(x))
        for l in self.layers:
            x = l(x, s, d, N)
        return x, sim

    def rl(self, d_np, epoch, rl_idx):
        mask = torch.tensor(np.isin(d_np, rl_idx))
        for l in self.layers:
            if l.cvg: continue
            avg = l.dist[mask].mean().item()
            if l.last < avg:
                if l.p - self.step > 0: l.p -= self.step
                l.f.append(-1)
            else:
                if l.p + self.step <= 1: l.p += self.step
                l.f.append(1)
            l.last = avg
            if epoch >= 9 and abs(sum(l.f[-10:])) <= 2: l.cvg = True


def init_cost(ytr, C):
    n = np.bincount(ytr, minlength=C).astype(float)
    M = n[:, None] / n[None, :]; np.fill_diagonal(M, 1.0)
    return np.log1p(M)


def cost_logits(feat, labels, Cm):  # original: row = true label of each node
    z = torch.tensor(Cm[labels], dtype=torch.float64) * torch.exp(feat.double())
    return z / z.sum(1, keepdim=True)


def calc_H(y, C):
    h = np.bincount(y, minlength=C) / len(y)
    H = np.maximum(h[:, None], h[None, :]); np.fill_diagonal(H, h)
    return H


def calc_S(X, y, C):
    SW = np.array([np.trace(np.cov(X[y == c].T, bias=True)) for c in range(C)])
    SW = np.tile(SW, (C, 1))
    mu = np.stack([X[y == c].mean(0) for c in range(C)])
    SB = np.linalg.norm(mu[:, None] - mu[None], axis=2); np.fill_diagonal(SB, 1)
    return SW.T * (1 / SB)


def train_model(a):
    """Train CSGNN as released (optionally with label-free early stopping). Returns final logits etc."""
    cfg = {**COMMON, **CFG[a.dataset]}
    if a.max_epoch: cfg["max_epoch"] = a.max_epoch
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    X, y, s, d, tr, va, te = build_graph(a.dataset, a.split)
    N, C = len(y), int(y.max() + 1)
    x, yt = torch.tensor(X), torch.tensor(y)
    model = CSGNN(X.shape[1], C, cfg["hid_dim"], cfg["num_layers"], cfg["step_size"])
    Cm = nn.Parameter(torch.tensor(init_cost(y[tr], C)))
    ce = nn.CrossEntropyLoss()
    og = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    oc = torch.optim.Adam([Cm], lr=cfg["lr_cost"], weight_decay=cfg["wd_cost"])
    rl_idx = tr[y[tr] > 0]
    trt, vat = torch.tensor(tr), torch.tensor(va)
    t0 = time.time(); best, best_ep = -1, -1
    for ep in range(cfg["max_epoch"]):
        model.train()
        lg, sim = model(x, s, d, N)
        lc = cost_logits(lg, y, Cm.detach().numpy())
        loss = ce(lc[trt], yt[trt]) + cfg["sim_weight"] * ce(sim[trt], yt[trt])
        H = calc_H(y[tr], C)
        S = calc_S(lc[vat].detach().numpy(), y[va], C)
        R = confusion_matrix(y[va], lc[vat].detach().argmax(1).numpy(), normalize="all", labels=list(range(C)))
        T = torch.tensor(cfg["cost_weight"] * H * S * R)
        verr = 1 - accuracy_score(y[va], lc[vat].detach().argmax(1).numpy())
        lcost = torch.norm(T - Cm) ** 2 + verr
        og.zero_grad(); loss.backward(); og.step()
        oc.zero_grad(); lcost.backward(); oc.step()
        model.rl(d, ep, rl_idx)
        if a.es:
            model.eval()
            with torch.no_grad():
                hv = torch.softmax(model(x, s, d, N)[0][vat].double(), 1).numpy()
            mv = metrics(y[va], hv); sc = mv["gmean"] + mv["auc"]
            if sc > best:
                best, best_ep = sc, ep
                snap = (copy.deepcopy(model.state_dict()), [l.p for l in model.layers], Cm.detach().clone())
            print(f"   val honest g={mv['gmean']:.4f} auc={mv['auc']:.4f}")
        print(f"ep{ep} loss={loss.item():.4f} p={[round(l.p, 2) for l in model.layers]}")
    if a.es:
        model.load_state_dict(snap[0]); Cm.data = snap[2]
        for l, pp in zip(model.layers, snap[1]): l.p = pp
        print("best epoch", best_ep)
    model.eval()
    with torch.no_grad():
        lg, _ = model(x, s, d, N)
    return lg, Cm, y, tr, va, te, cfg, t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sichuan")
    p.add_argument("--max_epoch", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", type=int, default=0, help="0 = shared split; k>0 = new stratified split")
    p.add_argument("--out_json", default=None, help="append test metrics of all scorings as one JSON line")
    p.add_argument("--es", type=int, default=0, help="1 = honest early stopping on val (G-Mean+AUC of label-free scores)")
    a = p.parse_args()
    lg, Cm, y, tr, va, te, cfg, t0 = train_model(a)
    leak = torch.softmax(cost_logits(lg, y, Cm.detach().numpy()), 1).numpy()   # as in main.py
    honest = torch.softmax(lg.double(), 1).numpy()
    # label-free repairs of the cost layer: index the cost matrix by the model's own prediction
    M = Cm.detach().numpy()
    yhat = lg.argmax(1).numpy()
    fix_pred = torch.softmax(cost_logits(lg, yhat, M), 1).numpy()
    q = torch.softmax(lg.double(), 1).numpy()                           # expected cost row
    z = (q @ M) * torch.exp(lg.double()).numpy(); z = z / z.sum(1, keepdims=True)
    fix_exp = torch.softmax(torch.tensor(z), 1).numpy()
    print(f"[{a.dataset}] epochs={cfg['max_epoch']} time={time.time()-t0:.1f}s")
    print("cost matrix:\n", np.round(Cm.detach().numpy(), 3))
    print("  TEST leak   (published protocol):", fmt(metrics(y[te], leak[te])))
    print("  TEST honest (no test labels)    :", fmt(metrics(y[te], honest[te])))
    print("  TEST fix: row = predicted class  :", fmt(metrics(y[te], fix_pred[te])))
    print("  TEST fix: row = expected (soft)  :", fmt(metrics(y[te], fix_exp[te])))
    np.save(f"{os.path.dirname(os.path.abspath(__file__))}/cache/csgnn_{a.dataset}_e{cfg['max_epoch']}_es{a.es}_s{a.seed}{'' if a.split == 0 else f'_split{a.split}'}.npy", dict(leak=leak[te], honest=honest[te], fix_pred=fix_pred[te], fix_exp=fix_exp[te], y=y[te]))

    if a.out_json:
        import json
        rec = dict(dataset=a.dataset, split=a.split, seed=a.seed, epochs=cfg["max_epoch"], es=a.es,
                   test={k: metrics(y[te], v[te]) for k, v in
                         [("leak", leak), ("honest", honest), ("fix_pred", fix_pred), ("fix_exp", fix_exp)]},
                   # validation scores of the label-free variants: used to choose ONE variant per run (no test-set selection)
                   val={k: metrics(y[va], v[va]) for k, v in
                        [("honest", honest), ("fix_pred", fix_pred), ("fix_exp", fix_exp)]})
        with open(a.out_json, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
