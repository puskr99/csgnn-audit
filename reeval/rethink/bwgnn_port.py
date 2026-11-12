"""Faithful pure-PyTorch port of BWGNN (Tang et al., ICML 2022; github.com/squareRoot3/Rethinking-Anomaly-Detection),
so it runs on CPU/GPU without DGL. Mirrors BWGNN.py and main.py:train() line by line:
  - PolyConv: h = sum_k theta_k L^k h with L h = h - D^-1/2 A D^-1/2 h, D = in-degree (clamped at 1),
    message passing = sum over incoming edges (A built with duplicates, like dgl.to_homogeneous)
  - BWGNN (homo): 2 linear+ReLU, Beta-wavelet filter bank, linear3, ReLU, linear4; graph = union of
    relations + one self loop per node
  - BWGNN_Hetero: per relation, conv bank + linear3; NOTE the official code re-assigns `h = linear3(...)`
    inside the relation loop, so relation r+1 receives relation r's output. We reproduce this as released.
  - train(): Adam lr 0.01, 100 epochs, CE weight neg/pos, threshold chosen on val macro-F1 (0.05 grid),
    test metrics recorded at the epoch with the best val macro-F1, probabilities taken from the
    training-mode forward pass of that epoch (as in the official code).
"""
import numpy as np, scipy.sparse as sp, scipy.special, sympy, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, recall_score, precision_score


def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d + 1):
        f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) / (scipy.special.beta(i + 1, d + 1 - i)))
        coeff = f.all_coeffs()
        thetas.append([float(coeff[d - j]) for j in range(d + 1)])
    return thetas


class Graph:
    """Sparse 'sum over incoming edges' operator with D^-1/2 normalisation, from a (src,dst)-indexed matrix."""
    def __init__(self, A, device):
        A = A.tocoo()
        indeg = np.asarray(A.sum(0)).ravel()            # in-degree = column sums (edge u->v at row u, col v)
        self.dinv = torch.tensor(np.power(np.clip(indeg, 1, None), -0.5), dtype=torch.float32, device=device).unsqueeze(-1)
        AT = A.T.tocoo()                                 # out[v] = sum_u A[u,v] h[u]
        self.M = torch.sparse_coo_tensor(np.vstack([AT.row, AT.col]), AT.data.astype(np.float32), AT.shape).coalesce().to(device)

    def lap(self, h):
        return h - torch.sparse.mm(self.M, h * self.dinv) * self.dinv


class PolyConv(nn.Module):
    def __init__(self, theta):
        super().__init__(); self.theta = theta

    def forward(self, g, feat):
        h = self.theta[0] * feat
        for k in range(1, len(self.theta)):
            feat = g.lap(feat)
            h = h + self.theta[k] * feat
        return h


class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2):
        super().__init__()
        self.g = graph
        self.conv = [PolyConv(t) for t in calculate_theta2(d)]
        self.linear = nn.Linear(in_feats, h_feats); self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats); self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.linear2(self.act(self.linear(x))))
        h = torch.cat([c(self.g, h) for c in self.conv], -1)
        return self.linear4(self.act(self.linear3(h)))


class BWGNN_Hetero(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graphs, d=2):
        super().__init__()
        self.gs = graphs
        self.conv = [PolyConv(t) for t in calculate_theta2(d)]
        self.linear = nn.Linear(in_feats, h_feats); self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats); self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        h = self.act(self.linear2(self.act(self.linear(x))))
        h_all = []
        for g in self.gs:
            h = self.linear3(torch.cat([c(g, h) for c in self.conv], -1))   # as released: h is overwritten
            h_all.append(h)
        return self.linear4(self.act(torch.stack(h_all).sum(0)))


def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = (probs[:, 1] > thres).astype(int)
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1: best_f1, best_thre = mf1, thres
    return best_f1, best_thre


def train(model, X, y, tr, va, te, epochs=100, device="cpu"):
    x = torch.tensor(X, dtype=torch.float32, device=device); yt = torch.tensor(y, device=device)
    trt = torch.tensor(tr, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    weight = (1 - y[tr]).sum() / y[tr].sum()
    best_f1, final = 0., None
    for e in range(epochs):
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits[trt], yt[trt], weight=torch.tensor([1., weight], device=device, dtype=torch.float32))
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        probs = logits.softmax(1).detach().cpu().numpy()
        f1, thres = get_best_f1(y[va], probs[va])
        if best_f1 < f1:
            best_f1 = f1
            pred = (probs[te, 1] > thres).astype(int)
            r = recall_score(y[te], pred, average=None)
            final = dict(auc=roc_auc_score(y[te], probs[te, 1]), ap=average_precision_score(y[te], probs[te, 1]),
                         f1=f1_score(y[te], pred, average='macro'), gmean=float(np.sqrt(r[0] * r[1])),
                         rec=recall_score(y[te], pred), prec=precision_score(y[te], pred, zero_division=0), epoch=e)
    return final


def build(rel_mats, homo, device):
    """rel_mats: list of scipy relation matrices. Homo graph = sum of relations (duplicates kept) + self loops."""
    rels = []
    for R in rel_mats:
        R = R.tocoo(); m = R.row != R.col
        rels.append(sp.csr_matrix((np.ones(m.sum(), np.float32), (R.row[m], R.col[m])), shape=R.shape))
    if homo:
        N = rels[0].shape[0]
        A = sum(rels) + sp.eye(N, dtype=np.float32, format="csr")
        return Graph(A, device)
    return [Graph(R, device) for R in rels]
