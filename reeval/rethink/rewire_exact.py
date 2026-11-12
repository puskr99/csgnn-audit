"""Exact degree- and reciprocity-preserving rewiring by double-edge swaps (Maslov & Sneppen 2002; mutual edges kept
as in Milo et al. 2002).

The edge list is split into mutual pairs (u->v and v->u both present) and one-way edges.
  * one-way edges: directed swaps (a->b, c->d) -> (a->d, c->b)
  * mutual pairs:  undirected swaps {a,b},{c,d} -> {a,d},{c,b} (random orientation of each pair)
A swap is rejected if it would create a self-loop, a duplicate edge, or a new mutual pair (and, for mutual pairs, if
any edge already joins the new endpoints). Every node therefore keeps exactly
  - its out-degree and in-degree,
  - its number of mutual neighbours,
  - hence its degree in the symmetrised graph that the GNNs aggregate over (binarised or not).
Sichuan is stored once per undirected pair (no mutual edges), so for it this is an undirected degree-preserving swap
that also keeps the stored orientation counts used as degree features."""
import numpy as np, time


def _swap_loop(s, d, present, rng, attempts, mutual):
    E = len(s); acc = 0; done = 0
    if E < 2: return 0
    while done < attempts:
        B = min(1_000_000, attempts - done)
        I = rng.integers(0, E, B).tolist(); J = rng.integers(0, E, B).tolist()
        F = rng.integers(0, 2, B).tolist() if mutual else None
        for t in range(B):
            i, j = I[t], J[t]
            if i == j: continue
            a, b, c, dd = s[i], d[i], s[j], d[j]
            if mutual and F[t]: c, dd = dd, c                    # undirected pair: random orientation of the second
            if a == dd or c == b or a == c or b == dd: continue
            n1, n2 = (a, dd), (c, b)
            if n1 in present or n2 in present or (dd, a) in present or (b, c) in present: continue
            if mutual:
                for e in [(a, b), (b, a), (c, dd), (dd, c)]: present.discard(e)
                for e in [(a, dd), (dd, a), (c, b), (b, c)]: present.add(e)
                s[i], d[i], s[j], d[j] = a, dd, c, b
            else:
                present.discard((a, b)); present.discard((c, dd)); present.add(n1); present.add(n2)
                d[i], d[j] = dd, b
            acc += 1
        done += B
    return acc


def rewire_exact(src, dst, n, seed, swaps_per_edge=10):
    rng = np.random.default_rng(20_000 + seed)
    src = np.asarray(src, np.int64); dst = np.asarray(dst, np.int64)
    present = set(zip(src.tolist(), dst.tolist()))
    mut = np.array([(b, a) in present for a, b in zip(src.tolist(), dst.tolist())], bool)
    one_s, one_d = src[~mut].tolist(), dst[~mut].tolist()
    keep = mut & (src < dst)                                   # one representative per mutual pair
    mu_s, mu_d = src[keep].tolist(), dst[keep].tolist()
    acc = _swap_loop(one_s, one_d, present, rng, swaps_per_edge * len(one_s), False)
    acc += _swap_loop(mu_s, mu_d, present, rng, swaps_per_edge * len(mu_s), True)
    s2 = np.array(one_s + mu_s + mu_d, np.int64); d2 = np.array(one_d + mu_d + mu_s, np.int64)
    rate = acc / max(1, swaps_per_edge * (len(one_s) + len(mu_s)))
    return s2, d2, rate


def check(src, dst, s2, d2, n):
    """Exactness checks: directed degrees, mutual degrees, symmetrised degrees; returns share of original edges left."""
    def stats(s, d):
        P = set(zip(s.tolist(), d.tolist()))
        m = np.array([(b, a) in P for a, b in zip(s.tolist(), d.tolist())], bool)
        und = set(zip(np.minimum(s, d).tolist(), np.maximum(s, d).tolist()))
        u = np.array(list(und)).T
        return (np.bincount(s, minlength=n), np.bincount(d, minlength=n), np.bincount(s[m], minlength=n),
                np.bincount(np.r_[u[0], u[1]], minlength=n), len(P), (s == d).sum())
    A, B = stats(src, dst), stats(s2, d2)
    ok = all((x == y).all() for x, y in zip(A[:4], B[:4])) and A[4] == B[4] == len(s2) and B[5] == 0
    left = len(set(zip(src.tolist(), dst.tolist())) & set(zip(s2.tolist(), d2.tolist()))) / len(src)
    return ok, left


if __name__ == "__main__":
    from common import load
    for ds in ["sichuan", "bupt"]:
        X, y, s, d, *_ = load(ds, 0) if load.__code__.co_argcount > 1 else load(ds); n = len(y)
        t = time.time(); s2, d2, rate = rewire_exact(s, d, n, 0)
        ok, left = check(s, d, s2, d2, n)
        do, di = np.bincount(s, minlength=n), np.bincount(d, minlength=n)
        print(ds, f"{time.time() - t:.0f}s accepted={rate:.2f} edges={len(s2)} (orig {len(s)}) exact={ok} "
                  f"original_edges_left={left:.3f} (configuration-model expectation {np.sum(do[s] * di[d]) / len(s) ** 2:.3f})")
