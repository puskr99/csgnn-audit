"""Revision experiment C: published methods (PC-GNN port, label-free CSGNN port) with the recipe's input representation
(signed log1p + log in/out degree, standardised on the training split, clipped), against their original inputs, on the
same five splits and seeds. PC-GNN additionally gets the validation-calibrated decision (fit_bias), as in the paper.
Usage: python rev_logpub.py --method pcgnn|csgnn --dataset sichuan|bupt"""
import argparse, inspect, json, os, time, types, numpy as np, torch
from common import load, metrics, fit_bias
import csboost_v5 as V

p = argparse.ArgumentParser()
p.add_argument("--method", required=True); p.add_argument("--dataset", required=True)
p.add_argument("--splits", type=int, nargs="*", default=[0, 1, 2, 3, 4]); p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
p.add_argument("--feats", default="orig,recipe")
p.add_argument("--out", default="results_rev/logpub.jsonl")
A = p.parse_args(); os.makedirs("results_rev", exist_ok=True); ds = A.dataset
done = {(r["dataset"], r["method"], r["feat"], r["split"], r["seed"]) for r in map(json.loads, open(A.out))} if os.path.exists(A.out) else set()


def recipe_X(k):
    X, y, src, dst, tr, va, te = load(ds, k)
    return V.base_features(X.astype(np.float32), src, dst, tr).astype(np.float64)


def write(rec):
    with open(A.out, "a") as fh: fh.write(json.dumps(rec) + "\n")


if A.method == "pcgnn":
    import pcgnn_port as PC
    src = inspect.getsource(PC.run).replace("pt = model.predict(te)", "pt = model.predict(te); PC_VAL[0] = model.predict(va)")
    PC.PC_VAL = [None]; exec(src, PC.__dict__)
    orig_pre = PC.preprocess
    for k in A.splits:
        RX = recipe_X(k)
        for f in A.feats.split(","):
            PC.preprocess = (lambda X, mode, tr=None: RX) if f == "recipe" else orig_pre
            for s in A.seeds:
                if (ds, "pcgnn", f, k, s) in done: continue
                t0 = time.time()
                r, pt, te = PC.run(ds, k, s, **PC.HP[ds], feat="zscore_train")
                X, y, _, _, tr, va, _ = load(ds, k); pv = PC.PC_VAL[0]
                b = fit_bias(y[va], np.log(pv + 1e-12))[0]
                cal = metrics(y[te], pt, (np.log(pt + 1e-12) + b).argmax(1))
                write(dict(dataset=ds, method="pcgnn", feat=f, split=k, seed=s, test=r["test"], test_cal=cal,
                           val=metrics(y[va], pv), best_epoch=r["best_epoch"], time=time.time() - t0))
                print(f"[pcgnn {ds} {f} split{k} s{s}] g={r['test']['gmean']:.4f} cal_g={cal['gmean']:.4f} auc={r['test']['auc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
else:
    import csgnn_port as CS
    orig_bg = CS.build_graph

    def bg_recipe(d, split=0):
        Xs, y, s_, d_, tr, va, te = orig_bg(d, split)
        return recipe_X(split).astype(np.float32), y, s_, d_, tr, va, te
    for k in A.splits:
        for f in A.feats.split(","):
            CS.build_graph = bg_recipe if f == "recipe" else orig_bg
            for s in A.seeds:
                if (ds, "csgnn", f, k, s) in done: continue
                t0 = time.time()
                a = types.SimpleNamespace(dataset=ds, max_epoch=200, seed=s, split=k, es=1)
                import contextlib, io
                with contextlib.redirect_stdout(io.StringIO()):
                    lg, Cm, y, tr, va, te, cfg, _ = CS.train_model(a)
                M = Cm.detach().numpy(); honest = torch.softmax(lg.double(), 1).numpy()
                fix_pred = torch.softmax(CS.cost_logits(lg, lg.argmax(1).numpy(), M), 1).numpy()
                z = (honest @ M) * torch.exp(lg.double()).numpy(); z = z / z.sum(1, keepdims=True)
                fix_exp = torch.softmax(torch.tensor(z), 1).numpy()
                V_ = {n: metrics(y[va], P[va]) for n, P in [("honest", honest), ("fix_pred", fix_pred), ("fix_exp", fix_exp)]}
                T_ = {n: metrics(y[te], P[te]) for n, P in [("honest", honest), ("fix_pred", fix_pred), ("fix_exp", fix_exp)]}
                best = max(V_, key=lambda n: V_[n]["gmean"] + V_[n]["auc"])
                write(dict(dataset=ds, method="csgnn", feat=f, split=k, seed=s, variant=best, val=V_[best], test=T_[best],
                           val_variants=V_, test_variants=T_, time=time.time() - t0))
                print(f"[csgnn {ds} {f} split{k} s{s}] {best} g={T_[best]['gmean']:.4f} auc={T_[best]['auc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
print("finished")
