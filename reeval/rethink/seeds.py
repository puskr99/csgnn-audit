import numpy as np, sys, json
import sign, gbdt
from common import load
out={}
for ds in ['sichuan','bupt']:
    data=load(ds)
    for name,mod,argv in [('MLP (no graph)',sign,['--ops','none','--logx','1']),
                          ('Mean-agg GNN (1-hop, decoupled)',sign,['--ops','sym1','--logx','1']),
                          ('GBDT (no graph)',gbdt,['--opts','none'])]:
        rs=[]
        for sd in range(5):
            a=mod.parser().parse_args(['--dataset',ds,'--seed',str(sd)]+argv)
            r=mod.run(a,data=data,verbose=False); rs.append(r)
        for k in ['test_raw','test_cal']:
            M={m:(np.mean([r[k][m] for r in rs]),np.std([r[k][m] for r in rs])) for m in rs[0][k]}
            out[f'{ds}|{name}|{k}']=M
            print(ds,name,k,' '.join(f'{m}={v[0]:.4f}±{v[1]:.4f}' for m,v in M.items()),flush=True)
json.dump(out,open('seeds5.json','w'),indent=1)
