"""Compare RBF and each polynomial degree 2–9 on fixed splits."""
import argparse
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"models"))
from train import FEATURES,load_data,fit

def run(data,seed,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    rows=[]; curves=[]
    for name,(tr,va,te) in load_data(data,seed).items():
        for degree in [None,*range(2,10)]:
            kernel="rbf" if degree is None else "poly"
            tag="RBF" if degree is None else f"Polynomial {degree}"
            best=None
            for c in (.1,1,10):
                for gamma in ("scale",.01,.1,1):
                    params=dict(kernel=kernel,C=c,gamma=gamma)
                    if degree is not None: params.update(degree=degree,coef0=1)
                    m=fit(tr,FEATURES,params)
                    auc=roc_auc_score(va.label,m.decision_function(va[FEATURES]))
                    if best is None or auc>best[0]: best=(auc,m,params)
            auc,m,params=best
            scores=m.decision_function(te[FEATURES])
            fpr,tpr,_=roc_curve(te.label,scores)
            rows.append(dict(model=name,kernel=tag,validation_auc=auc,test_auc=roc_auc_score(te.label,scores),test_accuracy=accuracy_score(te.label,m.predict(te[FEATURES])),**{k:v for k,v in params.items() if k!="kernel"}))
            curves.extend(dict(model=name,kernel=tag,fpr=x,tpr=y) for x,y in zip(fpr,tpr))
    pd.DataFrame(rows).to_csv(out/"kernel_metrics.csv",index=False)
    pd.DataFrame(curves).to_csv(out/"roc.csv",index=False)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--data",default="datasets"); p.add_argument("--seed",type=int,default=42); p.add_argument("--output",default="experiment_outputs")
    a=p.parse_args(); run(a.data,a.seed,a.output)
