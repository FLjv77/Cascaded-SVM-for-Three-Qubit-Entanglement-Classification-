"""Exploratory rotated-state sweeps and batch stability."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/x) for x in ("dataset_generation","models")]
import generate as g
from train import FEATURES,load_data,fit,rank_features

def run(bundle,data,seed,out,n):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed); rows=[]
    m=bundle["models"]["GHZ"]; f=bundle["features"]
    if isinstance(f,dict): f=f["GHZ"]
    for family in ("rotated_GHZ","GHZ_white_noise","GHZ_W"):
        for e in ([0] if family=="rotated_GHZ" else np.linspace(0,1,101)):
            rho=(1-e)*g.GHZ+e*(g.W if family=="GHZ_W" else np.eye(8)/8)
            x=pd.DataFrame([g.features(g.rotate(rng,rho)) for _ in range(n)],columns=g.LABELS)
            rows.append(dict(family=family,noise_or_W_weight=e,positive_rate=float(m.predict(x[f]).mean())))
    pd.DataFrame(rows).to_csv(out/"state_sweeps.csv",index=False)
    rows=[]; ranks=[]
    for name,(tr,va,te) in load_data(data,seed).items():
        # Use training partition only; keep global validation and test untouched.
        batches=[np.array_split(tr.loc[tr.label==label].sample(frac=1,random_state=seed),10) for label in (0,1)]
        for i in range(10):
            d=pd.concat([batches[0][i],batches[1][i]])
            if d.label.value_counts().min()<4: continue
            a,b=train_test_split(d,test_size=.3,stratify=d.label,random_state=seed+i)
            model=fit(a,FEATURES,bundle["parameters"][name])
            rows.append(dict(model=name,batch=i,accuracy=accuracy_score(b.label,model.predict(b[FEATURES]))))
            ranks.extend(dict(model=name,batch=i,feature=k,importance=v) for k,v in rank_features(model,b,seed+i,10,2000))
    pd.DataFrame(rows,columns=["model","batch","accuracy"]).to_csv(out/"batch_accuracy.csv",index=False)
    pd.DataFrame(ranks,columns=["model","batch","feature","importance"]).to_csv(out/"batch_importance.csv",index=False)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model",required=True); p.add_argument("--data",default="datasets"); p.add_argument("--seed",type=int,default=42); p.add_argument("--output",default="experiment_outputs"); p.add_argument("--rotations",type=int,default=20)
    a=p.parse_args(); run(load(a.model),a.data,a.seed,a.output,a.rotations)
