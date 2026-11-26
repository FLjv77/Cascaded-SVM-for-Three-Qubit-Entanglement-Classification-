"""Local PennyLane-convention channels; report retention of original labels."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/x) for x in ("dataset_generation","models")]
import generate as g
from train import cascade_predict

def kraus(channel,p):
    I=np.eye(2)
    if channel=="amplitude": return [np.diag([1,np.sqrt(1-p)]),np.array([[0,np.sqrt(p)],[0,0]])]
    if channel=="phase": return [np.diag([1,np.sqrt(1-p)]),np.diag([0,np.sqrt(p)])]
    if channel=="depolarizing": return [np.sqrt(1-p)*I]+[np.sqrt(p/3)*g.PAULI[k] for k in "XYZ"]
    raise ValueError(channel)

def noisy(rho,channel,p):
    if not 0<=p<=1: raise ValueError("invalid probability")
    for q in range(3):
        ks=[g.kron3(*(k if j==q else np.eye(2) for j in range(3))) for k in kraus(channel,p)]
        rho=sum(k@rho@k.conj().T for k in ks)
    return rho

def run(bundle,n,seed,output):
    rng=np.random.default_rng(seed)
    truth=np.repeat(["S","B","W","GHZ"],n)
    states=[g.sample(rng, "B" if c=="S" else c, int(c!="S")) for c in truth]
    rows=[]
    for channel in ("amplitude","phase","depolarizing"):
        for p in np.arange(51)/100:
            x=pd.DataFrame([g.features(noisy(r,channel,p)) for r in states],columns=g.LABELS)
            pred=cascade_predict(bundle["models"],x,bundle["features"])
            row=dict(channel=channel,strength=p,original_label_agreement=float(np.mean(pred==truth)),n=len(truth))
            for c in ("S","B","W","GHZ"):
                row[c+"_agreement"]=float(np.mean(pred[truth==c]==c))
            rows.append(row)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output,index=False)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",required=True); p.add_argument("--samples-per-class",type=int,default=100)
    p.add_argument("--seed",type=int,default=3027); p.add_argument("--output",default="experiment_outputs/noise.csv")
    a=p.parse_args()
    if a.samples_per_class<1: p.error("positive sample count required")
    run(load(a.model),a.samples_per_class,a.seed,a.output)
