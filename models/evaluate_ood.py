"""Evaluate certified binary labels only; never tune on OOD."""
import argparse
import json
from pathlib import Path
import pandas as pd
from joblib import load
from train import cascade_predict

def evaluate(model,data,output):
    bundle=load(model)
    features=bundle["features"]
    reports={}
    for family in ("horodecki","edge","upb","x"):
        d=pd.read_csv(Path(data)/(family+".csv"))
        result={}
        for name,m in bundle["models"].items():
            f=features[name] if isinstance(features,dict) else features
            y=d["label_"+name]
            keep=y.notna()
            pred=m.predict(d[f])
            result[name]={"n_total":len(d),"n_certified":int(keep.sum()),
                "n_positive":int((y==1).sum()),"n_negative":int((y==0).sum()),
                "accuracy":float((pred[keep]==y[keep]).mean()) if keep.any() else None}
        keep=d[["label_B","label_W","label_GHZ"]].notna().all(axis=1)
        if keep.any():
            truth=d.loc[keep].apply(lambda x: "GHZ" if x.label_GHZ else "W" if x.label_W else "B" if x.label_B else "S",axis=1)
            pred=cascade_predict(bundle["models"],d.loc[keep],features)
            result["cascade"]={"n_certified":int(keep.sum()),"accuracy":float((pred==truth).mean())}
        reports[family]=result
    p=Path(output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",required=True)
    p.add_argument("--data",default="ood_datasets")
    p.add_argument("--output",default="ood_results/metrics.json")
    a=p.parse_args(); evaluate(a.model,a.data,a.output)
