"""Paired comparison: only B positive training/validation examples differ."""
import argparse
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.metrics import accuracy_score
from train import FEATURES, FILES, CLASSES, load_data, select, metrics, cascade_predict, cascade_set, cascade_trace

def run(data,out,seed):
    data,out=Path(data),Path(out)
    out.mkdir(parents=True,exist_ok=False)
    splits={v:load_data(data/v,seed) for v in ("reference","mixed_b")}
    for name in FILES:
        for a,b in zip(splits["reference"][name],splits["mixed_b"][name]):
            if not a.index.equals(b.index) or not a.label.equals(b.label):
                raise ValueError("Paired split mismatch")
            if name!="B": pd.testing.assert_frame_equal(a,b)
    split_rows=[]
    for name,parts in splits["reference"].items():
        for partition,d in zip(("train","validation","test"),parts):
            split_rows.extend(dict(model=name,row_index=int(i),partition=partition) for i in d.index)
    pd.DataFrame(split_rows).to_csv(out/"split_indices.csv",index=False)
    shared={}; params={}; validation={}
    for name in ("GHZ","W"):
        tr,va,_=splits["reference"][name]
        print("Training shared",name,flush=True)
        start=time.monotonic()
        auc,p,m=select(tr,va,FEATURES,"rbf")
        shared[name]=m; params[name]=p
        validation[name]=dict(auc=float(auc),accuracy=float(accuracy_score(va.label,m.predict(va[FEATURES]))),seconds=time.monotonic()-start)
    reports={}
    rows=[]
    for variant in ("reference","mixed_b"):
        tr,va,_=splits[variant]["B"]
        print("Training",variant,"B",flush=True)
        start=time.monotonic()
        auc,p,m=select(tr,va,FEATURES,"rbf")
        models={**shared,"B":m}; chosen={**params,"B":p}
        report=dict(parameters=chosen,validation={**validation,"B":dict(auc=float(auc),accuracy=float(accuracy_score(va.label,m.predict(va[FEATURES]))),seconds=time.monotonic()-start)},test={})
        dump(dict(models=models,features=FEATURES,parameters=chosen,classes=CLASSES),out/(variant+".joblib"))
        for test_variant in ("reference","mixed_b"):
            tests=splits[test_variant]
            scores={}
            for name in FILES:
                te=tests[name][2]
                scores[name]=metrics(te.label,models[name].predict(te[FEATURES]),[0,1],models[name].decision_function(te[FEATURES]))
            test=cascade_set(tests,2)
            pred=cascade_predict(models,test,FEATURES)
            scores["cascade"]=metrics(test.entanglement,pred,CLASSES)
            pd.DataFrame(cascade_trace(models,test,FEATURES,test.entanglement)).to_csv(out/f"trace_{variant}_on_{test_variant}.csv",index=False)
            report["test"][test_variant]=scores
            for name,sc in scores.items():
                rows.append(dict(training=variant,test=test_variant,model=name,accuracy=sc["accuracy"],roc_auc=sc.get("roc_auc"),n=int(np.sum(sc["confusion_matrix"]))))
        reports[variant]=report
        (out/"metrics.json").write_text(json.dumps(reports,indent=2))
    df=pd.DataFrame(rows); df.to_csv(out/"comparison.csv",index=False)
    lines=["# Fixed-data comparison: 63 features",f"Split seed: {seed}. RBF, validation-AUC selection. No OOD, noise or feature selection.","","| Common test | Model | Reference training | Mixed-B training | Change (pp) |","|---|---|---:|---:|---:|"]
    for test_variant in ("reference","mixed_b"):
        for name in (*FILES,"cascade"):
            a=reports["reference"]["test"][test_variant][name]["accuracy"]
            b=reports["mixed_b"]["test"][test_variant][name]["accuracy"]
            lines.append(f"| {test_variant} | {name} | {a:.2%} | {b:.2%} | {100*(b-a):+.2f} |")
    lines.extend(["","GHZ and W training data/models are identical across variants. Only positive B samples change. Compare models on the SAME test column; diagonal results alone confound training and test distribution. Single seed, no uncertainty estimate."])
    (out/"summary.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--output",required=True); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); run(a.data,a.output,a.seed)
