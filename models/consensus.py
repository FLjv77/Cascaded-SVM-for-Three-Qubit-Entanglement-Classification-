"""Rank-aligned Feature Consensus using validation accuracy increments."""
import json
import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score
from train import FILES, FEATURES, CLASSES, fit, cascade_set, cascade_predict, metrics

def consensus_order(ranks,gains):
    selected=[]; reserve=[]
    for name in FILES:
        f=ranks[name][0]
        if f not in selected: selected.append(f)
    for i in range(1,len(FEATURES)):
        candidates=sorted(((gains[name][i],ranks[name][i]) for name in FILES),
                          key=lambda p:-p[0])
        top=candidates[0][1]
        reserve.extend(f for _,f in candidates[1:])
        if top in selected:
            top=next((f for f in reserve if f not in selected),None)
        if top is not None and top not in selected: selected.append(top)
    for f in reserve+FEATURES:
        if f not in selected: selected.append(f)
    return selected

def run_consensus(splits,params,rankings,args,out):
    ranks={name:[f for f,_ in rankings[name]] for name in FILES}
    gains={}; curves=[]
    for name,(tr,va,_) in splits.items():
        previous=0; gains[name]=[]
        for k in range(1,64):
            f=ranks[name][:k]
            m=fit(tr,f,params[name])
            score=float(accuracy_score(va.label,m.predict(va[f])))
            gains[name].append(score-previous); previous=score
            curves.append(dict(model=name,features=k,validation_accuracy=score))
    order=consensus_order(ranks,gains)
    pd.DataFrame(curves).to_csv(out/"individual_ablation.csv",index=False)
    (out/"consensus_order.json").write_text(json.dumps(order,indent=2))
    val=cascade_set(splits,1)
    test=cascade_set(splits,2)
    results=[]; reports={}
    for k in sorted(set(args.budgets)):
        if not 1<=k<=63: raise ValueError("budgets must be in 1..63")
        f=order[:k]; models={}
        row={"features":k}
        for name,(tr,va,te) in splits.items():
            m=fit(tr,f,params[name]); models[name]=m
            row[name+"_validation"]=float(accuracy_score(va.label,m.predict(va[f])))
            row[name+"_test"]=float(accuracy_score(te.label,m.predict(te[f])))
        row["cascade_validation"]=float(accuracy_score(val.entanglement,cascade_predict(models,val,f)))
        pred=cascade_predict(models,test,f)
        row["cascade_test"]=float(accuracy_score(test.entanglement,pred))
        reports[str(k)]=metrics(test.entanglement,pred,CLASSES)
        results.append(row)
        dump(dict(models=models,features=f,parameters=params,classes=CLASSES),out/f"consensus_{k}.joblib")
    pd.DataFrame(results).to_csv(out/"consensus_ablation.csv",index=False)
    (out/"consensus_metrics.json").write_text(json.dumps(reports,indent=2))
