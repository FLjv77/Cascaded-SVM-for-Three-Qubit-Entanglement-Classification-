"""Train and evaluate the manuscript's three nested SVM classifiers."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.inspection import permutation_importance

FILES = {"GHZ":"ghz_vs_other.csv","W":"w_vs_other.csv","B":"bi_vs_other.csv"}
FEATURES = sorted("".join(p) for p in itertools.product("IXYZ", repeat=3) if p != ("I","I","I"))
CLASSES = ["S","B","W","GHZ"]

def load_data(root, seed):
    result = {}
    for name, filename in FILES.items():
        d = pd.read_csv(Path(root)/filename)
        if set(d.columns) != set(FEATURES+["label"]) or len(d.columns)!=64:
            raise ValueError(f"{filename}: expected 63 named Pauli features and label")
        x = d[FEATURES]
        y = d.label
        if set(y.unique()) != {0,1} or not np.isfinite(x.to_numpy()).all() or (x.abs()>1+1e-8).any().any():
            raise ValueError(f"{filename}: invalid labels or features")
        a,b = train_test_split(d, test_size=.30, stratify=y, random_state=seed)
        v,t = train_test_split(b, test_size=.5, stratify=b.label, random_state=seed)
        result[name] = (a,v,t)
    return result

def fit(d, features, params):
    return SVC(cache_size=512, **params).fit(d[features],d.label)

def select(train,val,features,kernel,degree=None):
    candidates = [{"kernel":kernel,"C":c,"gamma":g,**({"degree":d,"coef0":1} if kernel=="poly" else {})}
                  for c in (.1,1,10) for g in ("scale",.01,.1,1)
                  for d in (([degree] if degree is not None else range(2,10)) if kernel=="poly" else [0])]
    best = None
    for params in candidates:
        m = fit(train,features,params)
        score = roc_auc_score(val.label,m.decision_function(val[features]))
        if best is None or score>best[0]:
            best=(score,params,m)
    return best

def metrics(y,p,labels,scores=None):
    result = dict(accuracy=float(accuracy_score(y,p)), confusion_matrix=confusion_matrix(y,p,labels=labels).tolist(),
                  classification_report=classification_report(y,p,labels=labels,output_dict=True,zero_division=0))
    if scores is not None:
        result["roc_auc"]=float(roc_auc_score(y,scores))
    return result

def cascade_predict(models,x,features):
    prediction=np.full(len(x),"S",dtype="<U3")
    active=np.arange(len(x))
    for name in ("GHZ","W","B"):
        if not len(active): break
        f=features[name] if isinstance(features,dict) else features
        hit=models[name].predict(x.iloc[active][f])==1
        prediction[active[hit]]=name
        active=active[~hit]
    return prediction

def cascade_trace(models,x,features,truth):
    truth=np.asarray(truth)
    active=np.ones(len(x),dtype=bool)
    rows=[]
    for stage in ("GHZ","W","B"):
        f=features[stage] if isinstance(features,dict) else features
        hit=np.zeros(len(x),dtype=bool)
        if active.any(): hit[active]=models[stage].predict(x.loc[active,f])==1
        for c in CLASSES:
            mask=truth==c
            n=int((mask&active).sum()); accepted=int((mask&hit).sum())
            rows.append(dict(stage=stage,true_class=c,entered=n,accepted=accepted,
                             rejected=n-accepted,errors=(n-accepted if c==stage else accepted),
                             upstream_excluded=int(mask.sum())-n))
        active &= ~hit
    return rows

def cascade_set(splits,index):
    # Positive samples have unambiguous exclusive classes; B negatives are S.
    frames=[]
    for name in FILES:
        d=splits[name][index]
        frames.append(d.loc[d.label==1,FEATURES].assign(entanglement=name))
    d=splits["B"][index]
    frames.append(d.loc[d.label==0,FEATURES].assign(entanglement="S"))
    return pd.concat(frames,ignore_index=True)

def rank_features(model,val,seed,repeats,limit):
    if len(val)>limit:
        val=val.sample(limit,random_state=seed)
    imp=permutation_importance(model,val[FEATURES],val.label,n_repeats=repeats,
                               random_state=seed,scoring="accuracy",n_jobs=1)
    return sorted(zip(FEATURES,imp.importances_mean),key=lambda p:(-p[1],p[0]))

def train_all(args):
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    splits=load_data(args.data,args.seed)
    models={}; params={}; rankings={}; report={}
    for name,(tr,va,te) in splits.items():
        score,p,m=select(tr,va,FEATURES,args.kernel)
        params[name]=p; models[name]=m
        rankings[name]=rank_features(m,va,args.seed,args.repeats,args.importance_limit)
        report[name]=dict(parameters=p,validation_auc=float(score),validation_accuracy=float(accuracy_score(va.label,m.predict(va[FEATURES]))),
            test=metrics(te.label,m.predict(te[FEATURES]),[0,1],m.decision_function(te[FEATURES])))
    test=cascade_set(splits,2)
    report["cascade"]=metrics(test.entanglement,cascade_predict(models,test,FEATURES),CLASSES)
    pd.DataFrame(cascade_trace(models,test,FEATURES,test.entanglement)).to_csv(out/"cascade_trace.csv",index=False)
    bundle=dict(models=models,features=FEATURES,parameters=params,classes=CLASSES)
    dump(bundle,out/"full_models.joblib")
    (out/"metrics.json").write_text(json.dumps(report,indent=2))
    (out/"rankings.json").write_text(json.dumps({k:[[f,float(s)] for f,s in v] for k,v in rankings.items()},indent=2))
    if args.consensus:
        from consensus import run_consensus
        run_consensus(splits,params,rankings,args,out)
    print(json.dumps({k:v["test"]["accuracy"] if k in FILES else v["accuracy"] for k,v in report.items()},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",default="datasets")
    p.add_argument("--output",default="model_outputs")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--kernel",choices=["rbf","linear","poly"],default="rbf")
    p.add_argument("--repeats",type=int,default=10)
    p.add_argument("--importance-limit",type=int,default=2000)
    p.add_argument("--consensus",action="store_true")
    p.add_argument("--budgets",type=int,nargs="+",default=[3,10,20,30,40,50,63])
    train_all(p.parse_args())
