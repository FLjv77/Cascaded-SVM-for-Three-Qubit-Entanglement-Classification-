"""Predict exclusive entanglement classes from named Pauli features."""
import argparse
import pandas as pd
from joblib import load
from train import cascade_predict

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",required=True,help="Trusted locally generated joblib bundle")
    p.add_argument("--input",required=True)
    p.add_argument("--output",default="predictions.csv")
    a=p.parse_args()
    bundle=load(a.model)
    x=pd.read_csv(a.input)
    x["predicted_class"]=cascade_predict(bundle["models"],x,bundle["features"])
    x.to_csv(a.output,index=False)
