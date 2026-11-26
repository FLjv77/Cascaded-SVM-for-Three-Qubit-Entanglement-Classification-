"""Render paper figure filenames from measured results only."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def render(models,experiments,out):
    models,experiments,out=map(Path,(models,experiments,out)); out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.tight_layout(); fig.savefig(out/name,dpi=200); plt.close(fig)
    suffix={"GHZ":"ghz","W":"w","B":"s","cascade":"holeModel"}
    if (models/"metrics.json").exists():
        report=json.loads((models/"metrics.json").read_text())
        for name,s in suffix.items():
            a=np.array(report[name]["confusion_matrix"] if name=="cascade" else report[name]["test"]["confusion_matrix"])
            labels=["S","B","W","GHZ"] if name=="cascade" else ["0","1"]
            fig,ax=plt.subplots(); im=ax.imshow(a)
            for i,j in np.ndindex(a.shape): ax.text(j,i,str(a[i,j]),ha="center",va="center",color="white" if a[i,j]<a.max()/2 else "black")
            ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,xlabel="Predicted",ylabel="True")
            fig.colorbar(im,ax=ax); save(fig,"CM_"+s+".png")
    if (experiments/"roc.csv").exists():
        d=pd.read_csv(experiments/"roc.csv"); metrics=pd.read_csv(experiments/"kernel_metrics.csv")
        for name in ("GHZ","W","B"):
            fig,ax=plt.subplots()
            for k,g in d[d.model==name].groupby("kernel"):
                auc=metrics.loc[(metrics.model==name)&(metrics.kernel==k),"test_auc"].iloc[0]
                ax.plot(g.fpr,g.tpr,label=f"{k} ({auc:.3f})")
            ax.plot([0,1],[0,1],"--"); ax.set(xlabel="False positive rate",ylabel="True positive rate"); ax.legend(fontsize=7); save(fig,"ROC_"+suffix[name]+".png")
    if (experiments/"noise.csv").exists():
        d=pd.read_csv(experiments/"noise.csv"); fig,ax=plt.subplots()
        for k,g in d.groupby("channel"): ax.plot(g.strength,g.original_label_agreement,label=k)
        ax.set(xlabel="Noise probability",ylabel="Agreement with original class",ylim=(0,1.02)); ax.legend(); save(fig,"accuracy_noisy.png")
    if (models/"individual_ablation.csv").exists():
        d=pd.read_csv(models/"individual_ablation.csv"); fig,ax=plt.subplots()
        for k,g in d.groupby("model"): ax.plot(g.features,g.validation_accuracy,label=k+" validation")
        if (models/"consensus_ablation.csv").exists():
            c=pd.read_csv(models/"consensus_ablation.csv"); ax.plot(c.features,c.cascade_validation,label="Cascade validation")
        ax.set(xlabel="Number of features",ylabel="Validation accuracy"); ax.legend(); save(fig,"accuracy_per_features.png")
        fig,ax=plt.subplots(figsize=(14,4))
        for i,k in enumerate(("GHZ","W","B")):
            g=d[d.model==k].sort_values("features")
            ax.bar(g.features.iloc[1:]+(i-1)*.25,g.validation_accuracy.diff().iloc[1:],width=.25,label=k)
        ax.set(xlabel="Number of features",ylabel="Change in validation accuracy"); ax.legend(); save(fig,"fluctuations_in_accuracy.png")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--models",default="model_outputs"); p.add_argument("--experiments",default="experiment_outputs"); p.add_argument("--output",default="figures"); a=p.parse_args(); render(a.models,a.experiments,a.output)
