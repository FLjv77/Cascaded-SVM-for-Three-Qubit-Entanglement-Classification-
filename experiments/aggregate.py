"""Aggregate completed independent seeds without pooling test observations."""
import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

def stats(values):
    values=[float(v) for v in values if v is not None and math.isfinite(float(v))]
    return dict(n_runs=len(values),mean=statistics.mean(values) if values else None,
                sample_std=statistics.stdev(values) if len(values)>1 else None)

def read_csv(path):
    with path.open(newline="") as f: return list(csv.DictReader(f))

def write_csv(path,rows):
    if not rows: return
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def aggregate_experiments(folders,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list); coverage=defaultdict(list); rankings=[]
    for folder in folders:
        seed=json.loads((folder/"manifest.json").read_text())["seed"]
        p=folder/"ood_results"/"metrics.json"
        if p.exists():
            for family,models in json.loads(p.read_text()).items():
                for model,m in models.items():
                    key=(family,model)
                    groups[("ood",*key,"accuracy")].append(m["accuracy"])
                    coverage[key].append((m["n_certified"],m["n_total"]))
        for kind,path,keys,metrics in [
            ("noise", "experiments/noise.csv",["channel","strength"],["original_label_agreement","S_agreement","B_agreement","W_agreement","GHZ_agreement"]),
            ("consensus","models/consensus_ablation.csv",["features"],[m+"_"+s for m in ("GHZ","W","B","cascade") for s in ("validation","test")]),
            ("individual","models/individual_ablation.csv",["model","features"],["validation_accuracy"]),
            ("kernel","experiments/kernel_metrics.csv",["model","kernel"],["test_auc","test_accuracy","validation_auc"]),
            ("trace","models/cascade_trace.csv",["stage","true_class"],["entered","accepted","rejected","errors","upstream_excluded"])]:
            p=folder/path
            if not p.exists(): continue
            seen=set()
            for row in read_csv(p):
                key=tuple(row[k] for k in keys)
                if key in seen: raise ValueError(f"Duplicate aggregation key: {path} {key}")
                seen.add(key)
                for metric in metrics: groups[(kind,*key,metric)].append(float(row[metric]))
        p=folder/"models"/"consensus_order.json"
        if p.exists():
            order=json.loads(p.read_text())
            if len(order)!=63 or len(set(order))!=63: raise ValueError("Invalid consensus order")
            rankings.append((seed,order))
    specs={"ood":["family","model"],"noise":["channel","strength"],"consensus":["features"],"individual":["model","features"],"kernel":["model","kernel"],"trace":["stage","true_class"]}
    result={}
    for kind,keys in specs.items():
        rows=[]
        for key,values in sorted(groups.items()):
            if key[0]!=kind: continue
            row=dict(zip(keys,key[1:-1]),metric=key[-1],**stats(values))
            if kind=="ood":
                counts=coverage[key[1:-1]]
                row.update(certified=sum(x for x,_ in counts),total=sum(y for _,y in counts),seeds_evaluated=len(counts))
            rows.append(row)
        if rows:
            write_csv(out/(kind+"_summary.csv"),rows)
            result[kind]=rows
    if rankings:
        features=sorted(rankings[0][1])
        if any(set(order)!=set(features) for _,order in rankings): raise ValueError("Inconsistent feature sets")
        rows=[]
        for feature in features:
            ranks=[order.index(feature)+1 for _,order in rankings]
            rows.append(dict(feature=feature,mean_rank=statistics.mean(ranks),rank_std=stats(ranks)["sample_std"],
                             top3_frequency=sum(k<=3 for k in ranks)/len(ranks),
                             top10_frequency=sum(k<=10 for k in ranks)/len(ranks),n_runs=len(ranks)))
        write_csv(out/"feature_stability.csv",sorted(rows,key=lambda r:r["mean_rank"]))
        pairs=[]
        for (a,x),(b,y) in itertools.combinations(rankings,2):
            d=sum((x.index(f)-y.index(f))**2 for f in features)
            pairs.append(dict(seed_a=a,seed_b=b,spearman=1-6*d/(63*(63**2-1)),top10_jaccard=len(set(x[:10])&set(y[:10]))/len(set(x[:10])|set(y[:10]))))
        write_csv(out/"ranking_pairs.csv",pairs)
        result["ranking_pairs"]=pairs
    return result

def plot_aggregates(results,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for kind,key,metric,xlabel,ylabel in [
        ("noise","strength","original_label_agreement","Noise probability","Original-label agreement"),
        ("consensus","features","cascade_test","Number of features","Cascade test accuracy")]:
        rows=[r for r in results.get(kind,[]) if r["metric"]==metric]
        if not rows: continue
        fig,ax=plt.subplots()
        for label in sorted({r.get("channel","Cascade") for r in rows}):
            g=sorted([r for r in rows if r.get("channel","Cascade")==label],key=lambda r:float(r[key]))
            x=[float(r[key]) for r in g]; y=[r["mean"] for r in g]
            ax.plot(x,y,label=label)
            if all(r["sample_std"] is not None for r in g):
                ax.fill_between(x,[max(0,r["mean"]-r["sample_std"]) for r in g],
                                [min(1,r["mean"]+r["sample_std"]) for r in g],alpha=.2)
        ax.set(xlabel=xlabel,ylabel=ylabel,ylim=(0,1.02)); ax.legend()
        fig.tight_layout(); fig.savefig(Path(out)/(kind+"_multi_seed.png"),dpi=200); plt.close(fig)

def markdown(summary,results):
    def fmt(v): return "—" if v is None else f"{100*v:.2f}%"
    lines=["# End-to-end results",f"Completed seeds: {summary['n_runs']}","","| Model | Mean test accuracy | Sample SD |","|---|---:|---:|"]
    for k,v in summary["accuracy"].items(): lines.append(f"| {k} | {fmt(v['mean'])} | {fmt(v['sample_std'])} |")
    if results.get("ood"):
        lines += ["","## OOD (certified subset only)","| Family | Model | Mean | SD | Certified / total | Valid seeds |","|---|---|---:|---:|---:|---:|"]
        for r in results["ood"]:
            lines.append(f"| {r['family']} | {r['model']} | {fmt(r['mean'])} | {fmt(r['sample_std'])} | {r['certified']} / {r['total']} | {r['n_runs']} |")
    if results.get("consensus"):
        lines += ["","## Consensus (held-out test)","| Features | Cascade mean | SD |","|---:|---:|---:|"]
        for r in sorted(results["consensus"],key=lambda r:int(r["features"])):
            if r["metric"]=="cascade_test": lines.append(f"| {r['features']} | {fmt(r['mean'])} | {fmt(r['sample_std'])} |")
    lines += ["","SD is undefined for one seed. Noise measures retention of original labels, not post-noise class accuracy. Smoke is for software validation only.","","See the per-seed artifacts and multi-seed-summary for CSVs and plots."]
    return "\n".join(lines)+"\n"
