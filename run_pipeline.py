"""Run one or more reproducible end-to-end experiment configurations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import platform
import importlib.metadata
import statistics

ROOT=Path(__file__).resolve().parent

def run_step(script,*args):
    command=[sys.executable,str(ROOT/script),*map(str,args)]
    print("RUN",*command,flush=True)
    subprocess.run(command,cwd=ROOT,check=True)

def run_one(config,profile,seed,out):
    p=config["profiles"][profile]
    offsets=config["seed_offsets"]
    folder=out/profile/str(seed)
    folder.mkdir(parents=True,exist_ok=False)
    d=folder/"datasets"; m=folder/"models"; o=folder/"ood_datasets"; e=folder/"experiments"
    manifest=dict(profile=profile,seed=seed,settings=p,ood_seed=seed+offsets["ood"],noise_seed=seed+offsets["noise"],python=platform.python_version(),revision=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True).stdout.strip(),dependencies={x:importlib.metadata.version(x) for x in ["numpy","pandas","scikit-learn","joblib","matplotlib"]},status="running")
    def save():
        (folder/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    save()
    try:
        run_step("dataset_generation/generate.py","--output",d,"--seed",seed,"--samples-per-model",p["samples_per_model"])
        args=["--data",d,"--output",m,"--seed",seed,"--kernel",p["kernel"],"--repeats",p["repeats"],"--importance-limit",p["importance_limit"],"--budgets",*p["budgets"]]
        if p["consensus"]: args.append("--consensus")
        run_step("models/train.py",*args)
        model=m/"full_models.joblib"
        if p["ood"]:
            run_step("dataset_generation/generate_ood.py","--output",o,"--seed",manifest["ood_seed"],"--samples-per-family",p["ood_samples_per_family"])
            run_step("models/evaluate_ood.py","--model",model,"--data",o,"--output",folder/"ood_results"/"metrics.json")
        if p["noise"]:
            run_step("experiments/noise.py","--model",model,"--seed",manifest["noise_seed"],"--samples-per-class",p["noise_samples_per_class"],"--output",e/"noise.csv")
        if p["compare_kernels"]: run_step("experiments/kernels.py","--data",d,"--seed",seed,"--output",e)
        if p["diagnostics"]: run_step("experiments/diagnostics.py","--model",model,"--data",d,"--seed",seed,"--rotations",p["rotations"],"--output",e)
        if p["figures"]: run_step("experiments/plots.py","--models",m,"--experiments",e,"--output",folder/"figures")
    except BaseException:
        manifest["status"]="failed"; save(); raise
    manifest["status"]="complete"; save()
    return folder

def aggregate(folders,output):
    records=[]
    for folder in folders:
        manifest=json.loads((folder/"manifest.json").read_text())
        if manifest["status"]!="complete": raise ValueError(f"Incomplete run: {folder}")
        report=json.loads((folder/"models"/"metrics.json").read_text())
        records.append({"seed":manifest["seed"],**{k:report[k]["test"]["accuracy"] for k in ("GHZ","W","B")},"cascade":report["cascade"]["accuracy"]})
    if not records: raise ValueError("No completed runs to aggregate")
    summary={k:{"mean":statistics.mean(r[k] for r in records),"sample_std":statistics.stdev(r[k] for r in records) if len(records)>1 else None} for k in ("GHZ","W","B","cascade")}
    output.write_text(json.dumps(dict(n_runs=len(records),per_seed=records,accuracy=summary),indent=2)+"\n")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default=str(ROOT/"configs"/"end_to_end.json"))
    p.add_argument("--profile",default="smoke")
    p.add_argument("--seed",type=int,help="Override profile seed list with one seed")
    p.add_argument("--output",default="runs")
    a=p.parse_args()
    c=json.loads(Path(a.config).read_text())
    if c["schema_version"]!=1: p.error("unsupported config schema")
    if a.profile not in c["profiles"]: p.error("unknown profile")
    seeds=[a.seed] if a.seed is not None else c["profiles"][a.profile].get("seeds",c["seeds"])
    if not seeds or len(set(seeds))!=len(seeds) or any(not isinstance(s,int) or not 0<=s<2**32-200000 for s in seeds): p.error("invalid seeds")
    out=Path(a.output).resolve()
    os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    os.environ.setdefault("OMP_NUM_THREADS","1")
    folders=[run_one(c,a.profile,s,out) for s in seeds]
    aggregate(folders,out/a.profile/("summary.json" if a.seed is None else f"summary_{a.seed}.json"))

if __name__=="__main__":
    main()
