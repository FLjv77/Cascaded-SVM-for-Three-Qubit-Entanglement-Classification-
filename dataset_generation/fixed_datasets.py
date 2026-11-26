"""Create fixed matched baseline and mixed-B datasets; no OOD or SDP."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import generate as g

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def build(root,n,seed,fraction):
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    old=root/"reference"; new=root/"mixed_b"
    g.generate(old,n,seed)
    shutil.copytree(old,new)
    d=pd.read_csv(old/g.FILES["B"])
    rng=np.random.default_rng(np.random.SeedSequence([seed,991]))
    positive=np.flatnonzero(d.label.to_numpy()==1)
    selected=np.sort(rng.choice(positive,size=round(len(positive)*fraction),replace=False))
    for idx in selected:
        d.loc[idx,g.LABELS]=g.features(g.mixed_bi(rng))
    d.to_csv(new/g.FILES["B"],index=False)
    meta=json.loads((new/"metadata.json").read_text())
    meta.update(b_generation="cross-cut mixture certified NPT",mixed_positive_count=len(selected),mixed_positive_fraction=fraction)
    (new/"metadata.json").write_text(json.dumps(meta,indent=2))
    # Preserve row positions and labels across variants. W and GHZ are byte-identical.
    pd.DataFrame({"row_index":selected}).to_csv(root/"replaced_B_rows.csv",index=False)
    manifest=dict(schema=1,seed=seed,samples_per_model=n,mixed_fraction=fraction,
                  files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob("*")) if p.is_file()})
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2))

def verify(root,n,seed,fraction):
    root=Path(root)
    m=json.loads((root/"manifest.json").read_text())
    if (m["schema"],m["samples_per_model"],m["seed"],m["mixed_fraction"])!=(1,n,seed,fraction):
        raise ValueError("Dataset configuration mismatch")
    expected={f"{v}/{f}" for v in ("reference","mixed_b") for f in [*g.FILES.values(),"metadata.json"]}|{"replaced_B_rows.csv"}
    if set(m["files"])!=expected: raise ValueError("Unexpected file list")
    for name,digest in m["files"].items():
        if sha(root/name)!=digest: raise ValueError(f"Checksum mismatch: {name}")
    for name in ("W","GHZ"):
        if m["files"]["reference/"+g.FILES[name]]!=m["files"]["mixed_b/"+g.FILES[name]]: raise ValueError("Non-B datasets changed")
    a=pd.read_csv(root/"reference"/g.FILES["B"]); b=pd.read_csv(root/"mixed_b"/g.FILES["B"])
    if not a.label.equals(b.label) or len(a)!=n: raise ValueError("Paired labels changed")
    selected=pd.read_csv(root/"replaced_B_rows.csv").row_index.to_numpy()
    if len(selected)!=round(n/2*fraction) or len(set(selected))!=len(selected) or not a.iloc[selected].label.eq(1).all(): raise ValueError("Invalid replaced indices")
    keep=np.ones(n,dtype=bool); keep[selected]=False
    np.testing.assert_allclose(a.loc[keep,g.LABELS],b.loc[keep,g.LABELS],atol=1e-14,rtol=0)
    for row in b.iloc[selected][g.LABELS].to_numpy():
        rho=(np.eye(8)+np.einsum("k,kij->ij",row,g.OPS))/8
        g.features(rho)
        if min(g.ppt_min_three(rho,k) for k in range(3))>=-1e-8: raise ValueError("Missing NPT certificate")
    print("Verified fixed datasets",root)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--output",required=True); p.add_argument("--samples",type=int,default=100000); p.add_argument("--seed",type=int,default=42); p.add_argument("--fraction",type=float,default=.7); p.add_argument("--verify",action="store_true")
    a=p.parse_args()
    if not 0<=a.fraction<=1 or a.samples<20 or a.samples%2: p.error("invalid configuration")
    if not a.verify: build(a.output,a.samples,a.seed,a.fraction)
    verify(a.output,a.samples,a.seed,a.fraction)
