"""Independent OOD families from manuscript, with conservative certified labels."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from generate import LABELS, features, projector, kron3, rotate

TOL=1e-9

def horodecki(a):
    if not 0<a<1: raise ValueError("a must lie in (0,1)")
    rho=np.diag([a,a,a,a,(1+a)/2,a,a,(1+a)/2])
    for i,j in [(0,5),(1,6),(2,7)]:
        rho[i,j]=rho[j,i]=a
    rho[4,7]=rho[7,4]=np.sqrt(1-a*a)/2
    return rho/(7*a+1)

def edge(a,b,c):
    if min(a,b,c)<=0: raise ValueError("positive parameters required")
    rho=np.diag([1,a,b,c,1/c,1/b,1/a,1])
    rho[0,7]=rho[7,0]=1
    return rho/np.trace(rho)

def upb():
    zero=np.array([1,0]); one=np.array([0,1])
    plus=np.array([1,1])/np.sqrt(2); minus=np.array([1,-1])/np.sqrt(2)
    vectors=[kron3(zero,zero,zero),kron3(minus,plus,one),
             kron3(plus,one,minus),kron3(one,minus,plus)]
    return (np.eye(8)-sum(projector(v) for v in vectors))/4

def x_state(rng,concentrated=False):
    # Original notebook uses normalized uniform diagonal and disk-uniform coherences.
    d=rng.random(8)
    if concentrated:
        # Additional explicitly documented proposal for certified GHZ examples.
        i=int(rng.integers(4)); d*=.03; d[i]=d[7-i]=1
    d/=d.sum()
    z=np.sqrt(d[:4]*d[::-1][:4])*np.sqrt(rng.random(4))*np.exp(2j*np.pi*rng.random(4))
    rho=np.diag(d.astype(complex))
    for i in range(4):
        rho[i,7-i]=z[i]; rho[7-i,i]=z[i].conjugate()
    return rho

def x_scores(rho):
    d=rho.diagonal().real; z=np.abs(np.fliplr(rho).diagonal()[:4])
    s=np.sqrt(d[:4]*d[::-1][:4])
    gme=2*max(0,float(np.max(z+s)-s.sum()))
    fidelity=float(np.max((d[:4]+d[::-1][:4])/2+z))
    return gme,fidelity

def ppt_min(rho,cut):
    axes=list(range(6)); axes[cut],axes[cut+3]=axes[cut+3],axes[cut]
    return float(np.linalg.eigvalsh(rho.reshape((2,)*6).transpose(axes).reshape(8,8))[0])

def rows(rng,family,n):
    for i in range(n):
        if family=="horodecki":
            rho=horodecki(rng.uniform(.001,.999))
            # Entangled across A|BC; no unverified higher-class labels.
            labels=(None,None,None); gme=fidelity=None
        elif family=="edge":
            while True:
                a,b,c=np.exp(rng.uniform(-2,2,3))
                if abs(np.log(a)+np.log(b)-np.log(c))>1e-3: break
            rho=edge(a,b,c)
            labels=(1,0,0); gme=fidelity=None
        elif family=="upb":
            rho=rotate(rng,upb()); labels=(1,0,0); gme=fidelity=None
        else:
            # Preserve original random proposal for non-GME and uncertain GME;
            # alternate with a GHZ-focused proposal; never relabel ambiguous GME.
            rho=x_state(rng,concentrated=bool(i%2))
            gme,fidelity=x_scores(rho)
            labels=(None,0,0) if gme<=TOL else (1,1,1 if fidelity>.75+TOL else None)
            # M_W is only defined on W: GME does not certify W membership.
            if gme>TOL: labels=(None,None,labels[2])
        minima=[ppt_min(rho,k) for k in range(3)]
        if family in ("edge","upb") and min(minima)<-TOL:
            raise ValueError("Expected PPT across every cut")
        if family=="horodecki" and minima[0]<-TOL:
            raise ValueError("Expected PPT across A|BC")
        if family=="x" and gme<=TOL and min(minima)<-TOL:
            labels=(1,0,0)  # biseparable and demonstrably not fully separable
        yield [*features(rho),*labels,gme,fidelity,*minima]

def generate_ood(out,n,seed):
    if n<1: raise ValueError("positive sample count required")
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    families=["horodecki","edge","upb","x"]
    for family,child in zip(families,np.random.SeedSequence(seed).spawn(4)):
        with (out/(family+".csv")).open("w",newline="") as f:
            w=csv.writer(f)
            w.writerow(LABELS+["label_B","label_W","label_GHZ","gme_concurrence","ghz_fidelity","ppt_A","ppt_B","ppt_C"])
            w.writerows(rows(np.random.default_rng(child),family,n))
    (out/"metadata.json").write_text(json.dumps(dict(seed=seed,samples_per_family=n,
        unknown_labels="blank; excluded from accuracy",x_positive_criterion="GHZ fidelity > 0.75",
        edge_parameters="log-uniform exp(-2) to exp(2), exclude abc separable surface c=ab",
        horodecki_parameters="uniform 0.001 to 0.999"),indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default="ood_datasets")
    p.add_argument("--samples-per-family",type=int,default=1000)
    p.add_argument("--seed",type=int,default=2027)
    a=p.parse_args(); generate_ood(a.output,a.samples_per_family,a.seed)
