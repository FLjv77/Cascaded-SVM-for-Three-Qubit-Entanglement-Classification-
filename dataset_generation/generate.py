"""Reproducible three-qubit dataset generation; NumPy only."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np

ALPHA_C = 0.4790  # rounded W + white noise biseparability boundary
ALPHA_MARGIN = 0.0001
EPSILON_C = 3 / (3 + 4 * np.cbrt(2))
EPSILON_MARGIN = 1e-8
I = np.eye(2)
PAULI = dict(I=I, X=np.array([[0,1],[1,0]]),
             Y=np.array([[0,-1j],[1j,0]]), Z=np.diag([1,-1]))
LABELS = sorted(("".join(x) for x in itertools.product("IXYZ", repeat=3)
                 if x != ("I","I","I")), key=lambda x: (3-x.count("I"), x))

def kron3(a,b,c):
    return np.kron(np.kron(a,b),c)

OPS = np.array([kron3(*(PAULI[c] for c in name)) for name in LABELS])

def projector(v):
    v = np.asarray(v, dtype=complex)
    v = v / np.linalg.norm(v)
    return np.outer(v,v.conj())

GHZ = projector([1,0,0,0,0,0,0,1])
W = projector([0,1,1,0,1,0,0,0])

def haar_vector(rng,d):
    v = rng.normal(size=d)+1j*rng.normal(size=d)
    return v/np.linalg.norm(v)

def haar_unitary(rng):
    q,r = np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
    phases = np.diag(r)
    return q * (phases/np.abs(phases))

def rotate(rng,rho):
    u = kron3(*(haar_unitary(rng) for _ in range(3)))
    return u @ rho @ u.conj().T

def ma(rng,d):
    n = int(rng.integers(1,51))
    return sum(w*projector(haar_vector(rng,d))
               for w in rng.dirichlet(np.ones(n)))

def single(rng):
    return projector(haar_vector(rng,2)) if rng.random()<0.5 else ma(rng,2)

def partial_transpose(rho):
    return rho.reshape(2,2,2,2).transpose(0,3,2,1).reshape(4,4).copy()

def entangled_pair(rng):
    for _ in range(10000):
        rho = ma(rng,4)
        if np.linalg.eigvalsh(partial_transpose(rho))[0] < -1e-8:
            return rho
    raise RuntimeError("NPT rejection sampling exhausted")

def permute(rho,order):
    return rho.reshape((2,)*6).transpose(tuple(order)+tuple(i+3 for i in order)).reshape(8,8)

def sep(rng):
    return kron3(*(single(rng) for _ in range(3)))

def bi(rng):
    return permute(np.kron(single(rng),entangled_pair(rng)),rng.permutation(3))

def mixture(rng, factories):
    n = int(rng.integers(2,5))
    return sum(w*factories[int(rng.integers(len(factories)))](rng)
               for w in rng.dirichlet(np.ones(n)))

def pure_family(rng, family):
    indices = [0,7] if family == "GHZ" else [1,2,4]
    if rng.random()<0.5:
        return rotate(rng, GHZ if family=="GHZ" else W)
    v = np.zeros(8,complex)
    while True:
        coeff = haar_vector(rng,len(indices))
        if np.min(np.abs(coeff)) > 1e-5:
            break
    v[indices] = coeff
    return rotate(rng,projector(v))

def werner_w(rng,label):
    lo,hi = (0,ALPHA_C-ALPHA_MARGIN) if label==0 else (ALPHA_C+ALPHA_MARGIN,1)
    a = rng.uniform(lo,hi)
    return rotate(rng,a*W+(1-a)*np.eye(8)/8)

def ghz_w(rng,label):
    lo,hi = (EPSILON_C+EPSILON_MARGIN,1) if label==0 else (0,EPSILON_C-EPSILON_MARGIN)
    e = rng.uniform(lo,hi)
    return rotate(rng,(1-e)*GHZ+e*W)

def w_pure(rng):
    return pure_family(rng,"W")

def w_positive(rng):
    return w_pure(rng) if rng.random()<0.5 else werner_w(rng,1)

def b_negative(rng):
    return sep(rng) if rng.random()<0.5 else mixture(rng,[sep])

def w_negative(rng):
    choice = rng.integers(4)
    if choice==0: return sep(rng)
    if choice==1: return bi(rng)
    if choice==2: return mixture(rng,[sep,bi])
    return werner_w(rng,0)

def ghz_negative(rng):
    choice = rng.integers(4)
    if choice==0: return w_negative(rng)
    if choice==1: return w_positive(rng)
    if choice==2: return mixture(rng,[w_negative,w_positive])
    return ghz_w(rng,0)

def sample(rng,model,label):
    if model=="B": return bi(rng) if label else b_negative(rng)
    if model=="W": return w_positive(rng) if label else w_negative(rng)
    if model=="GHZ":
        if not label: return ghz_negative(rng)
        return pure_family(rng,"GHZ") if rng.random()<0.5 else ghz_w(rng,1)
    raise ValueError(model)

def features(rho):
    if not np.allclose(rho,rho.conj().T,atol=1e-10) or not np.isclose(np.trace(rho),1,atol=1e-10):
        raise ValueError("Invalid Hermiticity or trace")
    if np.linalg.eigvalsh(rho)[0] < -1e-9:
        raise ValueError("Nonpositive density matrix")
    v = np.einsum("ij,kji->k",rho,OPS)
    if np.max(np.abs(v.imag))>1e-9 or np.max(np.abs(v.real))>1+1e-9:
        raise ValueError("Invalid expectation values")
    return v.real

FILES = {"B":"bi_vs_other.csv","W":"w_vs_other.csv","GHZ":"ghz_vs_other.csv"}

def generate(out,n,seed):
    if n<2 or n%2: raise ValueError("samples-per-model must be positive and even")
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    for model,child in zip(FILES,np.random.SeedSequence(seed).spawn(3)):
        rng=np.random.default_rng(child)
        labels=np.repeat([0,1],n//2)
        rng.shuffle(labels)
        with (out/FILES[model]).open("w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(LABELS+["label"])
            for label in labels:
                writer.writerow([*features(sample(rng,model,int(label))),int(label)])
    (out/"metadata.json").write_text(json.dumps(dict(seed=seed,samples_per_model=n,
        alpha_c=ALPHA_C,alpha_margin=ALPHA_MARGIN,epsilon_c=float(EPSILON_C),
        epsilon_margin=EPSILON_MARGIN,numpy_version=np.__version__,
        pauli_order=LABELS),indent=2)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default="datasets")
    p.add_argument("--samples-per-model",type=int,default=100000)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    generate(a.output,a.samples_per_model,a.seed)
