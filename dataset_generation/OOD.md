# Independent OOD evaluation

The four matrix families follow manuscript section Performance for Out-Of-Distribution dataset. UPB normalization is corrected by dividing by four. Edge parameters exclude c=ab (fully separable surface); GME concurrence of these X-shaped edge matrices is zero. Horodecki is PPT across A|BC and entangled across that cut; higher-class labels are left blank pending a separate biseparability proof. UPB is separable across bipartitions but not fully separable.

X-matrix GME concurrence follows https://doi.org/10.1103/PhysRevA.86.062303. GME does not certify GHZ class. Non-GME X states have W and GHZ labels zero. Positive GHZ labels require fidelity >3/4 with a phase-adjusted GHZ basis vector (https://doi.org/10.1103/PhysRevLett.87.040401). Uncertified GME examples are excluded from GHZ accuracy, not treated as negatives. The W model is not evaluated on uncertified W membership. X sampling alternates the original notebook proposal and a new GHZ-focused proposal; this changes the distribution and does not reproduce the original reported 98% result.

Run from repository root:
```bash
python dataset_generation/generate_ood.py --samples-per-family 1000 --seed 2027
python models/evaluate_ood.py --model model_outputs/full_models.joblib
```

Or run Evaluate independent OOD datasets in Actions with a successful training run ID. Only trusted model artifacts should be used. The evaluation reports coverage and label counts alongside accuracy. No retraining or tuning on OOD takes place. Parameter ranges and sample counts are new defaults, not recovered original settings. The original manuscript table requires reevaluation.
