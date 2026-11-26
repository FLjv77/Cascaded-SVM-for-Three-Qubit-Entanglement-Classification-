# Paper experiments

Consumes the model bundle and metrics from models/train.py. The noise sweep uses new independent states sampled from the corrected generator, not the exact original training/test density matrices. The same clean states are reused for all channels and strengths. The output is agreement with original labels, not verified accuracy against the unknown post-noise entanglement class. Channels use the PennyLane probability conventions with local Kraus operators and strengths 0–0.50 in steps of 0.01.

Kernel comparison uses fixed 70/15/15 splits and validation AUC for tuning each kernel (RBF and polynomial degrees 2–9); the grid is a new implementation choice. Batch diagnostics use ten stratified training-only batches, and local state sweeps use the frozen GHZ classifier. Sweep outputs are positive prediction rates, not certified class labels. The original exploratory segmented SVM is intentionally excluded: it is not the manuscript's cascade.

From repository root:
```bash
python -m pip install -r experiments/requirements.txt
python experiments/noise.py --model model_outputs/full_models.joblib
python experiments/kernels.py --data datasets
python experiments/diagnostics.py --model model_outputs/full_models.joblib --data datasets
python experiments/plots.py
```

For the manuscript feature-selection settings run models/train.py with --consensus --repeats 50 --budgets followed by all integers 3 through 63. The current training implementation caps permutation importance validation samples at 2000 by default; set --importance-limit to the desired validation size. The ablation plots show validation accuracy and its increments, not test curves. Kernel comparison and full ablation can be expensive.

Actions: Run paper experiments with a successful training run ID. It downloads the previous model/metrics artifact. Optional kernel and batch runs regenerate datasets from metadata; exact reproduction requires the same generator revision and dependencies as training. Use matching revisions. Only use trusted model artifacts. Figures are generated only when their source metrics exist; no fabricated results. OOD is handled by the existing separate workflow. The manuscript and old notebooks are unchanged.

Multi-seed aggregation: experiments/aggregate.py produces grouped means, sample SD, contributing seed counts, OOD certificate coverage, rank frequencies and pairwise Spearman/top-10 Jaccard scores. Noise and consensus plots show mean ± one sample SD only when multiple seeds contribute. No test result is used for model selection.
