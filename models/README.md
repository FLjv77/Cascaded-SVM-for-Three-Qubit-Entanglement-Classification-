# Cascaded SVM models

Run from the repository root, after generating the CSV datasets:

```bash
python -m pip install -r models/requirements.txt
python models/train.py --data datasets --output model_outputs
python models/train.py --data datasets --output model_outputs --consensus --budgets 3 10 20 30 40 50 63
python models/predict.py --model model_outputs/full_models.joblib --input datasets/bi_vs_other.csv --output predictions.csv
```

Only load trusted joblib files. Predictions are S, B, W, GHZ, referring to the exclusive classes S, B minus S, W minus B and GHZ minus W.

## Design

The input files are bi_vs_other.csv, w_vs_other.csv and ghz_vs_other.csv. Features are matched by Pauli column name, not column position. All labels use the new generator's convention: positive B means B minus S, positive W means W minus B, positive GHZ means GHZ minus W.

Training uses stratified 70/15/15 splits, matching the manuscript. RBF SVC is the default. Hyperparameters C and gamma are selected on the validation set using accuracy; the grid is an implementation choice because the exact original grid is unavailable. Linear and polynomial alternatives are optional (polynomial degrees 2–9). The test split is not used for tuning or permutation importance. No scaling is applied to the bounded Pauli features.

The cascade evaluates GHZ first, then W, then B, otherwise returning S. Its four-class evaluation set combines held-out positive rows from each binary dataset and negative rows from B. This is a structured held-out test, not an independent OOD dataset or proof of performance on arbitrary states. The same construction is used for validation.

Permutation importance uses validation data, ten repeats and at most 2,000 validation samples by default for computational feasibility. The optional consensus procedure follows the manuscript's rank-aligned accuracy-increment and reserve-list method rather than the exploratory frequency-count notebook. Ties are deterministic, duplicate initial features are removed, and unused features are appended to complete the ranking. Models are retrained for each requested feature budget with the full-feature hyperparameters fixed. Test curves are for reporting, not selecting the budget.

## Outputs

- full_models.joblib: three fitted classifiers and feature names.
- metrics.json: binary accuracy, ROC AUC, confusion matrices, classification reports and four-class cascade metrics.
- rankings.json: per-model permutation importance.
- individual_ablation.csv: validation accuracy for each individual feature prefix (consensus mode).
- consensus_order.json: shared feature ordering.
- consensus_ablation.csv and consensus_metrics.json: validation and held-out test results by budget.
- consensus_N.joblib: fitted cascade at budget N.

The original notebooks' OOD, quantum-noise, PPT edge-state, and canonical-segmentation experiments are not ported here. The original trained model files and exact tuning parameters were unavailable. Reported manuscript accuracy is not guaranteed; new results must be measured. Full kernel SVM grid searches and 189 individual ablation fits can be expensive for 100,000 rows per model. Start with a small generated dataset.

Use GitHub Actions > Generate and train cascaded SVM for a self-contained run. This generates a fresh dataset with the requested seed before training. Outputs are uploaded as an artifact rather than committed.
