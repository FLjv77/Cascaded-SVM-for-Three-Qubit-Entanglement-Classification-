# End-to-end configurations

Edit end_to_end.json to control dataset size, seed list, kernel, permutation repeats, importance sample limit, consensus feature budgets, OOD size, noise size, diagnostic rotations and optional stages.

Run locally from the repository root:

```bash
python -m pip install -r experiments/requirements.txt
python run_pipeline.py --profile smoke
python run_pipeline.py --profile pilot
python run_pipeline.py --profile paper --seed 42 --output new_runs
```

GitHub Actions: choose End-to-end research pipeline, select a profile and optionally override its seed. Each seed has a separate matrix job and output folder. The successful jobs produce individual artifacts and a combined accuracy summary (mean and sample standard deviation across seeds). The workflow fails rather than reporting partial runs as a complete multi-seed result.

The smoke profile is for integration testing, not scientific conclusions. Pilot is for initial estimates. Paper follows the manuscript's sample size, 50 permutation repeats and budgets 3–63; full kernel searches and ablation are costly and may exceed hosted runner time limits. Baseline skips the expensive consensus, kernel and diagnostic stages. These profiles and seed choices are new experimental settings, not recovered original settings.

OOD and noise seeds use distinct offsets from the base seed. Dataset and stratified split use the base seed. Each run saves resolved settings, revision, dependency versions and status in manifest.json. Output directories are never overwritten: choose a fresh --output for reruns. Aggregation currently covers full-feature binary and cascade accuracy; OOD, ablation and noise results remain available per seed. Noise agreement is not post-noise class accuracy. OOD results still depend on the validity of the implemented certificates.

No manuscript files are modified.
