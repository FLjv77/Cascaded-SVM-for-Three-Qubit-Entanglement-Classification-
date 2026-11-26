# Dataset generation

Run locally from the repository root:

```bash
python -m pip install -r dataset_generation/requirements.txt
python -m unittest discover -s dataset_generation -p 'test_*.py'
python dataset_generation/generate.py --samples-per-model 100000 --seed 42 --output datasets
```

Or use Actions > Generate three-qubit datasets > Run workflow. Download the artifact three-qubit-datasets after successful completion.

Each CSV contains 63 Pauli expectation values in the original Paulis notebook order and a binary label. Each model has 100,000 total samples by default, equally split between labels. This follows the manuscript rather than the original notebook's 200,000 total per model. The old file names bi_vs_other.csv, w_vs_other.csv and ghz_vs_other.csv are retained. metadata.json records seed, thresholds and feature order. No train/validation/test split is performed here.

## Label guarantees and corrections

- B: 0 is fully separable, including convex mixtures of product states; 1 is a single-qubit state tensored with an NPT two-qubit state, with random qubit permutation. Arbitrary mixtures of biseparable entangled states are not automatically labeled 1.
- W: 0 is constructed from separable and biseparable states, their convex mixtures, or standard W plus white noise below the biseparability threshold. 1 is a pure W-family state or standard W plus white noise above the threshold.
- GHZ: 0 is constructed from W-class states and convex mixtures, or standard GHZ/W mixtures on the zero-three-tangle side; 1 is a pure GHZ-family state or standard GHZ/W mixture on the positive-three-tangle side.
- The W white-noise threshold is approximately 0.4790 (Table I, https://doi.org/10.1103/PhysRevResearch.4.023238); an exclusion margin of 0.0001 avoids ambiguity due to rounding. The GHZ/W threshold for W weight is 3/(3+4*cuberoot(2)), from https://doi.org/10.1088/1367-2630/10/4/043014; a numerical margin of 1e-8 is used.
- Local Haar rotations preserve labels. Partial transpose does not mutate inputs; qubit permutations are tensor-axis permutations. Every emitted state is checked for trace, Hermiticity and positivity.
- Mixed single- and two-qubit states use Haar vectors with symmetric Dirichlet(1) weights and 1–50 components, consistent with the manuscript's MA construction (the concentration parameter is a chosen default).
- Family sampling proportions and mixture sizes (2–4 components) are implementation choices, not recovered original experimental settings. These are structured ensembles, not exhaustive or uniform samples of the entanglement classes. The generator does not reproduce previously reported metrics without retraining and evaluation.

The original X-state exploratory generator and model-training notebooks are not part of this workflow.
