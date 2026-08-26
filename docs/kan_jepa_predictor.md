# KAN-JEPA Predictor

`ipgdo_kan_jepa` is the final proposed-method adapter. It keeps the
Iterative Project-Guided Diffusion Oversampling (IPGDO) module and replaces the temporary MLP
backend with a transductive KAN-JEPA predictor.

## Data Contract

For one outer LOPO target, augmented source projects contain original rows with
nonnegative row IDs and synthetic defects with negative row IDs. The predictor
uses that invariant to construct the configured training views:

- Real and synthetic source rows: JEPA representation learning in the current
  `shared_augmented_source` mode.
- Real and synthetic source rows: supervised defect classification.
- Real source rows and unlabeled target rows: weighted CORAL alignment.

Synthetic rows never fit the predictor's `RobustScaler`, source distance
weights, or CORAL statistics. Target feature statistics never fit the scaler.
The target object is `UnlabeledProject`, so no target label can enter the
predictor call.

## Architecture

The context encoder and EMA target encoder are architecturally identical KANs.
Each KAN edge combines a SiLU base path and a fixed-grid B-spline path. A masked
context row is concatenated with its binary mask before encoding. The target
encoder sees the complete, lightly perturbed row and is updated only by EMA.

The JEPA predictor is a lightweight MLP. It receives the context latent and an
embedding of the feature mask, then predicts the target encoder's normalized
latent representation.

The classifier is TabR's encoder-retrieval-predictor architecture adapted to
the 64-dimensional KAN-JEPA latent space. For every query, TabR retrieves the
nearest latent keys from the labeled augmented-source candidate pool, encodes
their labels and query-to-neighbor differences, and adds the weighted context
to the query representation before producing one binary logit. Training
queries cannot retrieve themselves. Held-out source validation and target
prediction retrieve only from the source projects used to fit that model; no
target label or pseudo-label enters the candidate pool.

Candidate search uses batched PyTorch squared Euclidean distance instead of
FAISS. This keeps the original TabR retrieval and context-fusion computation
while avoiding another runtime dependency for the current CPDP-scale datasets.

Joint training minimizes latent prediction, source classification, weighted
CORAL, target dual-view consistency, latent variance, and latent covariance
losses. Target pseudo-labels are deliberately disabled.

## Model Selection

One source project is selected deterministically from the outer seed and held
out for early stopping and MCC threshold selection. The model is then refit on
all source projects for the selected number of joint epochs. Target labels are
not used for either decision.

All architecture, TabR classifier, view, loss, optimization, and smoke/full values are under
`profiles.<profile>.ipgdo_kan_jepa.predictor` in
`configs/experiment.yaml`.

Run one connectivity check with:

```powershell
.\.venv\Scripts\python.exe run_experiment.py run `
  --method ipgdo_kan_jepa `
  --profile smoke `
  --target ReLink/Apache `
  --seed 2026072201
```
