# Transformer-JEPA Architecture Variant

`ipgdo_transformer_jepa` is a controlled architecture variant of
`ipgdo_kan_jepa`. It preserves the same IPGDO oversampler,
CPDP protocol, source-only model selection, data partitions, losses, training
budget, threshold selection, and target-label policy. Only the context encoder,
EMA target encoder, and JEPA predictor are replaced.

## Tokenization

Each scalar software metric becomes one token. The token is the sum of a scalar
value projection, a learnable feature-identity embedding, and an observed/masked
state embedding. A learnable CLS token is prepended. This representation keeps
the identity of each tabular metric while allowing self-attention to model
cross-metric interactions.

## Context And Target Encoders

The context and target encoders use identical pre-normalized Transformer encoder
stacks. The context encoder receives masked rows and supplies its CLS latent to
classification, CORAL alignment, variance, and covariance losses. It also sends
its complete token sequence to the JEPA predictor.

The target encoder receives a complete, lightly perturbed row. It is initialized
from the context encoder, has gradients disabled, and changes only through the
same exponential moving average update used by KAN-JEPA.

## Transformer Predictor

The predictor projects the context token sequence into its own model dimension.
A learnable prediction token is conditioned on the binary feature mask, prepended
to the context sequence, and processed by an independent Transformer encoder.
The prediction-token output is projected to the JEPA latent dimension and is
trained to match the normalized target-encoder latent.

The classifier remains linear because the requested comparison changes only the
three JEPA modules. Synthetic source defects still enter supervised
classification only; they do not enter JEPA, CORAL, normalization, or source
distance estimation. No target labels or pseudo-labels are used.

Run the smoke profile with:

```powershell
.\.venv\Scripts\python.exe run_experiment.py run `
  --method ipgdo_transformer_jepa `
  --profile smoke `
  --target ReLink/Apache `
  --seed 2026072201
```

Results are written under the independent method directory
`outputs/runs/<profile>/ipgdo_transformer_jepa/`, so existing KAN-JEPA
results are never overwritten.
