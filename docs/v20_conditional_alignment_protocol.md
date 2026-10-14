# V20 Class-Conditional Transfer Protocol

Baseline: frozen V18 `dc9821cf4e36`; this candidate is independent and is not an adopted replacement.

## Sole training change

Replace the weighted global CORAL term with class-conditional prototype alignment.
All six V18 loss weights remain unchanged; there is no extra pseudo-label classification loss.
Keep JEPA, Chebyshev KAN, TabR, source-distance weights, optimizer, learning-rate schedule,
source validation, fixed IPGDO seeds, ratio 0.7 and equal threshold-centered fusion.
The `shared_augmented_source` JEPA view remains augmented-source-only.

## Teacher and prototypes

A separate, frozen evaluation teacher contains the online encoder AND classifier.
Initialize by copying the student at the start of joint training. At every subsequent
epoch boundary, update parameters as `teacher = 0.9*teacher + 0.1*student`;
copy buffers. The teacher and its retrieval keys remain fixed within that epoch.
This epoch teacher is separate from the unchanged JEPA target encoder EMA (0.996 per step).

Encode all real fold-training source rows using the teacher; normalize each latent vector.
For each class, average rows inside each source project, average project class means using
the unchanged source-distance weights, then normalize the final class prototype.
Synthetic source rows, held-out source rows, and target labels never form prototypes.
Missing/degenerate prototypes disable alignment for that class.

## Unlabeled target selection and loss

Teacher predicts two independently Gaussian-perturbed target views (std 0.02).
Keep rows only if both predictions agree at the fixed 0.5 boundary and both have
confidence >=0.9. Noise has an independent per-fold RNG, not the student's training RNG.
Teacher confidence is a heuristic, not a calibrated probability guarantee.

For each accepted class, compute the confidence-weighted mean of normalized STUDENT
target latents and normalize that mean. Minimize `1 - dot(target_mean, source_prototype)`.
Average equally over present classes, never force target defect prevalence.
Teacher/prototypes/pseudo-labels receive no gradient. All-rejected batches have zero
alignment loss, not CORAL fallback and not forced pseudo-labels.

No alignment in joint epochs 1-20. Ramp `(epoch-20)/20` clipped to [0,1]; full strength
from epoch 40 onward. Multiply by the unchanged alignment loss weight 0.1.
Rules are fixed before this experiment, not selected using target evaluation metrics.

## Evaluation and reproducibility

25 targets x the original ten prediction seeds, 178 folds per seed (1780 model fits).
Reuse only V18 fold IPGDO caches matching target, held-out project, training-source
fingerprint, fixed oversampling seed, original IPGDO settings and implementation hash.
Every predictor is trained from initialization: 160 pretrain, up to 300 joint epochs,
patience 25. Retain best source-validation BCE checkpoint and its source G-mean threshold.
Unchanged odds-based threshold centering and equal fold mean; final threshold 0.5.
Target labels are accessed only by the common final evaluator.

Save per-fold model/scaler/config, raw/centered predictions, source validation curve,
best epoch/threshold, batch losses and per-epoch pseudo-label coverage/class counts,
prototype validity, ramp and conditional loss. Coverage counts are batch exposures,
NOT unique target row counts. Record whether the selected checkpoint is after warmup.
Formal data cleaning and inherited (not nested-fold-reselected) oversampling seed map
retain the V18 protocol limitations.

Smoke uses one pretrain/two joint epochs, warmup=0/ramp=1, with four representative
targets and all their source folds. It is an execution check only, never a reported
performance experiment or a criterion for selecting hyperparameters. Unit training
tests force reliable pseudo-labels to verify active alignment gradients end to end.

## Risks

Confident pseudo-labels can still be wrong; confidence gating may accept few defect rows.
Class prototypes assume sufficient cross-project class semantics and can miss multimodal
class structure. Early source-validation selection may choose a pre-alignment checkpoint.
Improvement is not guaranteed; compare the complete candidate with V18 before adoption.
No Git commit/push, local-code sync, Excel update or V18 artifact overwrite is part of this run.
