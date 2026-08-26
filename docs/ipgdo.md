# IPGDO Oversampling Module

IPGDO is short for **Iterative Project-Guided Diffusion Oversampling**. The
active model deliberately contains only two algorithmic components:

- A project-conditioned tabular DDPM that generates defect candidates.
- A KNN filter that keeps candidates inside the normal internal-distance range
  of real defects from the same source project.

The oversampler remains independent from every prediction backend. Its public
API is `fit_resample(sources, seed)`, which has no target-project argument and
therefore cannot use target features or labels during sample generation.

## Data Flow

1. Collect real defect rows from every source project.
2. Fit an independent robust affine scaler on each source project's real
   defects so cross-project metric scales cannot distort generation.
3. Train one shared project-conditioned DDPM using only standard noise MSE.
4. For each source project, compute the required synthetic defect count.
5. Fit a KNN reference distribution from leave-one-out distances among that
   project's real defects.
6. Generate one batch of project-conditioned candidates from Gaussian noise.
7. Accept only candidates whose mean KNN distance lies between the configured
   lower and upper quantiles of the real-defect internal distances.
8. If the accepted count is insufficient, generate and filter another batch
   without relaxing the KNN interval.
9. Once enough candidates are available, rank them by proximity to the median
   real-defect internal distance and keep exactly the required count.
10. If the quota is still unmet after `max_generation_rounds`, stop with an
    explicit error instead of returning too few or lower-quality samples.

The lower KNN boundary rejects candidates that are unrealistically close to
memorized training rows. The upper boundary rejects candidates that leave the
real defect manifold. KNN is a quality filter, not a defect classifier.

## Configuration

The active `oversampling` section in `configs/experiment.yaml` contains only:

- `target_minority_ratio`: final defect-to-clean ratio for each source project.
- `candidate_batch_multiplier`: candidate batch size relative to the remaining
  quota.
- `max_generation_rounds`: hard safety limit for iterative generation.
- `diffusion`: standard DDPM architecture, optimization, and sampling budget.
- `knn`: `neighbors`, `lower_quantile`, and `upper_quantile`.

The current formal defaults use a target ratio of `0.6`, candidate batch
multiplier of `2.0`, at most `100` generation rounds, and a KNN interval from
the `0.10` to `0.90` internal-distance quantiles. Old formula-guidance,
semantic-role, numeric-validity, anchor, classifier-free-guidance, support
projection, and farthest-point diversity parameters are not part of the active
IPGDO model.

Old augmentation caches are disabled because their samples were produced by a
different architecture. A formal run of the clean IPGDO must generate a new
augmentation before training the predictor.

## Predictor Boundary

No active IPGDO component imports or understands KAN-JEPA. The method adapter
passes augmented projects through the stable predictor interface. See
`docs/kan_jepa_predictor.md` for the prediction architecture and data contract.
