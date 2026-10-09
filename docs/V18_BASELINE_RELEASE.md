# V18 reproducible baseline snapshot / V18 可复现版本

This commit records the server code used for `v18_multisource_dc9821cf4e36`.
Experiment completed 2026-10-08 08:11 Asia/Shanghai: 25 targets x 10 seeds,
250 successful final evaluations, 1780 source-project folds.

## Reproduction / 复现

- Entry: `run_v18_multisource.py`; preflight: `python run_v18_multisource.py --dry-run`.
- Environment: `pyproject.toml` and the requirements files; install the deep-learning extra.
- Baseline configuration: `configs/server_runtime_9f90f5526a55.yaml`.
- Fixed source-loss-selected seeds: `artifacts/ipgdo_ratio0p7/best_oversampling_seeds.yaml`.
- Required baseline fingerprint: `backups/v16_before_multisource_20261006T063845Z/manifest.json`.
- Expected server deployment root: `/mnt/proj/IPGDO_NetworkCapacity_Linux_Package_20260809`.
  The archived runtime YAML contains absolute server output paths. For a different
  machine, regenerate the runtime configuration via the entry point and supply the
  same datasets; do not silently change fingerprinted source/configuration files.
- Large trained checkpoints, caches and predictions stay in the server run directory;
  this is a code/configuration release, not a portable trained-model package.

## Protocol / 协议

IPGDO ratio 0.7; Chebyshev KAN degree 4; six losses; 160 pretrain epochs;
300 joint-epoch ceiling; original learning-rate schedule. Each source project is
held out in turn. The same selected model determines its source G-mean threshold
and scores the target. Threshold-centered scores are averaged with equal weights.
No full-source refit. Target defect labels are not used for model/fusion selection.
Inherited seed selection is source-only but not reselected inside each fold;
the original master-style data cleaning is retained.

## Verified results / 已核验结果

25-project macro means: AUC 0.7495985138, AUPRC 0.4940593985,
MCC 0.2777144951, F1 0.4443781717, G-mean 0.6435951400,
Popt@20% 0.4150244536. These are descriptive results, not significance claims.
See `docs/v18_multisource_protocol.md` for details and limitations.
