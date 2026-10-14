"""Independent V20 registration; V18 artifacts are read-only inputs."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

from cpdp.methods.base import BaseMethod, MethodProvenance
from cpdp.oversampling_seed_search.cache import (
    _atomic_json, load_augmentation_cache, mapping_signature, source_data_signature,
)
from cpdp.predictors.kan_jepa import _flush_batch_loss_log
from cpdp.randomness import seed_everything
from cpdp.types import PredictionResult
from v18_multisource import save_npz, target_feature_signature, threshold_center
from v20_conditional_alignment import ConditionalFoldPredictor as FoldPredictor


def augmentation_path(sources, validation, target, seed, params):
    reuse = params['multisource']['augmentation_reuse']
    identity = dict(target=target.key, validation=validation.key,
        source_signature=source_data_signature(sources), oversampling_seed=int(seed),
        oversampling=params['oversampling'], implementation_signature=reuse['implementation_signature'])
    return Path(reuse['artifact_dir'])/'augmentation'/target.key/(validation.features.name+'_'+mapping_signature(identity)[:20]+'.npz')


def validate_cache_metadata(cache, sources, target, seed, config):
    if not cache.is_file() or not cache.with_suffix('.json').is_file():
        raise FileNotFoundError('Required V18 fold augmentation missing: '+str(cache))
    metadata = json.loads(cache.with_suffix('.json').read_text())
    expected = dict(target=target.key, oversampling_seed=int(seed),
        oversampling_config_signature=mapping_signature(config),
        source_data_signature=source_data_signature(sources), source_projects=[s.key for s in sources])
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError('V18 augmentation cache mismatch: '+name)
    if metadata['augmentation_metadata'].get('target_data_used', False):
        raise ValueError('Target data entered IPGDO')
    if [p['family']+'/'+p['name'] for p in metadata['projects']] != expected['source_projects']:
        raise ValueError('Augmented project membership changed')
    return metadata


class MultiSourceMethod(BaseMethod):
    name = 'ipgdo_kan_jepa'
    provenance = MethodProvenance(
        display_name='IPGDO + KAN-JEPA class-conditional source-fold ensemble',
        implementation='V18 ensemble with real-source prototypes and confidence-gated target pseudo-labels',
        fidelity='proposed-method-variant', source_url='local', source_revision='v20-conditional-v1',
        target_label_policy='Target labels are used only by the common evaluator',
        notes='No pseudo-label BCE; replace CORAL only; unchanged equal threshold-centered fold fusion.',
    )

    def _augmentation(self, sources, validation, target, oversampling_seed):
        cache = augmentation_path(sources, validation, target, oversampling_seed, self.params)
        validate_cache_metadata(cache, sources, target, oversampling_seed, self.params['oversampling'])
        augmentation = load_augmentation_cache(cache, target_key=target.key,
            oversampling_seed=oversampling_seed, oversampling_config=self.params['oversampling'], sources=sources)
        return augmentation, cache

    def fit_predict(self, sources, target, seed):
        sources = tuple(sources)
        if len(sources) < 2 or hasattr(target, 'labels'):
            raise ValueError('Need at least two source projects and an unlabeled target')
        probe = FoldPredictor.from_mapping(self.params['predictor'])
        probe._partition_sources(sources, target)
        if probe.config.optimization.refit_all_sources:
            raise ValueError('V20 must retain V18 fold models without all-source refit')
        if probe.config.threshold_policy != 'gmean' or probe.config.probability_calibration != 'none':
            raise ValueError('V20 preserves source G-mean threshold and no calibration')
        if any(np.any(s.features.row_ids < 0) for s in sources):
            raise ValueError('Expected real sources before fold-specific augmentation')
        seed_map = yaml.safe_load((self.project_root/self.params['oversampling_reuse']['seed_map']).read_text())
        if seed_map.get('target_labels_used') is not False:
            raise ValueError('Invalid oversampling seed-map label boundary')
        oversampling_seed = int(seed_map['targets'][target.key]['best_seed'])
        root = Path(self.params['multisource']['artifact_dir'])/'folds'/target.key/f'seed_{seed}'
        signature = mapping_signature(dict(params=self.params, seed=int(seed),
            source_data=source_data_signature(sources), target_features=target_feature_signature(target),
            oversampling_seed=oversampling_seed))
        records, raw_columns, centered_columns = [], [], []
        outer_log = os.environ.get('CPDP_PREDICTOR_BATCH_LOSS_LOG')
        try:
            for index, validation in enumerate(sources):
                training = tuple(s for s in sources if s.key != validation.key)
                directory = root/('holdout_'+validation.features.name)
                directory.mkdir(parents=True, exist_ok=True)
                record_path = directory/'result.json'
                record = None
                if record_path.is_file():
                    saved = json.loads(record_path.read_text())
                    if saved.get('signature') != signature:
                        raise ValueError('Refusing fold reuse from another experiment')
                    if all((directory/name).is_file() for name in ('predictions.npz','model.pt')):
                        record = saved
                if record is None:
                    print(f'FOLD {index+1}/{len(sources)} target={target.key} seed={seed} validation={validation.key}', flush=True)
                    started = time.perf_counter()
                    augmentation, cache = self._augmentation(training, validation, target, oversampling_seed)
                    loss_path = directory/'batch_losses.jsonl'
                    loss_path.unlink(missing_ok=True)
                    os.environ['CPDP_PREDICTOR_BATCH_LOSS_LOG'] = str(loss_path)
                    seed_everything(seed)
                    predictor = FoldPredictor.from_mapping(self.params['predictor'])
                    details = predictor.fit_fold(augmentation.projects, validation, target, seed, directory)
                    _flush_batch_loss_log()
                    record = dict(status='success', signature=signature, fold_index=index,
                        target=target.key, prediction_seed=int(seed), training_sources=[s.key for s in training],
                        oversampling_seed=oversampling_seed, augmentation_cache=str(cache),
                        augmentation_training_source_signature=source_data_signature(training),
                        validation_excluded_from_ipgdo_fit=True, duration_seconds=time.perf_counter()-started,
                        seed_selection='inherited v16 source-only seed, not reselected within folds',
                        augmentation_reused_from_v18=True, **details)
                    _atomic_json(record_path, record)
                    print(f'FOLD_OK {index+1}/{len(sources)} target={target.key} epoch={details["best_joint_epoch"]} threshold={details["threshold"]:.6f}', flush=True)
                with np.load(directory/'predictions.npz', allow_pickle=False) as archive:
                    if not np.array_equal(archive['target_row_ids'], target.features.row_ids):
                        raise ValueError('Saved target row IDs changed')
                    centered = threshold_center(archive['target_scores'], record['threshold'])
                    if not np.array_equal(centered, archive['target_centered_scores']):
                        raise ValueError('Saved threshold-centered predictions changed')
                    raw_columns.append(archive['target_scores'].copy())
                    centered_columns.append(centered)
                records.append({**record, 'result_file':str(record_path)})
        finally:
            _flush_batch_loss_log()
            if outer_log is None:
                os.environ.pop('CPDP_PREDICTOR_BATCH_LOSS_LOG', None)
            else:
                os.environ['CPDP_PREDICTOR_BATCH_LOSS_LOG'] = outer_log
        raw, centered = np.column_stack(raw_columns), np.column_stack(centered_columns)
        scores = centered.mean(axis=1)
        save_npz(root/'ensemble_predictions.npz', target_row_ids=target.features.row_ids,
            raw_fold_scores=raw, centered_fold_scores=centered, thresholds=np.array([r['threshold'] for r in records]),
            fold_validation_projects=np.array([r['validation_project'] for r in records]), scores=scores,
            raw_mean_scores=raw.mean(axis=1))
        return PredictionResult(scores=scores, threshold=0.5, metadata=dict(
            variant='v20-conditional-v1', aggregation='equal_mean; >=0.5 predicts defect',
            score_semantics='threshold-centered decision scores, NOT calibrated probabilities',
            fold_count=len(sources), folds=records, target_positive_fraction=float(np.mean(scores>=0.5)),
            raw_mean_scores_file=str(root/'ensemble_predictions.npz'),
            random_seeds=dict(oversampling_seed=oversampling_seed, predictor_seed=int(seed)),
            predictor=dict(configuration=self.params['predictor'], data_contract=dict(
                target_labels_used=False, pseudo_labels_used=True,
                source_prototypes='real fold-training-source labels only',
                validation_project_excluded_from_ipgdo_and_predictor_training=True),
                model_selection=dict(policy='all-source-project-folds-retained',
                    threshold_policy='per-fold-source-validation-gmean', refit_all_sources=False))))


def register_methods():
    from cpdp.methods.registry import register_method
    register_method('ipgdo_kan_jepa', MultiSourceMethod, replace=True)
