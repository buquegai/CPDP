"""V18 source-project cross-validation ensemble; the v16 implementation is unchanged."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from cpdp.methods.base import BaseMethod, MethodProvenance
from cpdp.oversampling.pipeline import IPGDOOversampler
from cpdp.oversampling_seed_search.cache import (
    _atomic_json, load_augmentation_cache, mapping_signature,
    save_augmentation_cache, source_data_signature,
)
from cpdp.predictors.kan_jepa import KANJEPAPredictor, _resolve_device, _flush_batch_loss_log
from cpdp.predictors.threshold import select_metric_threshold
from cpdp.randomness import seed_everything
from cpdp.types import PredictionResult


def threshold_center(scores, threshold):
    """Monotone odds shift: a fold's source-only threshold maps exactly to 0.5.

    Outputs are decision scores, not calibrated defect probabilities.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Nonfinite or out-of-range fold probability')
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('G-mean threshold must lie strictly between zero and one')
    positive = scores * (1 - threshold)
    negative = (1 - scores) * threshold
    return positive / (positive + negative)


def target_feature_signature(target):
    """No target labels or label-dependent statistics enter the fingerprint."""
    digest = hashlib.sha256(target.key.encode())
    digest.update('\0'.join(target.features.feature_names).encode())
    for array in (target.features.values, target.features.row_ids):
        a = np.ascontiguousarray(array)
        digest.update(str(a.dtype).encode())
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def save_npz(path, **arrays):
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


class FoldPredictor(KANJEPAPredictor):
    def _validation_loss(self, *args, **kwargs):
        value = super()._validation_loss(*args, **kwargs)
        self.validation_curve.append(float(value))
        return value

    def fit_fold(self, sources, validation, target, prediction_seed, directory):
        """Train once and retain its best source-validation checkpoint for inference."""
        import torch

        self.validation_curve = []
        partitions = self._partition_sources(sources, target)
        validation_partition = self._partition_sources((validation,), target)[0]
        if validation.key in {p.key for p in partitions}:
            raise ValueError('Validation project leaked into predictor training')
        device = _resolve_device(torch, self.config.device, self.model_name)
        train_seed = int(prediction_seed + 104729)  # Same selection seed convention as v16.
        outcome = self._train(
            torch, partitions, target, device, train_seed,
            self.config.optimization.joint_epochs, validation_partition,
            'fold_' + validation.features.name,
        )
        validation_values = self._transform(outcome.arrays.scaler, validation_partition.real_values)
        validation_scores = self._scores(torch, outcome.model, validation_values, outcome.arrays, device)
        threshold = select_metric_threshold(validation_partition.real_labels, validation_scores, 'gmean')
        target_scores = self._scores(torch, outcome.model, outcome.arrays.target_values, outcome.arrays, device)
        centered = threshold_center(target_scores, threshold)
        save_npz(
            directory / 'predictions.npz',
            target_row_ids=target.features.row_ids, target_scores=target_scores,
            target_centered_scores=centered,
            validation_row_ids=validation.features.row_ids,
            validation_labels=validation_partition.real_labels, validation_scores=validation_scores,
        )
        temporary = directory / 'model.tmp.pt'
        torch.save({
            'state_dict': {k: v.detach().cpu() for k, v in outcome.model.state_dict().items()},
            'scaler_center': torch.tensor(outcome.arrays.scaler.center_),
            'scaler_scale': torch.tensor(outcome.arrays.scaler.scale_),
            'feature_names': list(target.features.feature_names),
            'configuration': asdict(self.config),
            'threshold': threshold, 'training_seed': train_seed,
            'best_joint_epoch': outcome.best_joint_epoch,
        }, temporary)
        os.replace(temporary, directory / 'model.pt')
        metadata = {
            'validation_project': validation.key,
            'validation_rows': len(validation_scores),
            'validation_defects': int(validation_partition.real_labels.sum()),
            'validation_curve': self.validation_curve,
            'best_validation_loss': outcome.best_validation_loss,
            'best_joint_epoch': outcome.best_joint_epoch,
            'completed_joint_epochs': outcome.completed_joint_epochs,
            'pretrain_epochs': outcome.completed_pretrain_epochs,
            'threshold': threshold, 'training_seed': train_seed,
            'configuration': asdict(self.config),
            'normalization': 'fold-training-real-source-only RobustScaler',
            'real_source_rows': len(outcome.arrays.real_values),
            'augmented_source_rows': len(outcome.arrays.augmented_values),
            'source_weights': dict(zip(outcome.arrays.project_names, outcome.source_weights.tolist())),
            'last_executed_epoch_losses_not_best_checkpoint_losses': outcome.final_losses,
            'target_score_min': float(target_scores.min()),
            'target_score_max': float(target_scores.max()),
            'target_positive_fraction_at_fold_threshold': float(np.mean(target_scores >= threshold)),
            'validation_score_min': float(validation_scores.min()),
            'validation_score_max': float(validation_scores.max()),
            'target_labels_used': False,
            'refit_all_sources': False,
        }
        del outcome
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return metadata


class MultiSourceMethod(BaseMethod):
    # Registered only by the v18 launcher; the ordinary v16 registry is untouched.
    name = 'ipgdo_kan_jepa'
    provenance = MethodProvenance(
        display_name='IPGDO + KAN-JEPA source-fold ensemble',
        implementation='Fold-training-only IPGDO; retain each source-validation model and its G-mean threshold',
        fidelity='proposed-method-variant', source_url='local', source_revision='v18-multisource-v1',
        target_label_policy='Target labels are used only by the common evaluator',
        notes='Equal mean of threshold-centered fold scores; these scores are not calibrated probabilities.',
    )

    def _augmentation(self, sources, validation, target, oversampling_seed):
        config = self.params['oversampling']
        identity = {
            'target': target.key, 'validation': validation.key,
            'source_signature': source_data_signature(sources),
            'oversampling_seed': int(oversampling_seed), 'oversampling': config,
            'implementation_signature': self.params['multisource']['implementation_signature'],
        }
        digest = mapping_signature(identity)[:20]
        cache = Path(self.params['multisource']['artifact_dir']) / 'augmentation' / target.key / (validation.features.name + '_' + digest + '.npz')
        if cache.is_file() and cache.with_suffix('.json').is_file():
            return load_augmentation_cache(cache, target_key=target.key,
                oversampling_seed=oversampling_seed, oversampling_config=config, sources=sources), cache
        seed_everything(oversampling_seed)
        augmentation = IPGDOOversampler.from_mapping(config).fit_resample(sources, oversampling_seed)
        if augmentation.metadata.get('target_data_used', False):
            raise ValueError('IPGDO unexpectedly used target data')
        if [s.key for s in augmentation.projects] != [s.key for s in sources]:
            raise ValueError('Augmentation project membership changed')
        save_augmentation_cache(cache, augmentation, target_key=target.key,
            oversampling_seed=oversampling_seed, oversampling_config=config, sources=sources)
        return augmentation, cache

    def fit_predict(self, sources, target, seed):
        sources = tuple(sources)
        if len(sources) < 2:
            raise ValueError('Multi-source validation requires two original source projects')
        probe = FoldPredictor.from_mapping(self.params['predictor'])
        probe._partition_sources(sources, target)
        if probe.config.optimization.refit_all_sources:
            raise ValueError('V18 retains fold models; refit_all_sources must be false')
        if probe.config.threshold_policy != 'gmean' or probe.config.probability_calibration != 'none':
            raise ValueError('V18 requires source G-mean and no probability calibration')
        if any(np.any(s.features.row_ids < 0) for s in sources):
            raise ValueError('Expected real sources before fold-specific IPGDO')
        seed_map_path = self.project_root / self.params['oversampling_reuse']['seed_map']
        seed_map = yaml.safe_load(seed_map_path.read_text())
        if seed_map.get('target_labels_used') is not False:
            raise ValueError('Seed map must declare no target-label selection')
        oversampling_seed = int(seed_map['targets'][target.key]['best_seed'])
        root = Path(self.params['multisource']['artifact_dir']) / 'folds' / target.key / f'seed_{seed}'
        full_signature = mapping_signature({
            'params': self.params, 'seed': int(seed), 'source_data': source_data_signature(sources),
            'target_features': target_feature_signature(target), 'oversampling_seed': oversampling_seed,
        })
        fold_metadata, target_score_columns, centered_columns = [], [], []
        outer_loss_path = os.environ.get('CPDP_PREDICTOR_BATCH_LOSS_LOG')
        try:
            for index, validation in enumerate(sources):
                training = tuple(s for s in sources if s.key != validation.key)
                directory = root / ('holdout_' + validation.features.name)
                directory.mkdir(parents=True, exist_ok=True)
                record_path = directory / 'result.json'
                record = None
                if record_path.is_file():
                    saved = json.loads(record_path.read_text())
                    if saved.get('signature') != full_signature:
                        raise ValueError('Refusing to reuse fold from a different experiment')
                    if all((directory / name).is_file() for name in ('predictions.npz', 'model.pt')):
                        record = saved
                if record is None:
                    print(f'FOLD {index+1}/{len(sources)} target={target.key} seed={seed} validation={validation.key}', flush=True)
                    started = time.perf_counter()
                    augmentation, cache = self._augmentation(training, validation, target, oversampling_seed)
                    loss_path = directory / 'batch_losses.jsonl'
                    loss_path.unlink(missing_ok=True)
                    os.environ['CPDP_PREDICTOR_BATCH_LOSS_LOG'] = str(loss_path)
                    seed_everything(seed)
                    predictor = FoldPredictor.from_mapping(self.params['predictor'])
                    details = predictor.fit_fold(augmentation.projects, validation, target, seed, directory)
                    _flush_batch_loss_log()
                    record = {
                        'status': 'success', 'signature': full_signature,
                        'fold_index': index, 'target': target.key, 'prediction_seed': int(seed),
                        'training_sources': [s.key for s in training],
                        'oversampling_seed': oversampling_seed,
                        'augmentation_cache': str(cache),
                        'augmentation_training_source_signature': source_data_signature(training),
                        'validation_excluded_from_ipgdo_fit': True,
                        'seed_selection': 'inherited v16 source-only seed, not reselected within folds',
                        'duration_seconds': time.perf_counter() - started,
                        **details,
                    }
                    _atomic_json(record_path, record)
                    print(f'FOLD_OK {index+1}/{len(sources)} target={target.key} epoch={details["best_joint_epoch"]} threshold={details["threshold"]:.6f}', flush=True)
                with np.load(directory / 'predictions.npz', allow_pickle=False) as archive:
                    if not np.array_equal(archive['target_row_ids'], target.features.row_ids):
                        raise ValueError('Saved target row IDs changed')
                    target_score_columns.append(archive['target_scores'].copy())
                    centered = threshold_center(archive['target_scores'], record['threshold'])
                    if not np.array_equal(centered, archive['target_centered_scores']):
                        raise ValueError('Saved centered fold scores are inconsistent')
                    centered_columns.append(centered)
                fold_metadata.append({**record, 'result_file': str(record_path)})
        finally:
            _flush_batch_loss_log()
            if outer_loss_path is None:
                os.environ.pop('CPDP_PREDICTOR_BATCH_LOSS_LOG', None)
            else:
                os.environ['CPDP_PREDICTOR_BATCH_LOSS_LOG'] = outer_loss_path
        raw = np.column_stack(target_score_columns)
        centered = np.column_stack(centered_columns)
        scores = centered.mean(axis=1)
        save_npz(root / 'ensemble_predictions.npz', target_row_ids=target.features.row_ids,
            raw_fold_scores=raw, centered_fold_scores=centered,
            thresholds=np.array([r['threshold'] for r in fold_metadata]),
            fold_validation_projects=np.array([r['validation_project'] for r in fold_metadata]),
            scores=scores, raw_mean_scores=raw.mean(axis=1))
        return PredictionResult(scores=scores, threshold=0.5, metadata={
            'variant': 'v18-multisource-v1',
            'score_semantics': 'mean of source-threshold-centered fold decision scores, NOT calibrated probabilities',
            'aggregation': 'equal_mean; >=0.5 predicts defect; equality predicts defect',
            'fold_count': len(sources), 'folds': fold_metadata,
            'target_positive_fraction': float(np.mean(scores >= 0.5)),
            'raw_mean_scores_file': str(root / 'ensemble_predictions.npz'),
            'random_seeds': {'oversampling_seed': oversampling_seed, 'predictor_seed': int(seed)},
            'predictor': {'configuration': self.params['predictor'],
                'data_contract': {'target_labels_used': False, 'pseudo_labels_used': False,
                    'validation_project_excluded_from_ipgdo_and_predictor_training': True},
                'model_selection': {'policy': 'all-source-project-folds-retained',
                    'threshold_policy': 'per-fold-source-validation-gmean',
                    'refit_all_sources': False}},
        })


def register_methods():
    from cpdp.methods.registry import register_method
    register_method('ipgdo_kan_jepa', MultiSourceMethod, replace=True)
