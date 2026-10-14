"""Run V20 class-conditional alignment against the frozen V18 baseline."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))

import yaml
from cpdp.config import load_config
from cpdp.data import load_all_projects, sources_for_target
from cpdp.oversampling_seed_search.cache import _atomic_json, source_data_signature
from cpdp.runner import ExperimentRunner, validate_protocol
from cpdp.result_store import ResultStore
from v20_multisource import augmentation_path, validate_cache_metadata, register_methods, target_feature_signature
from v20_conditional_alignment import ConditionalAlignmentConfig

BASELINE = ROOT / 'configs/v18_multisource_dc9821cf4e36.yaml'
SEEDMAP = ROOT / 'artifacts/ipgdo_ratio0p7/best_oversampling_seeds.yaml'
SNAPSHOT = ROOT / 'outputs/server_runs/formal/full/v18_multisource_dc9821cf4e36/manifest.json'
SMOKE_TARGETS = ['NASA/PC1', 'AEEEM/LC', 'ReLink/Apache', 'PROMISE/ant-1.7']


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(smoke):
    snapshot = json.loads(SNAPSHOT.read_text())
    for name, expected in snapshot['code_hashes'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('V18 baseline code no longer matches: ' + name)
    files = dict(snapshot['code_hashes'])
    for name in ['configs/v18_multisource_dc9821cf4e36.yaml',
                 'tools/v20_conditional_alignment.py', 'tools/v20_multisource.py',
                 'run_v20_conditional_alignment.py', 'tests/test_v20_conditional_alignment.py',
                 'docs/v20_conditional_alignment_protocol.md']:
        files[name] = digest(ROOT / name)
    implementation = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    name = 'v20_conditional_' + ('smoke_' if smoke else '') + implementation[:12]
    output = ROOT / 'outputs/server_runs/formal/full' / name
    raw = yaml.safe_load(BASELINE.read_text())
    original = copy.deepcopy(raw['profiles']['full']['ipgdo_kan_jepa'])
    params = copy.deepcopy(original)
    params['result_signature'] = name
    params['predictor']['conditional_alignment'] = asdict(ConditionalAlignmentConfig())
    params['multisource'] = {'implementation_signature': implementation,
        'artifact_dir': str(output / 'artifacts'),
        'fold_policy': 'leave-each-source-project-out',
        'aggregation': 'equal_mean_of_threshold_centered_scores',
        'augmentation_reuse': {key: original['multisource'][key]
                              for key in ('artifact_dir', 'implementation_signature')}}
    if smoke:
        params['predictor']['optimization'].update(pretrain_epochs=1, joint_epochs=2, early_stopping_patience=1)
        params['predictor']['conditional_alignment'].update(warmup_epochs=0, ramp_epochs=1)
        raw['experiment']['repetitions'] = 1
        raw['experiment']['seeds'] = raw['experiment']['seeds'][:1]
    raw['profiles'] = {'full': {'ipgdo_kan_jepa': params}}
    raw['experiment']['output_dir'] = str(output)
    raw['experiment']['save_predictions'] = True
    # No existing V18 predictor, IPGDO, threshold, or fusion setting may change.
    if not smoke:
        comparison = copy.deepcopy(params)
        comparison['multisource'] = original['multisource']
        comparison['predictor'].pop('conditional_alignment')
        comparison['result_signature'] = original['result_signature']
        if comparison != original:
            raise AssertionError('Unexpected V18 baseline parameter change')
    baseline_config = load_config(BASELINE)
    validate_protocol(baseline_config)
    projects = load_all_projects(baseline_config.data)
    seed_map = yaml.safe_load(SEEDMAP.read_text())
    if seed_map.get('target_labels_used') is not False:
        raise ValueError('Seed map target-label boundary invalid')
    fold_plan = []
    for family, project in baseline_config.targets():
        key = family + '/' + project
        if smoke and key not in SMOKE_TARGETS:
            continue
        sources = tuple(sources_for_target(projects, family, project))
        if len(sources) < 2:
            raise ValueError('Need at least two sources before constructing folds')
        for validation in sources:
            training = tuple(s for s in sources if s.key != validation.key)
            if not training or key in {s.key for s in training}:
                raise ValueError('Invalid fold')
            cache = augmentation_path(training, validation, projects[key].without_labels(),
                int(seed_map['targets'][key]['best_seed']), params)
            validate_cache_metadata(cache, training, projects[key].without_labels(),
                int(seed_map['targets'][key]['best_seed']), params['oversampling'])
            fold_plan.append({'target': key, 'validation': validation.key,
                'training_sources': [s.key for s in training],
                'training_source_signature': source_data_signature(training),
                'oversampling_seed': int(seed_map['targets'][key]['best_seed']),
                'verified_v18_augmentation_cache': str(cache),
                'real_training_class_counts': [int(sum((s.labels == c).sum() for s in training)) for c in (0, 1)]})
    targets = list(dict.fromkeys(p['target'] for p in fold_plan))
    if not smoke and (len(targets) != 25 or len(fold_plan) != 178):
        raise AssertionError('Expected 25 targets and 178 folds per prediction seed')
    manifest = {
        'variant': 'v20-conditional-v1', 'baseline': 'v18/dc9821cf4e36',
        'created_at': now(), 'status': 'prepared', 'output_dir': str(output),
        'baseline_snapshot': str(SNAPSHOT.parent), 'code_hashes': files,
        'implementation_signature': implementation, 'smoke_only': smoke,
        'target_labels': 'evaluation-only', 'target_count': len(targets),
        'prediction_seeds': raw['experiment']['seeds'],
        'expected_results': len(targets) * len(raw['experiment']['seeds']),
        'folds_per_prediction_seed': len(fold_plan),
        'expected_fold_fits': len(fold_plan) * len(raw['experiment']['seeds']),
        'fold_plan': fold_plan,
        'target_feature_signatures': {k: target_feature_signature(projects[k].without_labels()) for k in targets},
        'seed_map_note': 'Fixed v16 source-only seeds inherited; no nested fold-specific seed reselection.',
        'data_cleaning_note': 'v16 master-style dataset cleaning preserved; unchanged protocol limitation.',
        'score_semantics': 'decision scores, not calibrated probabilities',
        'conditional_alignment': params['predictor']['conditional_alignment'],
        'changed_component': 'replace weighted CORAL with confidence-gated class prototype alignment',
        'augmentation_policy': 'read-only reuse of validated V18 fold caches; retrain every predictor',
        'metrics': raw['experiment']['metrics'], 'targets': targets,
    }
    return raw, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    raw, manifest = prepare(args.smoke)
    output = Path(manifest['output_dir'])
    config_path = ROOT / 'configs' / (output.name + '.yaml')
    if args.dry_run:
        print(json.dumps({k: v for k, v in manifest.items() if k not in ('code_hashes','fold_plan','target_feature_signatures')}, indent=2))
        print('DRY_RUN_OK: V18 fingerprints/settings unchanged, source isolation, 0.7 ratio, fixed seeds and all fold caches verified')
        return 0
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another V20 launcher holds this experiment lock')
        (output / 'launcher.pid').write_text(str(os.getpid()) + '\n')
        if config_path.exists():
            if yaml.safe_load(config_path.read_text()) != raw:
                raise ValueError('Existing runtime configuration mismatch')
        else:
            config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8')
        manifest['runtime_config'] = str(config_path)
        previous = output / 'manifest.json'
        if previous.exists():
            old = json.loads(previous.read_text())
            if old['implementation_signature'] != manifest['implementation_signature']:
                raise ValueError('Experiment identity mismatch')
            manifest['created_at'] = old['created_at']
        manifest.update(status='running', started_or_resumed_at=now(), pid=os.getpid())
        _atomic_json(previous, manifest)
        register_methods()
        config = load_config(config_path)
        runner = ExperimentRunner(config)
        try:
            # Complete all projects for one seed before moving to the next seed.
            for seed in config.seeds:
                report = runner.run(['ipgdo_kan_jepa'], profile='full',
                    targets=manifest['targets'], seeds=[seed], workers=1, fail_fast=True)
                ResultStore(output, True).summarize(expected_seeds=config.seeds, expected_repetitions=config.repetitions)
                manifest['last_finished_prediction_seed'] = seed
                manifest['last_seed_report'] = asdict(report)
                _atomic_json(previous, manifest)
            results = [json.loads(p.read_text()) for p in (output/'runs').glob('*/*/*/*/seed_*.json')]
            successes = sum(r['status'] == 'success' for r in results)
            if successes != manifest['expected_results']:
                raise RuntimeError(f'Incomplete output: {successes}/{manifest["expected_results"]}')
            manifest.update(status='completed', finished_at=now(), completed_successes=successes)
        except BaseException as exc:
            manifest.update(status='interrupted_or_failed', finished_at=now(), error=repr(exc))
            raise
        finally:
            _atomic_json(previous, manifest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
