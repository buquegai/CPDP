"""Invariant tests for validation isolation and same-model thresholds."""
import sys
from pathlib import Path
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import v18_multisource as module
from cpdp.oversampling.types import AugmentationResult
from cpdp.types import ProjectFeatures, LabeledProject


def project(name):
    values = np.arange(16, dtype=float).reshape(8, 2) + len(name)
    return LabeledProject(ProjectFeatures('toy', name, values, ('x','loc'),
        values[:, 1], 'loc', np.arange(8)), np.array([0,0,0,0,1,1,1,1]))


def test_centering_preserves_fold_decisions_and_order():
    for threshold in [0.02, 0.2388, 0.5, 0.92]:
        scores = np.unique(np.r_[np.linspace(0, 1, 1001), threshold])
        centered = module.threshold_center(scores, threshold)
        assert np.array_equal(scores >= threshold, centered >= 0.5)
        assert np.all(np.diff(centered) > 0)
        assert module.threshold_center(np.array([threshold]), threshold)[0] == 0.5


def test_holdout_isolation_resume_and_one_training_source(tmp_path, monkeypatch):
    sources = (project('A'), project('B'))
    target = project('T').without_labels()
    seedmap = tmp_path/'seeds.yaml'
    seedmap.write_text(yaml.safe_dump({'target_labels_used': False,
        'targets': {target.key: {'best_seed': 71}}}))
    predictor = {'threshold_policy':'gmean', 'probability_calibration':'none',
                 'optimization':{'refit_all_sources':False}}
    params = {'predictor':predictor, 'oversampling':{},
              'oversampling_reuse':{'enabled':True,'seed_map':str(seedmap)},
              'multisource':{'artifact_dir':str(tmp_path/'artifacts'), 'implementation_signature':'test'}}
    augmented_keys, fit_keys = [], []
    def fake_resample(self, training, seed):
        assert seed == 71
        assert len(training) == 1
        augmented_keys.append(training[0].key)
        return AugmentationResult(tuple(training), {'target_data_used':False})
    monkeypatch.setattr(module.IPGDOOversampler, 'fit_resample', fake_resample)
    def fake_fit(self, training, validation, unlabeled, seed, directory):
        assert not hasattr(unlabeled, 'labels')
        assert validation.key not in [p.key for p in training]
        fit_keys.append((training[0].key, validation.key))
        scores=np.linspace(0.1,0.9,8)
        module.save_npz(directory/'predictions.npz',target_row_ids=unlabeled.features.row_ids,
            target_scores=scores,target_centered_scores=module.threshold_center(scores,0.3))
        (directory/'model.pt').write_bytes(b'test')
        return {'validation_project':validation.key,'best_joint_epoch':1,'threshold':0.3}
    monkeypatch.setattr(module.FoldPredictor, 'fit_fold', fake_fit)
    method=module.MultiSourceMethod(tmp_path,params)
    result=method.fit_predict(sources,target,12)
    assert fit_keys == [('toy/B','toy/A'),('toy/A','toy/B')]
    assert augmented_keys == ['toy/B','toy/A']
    again=method.fit_predict(sources,target,12)
    assert np.array_equal(result.scores,again.scores)
    assert len(fit_keys)==2
    # New prediction seed reuses only correctly scoped augmentation, not fold predictions.
    method.fit_predict(sources,target,13)
    assert len(fit_keys)==4 and len(augmented_keys)==2
    # A changed source label invalidates fold reuse instead of silently mixing results.
    altered=LabeledProject(sources[0].features,1-sources[0].labels)
    with pytest.raises(ValueError,match='different experiment'):
        method.fit_predict((altered,sources[1]),target,12)


def test_predictor_uses_selected_model_for_validation_and_target(tmp_path, monkeypatch):
    predictor=module.FoldPredictor.from_mapping({'device':'cpu',
        'threshold_policy':'gmean','optimization':{'refit_all_sources':False}})
    training=(project('A'),); validation=project('B'); target=project('T').without_labels()
    marker=SimpleNamespace(state_dict=lambda:{})
    arrays=SimpleNamespace(scaler=SimpleNamespace(center_=np.zeros(2),scale_=np.ones(2)),
        target_values=target.features.values, real_values=training[0].features.values,
        augmented_values=training[0].features.values, project_names=('toy/A',))
    outcome=SimpleNamespace(model=marker,arrays=arrays,best_joint_epoch=2,
        best_validation_loss=0.4,completed_joint_epochs=4,completed_pretrain_epochs=160,
        source_weights=np.array([1.]),final_losses={'classification':0.3})
    calls=[]
    def train(*args):
        calls.append('train'); assert args[-2].key=='toy/B'; return outcome
    def scores(torch, model, values, supplied_arrays, device):
        assert model is marker and supplied_arrays is arrays
        calls.append('scores'); return np.array([.1,.2,.3,.4,.6,.7,.8,.9])
    monkeypatch.setattr(predictor,'_train',train)
    monkeypatch.setattr(predictor,'_transform',lambda scaler,values:values)
    monkeypatch.setattr(predictor,'_scores',scores)
    result=predictor.fit_fold(training,validation,target,100,tmp_path)
    assert calls==['train','scores','scores']
    assert result['refit_all_sources'] is False
