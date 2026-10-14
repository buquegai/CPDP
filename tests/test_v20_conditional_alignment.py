"""V20 numerical, training, source isolation, and cache/resume regression tests."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import v20_multisource as module
from v20_conditional_alignment import (
    ConditionalAlignmentConfig, ConditionalFoldPredictor, class_prototypes,
    conditional_prototype_loss, confident_agreement, update_epoch_teacher,
)
from cpdp.oversampling.types import AugmentationResult
from cpdp.oversampling_seed_search.cache import save_augmentation_cache
from cpdp.types import LabeledProject, ProjectFeatures


def project(name):
    values = np.array([[-2,-1],[-1,-2],[-2,-2],[-1,-1],[1,1],[2,1],[1,2],[2,2]], dtype=float)
    features = ProjectFeatures('toy', name, values, ('x','loc'), np.abs(values[:,1]), 'loc', np.arange(8))
    return LabeledProject(features, np.array([0,0,0,0,1,1,1,1]))


def test_confidence_two_view_agreement_and_nonfinite():
    first = torch.tensor([.95,.05,.95,.8,.95,float('nan')])
    second = torch.tensor([.94,.06,.05,.9,.85,.99])
    labels, accepted, certainty = confident_agreement(torch, first, second, .9)
    assert labels.tolist()[:5] == [1,0,1,1,1]
    assert accepted.tolist() == [True,True,False,False,False,False]
    assert torch.isfinite(certainty).all()


def test_project_balanced_real_class_prototypes_and_missing_class():
    latent = torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
    labels = torch.tensor([0,0,0,1])
    ids = torch.tensor([0,0,1,1])
    prototypes, valid = class_prototypes(torch, latent, labels, ids, torch.tensor([.25,.75]))
    expected = torch.nn.functional.normalize(torch.tensor([.25,.75]), dim=0)
    assert torch.allclose(prototypes[0], expected) and valid.tolist() == [True,True]
    _, missing = class_prototypes(torch, latent, labels*0, ids, torch.tensor([.25,.75]))
    assert missing.tolist() == [True,False]


def test_conditional_gradient_balanced_classes_and_empty_selection():
    latent = torch.tensor([[1.,.2],[.2,1.]], requires_grad=True)
    prototype = torch.tensor([[0.,1.],[1.,0.]], requires_grad=True)
    labels, selected, confidence = torch.tensor([0,1]), torch.tensor([True,True]), torch.tensor([.95,.96])
    loss, counts, classes = conditional_prototype_loss(torch, latent, labels, selected, confidence, prototype, torch.tensor([True,True]))
    loss.backward()
    assert 0 < loss < 2 and classes == 2 and counts.tolist() == [1,1]
    assert latent.grad.abs().sum() > 0 and prototype.grad is None
    empty = latent.detach().clone().requires_grad_(True)
    zero, counts, classes = conditional_prototype_loss(torch, empty, labels, ~selected, confidence, prototype, torch.tensor([True,True]))
    zero.backward()
    assert zero == 0 and classes == 0 and torch.isfinite(empty.grad).all()
    invalid, _, _ = conditional_prototype_loss(torch, empty, labels, selected, confidence, prototype, torch.tensor([False,False]))
    assert invalid == 0


def test_teacher_ema_ramp_and_config_validation():
    student = torch.nn.Linear(2,1)
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        student.weight.fill_(2); teacher.weight.fill_(0)
    update_epoch_teacher(torch, teacher, student, .9)
    assert torch.allclose(teacher.weight, torch.full_like(teacher.weight,.2))
    assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
    cfg = ConditionalAlignmentConfig()
    assert [cfg.ramp(e) for e in (1,20,21,30,40,300)] == [0,0,.05,.5,1,1]
    with pytest.raises(ValueError):
        ConditionalAlignmentConfig.from_mapping({'confidence':.4})
    with pytest.raises(ValueError):
        ConditionalAlignmentConfig.from_mapping({'unknown':1})


def test_full_toy_fold_trains_new_loss_and_saves_same_student_checkpoint(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    config = dict(device='cpu', threshold_policy='gmean', probability_calibration='none',
        data_flow_mode='shared_augmented_source', encoder=dict(hidden_dims=[8],latent_dim=4),
        classifier=dict(context_size=2,hidden_dim=8,candidate_batch_size=16),
        jepa_predictor=dict(hidden_dims=[8],mask_embedding_dim=4),
        conditional_alignment=dict(warmup_epochs=0,ramp_epochs=1),
        optimization=dict(pretrain_epochs=1,joint_epochs=2,batch_size=4,refit_all_sources=False,use_amp=False))
    predictor = ConditionalFoldPredictor.from_mapping(config)
    target = project('T').without_labels()
    def reliable(torch_module, values, context):
        labels = (values[:,0]>0).long()
        return labels, torch.ones(len(values),dtype=torch.bool), torch.full((len(values),),.95)
    monkeypatch.setattr(predictor, '_pseudo_labels', reliable)
    seen = []
    original_teacher = predictor._teacher_epoch
    def verify(*args):
        arrays = predictor._real_arrays
        assert arrays.project_names == ('toy/A',)
        assert len(np.concatenate(arrays.real_labels_by_project)) == 8
        seen.append(True)
        return original_teacher(*args)
    monkeypatch.setattr(predictor, '_teacher_epoch', verify)
    try:
        metadata = predictor.fit_fold((project('A'),), project('B'), target, 42, tmp_path)
    finally:
        torch.set_num_threads(previous_threads)
    assert seen and predictor._teacher is None
    assert metadata['target_labels_used'] is False and metadata['pseudo_labels_used'] is True
    assert all(d['active_batch_count']>0 and d['alignment_loss']>0 for d in metadata['conditional_alignment_diagnostics'])
    checkpoint = torch.load(tmp_path/'model.pt',map_location='cpu',weights_only=True)
    assert checkpoint['configuration']['conditional_alignment']['confidence'] == .9
    assert not any('teacher' in name for name in checkpoint['state_dict'])
    with np.load(tmp_path/'predictions.npz') as archive:
        assert np.array_equal(archive['target_centered_scores'], module.threshold_center(archive['target_scores'],metadata['threshold']))


def test_v18_cache_isolation_fixed_seed_equal_fusion_and_resume(tmp_path, monkeypatch):
    sources = (project('A'),project('B'))
    target = project('T').without_labels()
    seedmap = tmp_path/'seeds.yaml'
    seedmap.write_text(yaml.safe_dump(dict(target_labels_used=False,targets={target.key:dict(best_seed=71)})))
    params = dict(predictor=dict(threshold_policy='gmean',probability_calibration='none',optimization=dict(refit_all_sources=False)),
        oversampling={}, oversampling_reuse=dict(seed_map=str(seedmap)),
        multisource=dict(artifact_dir=str(tmp_path/'v20'),implementation_signature='v20-test',
            augmentation_reuse=dict(artifact_dir=str(tmp_path/'v18'),implementation_signature='v18-test')))
    for validation in sources:
        training = tuple(s for s in sources if s.key!=validation.key)
        cache = module.augmentation_path(training,validation,target,71,params)
        save_augmentation_cache(cache,AugmentationResult(training,dict(target_data_used=False)),
            target_key=target.key,oversampling_seed=71,oversampling_config={},sources=training)
    calls = []
    def fake_fit(self,training,validation,unlabeled,seed,directory):
        assert not hasattr(unlabeled,'labels') and validation.key not in [s.key for s in training]
        calls.append((training[0].key,validation.key,seed))
        scores = np.linspace(.1,.9,8)
        threshold = .3 if validation.key.endswith('/A') else .6
        module.save_npz(directory/'predictions.npz',target_row_ids=unlabeled.features.row_ids,
            target_scores=scores,target_centered_scores=module.threshold_center(scores,threshold))
        (directory/'model.pt').write_bytes(b'test')
        return dict(validation_project=validation.key,best_joint_epoch=1,threshold=threshold)
    monkeypatch.setattr(module.FoldPredictor,'fit_fold',fake_fit)
    method = module.MultiSourceMethod(tmp_path,params)
    result = method.fit_predict(sources,target,12)
    assert calls == [('toy/B','toy/A',12),('toy/A','toy/B',12)]
    expected = (module.threshold_center(np.linspace(.1,.9,8),.3)+module.threshold_center(np.linspace(.1,.9,8),.6))/2
    assert np.array_equal(result.scores,expected) and result.threshold == .5
    assert result.metadata['random_seeds']['oversampling_seed'] == 71
    again = method.fit_predict(sources,target,12)
    assert len(calls)==2 and np.array_equal(again.scores,result.scores)
    method.fit_predict(sources,target,13)
    assert len(calls)==4
    with pytest.raises(ValueError,match='another experiment'):
        method.fit_predict((LabeledProject(sources[0].features,1-sources[0].labels),sources[1]),target,12)
    old_cache = module.augmentation_path((sources[0],),sources[1],target,71,params)
    with pytest.raises(ValueError,match='source_data_signature'):
        module.validate_cache_metadata(old_cache,(LabeledProject(sources[0].features,1-sources[0].labels),),target,71,{})
    with pytest.raises(FileNotFoundError,match='augmentation missing'):
        method.fit_predict((LabeledProject(sources[0].features,1-sources[0].labels),sources[1]),target,14)
