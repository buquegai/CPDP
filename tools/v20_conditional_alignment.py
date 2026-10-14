"""V20: source-class prototypes and confidence-gated, unlabeled target alignment."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np

from cpdp.predictors.kan_jepa import (
    KANJEPAConfig, _cycle, _flush_batch_loss_log, _record_batch_losses,
)
from v18_multisource import FoldPredictor as V18FoldPredictor


@dataclass(frozen=True)
class ConditionalAlignmentConfig:
    confidence: float = 0.9
    warmup_epochs: int = 20
    ramp_epochs: int = 20
    teacher_epoch_momentum: float = 0.9
    noise_std: float = 0.02

    @classmethod
    def from_mapping(cls, values):
        if set(values) - set(cls.__dataclass_fields__):
            raise ValueError('Unknown conditional alignment option')
        config = cls(**values)
        if not 0.5 < config.confidence < 1:
            raise ValueError('confidence must lie between 0.5 and 1')
        if (not isinstance(config.warmup_epochs, int) or config.warmup_epochs < 0
                or not isinstance(config.ramp_epochs, int) or config.ramp_epochs < 1):
            raise ValueError('Invalid warmup/ramp epoch count')
        if not 0 <= config.teacher_epoch_momentum < 1 or config.noise_std < 0:
            raise ValueError('Invalid teacher momentum or noise')
        return config

    def ramp(self, epoch):
        return min(1.0, max(0.0, (epoch - self.warmup_epochs) / self.ramp_epochs))


@dataclass(frozen=True)
class ConditionalKANJEPAConfig(KANJEPAConfig):
    conditional_alignment: ConditionalAlignmentConfig = ConditionalAlignmentConfig()


def confident_agreement(torch, first, second, confidence):
    labels = (first >= 0.5).long()
    certainty = torch.minimum(torch.maximum(first, 1-first), torch.maximum(second, 1-second))
    accepted = (torch.isfinite(first) & torch.isfinite(second)
                & (labels == (second >= 0.5).long()) & (certainty >= confidence))
    return labels, accepted, torch.nan_to_num(certainty, nan=0.0, posinf=0.0, neginf=0.0)


def class_prototypes(torch, latent, labels, project_ids, project_weights):
    """Equal rows within each project/class, then source-distance project weights."""
    latent = torch.nn.functional.normalize(latent.detach().float(), dim=1)
    prototypes, valid = [], []
    for category in (0, 1):
        means, weights = [], []
        for project in range(len(project_weights)):
            selected = (labels == category) & (project_ids == project)
            if selected.any():
                means.append(latent[selected].mean(0))
                weights.append(project_weights[project])
        if means:
            weight = torch.stack(weights)
            center = (torch.stack(means) * weight[:, None]).sum(0) / weight.sum().clamp_min(1e-8)
            usable = bool(torch.isfinite(center).all() and center.norm() > 1e-8)
            prototypes.append(torch.nn.functional.normalize(center, dim=0))
            valid.append(usable)
        else:
            prototypes.append(latent.new_zeros(latent.shape[1]))
            valid.append(False)
    return torch.stack(prototypes).detach(), torch.tensor(valid, device=latent.device)


def conditional_prototype_loss(torch, latent, labels, accepted, certainty, prototypes, valid):
    """Only the student receives gradients; class means have equal loss weight."""
    latent = torch.nn.functional.normalize(latent.float(), dim=1)
    accepted = accepted & valid[labels]
    losses, counts = [], []
    for category in (0, 1):
        selected = accepted & (labels == category)
        counts.append(selected.sum())
        if selected.any():
            weight = certainty[selected].detach()
            center = (latent[selected] * weight[:, None]).sum(0) / weight.sum().clamp_min(1e-8)
            center = torch.nn.functional.normalize(center, dim=0)
            losses.append((1 - (center * prototypes[category].detach()).sum()).clamp(0, 2))
    loss = torch.stack(losses).mean() if losses else latent.sum() * 0
    return loss, torch.stack(counts), len(losses)


def update_epoch_teacher(torch, teacher, student, momentum):
    student_parameters = dict(student.named_parameters())
    student_buffers = dict(student.named_buffers())
    with torch.no_grad():
        for name, parameter in teacher.named_parameters():
            parameter.mul_(momentum).add_(student_parameters[name].detach(), alpha=1-momentum)
        for name, buffer in teacher.named_buffers():
            buffer.copy_(student_buffers[name])
    teacher.requires_grad_(False)
    teacher.eval()


class ConditionalFoldPredictor(V18FoldPredictor):
    progress_name = 'v20_conditional_kan_jepa'

    @classmethod
    def from_mapping(cls, values):
        original = dict(values)
        alignment = ConditionalAlignmentConfig.from_mapping(original.pop('conditional_alignment', {}))
        base = KANJEPAConfig.from_mapping(original)
        return cls(ConditionalKANJEPAConfig(**base.__dict__, conditional_alignment=alignment))

    def _loaders(self, torch, arrays, seed, pin_memory):
        self._real_arrays = arrays
        self._noise_seed = seed + 6001
        self._teacher = None
        self._noise_generator = None
        self.alignment_history = []
        return super()._loaders(torch, arrays, seed, pin_memory)

    def fit_fold(self, *args, **kwargs):
        try:
            metadata = super().fit_fold(*args, **kwargs)
            metadata['conditional_alignment_diagnostics'] = self.alignment_history
            metadata['best_checkpoint_after_alignment_warmup'] = (
                metadata['best_joint_epoch'] > self.config.conditional_alignment.warmup_epochs)
            metadata['alignment_method'] = 'confidence-gated class prototype cosine alignment'
            metadata['pseudo_labels_used'] = True
            return metadata
        finally:
            self._teacher = None
            self._real_arrays = None
            self._noise_generator = None

    def _teacher_epoch(self, torch, model, classification_context, weights, device, ramp):
        cfg = self.config.conditional_alignment
        if self._teacher is None:
            self._teacher = copy.deepcopy(model).requires_grad_(False).eval()
            self._noise_generator = torch.Generator(device=device).manual_seed(self._noise_seed)
        else:
            update_epoch_teacher(torch, self._teacher, model, cfg.teacher_epoch_momentum)
        if ramp == 0:
            return None, None, None
        with torch.no_grad():
            context = self._refresh_classification_context(self._teacher, classification_context)
            arrays = self._real_arrays
            outputs = []
            for start in range(0, len(arrays.real_values), self.config.optimization.batch_size):
                values = torch.from_numpy(arrays.real_values[start:start+self.config.optimization.batch_size]).to(device)
                outputs.append(self._teacher.context_latent(values))
            latent = torch.cat(outputs)
            labels = torch.from_numpy(np.concatenate(arrays.real_labels_by_project).astype(np.int64)).to(device)
            projects = torch.from_numpy(arrays.real_project_ids).to(device)
            prototypes, valid = class_prototypes(torch, latent, labels, projects, weights)
        return context, prototypes, valid

    def _pseudo_labels(self, torch, values, teacher_context):
        probabilities = []
        with torch.no_grad():
            for _ in range(2):
                noise = torch.randn(values.shape, dtype=values.dtype, device=values.device, generator=self._noise_generator)
                view = values + noise * self.config.conditional_alignment.noise_std
                probabilities.append(torch.sigmoid(self._classification_logits(self._teacher, view, teacher_context, None)).float())
        return confident_agreement(torch, *probabilities, self.config.conditional_alignment.confidence)

    # Frozen copy of the V18 epoch; only the former weighted-CORAL block is replaced.
    def _joint_epoch(self, torch, model, optimizer, gradient_scaler, loaders,
                     classification_context, source_weights, device, amp_enabled,
                     stage_name, epoch, epoch_count, total_steps):
        model.train()
        model.target_encoder.eval()
        iterators = {name: _cycle(loader) for name, loader in loaders.items()}
        steps = max(len(loader) for loader in loaders.values())
        weights = torch.as_tensor(source_weights, dtype=torch.float32, device=device)
        ramp = self.config.conditional_alignment.ramp(epoch)
        teacher_context, prototypes, valid = self._teacher_epoch(
            torch, model, classification_context, weights, device, ramp)
        loss_names = ['total', 'jepa', 'classification', 'variance']
        for name in ('alignment', 'consistency', 'covariance'):
            if getattr(self.config.losses, name) > 0:
                loss_names.append(name)
        totals = {name: 0.0 for name in loss_names}
        diagnostics = np.zeros(6, dtype=np.float64)
        active_steps = 0
        for batch_index in range(1, steps+1):
            step_index = (epoch-1)*steps + batch_index-1
            self._set_optimizer_learning_rate(optimizer, self._learning_rate_for_step(step_index, total_steps))
            ssl_values = next(iterators['ssl'])[0].to(device, non_blocking=True)
            class_values, class_labels, class_projects, class_indices = (
                value.to(device, non_blocking=True) for value in next(iterators['classification']))
            if self.config.losses.alignment > 0:
                next(iterators['real'])  # Preserve V18 sampler advancement and epoch length.
            target_values = next(iterators['target'])[0].to(device, non_blocking=True)
            context = self._refresh_classification_context(model, classification_context)
            optimizer.zero_grad(set_to_none=True)
            counts = weights.new_zeros(2)
            certainty = weights.new_zeros(len(target_values))
            accepted = torch.zeros(len(target_values), dtype=torch.bool, device=device)
            active_classes = 0
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                jepa, variance, covariance = self._jepa_terms(torch, model, ssl_values,
                    include_covariance=self.config.losses.covariance > 0)
                logits = self._classification_logits(model, class_values, context, class_indices)
                row_losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, class_labels, reduction='none')
                class_weights = weights[class_projects]
                classification = (row_losses*class_weights).sum() / class_weights.sum().clamp_min(1e-8)
                alignment = jepa.new_zeros(())
                if self.config.losses.alignment > 0 and ramp > 0:
                    pseudo, accepted, certainty = self._pseudo_labels(torch, target_values, teacher_context)
                    accepted = accepted & valid[pseudo]
                    target_latent = model.context_latent(target_values)
                    alignment, counts, active_classes = conditional_prototype_loss(
                        torch, target_latent, pseudo, accepted, certainty, prototypes, valid)
                    alignment = alignment * ramp
                consistency = jepa.new_zeros(())
                if self.config.losses.consistency > 0:
                    first_view = target_values + torch.randn_like(target_values)*self.config.views.consistency_noise_std
                    second_view = target_values + torch.randn_like(target_values)*self.config.views.consistency_noise_std
                    first_probability = torch.sigmoid(self._classification_logits(model, first_view, context, None))
                    second_probability = torch.sigmoid(self._classification_logits(model, second_view, context, None))
                    consistency = torch.nn.functional.mse_loss(first_probability, second_probability)
                loss = (self.config.losses.jepa*jepa + self.config.losses.classification*classification
                        + self.config.losses.alignment*alignment + self.config.losses.consistency*consistency
                        + self.config.losses.variance*variance + self.config.losses.covariance*covariance)
            self._optimizer_step(torch, model, optimizer, gradient_scaler, loss)
            loss_by_name = dict(total=loss, jepa=jepa, classification=classification,
                variance=variance, alignment=alignment, consistency=consistency, covariance=covariance)
            diagnostic_tensors = [counts[0], counts[1], certainty.sum(), accepted.sum(),
                                  (certainty*accepted).sum(), certainty.new_tensor(len(target_values))]
            numeric = torch.stack([loss_by_name[name].detach().float() for name in loss_names]
                                  + [value.detach().float() for value in diagnostic_tensors]).cpu().tolist()
            batch_losses = dict(zip(loss_names, numeric[:len(loss_names)]))
            diagnostics += np.array(numeric[len(loss_names):])
            active_steps += int(active_classes > 0)
            for name, value in batch_losses.items():
                if not np.isfinite(value):
                    raise FloatingPointError('Nonfinite V20 loss: '+name)
                totals[name] += value
            _record_batch_losses(dict(schema_version=2, stage=stage_name, phase='joint',
                epoch=epoch, epoch_count=epoch_count, batch=batch_index, batch_count=steps,
                batch_size=len(class_values), learning_rate=float(optimizer.param_groups[0]['lr']),
                losses=batch_losses, conditional_alignment=dict(ramp=ramp,
                    accepted_class_0=int(numeric[-6]), accepted_class_1=int(numeric[-5]),
                    target_batch_size=len(target_values), active_classes=active_classes)))
        _flush_batch_loss_log()
        diagnostics = diagnostics.tolist()
        history = dict(epoch=epoch, ramp=ramp, accepted_class_0=int(diagnostics[0]),
            accepted_class_1=int(diagnostics[1]), candidate_exposures=int(diagnostics[5]) if ramp > 0 else 0,
            accepted_exposures=int(diagnostics[3]), active_batch_count=active_steps,
            coverage=diagnostics[3]/max(diagnostics[5], 1),
            mean_confidence=diagnostics[2]/max(diagnostics[5], 1),
            mean_accepted_confidence=diagnostics[4]/max(diagnostics[3], 1),
            source_prototype_valid=None if valid is None else valid.cpu().tolist(),
            alignment_loss=totals.get('alignment', 0)/steps)
        self.alignment_history.append(history)
        if epoch == self.config.conditional_alignment.warmup_epochs+1 or epoch % 10 == 0:
            print(f'CONDITIONAL epoch={epoch} ramp={ramp:.2f} coverage={history["coverage"]:.4f} '
                  f'class0={history["accepted_class_0"]} class1={history["accepted_class_1"]}', flush=True)
        return {name: value/steps for name, value in totals.items()}
