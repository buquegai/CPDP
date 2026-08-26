from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np

from cpdp.oversampling.config import DiffusionConfig


def resolve_torch_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for diffusion, but it is unavailable")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    return torch.device(requested)


def _cosine_betas(torch, timesteps: int, device):
    positions = torch.linspace(
        0, timesteps, timesteps + 1, dtype=torch.float64, device=device
    )
    cumulative = torch.cos(
        ((positions / timesteps) + 0.008) / 1.008 * math.pi * 0.5
    ) ** 2
    cumulative = cumulative / cumulative[0]
    betas = 1.0 - cumulative[1:] / cumulative[:-1]
    return torch.clamp(betas, 1e-5, 0.999).float()


class SinusoidalTimeEmbedding:
    @staticmethod
    def module(torch, dimension: int):
        nn = torch.nn

        class Embedding(nn.Module):
            def forward(self, timesteps):
                half = max(1, dimension // 2)
                exponent = -math.log(10000.0) * torch.arange(
                    half, device=timesteps.device, dtype=torch.float32
                ) / max(half - 1, 1)
                angles = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
                embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
                if embedding.shape[1] < dimension:
                    embedding = torch.nn.functional.pad(embedding, (0, 1))
                return embedding[:, :dimension]

        return Embedding()


def _build_denoiser(torch, input_dim: int, project_count: int, config: DiffusionConfig):
    nn = torch.nn

    class ResidualBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.normalization = nn.LayerNorm(config.hidden_dim)
            self.network = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim * 2),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            )

        def forward(self, values):
            return values + self.network(self.normalization(values))

    class ProjectConditionedDenoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(input_dim, config.hidden_dim)
            self.time_embedding = nn.Sequential(
                SinusoidalTimeEmbedding.module(torch, config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.project_embedding = nn.Embedding(project_count, config.hidden_dim)
            self.blocks = nn.ModuleList(
                [ResidualBlock() for _ in range(config.residual_blocks)]
            )
            self.output = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, input_dim),
            )

        def forward(self, values, timesteps, project_ids):
            hidden = (
                self.input_projection(values)
                + self.time_embedding(timesteps)
                + self.project_embedding(project_ids)
            )
            for block in self.blocks:
                hidden = block(hidden)
            return self.output(hidden)

    return ProjectConditionedDenoiser()


class ProjectConditionedTabularDDPM:
    """Project-conditioned DDPM trained only on real source defects."""

    def __init__(
        self,
        input_dim: int,
        project_count: int,
        config: DiffusionConfig,
        device: str,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "IPGDO requires PyTorch; install the project 'deep' extra"
            ) from exc
        if input_dim < 1 or project_count < 1:
            raise ValueError("input_dim and project_count must be positive")
        self.torch = torch
        self.input_dim = int(input_dim)
        self.project_count = int(project_count)
        self.config = config
        self.device = resolve_torch_device(torch, device)
        self.model = None
        self._sampling_generator = None

        if config.schedule == "cosine":
            betas = _cosine_betas(torch, config.timesteps, self.device)
        else:
            betas = torch.linspace(
                1e-4, 0.02, config.timesteps, device=self.device, dtype=torch.float32
            )
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_cumulative = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumulative_previous = torch.cat(
            (torch.ones(1, device=self.device), self.alpha_cumulative[:-1])
        )
        self.posterior_variance = torch.clamp(
            betas
            * (1.0 - self.alpha_cumulative_previous)
            / (1.0 - self.alpha_cumulative),
            min=1e-20,
        )
        self.posterior_clean_coefficient = (
            betas
            * torch.sqrt(self.alpha_cumulative_previous)
            / (1.0 - self.alpha_cumulative)
        )
        self.posterior_noisy_coefficient = (
            (1.0 - self.alpha_cumulative_previous)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alpha_cumulative)
        )
        self.training_metadata: Dict[str, Any] = {}

    def _extract(self, values, timesteps, shape):
        selected = values.gather(0, timesteps)
        return selected.reshape(len(timesteps), *((1,) * (len(shape) - 1)))

    def _add_noise(self, clean, timesteps, noise):
        alpha = self._extract(self.alpha_cumulative, timesteps, clean.shape)
        return _safe_sqrt(alpha) * clean + _safe_sqrt(1.0 - alpha) * noise

    def fit(
        self,
        normalized_defects: np.ndarray,
        project_ids: np.ndarray,
        seed: int,
        *,
        raw_defects: Optional[np.ndarray] = None,
        scaler_centers: Optional[np.ndarray] = None,
        scaler_scales: Optional[np.ndarray] = None,
        formula_guidance: Optional[Any] = None,
    ) -> Dict[str, Any]:
        values_array = np.asarray(normalized_defects, dtype=np.float32)
        projects_array = np.asarray(project_ids, dtype=np.int64)
        if values_array.ndim != 2 or values_array.shape[1] != self.input_dim:
            raise ValueError("normalized_defects has an incompatible shape")
        if len(values_array) == 0 or len(projects_array) != len(values_array):
            raise ValueError("Defect values and project IDs must be non-empty and aligned")
        if projects_array.min() < 0 or projects_array.max() >= self.project_count:
            raise ValueError("project_ids contains an unknown project")

        if formula_guidance is not None:
            if raw_defects is None:
                raise ValueError("raw_defects is required for formula guidance")
            if scaler_centers is None or scaler_scales is None:
                raise ValueError(
                    "scaler centers and scales are required for formula guidance"
                )
            raw_array = np.asarray(raw_defects, dtype=np.float32)
            centers_array = np.asarray(scaler_centers, dtype=np.float32)
            scales_array = np.asarray(scaler_scales, dtype=np.float32)
            if raw_array.shape != values_array.shape:
                raise ValueError("raw_defects must align with normalized_defects")
            if centers_array.shape != (self.project_count, self.input_dim):
                raise ValueError("scaler_centers has an incompatible shape")
            if scales_array.shape != (self.project_count, self.input_dim):
                raise ValueError("scaler_scales has an incompatible shape")
        else:
            raw_array = None
            centers_array = None
            scales_array = None

        torch = self.torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        self.model = _build_denoiser(
            torch, self.input_dim, self.project_count, self.config
        ).to(self.device)
        self._sampling_generator = torch.Generator(device=self.device.type)
        self._sampling_generator.manual_seed(int(seed) + 104729)

        values = torch.as_tensor(values_array, dtype=torch.float32)
        projects = torch.as_tensor(projects_array, dtype=torch.long)
        if raw_array is None:
            dataset = torch.utils.data.TensorDataset(values, projects)
        else:
            raw_values = torch.as_tensor(raw_array, dtype=torch.float32)
            dataset = torch.utils.data.TensorDataset(values, projects, raw_values)
        loader_generator = torch.Generator().manual_seed(int(seed))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.config.batch_size, len(dataset)),
            shuffle=True,
            generator=loader_generator,
            pin_memory=self.device.type == "cuda",
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs
        )
        amp_enabled = bool(self.config.use_amp and self.device.type == "cuda")
        gradient_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        if centers_array is not None:
            centers_tensor = torch.as_tensor(
                centers_array, dtype=torch.float32, device=self.device
            )
            scales_tensor = torch.as_tensor(
                scales_array, dtype=torch.float32, device=self.device
            )
        else:
            centers_tensor = None
            scales_tensor = None
        warmup_epochs = max(
            1,
            int(math.ceil(self.config.epochs * self.config.physics_warmup_fraction)),
        )
        minimum_total_loss = math.inf
        minimum_noise_loss = math.inf
        final_loss = math.nan
        final_noise_loss = math.nan
        final_physics_loss = 0.0
        final_consistency_loss = 0.0
        final_semantic_loss = 0.0
        report_every = max(1, self.config.epochs // 10)

        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0.0
            total_noise_loss = 0.0
            total_physics_loss = 0.0
            total_consistency_loss = 0.0
            total_semantic_loss = 0.0
            batches = 0
            for batch in loader:
                clean_cpu, project_cpu = batch[:2]
                clean = clean_cpu.to(self.device, non_blocking=True)
                batch_projects = project_cpu.to(self.device, non_blocking=True)
                timesteps = torch.randint(
                    0,
                    self.config.timesteps,
                    (len(clean),),
                    device=self.device,
                    generator=self._sampling_generator,
                )
                noise = torch.randn(
                    clean.shape,
                    dtype=clean.dtype,
                    device=self.device,
                    generator=self._sampling_generator,
                )
                noisy = self._add_noise(clean, timesteps, noise)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                    predicted_noise = self.model(noisy, timesteps, batch_projects)
                    noise_loss = torch.nn.functional.mse_loss(predicted_noise, noise)

                consistency_loss = noise_loss * 0.0
                semantic_loss = noise_loss * 0.0
                if (
                    formula_guidance is not None
                    and centers_tensor is not None
                    and scales_tensor is not None
                ):
                    alpha = self._extract(
                        self.alpha_cumulative, timesteps, clean.shape
                    )
                    predicted_clean = (
                        noisy - _safe_sqrt(1.0 - alpha) * predicted_noise
                    ) / _safe_sqrt(alpha)
                    predicted_raw = (
                        predicted_clean.float() * scales_tensor[batch_projects]
                        + centers_tensor[batch_projects]
                    )
                    clean_raw = batch[2].to(self.device, non_blocking=True)
                    valid_mask = torch.isfinite(predicted_raw).all(dim=1)
                    valid_mask = valid_mask & torch.isfinite(clean_raw).all(dim=1)
                    labels = torch.ones(
                        len(clean_raw), dtype=torch.float32, device=self.device
                    )
                    consistency_loss, semantic_loss = formula_guidance.losses(
                        predicted_raw,
                        clean_raw,
                        labels,
                        batch_projects,
                        valid_mask,
                    )
                physics_loss = (
                    self.config.formula_consistency_weight * consistency_loss
                    + self.config.formula_semantic_weight * semantic_loss
                )
                physics_scale = (
                    1.0
                    if self.config.physics_warmup_fraction == 0.0
                    else min(1.0, (epoch + 1) / warmup_epochs)
                )
                loss = noise_loss + physics_scale * physics_loss
                gradient_scaler.scale(loss).backward()
                gradient_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
                gradient_scaler.step(optimizer)
                gradient_scaler.update()
                total_loss += float(loss.detach().cpu())
                total_noise_loss += float(noise_loss.detach().cpu())
                total_physics_loss += float(physics_loss.detach().cpu())
                total_consistency_loss += float(consistency_loss.detach().cpu())
                total_semantic_loss += float(semantic_loss.detach().cpu())
                batches += 1

            scheduler.step()
            final_loss = total_loss / max(batches, 1)
            final_noise_loss = total_noise_loss / max(batches, 1)
            final_physics_loss = total_physics_loss / max(batches, 1)
            final_consistency_loss = total_consistency_loss / max(batches, 1)
            final_semantic_loss = total_semantic_loss / max(batches, 1)
            minimum_total_loss = min(minimum_total_loss, final_loss)
            minimum_noise_loss = min(minimum_noise_loss, final_noise_loss)
            if epoch == 0 or (epoch + 1) % report_every == 0 or epoch + 1 == self.config.epochs:
                print(
                    "PROGRESS ipgdo "
                    f"epoch={epoch + 1}/{self.config.epochs} "
                    f"noise_loss={final_noise_loss:.6f} "
                    f"physics_loss={final_physics_loss:.6f} "
                    f"physics_scale={physics_scale:.3f} "
                    f"total_loss={final_loss:.6f}",
                    flush=True,
                )

        self.training_metadata = {
            "device": str(self.device),
            "amp_enabled": amp_enabled,
            "training_samples": int(len(values_array)),
            "training_data": "real_source_defects_only",
            "objective": "ddpm_noise_mse_plus_frozen_formula_guidance",
            "formula_guidance_enabled": formula_guidance is not None,
            "formula_consistency_weight": float(
                self.config.formula_consistency_weight
            ),
            "formula_semantic_weight": float(self.config.formula_semantic_weight),
            "physics_warmup_fraction": float(self.config.physics_warmup_fraction),
            "parameters": int(
                sum(parameter.numel() for parameter in self.model.parameters())
            ),
            "configuration": asdict(self.config),
            "final_noise_loss": float(final_noise_loss),
            "final_total_loss": float(final_loss),
            "final_physics_loss": float(final_physics_loss),
            "final_ddpm_noise_loss": float(final_noise_loss),
            "final_formula_consistency_loss": float(final_consistency_loss),
            "final_formula_semantic_loss": float(final_semantic_loss),
            "minimum_total_loss": float(minimum_total_loss),
            "minimum_noise_loss": float(minimum_noise_loss),
        }
        return dict(self.training_metadata)

    def sample_defects(self, sample_count: int, project_id: int) -> np.ndarray:
        if sample_count < 1:
            return np.empty((0, self.input_dim), dtype=np.float64)
        if self.model is None or self._sampling_generator is None:
            raise RuntimeError("Diffusion model must be fitted before sampling")
        if not 0 <= project_id < self.project_count:
            raise ValueError(f"Unknown project_id: {project_id}")
        chunks = []
        for start in range(0, sample_count, self.config.sample_batch_size):
            chunk_size = min(self.config.sample_batch_size, sample_count - start)
            chunks.append(self._sample_chunk(chunk_size, project_id))
        return np.vstack(chunks).astype(np.float64, copy=False)

    def _sample_chunk(self, sample_count: int, project_id: int) -> np.ndarray:
        torch = self.torch
        if self.model is None or self._sampling_generator is None:
            raise RuntimeError("Diffusion model must be fitted before sampling")
        self.model.eval()
        values = torch.randn(
            (sample_count, self.input_dim),
            dtype=torch.float32,
            device=self.device,
            generator=self._sampling_generator,
        )
        projects = torch.full(
            (sample_count,), project_id, dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            for timestep in reversed(range(self.config.timesteps)):
                timestep_values = torch.full(
                    (sample_count,), timestep, dtype=torch.long, device=self.device
                )
                predicted_noise = self.model(values, timestep_values, projects)
                alpha_cumulative = self.alpha_cumulative[timestep]
                predicted_clean = (
                    values
                    - _safe_sqrt(1.0 - alpha_cumulative) * predicted_noise
                ) / _safe_sqrt(alpha_cumulative)
                predicted_clean = predicted_clean.clamp(
                    -self.config.normalized_clip, self.config.normalized_clip
                )
                mean = (
                    self.posterior_clean_coefficient[timestep] * predicted_clean
                    + self.posterior_noisy_coefficient[timestep] * values
                )
                if timestep > 0:
                    noise = torch.randn(
                        values.shape,
                        dtype=values.dtype,
                        device=self.device,
                        generator=self._sampling_generator,
                    )
                    values = mean + _safe_sqrt(
                        self.posterior_variance[timestep]
                    ) * noise
                else:
                    values = mean
                values = values.clamp(
                    -self.config.normalized_clip, self.config.normalized_clip
                )
        return values.cpu().numpy()


def _safe_sqrt(values):
    import torch

    return torch.sqrt(torch.clamp(values, min=1e-20))
