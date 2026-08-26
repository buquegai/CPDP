from __future__ import annotations

from typing import Sequence

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

from cpdp.methods.base import BaseMethod, MethodProvenance
from cpdp.types import LabeledProject, PredictionResult, UnlabeledProject
from cpdp.vendor import load_module


def _resolve_torch_device(torch, requested: str):
    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda" or normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "FD-HDP-U requested CUDA, but the installed PyTorch build cannot access CUDA"
            )
        device = torch.device(normalized)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {normalized}")
        return device
    raise ValueError("FD-HDP-U device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")


class FDHDPUMethod(BaseMethod):
    name = "fd_hdp_u"
    provenance = MethodProvenance(
        display_name="FD-HDP-U",
        implementation="Unlabeled-target adapter using official FD-HDP modules and losses",
        fidelity="official-core-unlabeled-adaptation",
        source_url="https://doi.org/10.5281/zenodo.15332940",
        source_revision="70a012f / Zenodo record 15332941",
        target_label_policy="Target labels and target-domain SMOTE are structurally unavailable",
        notes=(
            "Original FD-HDP is semi-supervised: its code SMOTEs target labels, tunes alpha on "
            "target labels, and fine-tunes/evaluates with target labels. Under the agreed unlabeled "
            "LOPO protocol this adapter trains the shared defect predictor from source labels only, "
            "uses unlabeled target features for reconstruction/domain losses, applies SMOTE to the "
            "source only, and fixes alpha_T=1.0. It must be reported as FD-HDP-U, not original FD-HDP."
        ),
    )

    def _modules(self):
        snapshots = tuple(
            (self.project_root / "third_party" / "original" / "FD-HDP-HDP").glob(
                "shiyanshi-yu-FD-HDP-*"
            )
        )
        if len(snapshots) != 1:
            raise RuntimeError(f"Expected one FD-HDP source snapshot, found {len(snapshots)}")
        root = snapshots[0] / "FD-HDP" / "FD-HDP"
        return {
            "preprocessing": load_module(
                "cpdp_vendor_fd_preprocessing", root / "data_preprocessing.py"
            ),
            "features": load_module(
                "cpdp_vendor_fd_features", root / "feature_disentanglement.py"
            ),
            "prediction": load_module(
                "cpdp_vendor_fd_prediction", root / "prediction_layer.py"
            ),
            "losses": load_module("cpdp_vendor_fd_losses", root / "loss_functions.py"),
        }

    @staticmethod
    def _source_smote(
        values: np.ndarray, labels: np.ndarray, seed: int
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        counts = np.bincount(labels, minlength=2)
        minority = int(np.argmin(counts))
        majority = int(np.argmax(counts))
        target_count = int(counts[majority] * 0.8)
        if target_count <= counts[minority] or counts[minority] < 2:
            return values, labels, False
        sampler = SMOTE(
            sampling_strategy={minority: target_count},
            k_neighbors=min(5, int(counts[minority]) - 1),
            random_state=seed % (2**32 - 1),
        )
        balanced_x, balanced_y = sampler.fit_resample(values, labels)
        return balanced_x, balanced_y, True

    def fit_predict(
        self,
        sources: Sequence[LabeledProject],
        target: UnlabeledProject,
        seed: int,
    ) -> PredictionResult:
        if not sources:
            raise ValueError("FD-HDP-U requires at least one source project")
        import torch

        requested_device = str(self.params.get("device", "auto"))
        device = _resolve_torch_device(torch, requested_device)
        modules = self._modules()
        source_x = np.vstack([source.features.values for source in sources])
        source_y = np.concatenate([source.labels for source in sources])
        source_x, source_y, smote_applied = self._source_smote(source_x, source_y, seed)
        target_x = target.features.values.copy()

        source_scaled = MinMaxScaler().fit_transform(source_x)
        target_scaled = MinMaxScaler().fit_transform(target_x)
        compressed_dimension = int(self.params.get("compressed_dim", 128))
        if compressed_dimension != 128:
            raise ValueError(
                "The official FD-HDP Predictor hard-codes a 128-dimensional input"
            )

        source_compressor = modules["preprocessing"].MLPFeatureCompression(
            input_dim=source_scaled.shape[1], compressed_dim=compressed_dimension
        ).to(device)
        target_compressor = modules["preprocessing"].MLPFeatureCompression(
            input_dim=target_scaled.shape[1], compressed_dim=compressed_dimension
        ).to(device)
        with torch.no_grad():
            source_tensor = source_compressor(
                torch.as_tensor(source_scaled, dtype=torch.float32, device=device)
            ).detach()
            target_tensor = target_compressor(
                torch.as_tensor(target_scaled, dtype=torch.float32, device=device)
            ).detach()
        source_labels = torch.as_tensor(source_y, dtype=torch.float32, device=device)

        disentanglement = modules["features"].FeatureDisentanglement(
            d_in=compressed_dimension, hidden_dims=[64, 32], disentangle_dim=64
        ).to(device)
        reconstructor = modules["features"].FeatureReconstructor(
            d_in=compressed_dimension,
            h_specific_dim=compressed_dimension,
            h_share_dim=compressed_dimension,
        ).to(device)
        domain_classifier = modules["features"].DomainSharedClassifier(
            disentangle_dim=compressed_dimension
        ).to(device)
        predictor = modules["prediction"].Predictor(
            d_in=compressed_dimension,
            hidden_dims=[128, 64],
            alpha_T=1.0,
            alpha_S=0.5,
        ).to(device)
        losses = modules["losses"].LossFunctions(alpha=1.0, beta=1.0, gamma=1.0)

        pretrain_epochs = int(self.params.get("pretrain_epochs", 50))
        pretrain_optimizer = torch.optim.Adam(
            list(disentanglement.parameters())
            + list(reconstructor.parameters())
            + list(domain_classifier.parameters()),
            lr=0.001,
        )
        pretrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            pretrain_optimizer, T_max=max(pretrain_epochs, 1)
        )
        for _ in range(pretrain_epochs):
            pretrain_optimizer.zero_grad()
            source_related, target_related, source_shared, target_shared = disentanglement(
                source_tensor, target_tensor
            )
            reconstructed_source = reconstructor(source_related, source_shared)
            reconstructed_target = reconstructor(target_related, target_shared)
            reconstruction_loss = losses.reconstruction_loss(
                source_tensor, reconstructed_source
            ) + losses.reconstruction_loss(target_tensor, reconstructed_target)
            source_domain = domain_classifier(source_shared)
            target_domain = domain_classifier(target_shared)
            domain_loss = losses.domain_adversarial_loss(
                torch.cat([source_domain, target_domain]),
                torch.cat(
                    [
                        torch.zeros(source_tensor.size(0), device=device),
                        torch.ones(target_tensor.size(0), device=device),
                    ]
                ),
            )
            orthogonality_loss = losses.orthogonality_loss(
                source_shared, source_related, target_shared, target_related
            )
            total_loss = reconstruction_loss + domain_loss + orthogonality_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(disentanglement.parameters())
                + list(reconstructor.parameters())
                + list(domain_classifier.parameters()),
                max_norm=1.0,
            )
            pretrain_optimizer.step()
            pretrain_scheduler.step()

        fine_tune_epochs = int(self.params.get("fine_tune_epochs", 100))
        fine_tune_optimizer = torch.optim.Adam(
            list(disentanglement.parameters())
            + list(reconstructor.parameters())
            + list(domain_classifier.parameters())
            + list(predictor.parameters()),
            lr=0.001,
            weight_decay=1e-4,
        )
        fine_tune_scheduler = torch.optim.lr_scheduler.StepLR(
            fine_tune_optimizer, step_size=30, gamma=0.1
        )
        for _ in range(fine_tune_epochs):
            fine_tune_optimizer.zero_grad()
            source_related, target_related, source_shared, target_shared = disentanglement(
                source_tensor, target_tensor
            )
            _, source_prediction = predictor(
                target_related, source_related, target_shared, source_shared
            )
            defect_loss = losses.defect_loss(source_prediction, source_labels)
            reconstruction_loss = losses.reconstruction_loss(
                source_tensor, reconstructor(source_related, source_shared)
            ) + losses.reconstruction_loss(
                target_tensor, reconstructor(target_related, target_shared)
            )
            source_domain = domain_classifier(source_shared)
            target_domain = domain_classifier(target_shared)
            domain_loss = losses.domain_adversarial_loss(
                torch.cat([source_domain, target_domain]),
                torch.cat(
                    [
                        torch.zeros(source_tensor.size(0), device=device),
                        torch.ones(target_tensor.size(0), device=device),
                    ]
                ),
            )
            orthogonality_loss = losses.orthogonality_loss(
                source_shared, source_related, target_shared, target_related
            )
            total_loss = defect_loss + reconstruction_loss + domain_loss + orthogonality_loss
            total_loss.backward()
            fine_tune_optimizer.step()
            fine_tune_scheduler.step()

        with torch.no_grad():
            _, _, _, target_shared = disentanglement(source_tensor, target_tensor)
            scores = predictor.domain_shared(target_shared).squeeze(1).cpu().numpy()
        return PredictionResult(
            scores=scores,
            threshold=0.5,
            metadata={
                "variant": "FD-HDP-U",
                "source_projects": [source.key for source in sources],
                "pretrain_epochs": pretrain_epochs,
                "fine_tune_epochs": fine_tune_epochs,
                "alpha_target": 1.0,
                "imbalance_handling": "source-only SMOTE to 80% of majority",
                "source_smote_applied": smote_applied,
                "target_labels_used": False,
                "requested_device": requested_device,
                "compute_device": str(device),
                "cuda_device_name": (
                    torch.cuda.get_device_name(device) if device.type == "cuda" else None
                ),
            },
        )
