from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Tuple

from cpdp.methods.base import BaseMethod
from cpdp.vendor import load_module


METHOD_NAMES = (
    "dssdpp",
    "master",
    "mbl_cpdp_sv",
    "fedcpdp",
    "fd_hdp_u",
    "ipgdo_kan_jepa",
    "ipgdo_transformer_jepa",
)
LEGACY_METHOD_ALIASES = {
    "apc_pc_tabddpm_mlp": "ipgdo_mlp",
    "apc_pc_tabddpm_kan_jepa": "ipgdo_kan_jepa",
    "apc_pc_tabddpm_transformer_jepa": "ipgdo_transformer_jepa",
}
MethodFactory = Callable[[Path, Mapping[str, Any]], BaseMethod]
_CUSTOM_METHODS: Dict[str, MethodFactory] = {}


def normalize_method_name(name: str) -> str:
    return name.lower().replace("-", "_")


def canonical_method_name(name: str) -> str:
    normalized = normalize_method_name(name)
    return LEGACY_METHOD_ALIASES.get(normalized, normalized)


def available_methods() -> Tuple[str, ...]:
    custom_only = sorted(name for name in _CUSTOM_METHODS if name not in METHOD_NAMES)
    return METHOD_NAMES + tuple(custom_only)


def register_method(
    name: str, factory: MethodFactory, *, replace: bool = False
) -> None:
    normalized = normalize_method_name(name)
    if not normalized:
        raise ValueError("Method name cannot be empty")
    if (normalized in METHOD_NAMES or normalized in _CUSTOM_METHODS) and not replace:
        raise KeyError(f"Method is already registered: {normalized}")
    _CUSTOM_METHODS[normalized] = factory


def load_method_plugin(path: Path) -> None:
    resolved = path.resolve()
    digest = hashlib.blake2b(str(resolved).encode("utf-8"), digest_size=8).hexdigest()
    module = load_module(f"cpdp_method_plugin_{digest}", resolved)
    register = getattr(module, "register_methods", None)
    if not callable(register):
        raise AttributeError(
            f"Method plugin must define a callable register_methods(): {resolved}"
        )
    register()


def create_method(name: str, project_root: Path, params: Mapping[str, Any]) -> BaseMethod:
    normalized = canonical_method_name(name)
    if normalized in _CUSTOM_METHODS:
        return _CUSTOM_METHODS[normalized](project_root, params)
    if normalized == "dssdpp":
        from cpdp.methods.dssdpp import DSSDPPMethod

        return DSSDPPMethod(project_root, params)
    if normalized == "master":
        from cpdp.methods.master import MASTERMethod

        return MASTERMethod(project_root, params)
    if normalized in {"mbl_cpdp", "mbl_cpdp_sv"}:
        from cpdp.methods.mbl_cpdp_sv import MBLCPDPSVMethod

        return MBLCPDPSVMethod(project_root, params)
    if normalized == "fedcpdp":
        from cpdp.methods.fedcpdp import FedCPDPMethod

        return FedCPDPMethod(project_root, params)
    if normalized in {"fd_hdp", "fd_hdp_u"}:
        from cpdp.methods.fd_hdp_u import FDHDPUMethod

        return FDHDPUMethod(project_root, params)
    if normalized == "ipgdo_mlp":
        from cpdp.methods.ipgdo_mlp import IPGDOMLPMethod

        return IPGDOMLPMethod(project_root, params)
    if normalized == "ipgdo_kan_jepa":
        from cpdp.methods.ipgdo_kan_jepa import IPGDOKANJEPAMethod

        return IPGDOKANJEPAMethod(project_root, params)
    if normalized == "ipgdo_transformer_jepa":
        from cpdp.methods.ipgdo_transformer_jepa import IPGDOTransformerJEPAMethod

        return IPGDOTransformerJEPAMethod(project_root, params)
    raise KeyError(f"Unknown method: {name}")
