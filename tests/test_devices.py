import pytest

from cpdp.methods.fd_hdp_u import _resolve_torch_device


def test_fd_hdp_auto_device_matches_torch_runtime() -> None:
    torch = pytest.importorskip("torch")

    device = _resolve_torch_device(torch, "auto")

    assert device.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_fd_hdp_rejects_unknown_device() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="device must be"):
        _resolve_torch_device(torch, "gpu")
