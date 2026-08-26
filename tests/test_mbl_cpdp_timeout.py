import time

import pytest

from cpdp.methods.mbl_cpdp_sv import HardTimeoutError, _call_with_hard_timeout


def test_hard_timeout_returns_subprocess_result() -> None:
    result = _call_with_hard_timeout(sum, ([1, 2, 3],), timeout_seconds=5.0)

    assert result == 6


def test_hard_timeout_terminates_long_subprocess() -> None:
    started = time.monotonic()

    with pytest.raises(HardTimeoutError, match="hard timeout"):
        _call_with_hard_timeout(time.sleep, (5.0,), timeout_seconds=0.25)

    assert time.monotonic() - started < 3.0
