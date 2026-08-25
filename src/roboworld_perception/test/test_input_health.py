"""Input-health state machine is kept ROS-independent for CI coverage."""
import pytest

from roboworld_perception.input_health import (
    DIAG_ERROR, DIAG_OK, DIAG_WARN, classify_input_health,
)


@pytest.mark.parametrize("valid,last_frame,error,expected", [
    (False, None, None, (DIAG_ERROR, "waiting for valid camera_info", None)),
    (True, None, None, (DIAG_WARN, "waiting for RGB-D frames", None)),
    (True, 8.0, None, (DIAG_WARN, "RGB-D input stale", 2.0)),
    (True, 9.0, None, (DIAG_OK, "RGB-D input healthy", 1.0)),
    (True, 9.0, "frame mismatch", (DIAG_ERROR, "input contract invalid", 1.0)),
])
def test_classify_input_health(valid, last_frame, error, expected):
    level, message, age = classify_input_health(
        now_monotonic=10.0,
        camera_info_valid=valid,
        last_frame_time=last_frame,
        stale_timeout=1.5,
        input_contract_error=error,
    )
    assert (level, message) == expected[:2]
    if expected[2] is None:
        assert age is None
    else:
        assert age == pytest.approx(expected[2])


def test_future_monotonic_sample_is_clamped_to_zero_age():
    _, _, age = classify_input_health(10.0, True, 11.0, 1.0, None)
    assert age == 0.0
