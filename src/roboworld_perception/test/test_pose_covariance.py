"""ROS-independent tests for the Detection3D covariance output contract."""
import math

import pytest

from roboworld_perception.pose_covariance import (
    POSITION_COVARIANCE_INDICES,
    ROTATION_COVARIANCE_INDICES,
    UNESTIMATED_ROTATION_VARIANCE_RAD2,
    published_pose_covariance,
)


def test_position_variances_keep_their_standard_pose_indices():
    covariance = published_pose_covariance((1e-4, 2e-4, 3e-4))

    assert len(covariance) == 36
    assert [covariance[index] for index in POSITION_COVARIANCE_INDICES] == [
        1e-4, 2e-4, 3e-4]


def test_unestimated_obb_rotation_is_conservative_not_exact():
    covariance = published_pose_covariance((0.0, 0.0, 0.0))

    assert [covariance[index] for index in ROTATION_COVARIANCE_INDICES] == [
        UNESTIMATED_ROTATION_VARIANCE_RAD2,
    ] * 3
    assert UNESTIMATED_ROTATION_VARIANCE_RAD2 == pytest.approx(math.pi ** 2)


@pytest.mark.parametrize("variance", [
    (1.0, 2.0),
    (1.0, 2.0, 3.0, 4.0),
    (1.0, -1.0, 3.0),
    (1.0, math.nan, 3.0),
])
def test_invalid_position_variance_is_rejected(variance):
    with pytest.raises(ValueError):
        published_pose_covariance(variance)
