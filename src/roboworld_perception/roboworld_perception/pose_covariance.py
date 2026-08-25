"""Covariance contract for published geometric OBB poses.

``PoseWithCovariance`` carries a 6x6 covariance in the order
``x, y, z, roll, pitch, yaw``.  This pipeline filters only position.  The
OBB quaternion is useful for visualisation, but is not a calibrated semantic
object orientation and has no orientation uncertainty model.  Never encode
that absence as zero variance: zero means exact certainty to ROS consumers.
"""
import math


POSITION_COVARIANCE_INDICES = (0, 7, 14)
ROTATION_COVARIANCE_INDICES = (21, 28, 35)
# One standard deviation of 180 degrees makes the unmodelled OBB orientation
# explicitly unusable for a typical orientation-gated grasp, while retaining a
# valid positive-semidefinite covariance for generic ROS covariance consumers.
UNESTIMATED_ROTATION_VARIANCE_RAD2 = math.pi ** 2


def published_pose_covariance(position_variance):
    """Build a valid 6x6 covariance for a position-filtered OBB pose.

    ``position_variance`` is the filter's x/y/z variance in m².  Rotational
    diagonals deliberately receive a conservative finite variance instead of
    an undocumented negative sentinel or an unsafe zero.
    """
    position_variance = tuple(position_variance)
    if len(position_variance) != 3:
        raise ValueError("position_variance must contain exactly three values")
    if any(not math.isfinite(value) or value < 0 for value in position_variance):
        raise ValueError("position_variance must be finite and non-negative")

    covariance = [0.0] * 36
    for index, variance in zip(POSITION_COVARIANCE_INDICES, position_variance):
        covariance[index] = float(variance)
    for index in ROTATION_COVARIANCE_INDICES:
        covariance[index] = UNESTIMATED_ROTATION_VARIANCE_RAD2
    return covariance
