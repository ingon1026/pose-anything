"""ROS-independent RGB-D input health classification."""

# diagnostic_msgs/DiagnosticStatus standard severity values.  Kept as plain
# integers so this module stays importable in the lightweight CI environment.
DIAG_OK = 0
DIAG_WARN = 1
DIAG_ERROR = 2


def classify_input_health(now_monotonic, camera_info_valid, last_frame_time,
                          stale_timeout, input_contract_error):
    """Classify camera readiness and return ``(level, message, frame_age_s)``."""
    frame_age_s = (None if last_frame_time is None else
                   max(0.0, now_monotonic - last_frame_time))
    if input_contract_error:
        return DIAG_ERROR, "input contract invalid", frame_age_s
    if not camera_info_valid:
        return DIAG_ERROR, "waiting for valid camera_info", frame_age_s
    if frame_age_s is None:
        return DIAG_WARN, "waiting for RGB-D frames", frame_age_s
    if frame_age_s > stale_timeout:
        return DIAG_WARN, "RGB-D input stale", frame_age_s
    return DIAG_OK, "RGB-D input healthy", frame_age_s
