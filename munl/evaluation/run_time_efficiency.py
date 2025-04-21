import math


def compute_run_time_efficiency(new_time: float, reference_time: float) -> float:
    """Compute the RTE metric given a current time and a reference time.

    Args:
        new_time (float): The new time obtained
        reference_time (float): The time used as reference

    Returns:
        float: The Run Time Efficiency metric
    """
    assert 0 < new_time, "reference time must be > 0"
    assert 0 < reference_time or math.isclose(
        reference_time, 0
    ), "reference time must be >= 0"
    return reference_time / new_time
