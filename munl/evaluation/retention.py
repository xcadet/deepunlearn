import math


def compute_accuracy_retention(new: float, reference: float) -> float:
    """_summary_

    Args:
        new (float): _description_
        reference (float): _description_

    Returns:
        float: _description_
    """
    assert (
        0 < new < 1 or math.isclose(new, 0) or math.isclose(new, 1)
    ), f"new accuracy must be between 0 and 1, found {new}"
    assert 0 < reference <= 1 or math.isclose(
        reference, 1
    ), "reference accuracy must be between 0 (excluded) and 1"
    return new / reference
