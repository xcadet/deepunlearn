# We consider the equation provided in zhangReviewMachineUnlearning2023
def sape_score(a: float, b: float) -> float:
    """Compute the SAPE between two values

    SAPE standes for Symmetric Absolute Percentage Error
    """
    assert isinstance(a, float)
    assert isinstance(b, float)
    upper = abs(b - a)
    lower = abs(b) + abs(a)
    combined = upper / lower
    as_percent = combined * 100
    return as_percent
