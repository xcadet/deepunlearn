def indiscernibility(mia):
    # Needs to be 1.0 when MIA = 0.5
    # Needs to b 0.0 when MIA = 0.0
    abs_mid_diff = abs(mia - 0.5) / 0.5

    return 1.0 - abs_mid_diff
