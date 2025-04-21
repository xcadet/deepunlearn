import matplotlib.pyplot as plt
from typing import Tuple


DEFAULT_SIZE_PER_AXIS = (10, 10)


def prepare_ax(
    ax: plt.Axes | None = None, size_per_axis: Tuple[int, int] = DEFAULT_SIZE_PER_AXIS
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=size_per_axis)
    return ax


def subplots(
    num_rows: int,
    num_columns: int,
    size_per_axis: Tuple[int, int] = DEFAULT_SIZE_PER_AXIS,
) -> plt.Axes:
    row_size = size_per_axis[0]
    col_size = size_per_axis[1]
    _, axs = plt.subplots(
        num_rows, num_columns, figsize=(col_size * num_columns, row_size * num_rows)
    )
    return axs
