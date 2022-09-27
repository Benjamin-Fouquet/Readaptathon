from scipy.signal import savgol_filter
import numpy as np


def interpolate(x, window_length=501, poly_order=3):
    """Interpolate a 1D array using savgol filter.

    Args:
        x: The array to interpolate.
        window_length: The length of the filter window. Defaults to 401.
        poly_order: The order of the polynomial. Defaults to 3.

    Returns:
        The interpolated array.
    """
    return savgol_filter(x, window_length, poly_order)


def remove_anomalies(x, x_interp, threshold):
    """Replace points which are above a given threshold from a given interpolation.

    Args:
        x: Array to remove anomalies from.
        x_interp: Interpolated array.
        threshold: The threshold.

    Returns:
        An array where anomalies have been removed.
    """

    return np.where(abs(x - x_interp) < threshold, x, x_interp)


def replacement_ratio(x, x_interp, threshold):
    """Compute the replacement ratio.

    Args:
        x: Initial array.
        x_interp: Interpolated array.
        threshold: Threshold from which the value is replaced.

    Returns:
        Replacement ratio.
    """
    return (abs(x - x_interp) >= threshold).sum() / x.shape[0]


def defect_score(x, x_interp, norm="L1", threshold=None):
    if threshold is not None:
        x = np.where(abs(x - x_interp) > threshold, x, x_interp)
        # x = np.where((x - x_interp) < -threshold, x, x_interp)
    if norm == "L1":
        return abs(x - x_interp).sum() / x.shape[0]  # / abs(x_interp).sum()
    if norm == "L2":
        return (
            np.sqrt(((x - x_interp) ** 2).sum()) / x.shape[0]
        )  # / np.sqrt((x_interp**2).sum())
    elif norm == "inf":
        return abs(x - x_interp).max() / x.shape[0]  # / abs(x_interp).max()
    else:
        raise RuntimeError("Unknown norm specified")
