import numpy as np

def mean(data, axis=None, ignore_nan=True):
    """
    Calculate the arithmetic mean of an array.

    Parameters
    ----------
    data : array-like
        Input numerical data.
    axis : int or None, optional
        Axis along which the mean is calculated.
        If None, calculate ove rthe entire array.
    ignore_nan:bool, default=True
        If True, Nan values are ignored.
        
    Returns
    -------
    float or np.ndarray
        Mean of the input data.
        
    Notes
    -----
    Time complexity: O(n)
    Space complexity: O(1)
    """
    data=np.asarray(data, dtype=float)
    if ignore_nan:
        return np.nanmean(data, axis=axis)
    return np.mean(data, axis=axis)

def median(data, axis=None, ignore_nan=True):
    """
    Calculate the median of an array.

    Parameters
    ----------
    data : array-like
        Input numerical data.
    axis : int or None, optional
        Axis along which the median is calculated.
        If None, calculate over the entire array.
    ignore_nan:bool, default=True
        If True, Nan values are ignored.
        
    Returns
    -------
    float or np.ndarray
        Median of the input data.
    """
    data=np.asarray(data, dtype=float)
    if ignore_nan:
        return np.nanmedian(data, axis=axis)
    return np.median(data, axis=axis)

def std(data, axis=None, ignore_nan=True, ddof=0):
    """
    Calculate the standard deviation of an array.

    Parameters
    ----------
    data : array-like
        Input numerical data.
    axis : int or None, optional
        Axis along which the standard deviation is calculated.
        If None, calculate over the entire array.
    ignore_nan:bool, default=True
        If True, Nan values are ignored.
        
    Returns
    -------
    float or np.ndarray
        Standard deviation of the input data.
    """
    data=np.asarray(data, dtype=float)
    if ignore_nan:
        return np.nanstd(data, axis=axis, ddof=ddof)
    return np.std(data, axis=axis, ddof=ddof)

def minimum(data, axis=None, ignore_nan=True):
    """
    Calculate the minimum of an array.

    Parameters
    ----------
    data : array-like
        Input numerical data.
    axis : int or None, optional
        Axis along which the minimum is calculated.
        If None, calculate over the entire array.
    ignore_nan:bool, default=True
        If True, Nan values are ignored.
        
    Returns
    -------
    float or np.ndarray
        Minimum value(s).
    """
    data=np.asarray(data, dtype=float)
    if ignore_nan:
        return np.nanmin(data, axis=axis)
    return np.min(data, axis=axis)

def maximum(data, axis=None, ignore_nan=True):
    """
    Calculate the maximum of an array.

    Parameters
    ----------
    data : array-like
        Input numerical data.
    axis : int or None, optional
        Axis along which the maximum is calculated.
        If None, calculate over the entire array.
    ignore_nan:bool, default=True
        If True, Nan values are ignored.
        
    Returns
    -------
    float or np.ndarray
        Maximum value(s).
    """
    data=np.asarray(data, dtype=float)
    if ignore_nan:
        return np.nanmax(data, axis=axis)
    return np.max(data, axis=axis)

def histogram(data, bins=10, range=None, ignore_nan=True):
    """
    Compute the histogram of an array.

    Parameters
    ----------
    data : array-like
        Input numerical data.
    bins : int or sequence of scalars, optional
        Number of bins or the bin edges.
    range : tuple, optional
        The lower and upper range of the bins.
    ignore_nan:bool, default=True
        If True, Nan values are ignored.
        
    Returns
    -------
    hist : np.ndarray
        The values of the histogram.
    bin_edges : np.ndarray
        The edges of the bins.
    """
    data=np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    return np.histogram(data, bins=bins, range=range)

def quantile(data, q, axis=None, ignore_nan=True):
    """
    Calculate quantiles of the input data.

    Parameters
    ----------
    data : array-like
        Input numerical data.
    q : float or array-like
        Quantile(s) to compute, which must be between 0 and 1 inclusive.
    axis : int or None, optional
        Axis along which the quantiles are computed.
    ignore_nan:bool, default=True
        If True, Nan values are ignored.
        
    Returns
    -------
    float or np.ndarray
        The calculated quantile(s).
    """
    data=np.asarray(data, dtype=float)
    if ignore_nan:
        return np.nanquantile(data, q, axis=axis)
    return np.quantile(data, q, axis=axis)
