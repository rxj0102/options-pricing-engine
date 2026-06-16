"""Normal and lognormal distribution helpers used across pricing models.

These thin wrappers around :mod:`scipy.stats` centralise the standard normal
CDF and PDF so every model shares an identical, vectorised implementation.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import norm

ArrayLike = Union[float, np.ndarray]


def norm_cdf(x: ArrayLike) -> ArrayLike:
    r"""Standard normal cumulative distribution function :math:`N(x)`.

    Parameters
    ----------
    x : float or numpy.ndarray
        Point(s) at which to evaluate the CDF.

    Returns
    -------
    float or numpy.ndarray
        :math:`N(x) = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi}} e^{-t^2/2}\,dt`.

    Examples
    --------
    >>> round(float(norm_cdf(0.0)), 4)
    0.5
    """
    return norm.cdf(x)


def norm_pdf(x: ArrayLike) -> ArrayLike:
    r"""Standard normal probability density function :math:`n(x)`.

    Parameters
    ----------
    x : float or numpy.ndarray
        Point(s) at which to evaluate the PDF.

    Returns
    -------
    float or numpy.ndarray
        :math:`n(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}`.

    Examples
    --------
    >>> round(float(norm_pdf(0.0)), 4)
    0.3989
    """
    return norm.pdf(x)


def lognormal_pdf(x: ArrayLike, mu: float, sigma: float) -> ArrayLike:
    r"""Lognormal probability density function.

    Parameters
    ----------
    x : float or numpy.ndarray
        Support point(s); must be positive.
    mu : float
        Mean of the underlying normal distribution (log-space).
    sigma : float
        Standard deviation of the underlying normal distribution (log-space).

    Returns
    -------
    float or numpy.ndarray
        Lognormal density evaluated at ``x``.

    Examples
    --------
    >>> round(float(lognormal_pdf(1.0, 0.0, 1.0)), 4)
    0.3989
    """
    x = np.asarray(x, dtype=float)
    coeff = 1.0 / (x * sigma * np.sqrt(2.0 * np.pi))
    return coeff * np.exp(-((np.log(x) - mu) ** 2) / (2.0 * sigma**2))
