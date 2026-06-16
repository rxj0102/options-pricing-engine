"""Volatility term-structure interpolation across expiries."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class VolTermStructure:
    r"""Interpolate ATM implied volatility across maturities.

    Interpolation is performed in **total variance** space
    (:math:`w = \sigma^2 T`) which is the arbitrage-consistent quantity,
    then converted back to volatility.

    Parameters
    ----------
    tenors : sequence of float
        Maturities in years (must be strictly positive).
    vols : sequence of float
        ATM implied vols aligned with ``tenors``.

    Examples
    --------
    >>> ts = VolTermStructure([0.25, 0.5, 1.0, 2.0], [0.25, 0.22, 0.20, 0.18])
    >>> round(float(ts.vol(0.75)), 4)
    0.2069
    """

    def __init__(self, tenors: Sequence[float], vols: Sequence[float]) -> None:
        order = np.argsort(np.asarray(tenors, dtype=float))
        self.tenors = np.asarray(tenors, dtype=float)[order]
        self.vols = np.asarray(vols, dtype=float)[order]
        self.total_var = self.vols**2 * self.tenors
        self._interp = interp1d(
            self.tenors,
            self.total_var,
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
        )

    @classmethod
    def flat(
        cls, sigma: float, tenors: Sequence[float] | None = None
    ) -> "VolTermStructure":
        """Construct a flat term structure at a single volatility level.

        Parameters
        ----------
        sigma : float
            Constant volatility.
        tenors : sequence of float, optional
            Tenors to populate. Default ``(0.25, 0.5, 1.0, 2.0)``.

        Returns
        -------
        VolTermStructure
            A flat structure where every tenor maps to ``sigma``.
        """
        tenors = list(tenors) if tenors is not None else [0.25, 0.5, 1.0, 2.0]
        return cls(tenors, [sigma] * len(tenors))

    def vol(self, T: float) -> float:
        """Interpolated volatility at maturity ``T``.

        Parameters
        ----------
        T : float
            Target maturity in years.

        Returns
        -------
        float
            Interpolated (or extrapolated) volatility.

        Examples
        --------
        >>> VolTermStructure([0.5, 1.0], [0.2, 0.2]).vol(0.75)
        0.2
        """
        total_var = float(self._interp(T))
        return float(np.sqrt(max(total_var, 0.0) / T))

    def curve(self, tenor_grid: np.ndarray) -> pd.DataFrame:
        """Tabulate the interpolated curve over a grid of maturities.

        Parameters
        ----------
        tenor_grid : numpy.ndarray
            Maturities at which to evaluate.

        Returns
        -------
        pandas.DataFrame
            Columns ``tenor`` and ``vol``.
        """
        grid = np.asarray(tenor_grid, dtype=float)
        return pd.DataFrame({"tenor": grid, "vol": [self.vol(t) for t in grid]})
