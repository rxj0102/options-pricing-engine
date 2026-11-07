"""Volatility smile construction: IV extraction, polynomial and SVI fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pricing.base import OptionType
from pricing.black_scholes import BlackScholesMerton


@dataclass
class SVIParams:
    """Raw SVI parameters for total variance ``w(k)``.

    Attributes
    ----------
    a : float
        Vertical level of variance.
    b : float
        Angle between the asymptotes (slope).
    rho : float
        Asymmetry / skew, in ``(-1, 1)``.
    m : float
        Horizontal shift of the smile minimum.
    sigma : float
        Curvature of the smile at the minimum.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float


class VolatilitySmile:
    r"""Construct and fit a volatility smile from market option prices.

    Parameters
    ----------
    quotes : sequence of (float, float, str)
        Tuples of ``(strike, market_price, option_type)``.
    S : float
        Spot price.
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate.
    q : float, optional
        Dividend yield. Default ``0.0``.

    Examples
    --------
    >>> from pricing.black_scholes import BlackScholesMerton
    >>> m = BlackScholesMerton()
    >>> quotes = [(k, m.price(100, k, 1.0, 0.05, 0.2), "call")
    ...           for k in (80, 90, 100, 110, 120)]
    >>> smile = VolatilitySmile(quotes, S=100, T=1.0, r=0.05)
    >>> df = smile.extract_ivols()
    >>> bool((df["ivol"].round(3) == 0.2).all())
    True
    """

    def __init__(
        self,
        quotes: Sequence[Tuple[float, float, Union[str, OptionType]]],
        S: float,
        T: float,
        r: float,
        q: float = 0.0,
    ) -> None:
        self.quotes = list(quotes)
        self.S = float(S)
        self.T = float(T)
        self.r = float(r)
        self.q = float(q)
        self._model = BlackScholesMerton()
        self.forward = self.S * np.exp((self.r - self.q) * self.T)

    def extract_ivols(self) -> pd.DataFrame:
        """Invert each market price to a Black-Scholes implied vol.

        Returns
        -------
        pandas.DataFrame
            Columns ``strike``, ``moneyness`` (``K/S``),
            ``log_moneyness`` (``ln(K/F)``), ``market_price``, ``ivol``,
            sorted by strike.
        """
        rows: List[dict] = []
        for strike, price, opt in self.quotes:
            iv = self._model.implied_vol(
                price, self.S, strike, self.T, self.r, self.q, opt
            )
            rows.append(
                {
                    "strike": float(strike),
                    "moneyness": float(strike) / self.S,
                    "log_moneyness": float(np.log(float(strike) / self.forward)),
                    "market_price": float(price),
                    "ivol": float(iv),
                }
            )
        return pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)

    def fit_polynomial(self, degree: int = 4) -> np.poly1d:
        """Fit a polynomial of IV as a function of moneyness ``K/S``.

        Parameters
        ----------
        degree : int, optional
            Polynomial degree. Default ``4``.

        Returns
        -------
        numpy.poly1d
            Callable polynomial ``ivol(moneyness)``.

        Examples
        --------
        >>> from pricing.black_scholes import BlackScholesMerton
        >>> m = BlackScholesMerton()
        >>> quotes = [(k, m.price(100, k, 1.0, 0.05, 0.2), "call")
        ...           for k in (80, 90, 100, 110, 120)]
        >>> poly = VolatilitySmile(quotes, 100, 1.0, 0.05).fit_polynomial(2)
        >>> round(float(poly(1.0)), 3)
        0.2
        """
        df = self.extract_ivols()
        coeffs = np.polyfit(df["moneyness"], df["ivol"], deg=degree)
        return np.poly1d(coeffs)

    def svi_fit(self) -> SVIParams:
        r"""Fit the raw SVI total-variance parameterisation.

        Total implied variance is modelled as

        .. math::

            w(k) = a + b\bigl(\rho (k - m) + \sqrt{(k - m)^2 + \sigma^2}\bigr),

        where :math:`k = \ln(K/F)` and total variance :math:`w = \sigma_{\rm
        BS}^2 T`.

        Returns
        -------
        SVIParams
            Calibrated parameters ``(a, b, rho, m, sigma)``.

        Examples
        --------
        >>> from pricing.black_scholes import BlackScholesMerton
        >>> m = BlackScholesMerton()
        >>> quotes = [(k, m.price(100, k, 1.0, 0.05, 0.2), "call")
        ...           for k in range(80, 125, 5)]
        >>> p = VolatilitySmile(quotes, 100, 1.0, 0.05).svi_fit()
        >>> p.b >= 0
        True
        """
        df = self.extract_ivols()
        k = df["log_moneyness"].to_numpy()
        total_var = (df["ivol"].to_numpy() ** 2) * self.T

        def svi(theta: np.ndarray, kk: np.ndarray) -> np.ndarray:
            a, b, rho, m, sig = theta
            return a + b * (rho * (kk - m) + np.sqrt((kk - m) ** 2 + sig**2))

        def objective(theta: np.ndarray) -> float:
            a, b, rho, m, sig = theta
            if b < 0 or sig <= 0 or abs(rho) >= 1.0:
                return 1e6
            return float(np.sum((svi(theta, k) - total_var) ** 2))

        x0 = np.array([float(np.min(total_var)), 0.1, -0.3, 0.0, 0.1])
        res = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 10000},
        )
        a, b, rho, m, sig = res.x
        return SVIParams(
            a=float(a), b=float(b), rho=float(rho), m=float(m), sigma=float(sig)
        )

    def svi_curve(self, params: SVIParams, log_moneyness: np.ndarray) -> np.ndarray:
        """Evaluate fitted SVI implied vols over a log-moneyness grid.

        Parameters
        ----------
        params : SVIParams
            Fitted SVI parameters.
        log_moneyness : numpy.ndarray
            Grid of ``ln(K/F)`` values.

        Returns
        -------
        numpy.ndarray
            Implied vols (annualised) on the grid.
        """
        k = np.asarray(log_moneyness, dtype=float)
        w = params.a + params.b * (
            params.rho * (k - params.m) + np.sqrt((k - params.m) ** 2 + params.sigma**2)
        )
        w = np.maximum(w, 0.0)
        return np.sqrt(w / self.T)

    def smile_plot_data(self) -> dict:
        """Assemble market and fitted curves for Streamlit plotting.

        Returns
        -------
        dict
            Keys: ``strike``, ``moneyness``, ``log_moneyness``,
            ``market_ivol``, ``poly_ivol``, ``svi_ivol``, ``svi_params``.
        """
        df = self.extract_ivols()
        poly = self.fit_polynomial()
        svi_params = self.svi_fit()
        svi_iv = self.svi_curve(svi_params, df["log_moneyness"].to_numpy())
        return {
            "strike": df["strike"].to_numpy(),
            "moneyness": df["moneyness"].to_numpy(),
            "log_moneyness": df["log_moneyness"].to_numpy(),
            "market_ivol": df["ivol"].to_numpy(),
            "poly_ivol": poly(df["moneyness"].to_numpy()),
            "svi_ivol": svi_iv,
            "svi_params": svi_params,
        }
