"""SABR stochastic-volatility smile via the Hagan (2002) approximation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

# Floor to avoid division by zero when F == K (the ATM limit).
_ATM_EPS: float = 1e-12


@dataclass
class SABRParams:
    """SABR parameter set.

    Attributes
    ----------
    alpha : float
        Initial volatility level (often denoted :math:`\\alpha`).
    beta : float
        CEV exponent in ``[0, 1]`` fixing the backbone.
    rho : float
        Correlation between spot and vol, in ``(-1, 1)``.
    nu : float
        Vol-of-vol.
    """

    alpha: float
    beta: float
    rho: float
    nu: float


class SABRModel:
    r"""SABR implied-volatility model (Hagan et al., 2002).

    The lognormal implied vol for strike :math:`K` and forward :math:`F` is
    approximated in closed form, giving a fast, arbitrage-light smile with
    four interpretable parameters :math:`(\alpha, \beta, \rho, \nu)`.

    Examples
    --------
    >>> m = SABRModel()
    >>> round(m.implied_vol(100, 100, 1.0, 0.2, 1.0, -0.3, 0.4), 4)
    0.2011
    """

    name = "SABR (Hagan 2002)"

    def implied_vol(
        self,
        F: float,
        K: float,
        T: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
    ) -> float:
        r"""Hagan (2002) lognormal implied volatility.

        Parameters
        ----------
        F : float
            Forward price.
        K : float
            Strike.
        T : float
            Time to expiry in years.
        alpha, beta, rho, nu : float
            SABR parameters.

        Returns
        -------
        float
            Approximate Black implied volatility for the strike.

        Examples
        --------
        >>> m = SABRModel()
        >>> round(m.implied_vol(100, 110, 1.0, 0.2, 1.0, -0.3, 0.4), 4)
        0.1965
        """
        F = float(F)
        K = float(K)
        one_minus_beta = 1.0 - beta

        if abs(F - K) < _ATM_EPS:  # ATM expansion
            f_pow = F**one_minus_beta
            term1 = ((one_minus_beta**2) / 24.0) * alpha**2 / (f_pow**2)
            term2 = 0.25 * rho * beta * nu * alpha / f_pow
            term3 = ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
            return float(alpha / f_pow * (1.0 + (term1 + term2 + term3) * T))

        log_fk = np.log(F / K)
        fk_pow = (F * K) ** (one_minus_beta / 2.0)
        z = (nu / alpha) * fk_pow * log_fk
        x_z = np.log((np.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho))

        denom = fk_pow * (
            1.0
            + (one_minus_beta**2 / 24.0) * log_fk**2
            + (one_minus_beta**4 / 1920.0) * log_fk**4
        )
        factor = z / x_z
        correction = (
            1.0
            + (
                (one_minus_beta**2 / 24.0) * alpha**2 / (fk_pow**2)
                + 0.25 * rho * beta * nu * alpha / fk_pow
                + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
            )
            * T
        )
        return float(alpha / denom * factor * correction)

    def calibrate(
        self,
        strikes: Sequence[float],
        ivols: Sequence[float],
        F: float,
        T: float,
        beta: float = 0.5,
    ) -> SABRParams:
        """Calibrate ``(alpha, rho, nu)`` to market vols via Nelder-Mead.

        Parameters
        ----------
        strikes : sequence of float
            Observed strikes.
        ivols : sequence of float
            Observed implied vols aligned with ``strikes``.
        F : float
            Forward price.
        T : float
            Time to expiry in years.
        beta : float, optional
            Fixed CEV exponent (commonly chosen a priori). Default ``0.5``.

        Returns
        -------
        SABRParams
            Best-fit parameters.

        Examples
        --------
        >>> m = SABRModel()
        >>> ks = [80, 90, 100, 110, 120]
        >>> ivs = [m.implied_vol(100, k, 1.0, 0.2, 0.5, -0.3, 0.4) for k in ks]
        >>> p = m.calibrate(ks, ivs, 100, 1.0)
        >>> abs(p.nu - 0.4) < 0.05
        True
        """
        strikes = np.asarray(strikes, dtype=float)
        ivols = np.asarray(ivols, dtype=float)

        def objective(theta: np.ndarray) -> float:
            alpha, rho, nu = theta
            if alpha <= 0 or nu <= 0 or abs(rho) >= 1.0:
                return 1e6
            model = np.array(
                [self.implied_vol(F, k, T, alpha, beta, rho, nu) for k in strikes]
            )
            return float(np.sum((model - ivols) ** 2))

        x0 = np.array([float(np.mean(ivols)), -0.2, 0.4])
        res = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 5000},
        )
        alpha, rho, nu = res.x
        return SABRParams(
            alpha=float(alpha), beta=float(beta), rho=float(rho), nu=float(nu)
        )

    def smile_from_params(
        self,
        F: float,
        T: float,
        strike_range: np.ndarray,
        params: SABRParams,
    ) -> np.ndarray:
        """Generate a vol curve over a range of strikes.

        Parameters
        ----------
        F : float
            Forward price.
        T : float
            Time to expiry in years.
        strike_range : numpy.ndarray
            Strikes at which to evaluate the smile.
        params : SABRParams
            SABR parameters.

        Returns
        -------
        numpy.ndarray
            Implied vols aligned with ``strike_range``.

        Examples
        --------
        >>> import numpy as np
        >>> m = SABRModel()
        >>> p = SABRParams(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
        >>> curve = m.smile_from_params(100, 1.0, np.array([90, 100, 110]), p)
        >>> curve.shape
        (3,)
        """
        return np.array(
            [
                self.implied_vol(
                    F, k, T, params.alpha, params.beta, params.rho, params.nu
                )
                for k in np.asarray(strike_range, dtype=float)
            ]
        )
