"""Boyle (1986) trinomial tree pricer."""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np

from pricing.base import ExerciseStyle, OptionPricer, OptionType

DEFAULT_N_STEPS: int = 150
#: Backward induction is ``O(n^2)`` in time; warn beyond this many steps.
MAX_RECOMMENDED_STEPS: int = 2000
#: Boyle's stretch parameter; lambda = sqrt(3/2) places three branches so the
#: lattice matches the first two moments of the lognormal increment.
BOYLE_LAMBDA: float = np.sqrt(1.5)


class BoyleTrinomial(OptionPricer):
    r"""Boyle (1986) trinomial tree for European and American options.

    With :math:`\Delta t = T/n` and :math:`\lambda = \sqrt{3/2}`,

    .. math::

        u = e^{\lambda\sigma\sqrt{\Delta t}}, \quad m = 1, \quad d = 1/u,

    and the branch probabilities are derived by matching the first two
    moments of the underlying's log-return:

    .. math::

        p_u &= \frac{1}{2\lambda^2}
               + \frac{(r - q - \tfrac12\sigma^2)\sqrt{\Delta t}}
                      {2\lambda\sigma}, \\
        p_m &= 1 - \frac{1}{\lambda^2}, \quad p_d = 1 - p_u - p_m.

    Parameters
    ----------
    n_steps : int, optional
        Number of time steps. Default ``150``.

    Examples
    --------
    >>> model = BoyleTrinomial(n_steps=200)
    >>> round(model.price(100, 100, 1.0, 0.05, 0.2), 2)
    10.45
    """

    name = "Boyle Trinomial"

    def __init__(self, n_steps: int = DEFAULT_N_STEPS) -> None:
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
        if n_steps > MAX_RECOMMENDED_STEPS:
            warnings.warn(
                f"n_steps={n_steps} exceeds {MAX_RECOMMENDED_STEPS}; the "
                "O(n^2) backward induction may be slow.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.n_steps = int(n_steps)
        self.tree: np.ndarray | None = None

    def _probabilities(
        self, r: float, q: float, sigma: float, dt: float
    ) -> "tuple[float, float, float, float, float]":
        """Return ``(u, d, pu, pm, pd)`` for the Boyle parameterisation."""
        lam = BOYLE_LAMBDA
        u = np.exp(lam * sigma * np.sqrt(dt))
        d = 1.0 / u
        drift = (r - q - 0.5 * sigma**2) * np.sqrt(dt) / (2.0 * lam * sigma)
        pu = 1.0 / (2.0 * lam**2) + drift
        pm = 1.0 - 1.0 / lam**2
        pd = 1.0 - pu - pm
        return u, d, pu, pm, pd

    def validate_probabilities(
        self, r: float, q: float, sigma: float, T: float
    ) -> dict:
        r"""Diagnostic check of the single-step branch probabilities.

        The Kamrad-Ritchken / Boyle scheme matches the first two moments of
        the **log-return**, so the probabilities sum to one exactly while the
        discrete *arithmetic* martingale identity
        :math:`p_u u + p_m + p_d d = e^{(r-q)\Delta t}` holds only to
        :math:`O(\Delta t^{3/2})` and vanishes as ``n_steps`` grows. This
        method reports both so callers can verify the construction.

        Parameters
        ----------
        r, q, sigma, T : float
            Rate, dividend yield, volatility and total time to expiry.

        Returns
        -------
        dict
            ``pu, pm, pd, prob_sum, martingale_lhs, martingale_rhs,
            martingale_abs_err, probabilities_valid``.

        Examples
        --------
        >>> chk = BoyleTrinomial(150).validate_probabilities(0.05, 0.0, 0.2, 1.0)
        >>> abs(chk["prob_sum"] - 1.0) < 1e-12 and chk["probabilities_valid"]
        True
        """
        dt = T / self.n_steps
        u, d, pu, pm, pd = self._probabilities(r, q, sigma, dt)
        lhs = pu * u + pm * 1.0 + pd * d
        rhs = float(np.exp((r - q) * dt))
        return {
            "pu": float(pu),
            "pm": float(pm),
            "pd": float(pd),
            "prob_sum": float(pu + pm + pd),
            "martingale_lhs": float(lhs),
            "martingale_rhs": rhs,
            "martingale_abs_err": float(abs(lhs - rhs)),
            "probabilities_valid": bool(
                0.0 <= pu <= 1.0 and 0.0 <= pm <= 1.0 and 0.0 <= pd <= 1.0
            ),
        }

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Union[OptionType, str] = OptionType.CALL,
        q: float = 0.0,
        exercise_style: Union[ExerciseStyle, str] = ExerciseStyle.EUROPEAN,
        adaptive_mesh: bool = False,
    ) -> float:
        """Price an option on the trinomial tree.

        Parameters
        ----------
        S, K, T, r, sigma, q : float
            Spot, strike, time to expiry, rate, volatility, dividend yield.
        option_type : OptionType or str, optional
            Call or put. Default call.
        exercise_style : ExerciseStyle or str, optional
            European or American. Default European.
        adaptive_mesh : bool, optional
            If ``True``, doubles the step count when the contract is near the
            money (``0.95 <= S/K <= 1.05``) to sharpen accuracy where the
            payoff kink sits. Default ``False``.

        Returns
        -------
        float
            Present value of the option.

        Examples
        --------
        >>> m = BoyleTrinomial(n_steps=150)
        >>> round(m.price(100, 100, 1.0, 0.05, 0.2), 2)
        10.45
        """
        option_type = OptionType.coerce(option_type)
        exercise_style = ExerciseStyle.coerce(exercise_style)

        n = self.n_steps
        if adaptive_mesh and 0.95 <= (S / K) <= 1.05:
            n *= 2

        dt = T / n
        u, d, pu, pm, pd = self._probabilities(r, q, sigma, dt)
        disc = np.exp(-r * dt)

        # Terminal asset prices: node index j in [0, 2n], net up moves = j - n
        j = np.arange(2 * n + 1)
        net_moves = j - n
        terminal = S * u ** net_moves.astype(float)
        self.tree = terminal

        sign = 1.0 if option_type is OptionType.CALL else -1.0
        values = np.maximum(sign * (terminal - K), 0.0)

        is_american = exercise_style is ExerciseStyle.AMERICAN
        for step in range(n - 1, -1, -1):
            width = 2 * step + 1
            cont = disc * (
                pu * values[2 : width + 2]
                + pm * values[1 : width + 1]
                + pd * values[0:width]
            )
            if is_american:
                net = np.arange(width) - step
                asset = S * u ** net.astype(float)
                intrinsic = np.maximum(sign * (asset - K), 0.0)
                cont = np.maximum(cont, intrinsic)
            values = cont

        return float(values[0])
