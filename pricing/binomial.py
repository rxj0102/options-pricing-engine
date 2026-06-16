"""Cox-Ross-Rubinstein (1979) binomial tree pricer."""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

from pricing.base import ExerciseStyle, OptionPricer, OptionType

DEFAULT_N_STEPS: int = 200
DEFAULT_CONVERGENCE_STEPS: Tuple[int, ...] = (10, 25, 50, 100, 200, 500)


class CRRBinomial(OptionPricer):
    r"""Cox-Ross-Rubinstein binomial tree for European and American options.

    With :math:`\Delta t = T/n` the CRR parameterisation sets

    .. math::

        u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u, \quad
        p = \frac{e^{(r-q)\Delta t} - d}{u - d}.

    American options take ``max(intrinsic, continuation)`` at every node.

    Parameters
    ----------
    n_steps : int, optional
        Number of time steps in the tree. Default ``200``.

    Examples
    --------
    >>> model = CRRBinomial(n_steps=500)
    >>> round(model.price(100, 100, 1.0, 0.05, 0.2), 2)
    10.45
    """

    name = "CRR Binomial"

    def __init__(self, n_steps: int = DEFAULT_N_STEPS) -> None:
        self.n_steps = int(n_steps)
        self.tree: np.ndarray | None = None

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
    ) -> float:
        """Price an option on the binomial tree.

        Parameters
        ----------
        S, K, T, r, sigma, q : float
            Spot, strike, time to expiry, rate, volatility, dividend yield.
        option_type : OptionType or str, optional
            Call or put. Default call.
        exercise_style : ExerciseStyle or str, optional
            European or American. Default European.

        Returns
        -------
        float
            Present value of the option.

        Examples
        --------
        >>> m = CRRBinomial(n_steps=300)
        >>> am = m.price(100, 100, 1.0, 0.05, 0.2, "put", exercise_style="american")
        >>> eu = m.price(100, 100, 1.0, 0.05, 0.2, "put", exercise_style="european")
        >>> am >= eu
        True
        """
        price, _, _ = self.price_with_tree(
            S, K, T, r, sigma, option_type, q, exercise_style
        )
        return price

    def price_with_tree(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Union[OptionType, str] = OptionType.CALL,
        q: float = 0.0,
        exercise_style: Union[ExerciseStyle, str] = ExerciseStyle.EUROPEAN,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Price and also return the stock and option lattices.

        Parameters
        ----------
        S, K, T, r, sigma, q : float
            Standard pricing inputs.
        option_type : OptionType or str, optional
            Call or put. Default call.
        exercise_style : ExerciseStyle or str, optional
            European or American. Default European.

        Returns
        -------
        tuple
            ``(price, stock_tree, option_tree)`` where each tree is a
            ``(n_steps + 1, n_steps + 1)`` lower-triangular array. The stock
            tree is cached on ``self.tree`` for visualisation.

        Examples
        --------
        >>> price, stock, opt = CRRBinomial(50).price_with_tree(
        ...     100, 100, 1.0, 0.05, 0.2)
        >>> stock.shape
        (51, 51)
        """
        option_type = OptionType.coerce(option_type)
        exercise_style = ExerciseStyle.coerce(exercise_style)
        n = self.n_steps
        dt = T / n
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        disc = np.exp(-r * dt)
        p = (np.exp((r - q) * dt) - d) / (u - d)

        # Stock price lattice: stock[i, j] = S * u**(i - j) * d**j  (j down moves)
        steps = np.arange(n + 1)
        stock_tree = np.zeros((n + 1, n + 1))
        for i in steps:
            j = np.arange(i + 1)
            stock_tree[i, : i + 1] = S * u ** (i - j) * d**j
        self.tree = stock_tree

        # Terminal payoffs
        sign = 1.0 if option_type is OptionType.CALL else -1.0
        option_tree = np.zeros((n + 1, n + 1))
        terminal = np.maximum(sign * (stock_tree[n, : n + 1] - K), 0.0)
        option_tree[n, : n + 1] = terminal

        # Backward induction
        is_american = exercise_style is ExerciseStyle.AMERICAN
        for i in range(n - 1, -1, -1):
            cont = disc * (
                p * option_tree[i + 1, : i + 1]
                + (1.0 - p) * option_tree[i + 1, 1 : i + 2]
            )
            if is_american:
                intrinsic = np.maximum(sign * (stock_tree[i, : i + 1] - K), 0.0)
                cont = np.maximum(cont, intrinsic)
            option_tree[i, : i + 1] = cont

        return float(option_tree[0, 0]), stock_tree, option_tree

    def convergence(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Union[OptionType, str] = OptionType.CALL,
        q: float = 0.0,
        exercise_style: Union[ExerciseStyle, str] = ExerciseStyle.EUROPEAN,
        n_steps_list: Tuple[int, ...] = DEFAULT_CONVERGENCE_STEPS,
    ) -> "tuple[List[int], List[float]]":
        """Price across increasing step counts for convergence analysis.

        Parameters
        ----------
        S, K, T, r, sigma, q : float
            Standard pricing inputs.
        option_type : OptionType or str, optional
            Call or put. Default call.
        exercise_style : ExerciseStyle or str, optional
            European or American. Default European.
        n_steps_list : tuple of int, optional
            Step counts to evaluate. Default ``(10, 25, 50, 100, 200, 500)``.

        Returns
        -------
        tuple
            ``(steps, prices)`` ready for a convergence plot.

        Examples
        --------
        >>> steps, prices = CRRBinomial().convergence(100, 100, 1.0, 0.05, 0.2)
        >>> len(steps) == len(prices)
        True
        """
        original = self.n_steps
        steps: List[int] = []
        prices: List[float] = []
        try:
            for n in n_steps_list:
                self.n_steps = int(n)
                steps.append(int(n))
                prices.append(
                    self.price(S, K, T, r, sigma, option_type, q, exercise_style)
                )
        finally:
            self.n_steps = original
        return steps, prices
