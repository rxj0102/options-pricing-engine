"""Tree-model convergence analysis utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from pricing.base import OptionContract
from pricing.binomial import CRRBinomial
from pricing.black_scholes import BlackScholesMerton
from pricing.trinomial import BoyleTrinomial

DEFAULT_STEPS: Sequence[int] = (5, 10, 25, 50, 100, 200, 400, 800)


class ConvergenceAnalyzer:
    """Study how tree prices converge to the analytical BSM benchmark.

    Examples
    --------
    >>> from pricing.base import OptionContract
    >>> c = OptionContract(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    >>> res = ConvergenceAnalyzer().analyze(c, [50, 400])
    >>> res["binomial_order"] > 0
    True
    """

    def __init__(self) -> None:
        self.bsm = BlackScholesMerton()

    def convergence_data(
        self,
        contract: OptionContract,
        n_steps_list: Sequence[int] = DEFAULT_STEPS,
    ) -> pd.DataFrame:
        """Compute tree prices and absolute errors across step counts.

        Parameters
        ----------
        contract : OptionContract
            Contract to value (European recommended for BSM benchmarking).
        n_steps_list : sequence of int, optional
            Step counts. Default geometric ladder up to 800.

        Returns
        -------
        pandas.DataFrame
            Columns ``n_steps``, ``binomial``, ``trinomial``, ``bsm``,
            ``binomial_abs_err``, ``trinomial_abs_err``.
        """
        bsm_price = self.bsm.price_contract(contract)
        rows = []
        for n in n_steps_list:
            bino = CRRBinomial(n_steps=int(n)).price(
                contract.S,
                contract.K,
                contract.T,
                contract.r,
                contract.sigma,
                contract.option_type,
                contract.q,
                contract.exercise_style,
            )
            tri = BoyleTrinomial(n_steps=int(n)).price(
                contract.S,
                contract.K,
                contract.T,
                contract.r,
                contract.sigma,
                contract.option_type,
                contract.q,
                contract.exercise_style,
            )
            rows.append(
                {
                    "n_steps": int(n),
                    "binomial": bino,
                    "trinomial": tri,
                    "bsm": bsm_price,
                    "binomial_abs_err": abs(bino - bsm_price),
                    "trinomial_abs_err": abs(tri - bsm_price),
                }
            )
        return pd.DataFrame(rows)

    def analyze(
        self,
        contract: OptionContract,
        n_steps_list: Sequence[int] = DEFAULT_STEPS,
    ) -> dict:
        r"""Estimate the empirical convergence order of each tree.

        The order :math:`p` is fitted via a log-log regression of absolute
        error against step count, ``err ~ C * n^{-p}``.

        Parameters
        ----------
        contract : OptionContract
            Contract to value.
        n_steps_list : sequence of int, optional
            Step counts. Default geometric ladder.

        Returns
        -------
        dict
            ``data`` (DataFrame), ``binomial_order``, ``trinomial_order``,
            ``bsm_price``.

        Examples
        --------
        >>> from pricing.base import OptionContract
        >>> c = OptionContract(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        >>> out = ConvergenceAnalyzer().analyze(c, [50, 100, 200, 400])
        >>> "data" in out and "binomial_order" in out
        True
        """
        df = self.convergence_data(contract, n_steps_list)

        def order(errors: np.ndarray) -> float:
            mask = errors > 0
            if mask.sum() < 2:
                return float("nan")
            log_n = np.log(df["n_steps"].to_numpy()[mask])
            log_e = np.log(errors[mask])
            slope = np.polyfit(log_n, log_e, 1)[0]
            return float(-slope)

        return {
            "data": df,
            "binomial_order": order(df["binomial_abs_err"].to_numpy()),
            "trinomial_order": order(df["trinomial_abs_err"].to_numpy()),
            "bsm_price": float(df["bsm"].iloc[0]),
        }
