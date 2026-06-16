"""Cross-model price comparison and tree convergence tables."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from pricing.base import ExerciseStyle, OptionContract, OptionType
from pricing.binomial import CRRBinomial
from pricing.black76 import Black76
from pricing.black_scholes import BlackScholesMerton
from pricing.trinomial import BoyleTrinomial

#: Deviation above this (fraction of BSM price) is flagged as a warning.
WARNING_THRESHOLD: float = 0.005
BPS_PER_UNIT: float = 1e4
DEFAULT_CONVERGENCE_STEPS: Sequence[int] = (10, 25, 50, 100, 200, 500)


class ModelComparison:
    """Price one contract across all four models and compare.

    Parameters
    ----------
    n_steps : int, optional
        Step count for the tree models. Default ``500``.

    Examples
    --------
    >>> from pricing.base import OptionContract
    >>> c = OptionContract(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    >>> table = ModelComparison().comparison_table(c)
    >>> set(table["model"]) >= {"Black-Scholes-Merton", "Black-76"}
    True
    """

    def __init__(self, n_steps: int = 500) -> None:
        self.n_steps = int(n_steps)
        self.bsm = BlackScholesMerton()
        self.black76 = Black76()
        self.binomial = CRRBinomial(n_steps=self.n_steps)
        self.trinomial = BoyleTrinomial(n_steps=self.n_steps)

    def _prices(self, contract: OptionContract) -> "dict[str, float]":
        forward = contract.S * np.exp((contract.r - contract.q) * contract.T)
        european = contract.exercise_style is ExerciseStyle.EUROPEAN
        prices = {
            self.bsm.name: self.bsm.price_contract(contract),
            self.black76.name: self.black76.price(
                forward,
                contract.K,
                contract.T,
                contract.r,
                contract.sigma,
                contract.option_type,
            ),
            self.binomial.name: self.binomial.price(
                contract.S,
                contract.K,
                contract.T,
                contract.r,
                contract.sigma,
                contract.option_type,
                contract.q,
                contract.exercise_style,
            ),
            self.trinomial.name: self.trinomial.price(
                contract.S,
                contract.K,
                contract.T,
                contract.r,
                contract.sigma,
                contract.option_type,
                contract.q,
                contract.exercise_style,
            ),
        }
        # Black-76 only reproduces BSM for European exercise.
        if not european:
            prices.pop(self.black76.name)
        return prices

    def comparison_table(self, contract: OptionContract) -> pd.DataFrame:
        """Tabulate all model prices with deviation from BSM.

        Parameters
        ----------
        contract : OptionContract
            Contract to value across models.

        Returns
        -------
        pandas.DataFrame
            Columns ``model``, ``price``, ``abs_dev_vs_bsm``,
            ``dev_bps_vs_bsm``, ``pct_dev_vs_bsm``, ``warning``.

        Examples
        --------
        >>> from pricing.base import OptionContract
        >>> c = OptionContract(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        >>> df = ModelComparison().comparison_table(c)
        >>> bool((~df["warning"]).all())
        True
        """
        prices = self._prices(contract)
        reference = prices[self.bsm.name]
        rows: List[dict] = []
        for model, price in prices.items():
            abs_dev = price - reference
            pct_dev = abs_dev / reference if reference else float("nan")
            rows.append(
                {
                    "model": model,
                    "price": price,
                    "abs_dev_vs_bsm": abs_dev,
                    "dev_bps_vs_bsm": (
                        (abs_dev / reference * BPS_PER_UNIT)
                        if reference
                        else float("nan")
                    ),
                    "pct_dev_vs_bsm": pct_dev * 100.0,
                    "warning": bool(abs(pct_dev) > WARNING_THRESHOLD),
                }
            )
        return pd.DataFrame(rows)

    def convergence_table(
        self,
        contract: OptionContract,
        n_steps_list: Sequence[int] = DEFAULT_CONVERGENCE_STEPS,
    ) -> pd.DataFrame:
        """Tabulate Binomial/Trinomial prices vs step count against BSM.

        Parameters
        ----------
        contract : OptionContract
            Contract to value.
        n_steps_list : sequence of int, optional
            Step counts. Default ``(10, 25, 50, 100, 200, 500)``.

        Returns
        -------
        pandas.DataFrame
            Columns ``n_steps``, ``binomial``, ``trinomial``, ``bsm``,
            ``binomial_err``, ``trinomial_err``.

        Examples
        --------
        >>> from pricing.base import OptionContract
        >>> c = OptionContract(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        >>> df = ModelComparison().convergence_table(c, [50, 200])
        >>> bool(df["trinomial_err"].iloc[-1] < df["trinomial_err"].iloc[0])
        True
        """
        bsm_price = self.bsm.price_contract(contract)
        rows: List[dict] = []
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
                    "binomial_err": abs(bino - bsm_price),
                    "trinomial_err": abs(tri - bsm_price),
                }
            )
        return pd.DataFrame(rows)
