"""Stress testing across spot/vol/rate/time parameter grids."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from pricing.base import OptionContract, OptionPricer
from pricing.black_scholes import BlackScholesMerton

#: Pre-built market stress scenarios as multiplicative/absolute shocks.
#: ``dS`` and ``dsigma`` are fractional spot/vol moves; ``dr`` is absolute;
#: ``dT`` is in years.
PREBUILT_SCENARIOS: Dict[str, Dict[str, float]] = {
    "2008 Crisis": {"dS": -0.40, "dsigma": 0.40, "dr": 0.0, "dT": 0.0},
    "2020 COVID": {"dS": -0.35, "dsigma": 0.60, "dr": 0.0, "dT": 0.0},
    "Rate Hike 100bps": {"dS": 0.0, "dsigma": 0.0, "dr": 0.01, "dT": 0.0},
    "Earnings Surprise +15%": {"dS": 0.15, "dsigma": -0.05, "dr": 0.0, "dT": 0.0},
    "Earnings Surprise -15%": {"dS": -0.15, "dsigma": 0.10, "dr": 0.0, "dT": 0.0},
}

_MIN_VOL: float = 1e-6
_MIN_TIME: float = 1e-6


class StressTester:
    """Revalue a contract across shocked market scenarios.

    Parameters
    ----------
    pricer : OptionPricer, optional
        Model used to revalue. Default Black-Scholes-Merton.

    Examples
    --------
    >>> from pricing.base import OptionContract
    >>> c = OptionContract(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    >>> grid = StressTester().spot_vol_grid(c, [-0.1, 0.0, 0.1], [-0.05, 0.05])
    >>> grid.shape
    (3, 2)
    """

    def __init__(self, pricer: OptionPricer | None = None) -> None:
        self.pricer = pricer or BlackScholesMerton()

    def _reprice(
        self,
        contract: OptionContract,
        dS: float,
        dsigma: float,
        dr: float = 0.0,
        dT: float = 0.0,
    ) -> float:
        shocked = contract.with_(
            S=contract.S * (1.0 + dS),
            sigma=max(contract.sigma * (1.0 + dsigma), _MIN_VOL),
            r=contract.r + dr,
            T=max(contract.T - dT, _MIN_TIME),
        )
        return self.pricer.price_contract(shocked)

    def spot_vol_grid(
        self,
        contract: OptionContract,
        S_shocks: Sequence[float],
        vol_shocks: Sequence[float],
    ) -> pd.DataFrame:
        """P&L grid over fractional spot and vol shocks.

        Parameters
        ----------
        contract : OptionContract
            Base contract.
        S_shocks : sequence of float
            Fractional spot moves (rows), e.g. ``-0.1`` for ``-10%``.
        vol_shocks : sequence of float
            Fractional vol moves (columns).

        Returns
        -------
        pandas.DataFrame
            P&L relative to the base price; index = spot shocks, columns =
            vol shocks (both formatted as percentages).

        Examples
        --------
        >>> from pricing.base import OptionContract
        >>> c = OptionContract(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        >>> g = StressTester().spot_vol_grid(c, [0.0], [0.0])
        >>> round(float(g.iloc[0, 0]), 6)
        0.0
        """
        base = self.pricer.price_contract(contract)
        data = np.array(
            [
                [self._reprice(contract, ds, dv) - base for dv in vol_shocks]
                for ds in S_shocks
            ]
        )
        return pd.DataFrame(
            data,
            index=[f"{s:+.0%}" for s in S_shocks],
            columns=[f"{v:+.0%}" for v in vol_shocks],
        )

    def rate_time_grid(
        self,
        contract: OptionContract,
        r_shocks: Sequence[float],
        T_values: Sequence[float],
    ) -> pd.DataFrame:
        """Price grid over absolute rate shocks and absolute maturities.

        Parameters
        ----------
        contract : OptionContract
            Base contract.
        r_shocks : sequence of float
            Absolute rate shocks (rows), e.g. ``0.01`` for +100bps.
        T_values : sequence of float
            Absolute times to expiry in years (columns).

        Returns
        -------
        pandas.DataFrame
            Repriced option values; index = rate shocks, columns = maturities.
        """
        data = np.array(
            [
                [
                    self.pricer.price_contract(
                        contract.with_(r=contract.r + dr, T=max(t, _MIN_TIME))
                    )
                    for t in T_values
                ]
                for dr in r_shocks
            ]
        )
        return pd.DataFrame(
            data,
            index=[f"{dr*1e4:+.0f}bps" for dr in r_shocks],
            columns=[f"{t:.2f}y" for t in T_values],
        )

    def scenario_pnl(
        self,
        contract: OptionContract,
        scenarios_dict: Mapping[str, Mapping[str, float]] | None = None,
    ) -> pd.DataFrame:
        """Evaluate named scenarios and report P&L impact.

        Parameters
        ----------
        contract : OptionContract
            Base contract.
        scenarios_dict : mapping, optional
            Map of ``scenario_name -> {dS, dsigma, dT, dr}`` where ``dS`` and
            ``dsigma`` are fractional moves. Default :data:`PREBUILT_SCENARIOS`.

        Returns
        -------
        pandas.DataFrame
            Columns ``scenario``, ``base_price``, ``stressed_price``,
            ``pnl``, ``pnl_pct``.

        Examples
        --------
        >>> from pricing.base import OptionContract
        >>> c = OptionContract(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        >>> df = StressTester().scenario_pnl(c)
        >>> "2008 Crisis" in set(df["scenario"])
        True
        """
        scenarios = dict(scenarios_dict) if scenarios_dict else PREBUILT_SCENARIOS
        base = self.pricer.price_contract(contract)
        rows = []
        for name, shock in scenarios.items():
            stressed = self._reprice(
                contract,
                dS=shock.get("dS", 0.0),
                dsigma=shock.get("dsigma", 0.0),
                dr=shock.get("dr", 0.0),
                dT=shock.get("dT", 0.0),
            )
            rows.append(
                {
                    "scenario": name,
                    "base_price": base,
                    "stressed_price": stressed,
                    "pnl": stressed - base,
                    "pnl_pct": (
                        (stressed - base) / base * 100.0 if base else float("nan")
                    ),
                }
            )
        return pd.DataFrame(rows)
