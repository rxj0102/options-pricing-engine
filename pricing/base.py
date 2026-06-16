"""Abstract base class and shared domain types for all pricing models.

This module defines the common contract every pricer implements so that the
Greek engine, validation suite, scenario tools, and Streamlit app can treat
all four models (Black-Scholes, Black-76, Binomial, Trinomial) uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Named constants (no magic numbers anywhere in the engine)
# ---------------------------------------------------------------------------
DAYS_PER_YEAR: int = 365
"""Calendar-day convention used for daily theta and time shocks."""

TRADING_DAYS_PER_YEAR: int = 252
"""Trading-day convention available for annualisation when required."""


class OptionType(str, Enum):
    """Right of the option holder."""

    CALL = "call"
    PUT = "put"

    @classmethod
    def coerce(cls, value: Union["OptionType", str]) -> "OptionType":
        """Coerce a string or enum into an :class:`OptionType`.

        Parameters
        ----------
        value : OptionType or str
            Either an existing enum member or a case-insensitive string
            (``"call"`` / ``"put"``).

        Returns
        -------
        OptionType
            The corresponding enum member.

        Examples
        --------
        >>> OptionType.coerce("CALL") is OptionType.CALL
        True
        """
        if isinstance(value, cls):
            return value
        text = getattr(value, "value", value)
        return cls(str(text).lower())


class ExerciseStyle(str, Enum):
    """Exercise style of the contract."""

    EUROPEAN = "european"
    AMERICAN = "american"

    @classmethod
    def coerce(cls, value: Union["ExerciseStyle", str]) -> "ExerciseStyle":
        """Coerce a string or enum into an :class:`ExerciseStyle`.

        Parameters
        ----------
        value : ExerciseStyle or str
            Either an existing enum member or a case-insensitive string.

        Returns
        -------
        ExerciseStyle
            The corresponding enum member.
        """
        if isinstance(value, cls):
            return value
        text = getattr(value, "value", value)
        return cls(str(text).lower())


@dataclass
class OptionContract:
    """Container describing a single option contract.

    Parameters
    ----------
    S : float
        Spot price of the underlying.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Continuously compounded risk-free rate (annualised).
    q : float, optional
        Continuous dividend yield (annualised). Default ``0.0``.
    sigma : float, optional
        Volatility (annualised). Default ``0.2``.
    option_type : OptionType or str, optional
        Call or put. Default :attr:`OptionType.CALL`.
    exercise_style : ExerciseStyle or str, optional
        European or American. Default :attr:`ExerciseStyle.EUROPEAN`.

    Examples
    --------
    >>> c = OptionContract(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    >>> c.option_type
    <OptionType.CALL: 'call'>
    """

    S: float
    K: float
    T: float
    r: float
    q: float = 0.0
    sigma: float = 0.2
    option_type: OptionType = OptionType.CALL
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN

    def __post_init__(self) -> None:
        self.option_type = OptionType.coerce(self.option_type)
        self.exercise_style = ExerciseStyle.coerce(self.exercise_style)

    def with_(self, **changes: float) -> "OptionContract":
        """Return a copy of this contract with selected fields overridden.

        Parameters
        ----------
        **changes
            Field overrides, e.g. ``contract.with_(S=105, sigma=0.25)``.

        Returns
        -------
        OptionContract
            A new contract instance; the original is left unchanged.

        Examples
        --------
        >>> c = OptionContract(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        >>> c.with_(S=110).S
        110
        """
        return replace(self, **changes)

    def intrinsic_value(self) -> float:
        """Undiscounted intrinsic value of the contract at the current spot.

        Returns
        -------
        float
            ``max(S - K, 0)`` for a call, ``max(K - S, 0)`` for a put.
        """
        if self.option_type is OptionType.CALL:
            return max(self.S - self.K, 0.0)
        return max(self.K - self.S, 0.0)


class OptionPricer(ABC):
    """Abstract base class implemented by every pricing model.

    Concrete subclasses must implement :meth:`price`. A default
    :meth:`price_batch` is provided that simply maps :meth:`price` across the
    rows of a DataFrame; vectorised models may override it for speed.
    """

    #: Human-readable model name, set by each subclass.
    name: str = "OptionPricer"

    @abstractmethod
    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Union[OptionType, str] = OptionType.CALL,
        q: float = 0.0,
    ) -> float:
        """Price a single option.

        Parameters
        ----------
        S : float
            Spot (or forward, for :class:`~pricing.black76.Black76`) price.
        K : float
            Strike price.
        T : float
            Time to expiry in years.
        r : float
            Continuously compounded risk-free rate.
        sigma : float
            Volatility (annualised).
        option_type : OptionType or str, optional
            Call or put. Default call.
        q : float, optional
            Continuous dividend yield. Default ``0.0``.

        Returns
        -------
        float
            Present value of the option.
        """
        raise NotImplementedError

    def price_contract(self, contract: OptionContract) -> float:
        """Price an :class:`OptionContract` instance.

        Parameters
        ----------
        contract : OptionContract
            The contract to value.

        Returns
        -------
        float
            Present value of the option.

        Examples
        --------
        >>> from pricing.black_scholes import BlackScholesMerton
        >>> c = OptionContract(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        >>> round(BlackScholesMerton().price_contract(c), 4)
        10.4506
        """
        return self.price(
            S=contract.S,
            K=contract.K,
            T=contract.T,
            r=contract.r,
            sigma=contract.sigma,
            option_type=contract.option_type,
            q=contract.q,
        )

    def price_batch(self, params_df: pd.DataFrame) -> pd.DataFrame:
        """Price a batch of contracts described by a DataFrame.

        Parameters
        ----------
        params_df : pandas.DataFrame
            Must contain columns ``S, K, T, r, sigma`` and optionally
            ``q`` and ``option_type``.

        Returns
        -------
        pandas.DataFrame
            Copy of the input with an additional ``price`` column.

        Examples
        --------
        >>> import pandas as pd
        >>> from pricing.black_scholes import BlackScholesMerton
        >>> df = pd.DataFrame({"S": [100], "K": [100], "T": [1.0],
        ...                    "r": [0.05], "sigma": [0.2]})
        >>> BlackScholesMerton().price_batch(df)["price"].round(4).tolist()
        [10.4506]
        """
        out = params_df.copy()
        out["price"] = [
            self.price(
                S=row.S,
                K=row.K,
                T=row.T,
                r=row.r,
                sigma=row.sigma,
                option_type=getattr(row, "option_type", OptionType.CALL),
                q=getattr(row, "q", 0.0),
            )
            for row in params_df.itertuples(index=False)
        ]
        return out

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}(name={self.name!r})"
