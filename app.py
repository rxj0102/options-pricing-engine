"""Streamlit dashboard for the options-pricing-engine.

Run with::

    streamlit run app.py

The app exposes six tabs: an option pricer, a Greek engine, pricing
validation, volatility smile construction, scenario analysis, and model
documentation. All numerics are delegated to the library modules so the UI
remains a thin presentation layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from greeks.greek_engine import GreekEngine
from pricing.base import ExerciseStyle, OptionContract, OptionType
from pricing.binomial import CRRBinomial
from pricing.black76 import Black76
from pricing.black_scholes import BlackScholesMerton
from pricing.trinomial import BoyleTrinomial
from scenarios.greek_pnl import GreekPnLAttribution
from scenarios.stress_test import PREBUILT_SCENARIOS, StressTester
from utils.plotting import (GREEK_COLORS, heatmap_plot, line_plot,
                            surface_plot, waterfall_plot)
from validation.convergence import ConvergenceAnalyzer
from validation.model_comparison import ModelComparison
from validation.put_call_parity import PutCallParityCheck
from volatility.sabr import SABRModel, SABRParams
from volatility.smile import VolatilitySmile

TREE_STEPS = 300

st.set_page_config(
    page_title="Options Pricing Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sidebar — global contract inputs
# ---------------------------------------------------------------------------
def sidebar_inputs() -> "tuple[OptionContract, str]":
    """Render the sidebar and return the contract and selected model."""
    st.sidebar.title("⚙️ Contract Inputs")
    S = st.sidebar.number_input("Spot (S)", value=100.0, min_value=0.01, step=1.0)
    K = st.sidebar.number_input("Strike (K)", value=100.0, min_value=0.01, step=1.0)
    T = st.sidebar.number_input(
        "Time to expiry (years, T)", value=1.0, min_value=0.01, step=0.25
    )
    r = st.sidebar.number_input(
        "Risk-free rate (r)", value=0.05, step=0.01, format="%.4f"
    )
    q = st.sidebar.number_input(
        "Dividend yield (q)", value=0.0, step=0.01, format="%.4f"
    )
    sigma = st.sidebar.number_input(
        "Volatility (σ)", value=0.20, min_value=0.001, step=0.01, format="%.4f"
    )
    option_type = st.sidebar.radio("Option type", ["Call", "Put"], horizontal=True)
    exercise = st.sidebar.radio(
        "Exercise style", ["European", "American"], horizontal=True
    )
    model = st.sidebar.selectbox(
        "Pricing model",
        ["Black-Scholes", "Black-76", "Binomial", "Trinomial", "ALL (compare)"],
    )
    contract = OptionContract(
        S=S,
        K=K,
        T=T,
        r=r,
        q=q,
        sigma=sigma,
        option_type=OptionType.coerce(option_type),
        exercise_style=ExerciseStyle.coerce(exercise),
    )
    return contract, model


def price_one(model_name: str, contract: OptionContract, opt_type: OptionType) -> float:
    """Price a contract with a named model for a specific option type."""
    c = contract.with_(option_type=opt_type)
    if model_name == "Black-Scholes":
        return BlackScholesMerton().price_contract(c)
    if model_name == "Black-76":
        fwd = c.S * np.exp((c.r - c.q) * c.T)
        return Black76().price(fwd, c.K, c.T, c.r, c.sigma, c.option_type)
    if model_name == "Binomial":
        return CRRBinomial(TREE_STEPS).price(
            c.S, c.K, c.T, c.r, c.sigma, c.option_type, c.q, c.exercise_style
        )
    if model_name == "Trinomial":
        return BoyleTrinomial(TREE_STEPS).price(
            c.S, c.K, c.T, c.r, c.sigma, c.option_type, c.q, c.exercise_style
        )
    return BlackScholesMerton().price_contract(c)


# ---------------------------------------------------------------------------
# Tab 1 — Option Pricer
# ---------------------------------------------------------------------------
def tab_pricer(contract: OptionContract, model: str) -> None:
    st.header("Option Pricer")
    base_model = "Black-Scholes" if model.startswith("ALL") else model

    call_price = price_one(base_model, contract, OptionType.CALL)
    put_price = price_one(base_model, contract, OptionType.PUT)
    is_call = contract.option_type is OptionType.CALL
    intrinsic = contract.intrinsic_value()
    selected = call_price if is_call else put_price
    time_value = selected - intrinsic

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Call Price", f"{call_price:,.4f}")
    c2.metric("Put Price", f"{put_price:,.4f}")
    c3.metric("Intrinsic Value", f"{intrinsic:,.4f}")
    c4.metric("Time Value", f"{time_value:,.4f}")

    if model.startswith("ALL"):
        st.subheader("Cross-Model Comparison")
        table = ModelComparison(n_steps=TREE_STEPS).comparison_table(contract)
        st.dataframe(
            table.style.format(
                {
                    "price": "{:.4f}",
                    "abs_dev_vs_bsm": "{:.4f}",
                    "dev_bps_vs_bsm": "{:.2f}",
                    "pct_dev_vs_bsm": "{:.4f}",
                }
            ),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Tab 2 — Greek Engine
# ---------------------------------------------------------------------------
def tab_greeks(contract: OptionContract) -> None:
    st.header("Greek Engine")
    greeks = GreekEngine(BlackScholesMerton()).compute_all(contract)

    cols = st.columns(5)
    for col, (name, color) in zip(cols, GREEK_COLORS.items()):
        value = getattr(greeks, name)
        col.markdown(
            f"<div style='border-left:4px solid {color};padding-left:8px'>"
            f"<small>{name.upper()}</small><br>"
            f"<span style='font-size:1.5rem'>{value:,.4f}</span></div>",
            unsafe_allow_html=True,
        )

    st.caption(
        f"Higher-order — Vanna {greeks.vanna:.4f} · Volga {greeks.volga:.4f} · "
        f"Charm {greeks.charm:.6f} · Speed {greeks.speed:.6f}"
    )

    st.subheader("Greek Surface (Spot × Vol)")
    greek_name = st.selectbox(
        "Greek", ["delta", "gamma", "vega", "theta", "rho"], key="surface_greek"
    )
    S_range = np.linspace(contract.S * 0.6, contract.S * 1.4, 40)
    sigma_range = np.linspace(0.05, 0.6, 40)
    z = GreekEngine().greek_surface(
        greek_name,
        S_range,
        sigma_range,
        contract.K,
        contract.T,
        contract.r,
        contract.q,
        contract.option_type,
    )
    st.plotly_chart(
        surface_plot(
            sigma_range,
            S_range,
            z,
            f"{greek_name.title()} surface",
            z_title=greek_name.title(),
        ),
        use_container_width=True,
    )

    col_a, col_b = st.columns(2)
    spots = np.linspace(contract.S * 0.6, contract.S * 1.4, 60)
    eng = GreekEngine()
    call_delta = [
        eng.compute_all(contract.with_(S=s, option_type=OptionType.CALL)).delta
        for s in spots
    ]
    put_delta = [
        eng.compute_all(contract.with_(S=s, option_type=OptionType.PUT)).delta
        for s in spots
    ]
    col_a.plotly_chart(
        line_plot(
            spots,
            {"Call Δ": call_delta, "Put Δ": put_delta},
            "Delta vs Spot",
            "Spot",
            "Delta",
        ),
        use_container_width=True,
    )
    gamma = [eng.compute_all(contract.with_(S=s)).gamma for s in spots]
    col_b.plotly_chart(
        line_plot(
            spots / contract.K,
            {"Gamma": gamma},
            "Gamma Smile",
            "Moneyness (S/K)",
            "Gamma",
        ),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Tab 3 — Pricing Validation
# ---------------------------------------------------------------------------
def tab_validation(contract: OptionContract) -> None:
    st.header("Pricing Validation")
    bsm = BlackScholesMerton()
    call = bsm.price_contract(contract.with_(option_type=OptionType.CALL))
    put = bsm.price_contract(contract.with_(option_type=OptionType.PUT))
    parity = PutCallParityCheck().check(
        call, put, contract.S, contract.K, contract.T, contract.r, contract.q
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        if parity["passes"]:
            st.success(f"✅ Put-Call Parity holds ({parity['violation_bps']:.3f} bps)")
        else:
            st.error(f"❌ Parity violated ({parity['violation_bps']:.3f} bps)")
        st.json(
            {k: round(v, 6) if isinstance(v, float) else v for k, v in parity.items()}
        )
    with col2:
        st.subheader("Implied Vol Roundtrip")
        recovered = bsm.implied_vol(
            call, contract.S, contract.K, contract.T, contract.r, contract.q, "call"
        )
        reprice = bsm.price(
            contract.S,
            contract.K,
            contract.T,
            contract.r,
            recovered,
            "call",
            contract.q,
        )
        st.metric("Input σ", f"{contract.sigma:.4f}")
        st.metric(
            "Recovered σ",
            f"{recovered:.4f}",
            delta=f"{recovered - contract.sigma:+.2e}",
        )
        st.metric("Reprice error", f"{abs(reprice - call):.2e}")

    st.subheader("Tree Convergence to BSM")
    analysis = ConvergenceAnalyzer().analyze(
        contract.with_(exercise_style=ExerciseStyle.EUROPEAN)
    )
    df = analysis["data"]
    fig = line_plot(
        df["n_steps"],
        {"Binomial": df["binomial"], "Trinomial": df["trinomial"]},
        "Convergence vs Steps",
        "n_steps",
        "Price",
    )
    fig.add_hline(
        y=analysis["bsm_price"],
        line_dash="dash",
        annotation_text="BSM",
        line_color="white",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Empirical convergence order — Binomial ≈ {analysis['binomial_order']:.2f}, "
        f"Trinomial ≈ {analysis['trinomial_order']:.2f}"
    )

    st.subheader("Cross-Model Comparison")
    st.dataframe(
        ModelComparison(n_steps=TREE_STEPS)
        .comparison_table(contract)
        .style.format(
            {
                "price": "{:.4f}",
                "abs_dev_vs_bsm": "{:.4f}",
                "dev_bps_vs_bsm": "{:.2f}",
                "pct_dev_vs_bsm": "{:.4f}",
            }
        ),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Tab 4 — Volatility Smile
# ---------------------------------------------------------------------------
def tab_volatility(contract: OptionContract) -> None:
    st.header("Volatility Smile")
    uploaded = st.file_uploader(
        "Upload CSV with columns: strike, call_price", type="csv"
    )

    forward = contract.S * np.exp((contract.r - contract.q) * contract.T)
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        quotes = [(row.strike, row.call_price, "call") for row in data.itertuples()]
    else:
        st.info("No file uploaded — using a synthetic SABR-generated smile.")
        sabr = SABRModel()
        params = SABRParams(alpha=0.2 * forward**0.0, beta=1.0, rho=-0.3, nu=0.5)
        strikes = np.linspace(contract.S * 0.7, contract.S * 1.3, 13)
        bsm = BlackScholesMerton()
        quotes = []
        for k in strikes:
            iv = sabr.implied_vol(
                forward, k, contract.T, params.alpha, params.beta, params.rho, params.nu
            )
            price = bsm.price(
                contract.S, k, contract.T, contract.r, iv, "call", contract.q
            )
            quotes.append((float(k), float(price), "call"))

    smile = VolatilitySmile(quotes, contract.S, contract.T, contract.r, contract.q)
    plot = smile.smile_plot_data()

    sabr = SABRModel()
    sabr_params = sabr.calibrate(
        plot["strike"], plot["market_ivol"], forward, contract.T, beta=1.0
    )
    sabr_curve = sabr.smile_from_params(
        forward, contract.T, plot["strike"], sabr_params
    )

    col1, col2, col3 = st.columns(3)
    col1.plotly_chart(
        line_plot(
            plot["strike"],
            {
                "Market": plot["market_ivol"],
                "SVI": plot["svi_ivol"],
                "SABR": sabr_curve,
            },
            "IV vs Strike",
            "Strike",
            "Implied Vol",
        ),
        use_container_width=True,
    )
    col2.plotly_chart(
        line_plot(
            plot["moneyness"],
            {"Market": plot["market_ivol"], "Poly fit": plot["poly_ivol"]},
            "IV vs Moneyness",
            "K/S",
            "Implied Vol",
        ),
        use_container_width=True,
    )
    deltas = [
        GreekEngine().compute_all(contract.with_(K=k, sigma=iv)).delta
        for k, iv in zip(plot["strike"], plot["market_ivol"])
    ]
    col3.plotly_chart(
        line_plot(
            deltas,
            {"Market": plot["market_ivol"]},
            "IV vs Delta",
            "Delta",
            "Implied Vol",
        ),
        use_container_width=True,
    )

    p = plot["svi_params"]
    st.caption(
        f"SVI: a={p.a:.4f}, b={p.b:.4f}, ρ={p.rho:.4f}, m={p.m:.4f}, σ={p.sigma:.4f}  |  "
        f"SABR: α={sabr_params.alpha:.4f}, β={sabr_params.beta:.2f}, "
        f"ρ={sabr_params.rho:.4f}, ν={sabr_params.nu:.4f}"
    )


# ---------------------------------------------------------------------------
# Tab 5 — Scenario Analysis
# ---------------------------------------------------------------------------
def tab_scenarios(contract: OptionContract) -> None:
    st.header("Scenario Analysis")
    st.subheader("Stress Test Heatmap — P&L vs (ΔSpot%, ΔVol%)")
    spot_shocks = np.round(np.linspace(-0.3, 0.3, 13), 3)
    vol_shocks = np.round(np.linspace(-0.5, 0.5, 11), 3)
    grid = StressTester().spot_vol_grid(contract, list(spot_shocks), list(vol_shocks))
    st.plotly_chart(
        heatmap_plot(vol_shocks, spot_shocks, grid.to_numpy(), "P&L Stress Grid"),
        use_container_width=True,
    )

    st.subheader("Pre-Built Crisis Scenarios")
    st.dataframe(
        StressTester()
        .scenario_pnl(contract)
        .style.format(
            {
                "base_price": "{:.4f}",
                "stressed_price": "{:.4f}",
                "pnl": "{:.4f}",
                "pnl_pct": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Greek P&L Attribution")
    c1, c2, c3, c4 = st.columns(4)
    dS_pct = c1.slider("ΔSpot %", -30.0, 30.0, 5.0, 0.5) / 100.0
    dsigma = c2.slider("Δσ (abs)", -0.30, 0.30, 0.02, 0.01)
    dT_days = c3.slider("ΔT (days)", 0.0, 60.0, 1.0, 1.0)
    dr_bps = c4.slider("Δr (bps)", -200.0, 200.0, 0.0, 5.0)

    waterfall = GreekPnLAttribution().waterfall_data(
        contract,
        dS=contract.S * dS_pct,
        dsigma=dsigma,
        dT=dT_days / 365.0,
        dr=dr_bps / 1e4,
    )
    st.plotly_chart(
        waterfall_plot(
            waterfall["component"],
            waterfall["pnl"],
            waterfall["measure"],
            "Greek P&L Waterfall",
        ),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Tab 6 — Model Documentation
# ---------------------------------------------------------------------------
def tab_documentation() -> None:
    st.header("Model Documentation")

    st.subheader("Black-Scholes-Merton (1973)")
    st.latex(
        r"d_1 = \frac{\ln(S/K) + (r - q + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}"
    )
    st.latex(r"C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)")

    st.subheader("Black-76 (1976)")
    st.latex(
        r"d_1 = \frac{\ln(F/K) + \tfrac{1}{2}\sigma^2 T}{\sigma\sqrt{T}}, \quad C = e^{-rT}\left[F N(d_1) - K N(d_2)\right]"
    )

    st.subheader("Cox-Ross-Rubinstein Binomial (1979)")
    st.latex(
        r"u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u, \quad p = \frac{e^{(r-q)\Delta t} - d}{u - d}"
    )

    st.subheader("Model Assumptions")
    st.table(
        pd.DataFrame(
            {
                "Model": ["Black-Scholes", "Black-76", "Binomial", "Trinomial"],
                "Underlier": [
                    "Spot + yield",
                    "Forward/Future",
                    "Spot + yield",
                    "Spot + yield",
                ],
                "Exercise": ["European", "European", "Euro/American", "Euro/American"],
                "Analytical Greeks": ["Yes", "Yes", "No (FD)", "No (FD)"],
            }
        )
    )

    st.subheader("Seminal References")
    st.markdown(
        "- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.*\n"
        "- Black, F. (1976). *The Pricing of Commodity Contracts.*\n"
        "- Cox, J., Ross, S. & Rubinstein, M. (1979). *Option Pricing: A Simplified Approach.*\n"
        "- Boyle, P. (1986). *Option Valuation Using a Three-Jump Process.*\n"
        "- Hagan, P. et al. (2002). *Managing Smile Risk.*"
    )

    st.subheader("Python Library Usage")
    st.code(
        "from pricing import BlackScholesMerton, OptionContract\n"
        "from greeks import GreekEngine\n\n"
        "contract = OptionContract(S=100, K=100, T=1.0, r=0.05, sigma=0.2)\n"
        "price = BlackScholesMerton().price_contract(contract)\n"
        "greeks = GreekEngine().compute_all(contract)",
        language="python",
    )


def main() -> None:
    st.title("📈 Options Pricing Engine")
    st.caption(
        "Multi-model option pricing, Greeks, volatility smiles, validation & stress testing."
    )
    contract, model = sidebar_inputs()

    tabs = st.tabs(
        [
            "Option Pricer",
            "Greek Engine",
            "Pricing Validation",
            "Volatility Smile",
            "Scenario Analysis",
            "Model Documentation",
        ]
    )
    with tabs[0]:
        tab_pricer(contract, model)
    with tabs[1]:
        tab_greeks(contract)
    with tabs[2]:
        tab_validation(contract)
    with tabs[3]:
        tab_volatility(contract)
    with tabs[4]:
        tab_scenarios(contract)
    with tabs[5]:
        tab_documentation()


if __name__ == "__main__":
    main()
