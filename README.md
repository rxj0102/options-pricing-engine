# 📈 Options Pricing Engine

[![CI](https://github.com/rxj0102/options-pricing-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/options-pricing-engine/actions/workflows/ci.yml)

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-ff4b4b.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/tests-pytest-success.svg)](tests/)

A multi-model options pricing engine supporting **Black-Scholes**, **Black-76**, **Binomial (CRR)**, and **Trinomial (Boyle)** models with a full **Greek engine**, **volatility smile construction**, **scenario analysis**, and **model validation** — packaged as both a Streamlit web app and a Python library.

What makes it institutional-grade: a single abstract pricing interface shared by every model, closed-form *and* finite-difference Greeks that agree to `1e-4`, NumPy-style docstrings with runnable examples on every public function, a put-call-parity / cross-model / convergence validation suite, and historically-motivated stress scenarios with Greek-based P&L attribution.

---

## 🚀 Quick Install

```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

```python
from pricing import BlackScholesMerton, OptionContract
from greeks import GreekEngine

# Price an at-the-money 1-year call
contract = OptionContract(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20)
price = BlackScholesMerton().price_contract(contract)
print(f"Call price: {price:.4f}")          # 10.4506

# Compute the full Greek set (delta, gamma, vega, theta, rho, vanna, volga, charm, speed)
greeks = GreekEngine().compute_all(contract)
print(f"Delta {greeks.delta:.4f}  Gamma {greeks.gamma:.4f}  Vega {greeks.vega:.4f}")
```

## 🖥️ Launch the Dashboard

```bash
streamlit run app.py
```

The six-tab dashboard covers the **Option Pricer**, **Greek Engine** (with 3-D surfaces), **Pricing Validation**, **Volatility Smile** (SVI + SABR fits), **Scenario Analysis** (stress heatmaps + P&L waterfall), and **Model Documentation**.

---

## 📊 Model Overview

| Model | Underlier | Exercise | Analytical Greeks |
|-------|-----------|----------|-------------------|
| Black-Scholes-Merton | Spot + dividend yield | European | ✅ Yes |
| Black-76 | Forward / Futures | European | ✅ Yes |
| CRR Binomial | Spot + dividend yield | European & American | ➖ Finite-difference |
| Boyle Trinomial | Spot + dividend yield | European & American | ➖ Finite-difference |

All four models agree to numerical tolerance for European contracts.

---

## 🧮 Mathematical Notation

**Black-Scholes-Merton**

```
d1 = (ln(S/K) + (r - q + ½σ²)·T) / (σ·√T)
d2 = d1 - σ·√T
C  = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
```

**Black-76 (forward measure)**

```
d1 = (ln(F/K) + ½σ²·T) / (σ·√T),   F = S·e^((r-q)T)
C  = e^(-rT)·[ F·N(d1) - K·N(d2) ]
```

**Cox-Ross-Rubinstein binomial**

```
u = e^(σ·√Δt),   d = 1/u,   p = (e^((r-q)Δt) - d) / (u - d)
```

---

## 📦 Module API Reference

### `pricing` — the four models behind a shared interface

```python
from pricing import BlackScholesMerton, Black76, CRRBinomial, BoyleTrinomial

BlackScholesMerton().price(100, 100, 1.0, 0.05, 0.2, "call")          # 10.4506
BlackScholesMerton().implied_vol(10.45, 100, 100, 1.0, 0.05)          # ≈ 0.20
Black76().price_from_spot(100, 100, 1.0, 0.05, 0.2)                   # 10.4506
CRRBinomial(300).price(100, 100, 1.0, 0.05, 0.2, "put",
                       exercise_style="american")                     # early-exercise premium
```

### `greeks` — analytical, numerical & unified engine

```python
from greeks import BSMGreeks, NumericalGreeks, GreekEngine
from pricing import CRRBinomial, OptionContract

BSMGreeks().compute(100, 100, 1.0, 0.05, 0.2, "call")                 # GreekSet(...)
NumericalGreeks(CRRBinomial(300)).compute(100, 100, 1.0, 0.05, 0.2)   # FD Greeks
GreekEngine().pnl_explain(OptionContract(S=100, K=100, T=1, r=0.05, sigma=0.2),
                          dS=1.0, dsigma=0.01, dT=1/365, dr=0.0)       # P&L attribution
```

### `volatility` — smile, SVI, SABR, term structure

```python
from volatility import VolatilitySmile, SABRModel, VolTermStructure

smile = VolatilitySmile(quotes, S=100, T=1.0, r=0.05)
smile.extract_ivols()            # strike / moneyness / ivol DataFrame
smile.svi_fit()                  # raw SVI parameters
SABRModel().calibrate(strikes, ivols, F=105, T=1.0)   # Hagan-2002 calibration
```

### `validation` — parity, cross-model, convergence

```python
from validation import PutCallParityCheck, ModelComparison, ConvergenceAnalyzer

PutCallParityCheck().check(call, put, 100, 100, 1.0, 0.05)   # {"passes": True, ...}
ModelComparison().comparison_table(contract)                  # all 4 models vs BSM in bps
ConvergenceAnalyzer().analyze(contract)                       # empirical convergence order
```

### `scenarios` — stress testing & P&L attribution

```python
from scenarios import StressTester, GreekPnLAttribution

StressTester().scenario_pnl(contract)        # 2008 Crisis, 2020 COVID, Rate Hike, Earnings
GreekPnLAttribution().waterfall_data(contract, dS=2.0, dsigma=0.01, dT=1/365, dr=0.0)
```

---

## 📈 Convergence

The lattice models converge to the analytical Black-Scholes price as the number of steps increases. `ConvergenceAnalyzer` fits a log-log regression of absolute error against step count and reports the empirical convergence order (≈ 1 for both trees on a smooth European payoff). The dashboard's **Pricing Validation** tab plots Binomial and Trinomial prices against `n_steps` with the BSM value as a horizontal reference.

---

## 🧪 Tests

```bash
pytest tests/
```

Covers known analytical values, put-call parity for all four models, Greek sign conventions, implied-vol roundtrips, tree convergence, the American-put early-exercise premium, and analytical-vs-numerical Greek agreement.

---

## 📄 SSRN Paper

Generate the SSRN-ready whitepaper PDF (tables built live from the library):

```bash
python paper/generate_paper.py
```

> 📎 SSRN link: _coming soon_.

---

## 📂 Repository Structure

```
options-pricing-engine/
├── app.py                  # Streamlit dashboard (6 tabs)
├── pricing/                # base + Black-Scholes, Black-76, Binomial, Trinomial
├── greeks/                 # analytical, numerical, unified engine
├── volatility/             # smile, SVI, SABR, term structure
├── validation/             # model comparison, put-call parity, convergence
├── scenarios/              # stress tests, Greek P&L attribution, runner
├── utils/                  # distributions, plotting
├── tests/                  # pytest suite
├── notebooks/              # options_pricing_deep_dive.ipynb
└── paper/                  # generate_paper.py (ReportLab)
```

---

## Reproducibility

### Data availability

No external data is downloaded. Inputs are generated or parameterised inside the code.

### Computational requirements

- Python 3.12.3. Every dependency version is pinned in `requirements.lock` (compiled from `requirements.txt` with uv).
- Checked on Linux x86_64 with 2 CPU cores and 8 GB RAM. No GPU is needed.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock   # exact pinned environment
pip install -e . --no-deps         # this package
pytest -q                          # runs the test suite
```

### Runtime

The test suite (80 tests) finishes in about 10 seconds on the machine above, including any data downloads it triggers.

### Citation

Citation metadata is in `CITATION.cff`. GitHub turns it into a Cite this repository button on the repo page.


## 📜 License

MIT — see [LICENSE](LICENSE).

## 🔗 GitHub

[github.com/rxj0102/options-pricing-engine](https://github.com/rxj0102/options-pricing-engine)
