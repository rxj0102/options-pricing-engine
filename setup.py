"""Packaging configuration for the options-pricing-engine library."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
LONG_DESCRIPTION = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

setup(
    name="options-pricing-engine",
    version="0.1.0",
    description=(
        "A multi-model options pricing engine (Black-Scholes, Black-76, Binomial, "
        "Trinomial) with a full Greek engine, volatility smile construction, "
        "scenario analysis, and model validation."
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="rxj0102",
    url="https://github.com/rxj0102/options-pricing-engine",
    license="MIT",
    packages=find_packages(
        exclude=("tests", "tests.*", "notebooks", "notebooks.*", "paper", "paper.*")
    ),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
        "streamlit>=1.30",
        "plotly>=5.18",
        "matplotlib>=3.7",
        "reportlab>=4.0",
    ],
    extras_require={"dev": ["pytest>=7.4", "black", "isort"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
