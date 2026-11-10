# make validate   run the numerical validation checks and re-render the README Validation table
# make check      verify the README Validation table matches results/ (fast, used in CI)
# make test       run the unit tests
.PHONY: setup validate check test

setup:
	python -m venv .venv && .venv/bin/pip install -r requirements.lock && .venv/bin/pip install -e . --no-deps

validate:
	python scripts/validate.py

check:
	python scripts/validate.py --check

test:
	MPLBACKEND=Agg python -m pytest -q
