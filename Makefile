PY ?= .venv/bin/python

.PHONY: setup data eda backtest inventory forecast all test lint

setup:            ## create venv and install the package
	uv venv .venv --allow-existing && uv pip install --python $(PY) -e ".[dev]"

data:             ## download M5 + build cleaned panel
	$(PY) -m demandcast.pipeline prepare
eda:
	$(PY) -m demandcast.pipeline eda
backtest:         ## ~5-10 min on a laptop
	$(PY) -m demandcast.pipeline backtest
inventory:
	$(PY) -m demandcast.pipeline inventory
forecast:
	$(PY) -m demandcast.pipeline forecast
all:
	$(PY) -m demandcast.pipeline all

test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check src tests
