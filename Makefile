PYTHON ?= .venv/bin/python
export MPLCONFIGDIR ?= /tmp/fabfederate-matplotlib
export FLWR_TELEMETRY_ENABLED ?= 0
CONFIG ?= configs/default.yaml
STUDY ?= artifacts/experiments/nasa-final
STUDY_OUTPUT ?= artifacts/experiments/new-study
MODEL ?= $(STUDY)/models/fedavg
INPUT ?= artifacts/inference/example_features.csv

.PHONY: help install frontend-install api frontend dashboard download audit prepare study report predict explain test check

help:
	@printf '%s\n' 'FabFederate-CNC commands (run from the repository root):' \
	  '  make api               Start the local Python backend on :8000' \
	  '  make frontend          Start the React website on :5173' \
	  '  make dashboard         Open the optional Streamlit research explorer on :8501' \
	  '  make download          Retrieve the NASA source dataset' \
	  '  make audit             Audit raw NASA recordings' \
	  '  make prepare           Generate features and split/client manifests' \
	  '  make study             Train and evaluate all configured methods' \
	  '  make report            Regenerate a saved study report' \
	  '  make predict           Predict from INPUT using MODEL' \
	  '  make explain           Explain INPUT using the saved training reference' \
	  '  make test              Run Python and frontend tests' \
	  '  make check             Run Python lint and the TypeScript production build'

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --no-build-isolation --no-deps -e .

frontend-install:
	npm --prefix frontend ci

api:
	$(PYTHON) -m uvicorn fabfederate.api.app:app --host 127.0.0.1 --port 8000

frontend:
	npm --prefix frontend run dev

dashboard:
	cd dashboard && $(abspath $(PYTHON)) -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501

download:
	$(PYTHON) -m fabfederate download

audit:
	$(PYTHON) -m fabfederate audit --output data/interim/nasa

prepare:
	$(PYTHON) -m fabfederate prepare --output data/processed/nasa

study:
	$(PYTHON) -m fabfederate study --config $(CONFIG) --output $(STUDY_OUTPUT)

report:
	$(PYTHON) -m fabfederate report $(STUDY)

predict:
	$(PYTHON) -m fabfederate predict --model $(MODEL) --input $(INPUT) --output artifacts/inference/prediction.csv

explain:
	$(PYTHON) -m fabfederate explain --model $(MODEL) --input $(INPUT) --background $(STUDY)/features.parquet --output artifacts/inference/explanation.json

test:
	MPLCONFIGDIR=/tmp/fabfederate-matplotlib FLWR_TELEMETRY_ENABLED=0 $(PYTHON) -m pytest -q
	npm --prefix frontend test

check:
	$(PYTHON) -m ruff check src tests dashboard scripts
	npm --prefix frontend run build
