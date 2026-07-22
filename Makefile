PYTHON ?= python3

.PHONY: install install-full demo ui api worker infra test doctor clean-outputs

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-deps

install-full: install
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m playwright install chromium

demo:
	MATTRESS_INTEL_LLM_PROVIDER=none MATTRESS_INTEL_SEARCH_PROVIDER=none $(PYTHON) -m mattress_intelligence demo

ui:
	$(PYTHON) -m streamlit run app.py

api:
	$(PYTHON) -m uvicorn mattress_intelligence.api:app --reload

worker:
	$(PYTHON) -m celery -A mattress_intelligence.celery_app:celery_app worker --loglevel=INFO --concurrency=2

infra:
	docker compose up --build

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

doctor:
	$(PYTHON) -m mattress_intelligence doctor

clean-outputs:
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('outputs').glob('*.xlsx')]"
