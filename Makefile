PYTHON ?= python3

.PHONY: install install-full demo ui api test doctor clean-outputs

install:
	$(PYTHON) -m pip install -e .

install-full:
	$(PYTHON) -m pip install -e ".[full,dev]"
	$(PYTHON) -m playwright install chromium

demo:
	$(PYTHON) -m brixta_mattress demo

ui:
	$(PYTHON) -m streamlit run app.py

api:
	$(PYTHON) -m uvicorn brixta_mattress.api:app --reload

test:
	$(PYTHON) -m unittest discover -s tests -v

doctor:
	$(PYTHON) -m brixta_mattress doctor

clean-outputs:
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('outputs').glob('*.xlsx')]"

