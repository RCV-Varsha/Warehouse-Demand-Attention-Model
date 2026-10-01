PYTHON ?= python
SEED ?= 0
.PHONY: setup test data all demo clean reset
setup:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-jax.txt
	$(PYTHON) -m pip install --no-build-isolation -e .
test:
	$(PYTHON) -m pytest -q
data:
	$(PYTHON) -m experiments.dataset --seed $(SEED)
all:
	$(PYTHON) -m experiments.run_all --seed $(SEED)
demo:
	$(PYTHON) -m scripts.demo --seed $(SEED)
clean:
	$(PYTHON) -m scripts.manage clean --seed $(SEED)
reset:
	$(PYTHON) -m scripts.manage reset --seed $(SEED)
