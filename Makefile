.PHONY: reproduce generate prepare features candidates train evaluate lodo recommend scenarios docs test lint

PY := uv run python
export PYTHONHASHSEED := 0

reproduce: generate prepare features candidates train evaluate scenarios docs test

generate:
	$(PY) -m poi_rank.cli generate

prepare:
	$(PY) -m poi_rank.cli prepare

features:
	$(PY) -m poi_rank.cli features

candidates:
	$(PY) -m poi_rank.cli candidates

train:
	$(PY) -m poi_rank.cli train

evaluate:
	$(PY) -m poi_rank.cli evaluate

# Leave-one-destination-out (spec.md section 11.8): 3 full LightGBM retrains,
# genuinely expensive -- deliberately NOT part of `reproduce`'s default chain
# (eval/cold_start.py's module docstring). Run after `evaluate`; merges a `lodo`
# key into the already-written results/metrics.json.
lodo:
	$(PY) -m poi_rank.cli lodo

recommend:
	$(PY) -m poi_rank.cli recommend

scenarios:
	$(PY) -m poi_rank.cli scenarios

docs:
	$(PY) -m poi_rank.eval.report

test:
	uv run pytest

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy
