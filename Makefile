.PHONY: reproduce reproduce-full generate prepare features candidates gate-dgp train evaluate \
	representation gate-representation compose lodo recommend scenarios docs audit test \
	test-fast test-slow lint

PY := uv run python
export PYTHONHASHSEED := 0

# --- reproduction (CPU, single seed, no cloud, no API keys) ---------------------------------
# `reproduce`: data -> gates -> train -> evaluate -> compose. Both acceptance gates block:
#   Gate-A (gate-dgp): simulator quality, frozen before any model work;
#   Gate-B (gate-representation): oracle-free candidate-recall rows (D9/D11/oracle rows are
#   reporting-only). pytest is CI, not reproduction, so it is NOT in this chain.
reproduce: generate prepare features candidates gate-dgp train evaluate representation \
	gate-representation compose

# `reproduce-full`: + the inspection artifacts (recommend capped at 300 holdout trips,
# scenarios), leave-one-destination-out, and the regenerated docs. Wall-clock for both
# targets is recorded in results/parts/timings.json by scripts/time_reproduce.py.
reproduce-full: reproduce recommend scenarios lodo docs

generate:
	$(PY) -m poi_rank.cli generate

prepare:
	$(PY) -m poi_rank.cli prepare

features:
	$(PY) -m poi_rank.cli features

candidates:
	$(PY) -m poi_rank.cli candidates

gate-dgp:
	$(PY) -m poi_rank.cli gate-dgp

train:
	$(PY) -m poi_rank.cli train

evaluate:
	$(PY) -m poi_rank.cli evaluate

representation:
	$(PY) -m poi_rank.cli representation

gate-representation:
	$(PY) -m poi_rank.cli gate-representation

# The ONLY writer of results/metrics.json (evaluate/lodo also call it after writing their part).
compose:
	$(PY) -m poi_rank.cli compose

# Leave-one-destination-out (spec.md section 11.8): 3 LightGBM retrains.
lodo:
	$(PY) -m poi_rank.cli lodo

recommend:
	$(PY) -m poi_rank.cli recommend

scenarios:
	$(PY) -m poi_rank.cli scenarios

docs:
	$(PY) -m poi_rank.eval.report

# Deterministic invariant audit (replaces the verifier subagent); JSON on stdout, non-zero on
# any failed check. `--deep` also regenerates the dataset twice and compares SHA-256s.
audit:
	$(PY) -m poi_rank.cli audit

# --- tests (CI) -------------------------------------------------------------------------------
# default tier: everything not marked slow, xdist across cores (light tests only).
test-fast:
	uv run pytest -m "not slow" -n 12

# slow tier: heavy pipeline/training tests, serial lane (they share session fixtures and each
# holds multi-GB frames -- 8 concurrent workers would exceed 31 GB).
test-slow:
	uv run pytest -m slow

test: test-fast test-slow

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy
