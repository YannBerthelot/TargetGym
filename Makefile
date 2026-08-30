SHELL=/bin/bash
LINT_PATHS=src/ tests/

# Strict CPU isolation -- prevents tests from touching GPU when a live
# experiment is running on this machine. CUDA_VISIBLE_DEVICES="" hides the
# GPU from the CUDA driver entirely so CUDA init can't probe it.
CPU_ENV := CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu

.PHONY: ci ci-format-check ci-test help install \
        all all-% figures figures-% videos videos-% tuning tuning-% \
        clear-tuning clear-mpc short-gifs test test-all mypy coverage \
        missing-annotations type lint format check-codestyle commit-checks

help:  ## Show this help message
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install:  ## Create/refresh the dev environment (.venv) with uv
	uv sync --group dev

# ---------------------------------------------------------------------------
# Canonical local-CI -- mirrors .github/workflows/python-app.yml exactly.
# ---------------------------------------------------------------------------

ci: ci-format-check ci-test  ## Full local CI (matches .github/workflows/python-app.yml)

ci-format-check:  ## Black --check on the whole tree
	uv run black --check .

ci-test:  ## Fast test suite on CPU (skips `slow` closed-loop checks)
	$(CPU_ENV) uv run pytest tests/ -v -m "not slow"

all:
	uv run python -m target_gym.runners.runners

all-%:
	uv run python -m target_gym.runners.runners --env $*

figures:
	uv run python -m target_gym.runners.runners --only figures

figures-%:
	uv run python -m target_gym.runners.runners --only figures --env $*

videos:
	uv run python -m target_gym.runners.runners --only videos

videos-%:
	uv run python -m target_gym.runners.runners --only videos --env $*

tuning:
	uv run python scripts/tune_pid.py

tuning-%:
	uv run python scripts/tune_pid.py --envs $*

clear-tuning:
	rm -f data/pid_gains.json
	@echo "Cleared PID gains cache (data/pid_gains.json)."

short-gifs:
	uv run python scripts/shorten_gifs.py

clear-mpc:
	rm -rf data/mpc_cache data/interpolators
	@echo "Cleared MPC trajectory cache (data/mpc_cache/, data/interpolators/)."

test:  ## Fast tests only
	$(CPU_ENV) uv run pytest --tb=short --disable-warnings -m "not slow"

test-all:  ## Every test, including the slow closed-loop controller checks
	$(CPU_ENV) uv run pytest --tb=short --disable-warnings

mypy:
	uv run mypy ${LINT_PATHS}

coverage:
	$(CPU_ENV) uv run coverage run --source target_gym -m pytest tests
	uv run coverage report -m --fail-under 80

missing-annotations:
	uv run mypy --disallow-untyped-calls --disallow-untyped-defs --ignore-missing-imports src

type: mypy

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://www.flake8rules.com/
	uv run ruff check ${LINT_PATHS} --select=E9,F63,F7,F82 --output-format=full
	# exit-zero treats all errors as warnings.
	uv run ruff check ${LINT_PATHS} --exit-zero --output-format=concise

format:
	# Sort imports
	uv run ruff check --select I $(LINT_PATHS) --fix
	# Reformat using black
	uv run black $(LINT_PATHS)

check-codestyle:
	# Sort imports
	uv run ruff check --select I ${LINT_PATHS}
	# Reformat using black
	uv run black --check ${LINT_PATHS}

commit-checks: format type lint
