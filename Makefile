SHELL=/bin/bash
LINT_PATHS=src/ tests/

# Strict CPU isolation -- prevents tests from touching GPU when a live
# experiment is running on this machine. CUDA_VISIBLE_DEVICES="" hides the
# GPU from the CUDA driver entirely so CUDA init can't probe it.
CPU_ENV := CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu

.PHONY: ci ci-lint ci-format-check ci-test help install \
        all all-% figures figures-% videos videos-% tuning tuning-% \
        clear-tuning clear-mpc short-gifs docs docs-build test test-all mypy coverage \
        missing-annotations type lint format check-codestyle commit-checks \
        mypy-all

help:  ## Show this help message
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install:  ## Create/refresh the dev environment (.venv) with uv
	uv sync --group dev

# ---------------------------------------------------------------------------
# Canonical local-CI -- mirrors .github/workflows/python-app.yml exactly.
# ---------------------------------------------------------------------------

ci: ci-lint ci-format-check ci-test  ## Full local CI (matches .github/workflows/python-app.yml)

ci-lint:  ## Ruff over the tree, the same set CI enforces
	uv run --frozen --only-group lint ruff check src/ tests/ scripts/

ci-format-check:  ## Black --check on the whole tree
	# --only-group lint installs black alone: formatting needs no runtime deps.
	uv run --frozen --only-group lint black --check .

# -n auto spreads the suite over every core; tests/conftest.py holds each
# worker to one compute thread so they do not fight over them. Pass -n0 to
# turn that off, which is what you want when reaching for --pdb.
ci-test:  ## Fast test suite on CPU (skips `slow` closed-loop checks)
	$(CPU_ENV) uv run pytest tests/ -q -n auto --durations=10 -m "not slow"

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

docs:  ## Serve the documentation site locally at :8000
	uv run --group docs mkdocs serve

docs-build:  ## Build the documentation site into site/ (what CI publishes)
	uv run --group docs mkdocs build --strict

short-gifs:
	uv run python scripts/shorten_gifs.py

clear-mpc:
	rm -rf data/mpc_cache data/interpolators
	@echo "Cleared MPC trajectory cache (data/mpc_cache/, data/interpolators/)."

test:  ## Fast tests only
	$(CPU_ENV) uv run pytest --tb=short --disable-warnings -n auto -m "not slow"

test-all:  ## Every test, including the slow closed-loop controller checks
	$(CPU_ENV) uv run pytest --tb=short --disable-warnings -n auto

# Modules that type-check cleanly today and are enforced in CI. Grow this
# list by making a module pass and then adding it; pyproject.toml's [tool.mypy]
# explains why it is not simply the whole tree.
MYPY_PATHS=src/target_gym/registry.py src/target_gym/runners/runners.py

mypy:  ## Type-check the enforced modules (what CI runs)
	uv run mypy $(MYPY_PATHS)

mypy-all:  ## Type-check everything -- exploratory, currently reports ~324
	uv run mypy src/

# pytest-cov supplies the xdist plumbing coverage.py lacks on its own, so this
# runs in parallel and still reports exactly the same total as a serial run --
# 86s rather than 278s, measured. Threshold and omissions are in pyproject.toml.
coverage:  ## Test suite with coverage, enforcing the threshold
	$(CPU_ENV) uv run pytest tests/ -q -n auto --cov=target_gym

missing-annotations:
	uv run mypy --disallow-untyped-calls --disallow-untyped-defs --ignore-missing-imports src

type: mypy

lint:  ## Ruff over src/ and tests/ (rule set lives in pyproject.toml)
	uv run ruff check ${LINT_PATHS}

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
