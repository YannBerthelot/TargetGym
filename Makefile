SHELL=/bin/bash
LINT_PATHS=src/ tests/

.PHONY: all all-% figures figures-% videos videos-% tuning tuning-% clear-tuning clear-mpc short-gifs test mypy coverage missing-annotations type lint format check-codestyle commit-checks

all:
	poetry run python -m target_gym.runners.runners

all-%:
	poetry run python -m target_gym.runners.runners --env $*

figures:
	poetry run python -m target_gym.runners.runners --only figures

figures-%:
	poetry run python -m target_gym.runners.runners --only figures --env $*

videos:
	poetry run python -m target_gym.runners.runners --only videos

videos-%:
	poetry run python -m target_gym.runners.runners --only videos --env $*

tuning:
	poetry run python scripts/tune_pid.py

tuning-%:
	poetry run python scripts/tune_pid.py --envs $*

clear-tuning:
	rm -f data/pid_gains.json
	@echo "Cleared PID gains cache (data/pid_gains.json)."

short-gifs:
	poetry run python scripts/shorten_gifs.py

clear-mpc:
	rm -rf data/mpc_cache data/interpolators
	@echo "Cleared MPC trajectory cache (data/mpc_cache/, data/interpolators/)."

test:
	poetry run pytest --tb=short --disable-warnings

mypy:
	mypy ${LINT_PATHS} 

coverage:
	poetry run coverage run --source target_gym -m pytest tests
	poetry run coverage report -m --fail-under 80

missing-annotations:
	mypy --disallow-untyped-calls --disallow-untyped-defs --ignore-missing-imports src

type: mypy

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://www.flake8rules.com/
	poetry run ruff check ${LINT_PATHS} --select=E9,F63,F7,F82 --output-format=full
	# exit-zero treats all errors as warnings.
	poetry run ruff check ${LINT_PATHS} --exit-zero --output-format=concise

format:
	# Sort imports
	poetry run ruff check --select I $(LINT_PATHS) --fix
	# Reformat using black
	poetry run black $(LINT_PATHS)

check-codestyle:
	# Sort imports
	ruff check --select I ${LINT_PATHS}
	# Reformat using black
	black --check ${LINT_PATHS}

commit-checks: format type lint
