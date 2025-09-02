SHELL=/bin/bash
LINT_PATHS=src/ tests/ 

test:
	poetry run pytest --tb=short --disable-warnings

mypy:
	mypy ${LINT_PATHS} 

coverage:
	poetry run coverage run -m pytest tests
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
