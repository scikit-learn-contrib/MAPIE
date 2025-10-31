### Config ###
.PHONY: tests doc build


### Checks that are run in GitHub CI ###
lint:
	ruff check examples mapie notebooks
	
format:
	ruff format --diff examples mapie notebooks

type-check:
	mypy mapie

pytest -vsx \
    --cov=mapie \
    --cov-branch \
    --cov-report term-missing \
    --pyargs mapie \
    --cov-fail-under=100 \
    --no-cov-on-fail \
    --doctest-modules \
    --ignore=mapie/tests/notebooks/_run_notebooks.py


### Checks that are run in ReadTheDocs CI ###
doc:
	$(MAKE) html -C doc

doctest:
	# Tests .. testcode:: blocks in documentation, among other things
	$(MAKE) doctest -C doc


### Other utilities (for local use) ###
all-checks:
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) coverage

tests:
	pytest -vs --doctest-modules mapie --ignore=mapie/tests/notebooks

clean-doc:
	$(MAKE) clean -C doc

build:
	python -m build

clean-build:
	rm -rf build dist MAPIE.egg-info

clean:
	rm -rf .mypy_cache .pytest_cache .coverage*
	rm -rf **__pycache__
	$(MAKE) clean-build
	$(MAKE) clean-doc

# Run all notebooks located in mapie/tests/notebooks/
notebook-tests:
	@echo "Executing all notebooks in mapie/tests/notebooks/..."
	python mapie/tests/notebooks/_run_notebooks.py
