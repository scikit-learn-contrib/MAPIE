### Config ###
.PHONY: tests doc build


### Checks that are run in GitHub CI ###
lint:
	ruff check examples mapie notebooks
	
format:
	ruff format --check --diff examples mapie notebooks

type-check:
	mypy mapie

coverage:
	pytest -vsx \
		--doctest-modules \
		--pyargs mapie \
		--cov-branch \
		--cov=mapie \
		--cov-report term-missing \
		--cov-fail-under=100 \
		--no-cov-on-fail \
		--ignore=mapie/tests/long_tests

long-tests:
	pytest -vsx \
		mapie/tests/long_tests

### Auto-formatting for local use ###
format-fix:
	ruff format examples mapie notebooks

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
	pytest -vs \
		--doctest-modules \
		--pyargs mapie \
		--ignore=mapie/tests/long_tests

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
