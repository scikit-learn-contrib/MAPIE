### Config ###

.PHONY: tests doc build
integration_tests_folder_name = tests_v1/integration_tests
mapie_v0_folder_name = mapie_v0_package


### To run when working locally ###

all-checks:
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) coverage

v1-all-checks:
	$(MAKE) v1-checks-not-in-ci
	$(MAKE) v1-tests
	$(MAKE) v1-docstring-tests
	$(MAKE) lint


### Checks that are run in GitHub CI ###

lint:
	flake8 . --max-line-length=88 --exclude=doc

type-check:
	mypy mapie

coverage:
	# We may need to add the v1 test suite here if we remove some v0 tests, to keep a 100% coverage
	pytest -vsx \
		--cov-branch \
		--cov=mapie \
		--cov-report term-missing \
		--pyargs mapie \
		--cov-fail-under=100 \
		--cov-config=.coveragerc \
		--no-cov-on-fail

v1-tests:
	# To replace with v1-coverage when we reach 100%
	python -m pytest -vs tests_v1 --ignore=$(integration_tests_folder_name)

v1-docstring-tests:
	pytest -vs --doctest-modules mapie_v1 --ignore=$(integration_tests_folder_name)


### Checks that are run in ReadTheDocs CI ###

doc:
	$(MAKE) html -C doc

doctest:
	# Tests .. testcode:: blocks in documentation, among other things
	$(MAKE) doctest -C doc


### Local utilities ###

tests:
	pytest -vs --doctest-modules mapie

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


### Local utilities (v1 specific) ###

v1-checks-not-in-ci:
	$(MAKE) v1-type-check # Issues when trying to include it in CI, see task "Dépendances v1 + MyPy stricter in CI" in project board
	$(MAKE) v1-integration-tests # We will include a different version at v1 release, so we're not adding this in CI for now

v1-type-check:
	mypy mapie_v1 --disallow-untyped-defs --exclude $(mapie_v0_folder_name)

v1-integration-tests:
	# Run `make v1-integration-tests params="-m classification"` to select only classification tests (for example)
	pytest -vs $(integration_tests_folder_name)/tests $(params)

v1-coverage:
	# To add in CI when we reach 100%
	pytest -vsx \
		--cov-branch \
		--cov=mapie_v1 \
		--cov-report term-missing \
		--pyargs tests_v1 --ignore=$(integration_tests_folder_name) \
		--cov-fail-under=100 \
		--no-cov-on-fail
