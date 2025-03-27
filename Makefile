### Config ###

.PHONY: tests doc build
mapie_v0_folder_name = mapie_v0_package


### To run when working locally ###

all-checks:
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) coverage

v1-all-checks:
	$(MAKE) v1-type-check
	$(MAKE) v1-tests
	$(MAKE) v1-docstring-tests
	$(MAKE) lint


### Checks that are run in GitHub CI ###

lint:
	flake8 . --max-line-length=88 --exclude=doc

type-check:
	mypy mapie

coverage:
	pytest -vsx \
		--cov-branch \
		--cov=mapie \
		--cov-report term-missing \
		--pyargs mapie \
		--cov-fail-under=100 \
		--cov-config=.coveragerc \
		--no-cov-on-fail

v1-tests:
	python -m pytest -vs tests_v1

v1-docstring-tests:
	pytest -vs --doctest-modules mapie_v1


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

# Issues when trying to include it in CI, see related task on the project board
v1-type-check:
	mypy mapie_v1 --disallow-untyped-defs --exclude $(mapie_v0_folder_name)

v1-coverage:
	pytest -vsx \
		--cov-branch \
		--cov=mapie_v1 \
		--cov-report term-missing \
		--pyargs tests_v1 \
		--cov-fail-under=100 \
		--no-cov-on-fail
