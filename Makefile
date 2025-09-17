### Config ###
.PHONY: tests doc build


### Checks that are run in GitHub CI ###
lint:
	flake8 examples mapie notebooks --max-line-length=88

type-check:
	mypy --version
	mypy mapie

coverage:
	pytest -vsx \
		--cov-branch \
		--cov=mapie \
		--cov-report term-missing \
		--pyargs mapie \
		--cov-fail-under=100 \
		--no-cov-on-fail \
		--doctest-modules


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
