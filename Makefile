.PHONY: tests doc build

integration_tests_folder_name = tests_v1/integration_tests
mapie_v0_folder_name = mapie_v0_package

lint:
	flake8 . --max-line-length=88 --exclude=doc

type-check:
	mypy mapie

v1-type-check:
	mypy mapie_v1 --disallow-untyped-defs --exclude $(mapie_v0_folder_name)

tests:
	pytest -vs --doctest-modules mapie
	$(MAKE) v1-tests
	$(MAKE) v1-docstring-tests

v1-tests:
	pytest -vs tests_v1 --ignore=$(integration_tests_folder_name)

v1-docstring-tests:
	pytest -vs --doctest-modules mapie_v1 --ignore=$(integration_tests_folder_name)

v1-integration-tests:
	@pip install git+https://github.com/scikit-learn-contrib/MAPIE@master --no-dependencies --target=./$(integration_tests_folder_name)/$(mapie_v0_folder_name) >/dev/null 2>&1
	@mv ./$(integration_tests_folder_name)/$(mapie_v0_folder_name)/mapie ./$(integration_tests_folder_name)/$(mapie_v0_folder_name)/mapiev0
	@- export PYTHONPATH="${PYTHONPATH}:./$(integration_tests_folder_name)/$(mapie_v0_folder_name)"; pytest -vs $(integration_tests_folder_name)/tests $(params)
	@mv ./$(integration_tests_folder_name)/$(mapie_v0_folder_name)/mapiev0 ./$(integration_tests_folder_name)/$(mapie_v0_folder_name)/mapie

v1-checks-not-in-ci:
	$(MAKE) v1-type-check # Issues when trying to include it in CI, see task "Dépendances v1 + MyPy stricter in CI" in project board
	$(MAKE) v1-integration-tests # We don't want to include this in CI, will be removed at v1 release
	$(MAKE) v1-coverage # To add in CI when we reach 100%

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

v1-coverage:
	pytest -vsx \
		--cov-branch \
		--cov=mapie_v1 \
		--cov-report term-missing \
		--pyargs tests_v1 --ignore=$(integration_tests_folder_name) \
		--cov-fail-under=100 \
		--no-cov-on-fail

doc:
	$(MAKE) html -C doc

doctest:
	$(MAKE) doctest -C doc

clean-doc:
	$(MAKE) clean -C doc

build:
	python setup.py sdist bdist_wheel

clean-build:
	rm -rf build dist MAPIE.egg-info

clean:
	rm -rf .mypy_cache .pytest_cache .coverage*
	rm -rf **__pycache__
	$(MAKE) clean-build
	$(MAKE) clean-doc
