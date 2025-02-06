.PHONY: tests doc build

lint:
	flake8 . --max-line-length=88 --exclude=doc

type-check:
	mypy mapie

tests:
	pytest -vs --doctest-modules mapie

coverage:
	pytest -vsx \
		--cov-branch \
		--cov=mapie \
		--cov-report term-missing \
		--pyargs mapie \
		--cov-fail-under=100 \
		--cov-config=.coveragerc \
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
