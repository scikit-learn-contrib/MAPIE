.PHONY: tests doc build
lint:	
	flake8 . --exclude=doc

type-check:
	mypy mapie examples --strict

tests:
	pytest -vs --doctest-modules mapie

coverage:
	pytest -vs --doctest-modules --cov-branch --cov=mapie --cov-report term-missing --pyargs mapie

doc:
	$(MAKE) clean -C doc
	$(MAKE) html -C doc

build:
	python setup.py sdist bdist_wheel

clean-build:
	rm -rf build dist MAPIE.egg-info

clean:
	rm -rf build dist MAPIE.egg-info .mypy_cache .pytest_cache .coverage*
	$(MAKE) clean -C doc