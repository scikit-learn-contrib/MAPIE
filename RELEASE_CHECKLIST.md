# Release checklist

- [ ] Edit HISTORY.rst and AUTHORS.rst to make sure it’s up-to-date and add release date
- [ ] Check whether any new files need to go in MANIFEST.in
- [ ] Make sure everything is committed and pushed
- [ ] Make sure tests run, pass and cover 100% of the package:
    * `flake8 . --exclude=doc`
    * `mypy mapie examples --strict --config-file mypy.ini`
    * `pytest -vs --doctest-modules mapie`
    * `pytest -vs --doctest-modules --cov-branch --cov=mapie --pyargs mapie`
- [ ] Update the version number with `bump2version major|minor|patch`.
- [ ] Push new tag to your commit: `git push --tags`
- [ ] Build source distribution:
    * `rm -rf build dist`
    * `python setup.py sdist bdist_wheel`
- [ ] Check that your package is ready for publication: `twine check dist/*`
- [ ] Upload it to TestPyPi: `twine upload --repository-url https://test.pypi.org/legacy/ dist/*`
- [ ] Test upload on TestPyPi:
    * `cd`
    * `conda activate`
    * `conda create -n test-mapie --yes python=3.9.2`
    * `conda activate test-mapie`
    * `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mapie`
    * `conda activate`
    * `conda env remove -n test-mapie`
- [ ] Create new release on GitHub for this tag, with all of the links.
