# Release checklist

- [ ] Edit HISTORY.rst and AUTHORS.rst to make sure it’s up-to-date and add release date
- [ ] Make sure tests run, pass and cover 100% of the package:
    * `make lint`
    * `make type-check`
    * `make tests`
    * `make coverage`
- [ ] Make sure documentation builds without warnings and shows nicely:
    * `make doc`
- [ ] Commit every change from the steps above
- [ ] Update the version number with `bump2version major|minor|patch` (a commit is automatically made)
- [ ] Build source distribution:
    * `make clean-build`
    * `make build`
- [ ] Check that your package is ready for publication: `twine check dist/*`
- [ ] Upload it to TestPyPi:
    * you need to create an account on test.pypi.org first if you don't have one, then an API key, and ask one the existing MAPIE maintainer to add you as a maintainer
    * `twine upload --repository-url https://test.pypi.org/legacy/ dist/*` (use `__token__` as username and your api token as password)
- [ ] Test upload on TestPyPi:
    * create a new empty virtual environment
    * `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mapie` (mapie should install from test.pypi.org, may not work if using uv)
- [ ] Push the commit created by bump2version: `git push origin master`
- [ ] Push the tag created by bump2version: `git push --tags`
- [ ] Create new release on GitHub for this tag.
- [ ] Merge the automatically created pull request on https://github.com/conda-forge/mapie-feedstock. You need to be added as a maintainer on this repo first. To create the pull request
  manually to avoid waiting for automation, create an issue with the name `@conda-forge-admin, please update version`
