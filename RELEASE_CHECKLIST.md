# Release checklist

- [ ] Edit HISTORY.rst and AUTHORS.rst to make sure it’s up-to-date and add release date
- [ ] Update the version number with `bump2version major|minor|patch`.
- [ ] Check whether any new files need to go in MANIFEST.in
- [ ] Make sure everything is committed and pushed, make sure tests run and pass
- [ ] Build source distribution: `rm -rf build ; rm -rf dist ; python setup.py sdist bdist_wheel`
- [ ] Check that your package is ready for publication: `twine check dist/*`
- [ ] Upload it to TestPyPi: `twine upload --repository-url https://test.pypi.org/legacy/ dist/*`
- [ ] Test upload on TestPyPi: `pip install -i https://testpypi.python.org/pypi mapie`
- [ ] Add new tag to your commit: `git tag x.y.z`
- [ ] Push new tag: `git push --tags`
- [ ] Push to master: `git push origin master`
- [ ] Create new release on GitHub for this tag, with all of the links.
- [ ] Switch default readthedocs version to the latest version
- [ ] Create maintenance branch if major version update
- [ ] Bump version to development version with `bumpversion patch`
- [ ] Start on next version changes in HISTORY file