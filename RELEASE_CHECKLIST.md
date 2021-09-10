# Release checklist

- [ ] Update the version number with `bump2version major|minor|patch`
- [ ] Push new tag to your commit: `git push --tags`
- [ ] Edit HISTORY.rst and AUTHORS.rst to make sure it’s up-to-date and add release date
- [ ] Check whether any new files need to go in MANIFEST.in
- [ ] Make sure tests run, pass and cover 100% of the package:
    * `make lint`
    * `make type-check`
    * `make tests`
    * `make coverage`
- [ ] Make sure documentation builds without warnings and shows nicely:
    * `make doc`
- [ ] Build source distribution:
    * `make clean-build`
    * `make build`
- [ ] Check that your package is ready for publication: `twine check dist/*`
- [ ] Make sure everything is committed and pushed: `git push origin master`
- [ ] Upload it to TestPyPi: `twine upload --repository-url https://test.pypi.org/legacy/ dist/*`
- [ ] Test upload on TestPyPi:
    * `cd`
    * `conda activate`
    * `conda create -n test-mapie --yes python=3.9`
    * `conda activate test-mapie`
    * `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mapie`
    * `conda activate`
    * `conda env remove -n test-mapie`
- [ ] Create new release on GitHub for this tag.
- [ ] Merge the automatically created pull request on https://github.com/conda-forge/mapie-feedstock
