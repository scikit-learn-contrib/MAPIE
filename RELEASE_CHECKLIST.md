# Release checklist

- [ ] Edit HISTORY.rst and AUTHORS.rst to make sure it’s up-to-date and add release date
- [ ] Make sure tests run, pass and cover 100% of the package:
    * `make lint`
    * `make type-check`
    * `make format`
    * `make coverage`
- [ ] Make sure documentation builds without warnings and shows nicely:
    * `make doc`
- [ ] Commit every change from the steps above
- [ ] Update the version number with `bump2version major|minor|patch` (only one option between the three, a commit is automatically made)
- [ ] Check that bump to version is the tagged commit `git log --decorate`
- [ ] Push the commit created by bump2version: `git push origin master`
- [ ] (Optional) Manually trigger the TestPyPI publish job on GitHub Actions:
    * Verify that the build and TestPyPI publish succeeded
    * Test installation from TestPyPI:
        - create a new empty virtual environment
        - `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mapie`
        - import mapie and verify version: `python -c "import mapie; print(mapie.__version__)"`
- [ ] Push the tag created by bump2version: `git push --tags`
- [ ] Monitor the PyPI publish job on GitHub Actions:
    * The workflow automatically triggers on tag pushes
    * The `pypi` environment requires manual approval (configured in repo settings)
    * Approve the deployment in the GitHub Actions UI when prompted
    * Verify the package appears on PyPI after approval
    * Test installation:
        - create a new empty virtual environment
        - `pip install mapie`
        - import mapie and verify version: `python -c "import mapie; print(mapie.__version__)"`
- [ ] Create new release on GitHub for this tag, using information from HISTORY.rst.
- [ ] Merge the automatically created pull request on https://github.com/conda-forge/mapie-feedstock. You need to be added as a maintainer on this repo first. To create the pull request manually to avoid waiting for automation, create an issue with the name `@conda-forge-admin, please update version`
