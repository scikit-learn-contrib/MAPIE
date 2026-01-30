# Release checklist

- [ ] Check that `master` contains all intended changes for the release.
- [ ] Make sure CI tests pass on GitHub Actions for the latest commit on master. Otherwise fix issues in a Pull Request and merge it.
- [ ] Look at the latest documentation version and check if it was compiled without issue or warning (https://app.readthedocs.org/projects/mapie/builds/). Otherwise fix issues in a Pull Request and merge it.
- [ ] Checkout `master` and pull latest changes: `git checkout master && git pull origin master`.
- [ ] Edit HISTORY.rst and AUTHORS.rst to make sure it’s up-to-date.
- [ ] Do a pre-release commit including every change from the steps above: `git add HISTORY.rst AUTHORS.rst && git commit -m "vX.Y.Z pre-release changes"`.
- [ ] Update the version number with `bump2version major|minor|patch` (only one option between the three, a commit and tag are automatically made)
- [ ] Check that bump to version is the tagged commit `git log --decorate`
- [ ] Push the commit created by bump2version: `git push origin master`
- [ ] Manually trigger the TestPyPI publish job on GitHub Actions:
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
- [ ] Check that the new stable version of the documentation is built and published and that the new version appears in the version selector (should be automatically made by a Read The Docs automation).
- [ ] Merge the automatically created pull request on https://github.com/conda-forge/mapie-feedstock (PR creation might take some time). You need to be added as a maintainer on this repo first. To create the pull request manually to avoid waiting for automation, create an issue with the name `@conda-forge-admin, please update version`
