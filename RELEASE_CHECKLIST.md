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
- [ ] (Optional) Monitor the TestPyPI publish job on GitHub Actions:
    * The workflow automatically publishes to TestPyPI on every push to master
    * The workflow can also be triggered manually
    * Check the Actions tab to verify the build and TestPyPI publish succeeded
    * Test installation from TestPyPI if desired:
        - create a new empty virtual environment
        - `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mapie`
- [ ] Push the tag created by bump2version: `git push --tags`
- [ ] Monitor the PyPI publish job on GitHub Actions:
    * The workflow automatically triggers on tag pushes
    * The `pypi` environment requires manual approval (configured in repo settings)
    * Approve the deployment in the GitHub Actions UI when prompted
    * Verify the package appears on PyPI after approval
- [ ] Create new release on GitHub for this tag.
- [ ] Merge the automatically created pull request on https://github.com/conda-forge/mapie-feedstock. You need to be added as a maintainer on this repo first. To create the pull request manually to avoid waiting for automation, create an issue with the name `@conda-forge-admin, please update version`
