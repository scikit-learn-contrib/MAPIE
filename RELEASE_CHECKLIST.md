# Release checklist

## 1) Manual pre-release checks

- [ ] Check that `master` contains all intended changes for the release.
- [ ] Make sure CI tests pass on GitHub Actions for the latest commit on master. Otherwise fix issues in a Pull Request and merge it.
- [ ] Look at the latest documentation version and check if it was compiled without issue or warning (https://app.readthedocs.org/projects/mapie/builds/). Otherwise fix issues in a Pull Request and merge it.
- [ ] Checkout `master` and pull latest changes: `git checkout master && git pull origin master`.
- [ ] Edit HISTORY.md and AUTHORS.md to make sure it’s up-to-date.
- [ ] Do a pre-release commit including every change from the steps above: `git add HISTORY.md AUTHORS.md && git commit -m "vX.Y.Z pre-release changes"`.
- [ ] Push the pre-release commit: `git push origin master`

## 2) Release candidate on TestPyPI

- [ ] Create and push a release candidate tag: `git tag vX.Y.ZrcN && git push origin vX.Y.ZrcN` (e.g., v1.5.0rc1)
- [ ] Check that RC tag points to the tagged commit `git log --decorate`
- [ ] Publish to TestPyPI to verify the build:
    * For RC tags (`vX.Y.ZrcN`), TestPyPI publish is automatic on tag push.
    * Smoke-test installation/import from TestPyPI is automatic in the workflow.
    * Verify that the build and TestPyPI publish succeeded (https://test.pypi.org/project/MAPIE/)
    * (Optional extra manual check) Test installation from TestPyPI:
        - create a new empty virtual environment
        - `pip install --pre -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mapie==X.Y.ZrcN` or if using `uv`: `uv pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --index-strategy unsafe-best-match mapie==X.Y.ZrcN`.
        - import mapie and verify version: `python -c "import mapie; print(mapie.__version__)"`

## 3) Final release on PyPI

- [ ] Create and push the final release tag: `git tag vX.Y.Z && git push origin vX.Y.Z`
- [ ] Check that final tag points to the tagged commit `git log --decorate`
- [ ] Monitor the PyPI publish job on GitHub Actions:
    * The workflow automatically triggers on final tag pushes (`vX.Y.Z`) (might take a few minutes to start)
    * The `pypi` environment requires manual approval (configured in repo settings)
    * Approve the deployment in the GitHub Actions UI when prompted
    * Verify the package appears on PyPI after approval (https://pypi.org/project/MAPIE/)
    * Test installation:
        - create a new empty virtual environment
        - `pip install mapie` or if using `uv`: `uv pip install mapie` (you might need to run `uv cache clean` first).
        - import mapie and verify version: `python -c "import mapie; print(mapie.__version__)"`

## 4) Post-release checks

- [ ] Create new release on GitHub for this tag, using information from HISTORY.md.
- [ ] Check that the new stable version of the documentation is built and published and that the new version appears in the version selector (should be automatically made by a Read The Docs automation).
- [ ] Merge the automatically created pull request on https://github.com/conda-forge/mapie-feedstock (PR creation might take some time). You need to be added as a maintainer on this repo first. To create the pull request manually to avoid waiting for automation, create an issue with the name `@conda-forge-admin, please update version`

