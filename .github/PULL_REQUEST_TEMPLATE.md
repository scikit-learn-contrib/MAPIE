# Description

Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context. List any dependencies that are required for this change.

Fixes #(issue)

## Type of change

Please check options that are relevant.

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

# How Has This Been Tested?

Please describe the tests that you ran to verify your changes. Provide instructions so we can reproduce. Please also list any relevant details for your test configuration

- [ ] Test A
- [ ] Test B

# Checklist:

- [ ] I have read the [contributing guidelines](https://github.com/simai-ml/MAPIE/blob/master/CONTRIBUTING.rst)
- [ ] I have updated the [HISTORY.rst](https://github.com/simai-ml/MAPIE/blob/master/HISTORY.rst) and [AUTHORS.rst](https://github.com/simai-ml/MAPIE/blob/master/AUTHORS.rst) files
- [ ] Linting passes successfully : `flake8 . --exclude=doc`
- [ ] Typing passes successfully : `mypy mapie examples --strict`
- [ ] Unit tests pass successfully : `pytest -vs --doctest-modules mapie`
- [ ] Coverage is 100% : `pytest -vs --doctest-modules --cov-branch --cov=mapie --pyargs mapie`
- [ ] Documentation builds successfully : `cd doc; make clean; make html`