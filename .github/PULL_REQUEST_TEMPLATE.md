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

**Test Configuration**:
* OS version:
* Python version:
* MAPIE version:

# Checklist:

- [ ] I have read the [contributing guidelines](https://github.com/simai-ml/MAPIE/blob/master/CONTRIBUTING.rst)
- [ ] The following command passes successfully : `flake8 . --exclude=doc`
- [ ] The following command passes successfully : `mypy mapie examples --strict --config-file mypy.ini`
- [ ] The following command passes successfully : `pytest -vs --doctest-modules mapie`
- [ ] The following command gives 100% coverage : `pytest -vs --doctest-modules --cov-branch --cov=mapie --pyargs mapie`
- [ ] The documentation builds successfully : `cd doc; make clean; make html`