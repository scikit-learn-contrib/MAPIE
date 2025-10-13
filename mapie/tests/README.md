# Overall recommendations

- Group tests in a class if more than one test is needed for a given function/functionality
- Prefer black-box tests (no mocks) if possible, to avoid testing implementation details.
- Avoid unnecessary comments/docstrings: the code must be self-explanatory as much as possible.

# Unit tests

## Scope

Testing one function, method, or functionality

## Recommendations

- Focus on the function goal to define the test cases.
- Testing corner cases is not mandatory. Sometimes we prefer a function to fail rather than being robust to unwanted scenarii.
- Unit tests on their own should provide a coverage close to 100%.

# Functional or end-to-end tests

## Scope

Testing the main functionalities of the API as seen from a user point-of-view, or testing behaviors hard to test in a unit style.

## Recommendations

- Such tests here should be added wisely (they take usually more time to run)
- Be careful of test time. Testing few _varied_ scenarios is more important than trying to test _all_ scenarios.
- This is not implemented yet, but ideally those tests should not count against coverage.
