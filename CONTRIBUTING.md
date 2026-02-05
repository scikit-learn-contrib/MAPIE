# Contribution guidelines

## How to contribute?

If you are new to open source, first of all, welcome! You might want to start with this guide: [Contributing to open source on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-open-source).

The typical workflow for contributing to `mapie` is:

1. Choose [what to work on](#what-to-work-on)
2. Fork the `master` branch from the [GitHub repository](https://github.com/scikit-learn-contrib/MAPIE).
3. Clone your fork locally.
4. Configure your [development environment](#development-environment).
5. Commit and push the changes to your fork. Do not forget to [make your contribution visible](#make-your-contribution-visible).
6. Send a pull request from your fork back to the original `master` branch. We encourage you to open a draft pull request early in the development process to get feedback on your idea and assistance from the `mapie` maintainers if needed.
7. Iteration phase: `mapie` maintainers will review your contribution and provide feedback. If needed, make changes and update your pull request, marking it as ready for review when you are done.
8.  Once your contribution is validated (see the [validation process](#validation-process)), it will be merged into the `master` branch.

## What to work on

You can start by checking the list of [open issues](https://github.com/scikit-learn-contrib/MAPIE/issues). Issues tagged "Good first issue" are perfect for open-source beginners. For the more experienced, issues tagged "Contributors welcome" are recommended if you want to help.

You are also welcome to propose and contribute to new ideas, such as improving an existing implementation, adding new methods from the literature, or expanding the documentation.
We encourage you to open a new issue so that we can align on the work to be done.
It is generally a good idea to have a quick discussion before opening a pull request that is potentially out-of-scope.

When implementing a new method, it should be supported by a journal or conference publication. If this is not the case, the contribution will be marked as 'experimental' until further validation from the community is available.

## Development environment

### using uv
We recommend using [uv](https://docs.astral.sh/uv/), an extremely fast Python package and project manager, to create an environment for `mapie`.
You'll want to activate it every time you want to work on `mapie`.
Here is how to install all dependencies at once.

The `dev` extra includes development dependencies such as linters and testing tools, the `docs` extra includes documentation dependencies (optional), and the `notebooks` extra includes Jupyter notebook dependencies (optional).

MacOS users should install `libomp` beforehand if it is not already present (`brew install libomp`) because LightGBM depends on it and is used in the documentation.


```sh
uv sync --python 3.12 --extra dev --extra docs --extra notebooks
```

Then, either activate the virtual environment created by `uv`:
```sh
source .venv/bin/activate
```

or use `uv run` to run commands inside the virtual environment without activating it:
```sh
uv run <command>
```

To run and edit the Jupyter notebooks located in the `notebooks/` folder, you need the `notebook` dependencies installed in your environment (see above).
You can then use Jupyter lab.

```sh
jupyter lab # if virtual environment is activated
uv run jupyter lab # if virtual environment is not activated
```

### using pip

Alternatively, using `pip`, you can install development dependencies with the following command:

```sh
python -m pip install -e '.[dev,docs,notebooks]'
```

If you don't have `pip` installed, you can install it by running:

```sh
python -m ensurepip --upgrade
```


## Make your contribution visible

1. If your change is user-facing (bug fix, feature, ...), add a line to describe it in [HISTORY.md](https://github.com/scikit-learn-contrib/MAPIE/blob/master/HISTORY.md)
2. If you want to be acknowledged for your contribution, add your name to the Contributors section of [AUTHORS.md](https://github.com/scikit-learn-contrib/MAPIE/blob/master/AUTHORS.md)


## Validation process

Many aspects of your contribution will be automatically checked by Continuous Integration (CI) tools, and the result will be displayed in the pull request. To debug your contribution, you can run the following checks locally.

### Code compatibility

For public classes or functions, the API must be compatible with `mapie`. For instance, when implementing a new conformity score for regression, the new class must inherit from `BaseRegressionScore` and implement the `get_signed_conformity_scores` and `get_estimation_distribution` methods.

### Code quality

The linter must pass:

```sh
make lint
```
The formatting must pass (if you are not already using `ruff`, you can run `make format-fix` to auto-format your code before committing):

```sh
make format
```

The typing must pass.

```sh
make type-check
```

### Testing your change

A good development practice is to add tests for your change if applicable. You can look at existing tests for inspiration in `mapie/tests` and please read [the tests README.md](https://github.com/scikit-learn-contrib/MAPIE/blob/master/mapie/tests/README.md) for guidance.

The coverage should absolutely be 100%, meaning that all lines of code are covered by tests. You can check the coverage locally with:

```sh
make coverage
```

The tests absolutely have to pass. You can run the test suite directly (optional) with:

```sh
make tests
```

### Documenting your change

If you're adding a public class or function, then you'll need to add a docstring with a doctest to describe the API for the users. We follow the [numpy docstring convention](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html), so please do too. You can look at e.g., `BinaryClassificationController` for an example of a well-documented class.

Any estimator should follow the [scikit-learn API](https://scikit-learn.org/stable/developers/develop.html), so please follow these guidelines.



### Adding examples

We highly recommand adding an example to illustrate your contribution if applicable. This gives users a quick way to discover and understand how to use your new feature. The `mapie` documentation already contains many examples that you can look at for inspiration. Python scripts located in the `examples/` folder are automatically included in the documentation.

In order to build the documentation locally, you need you need the `docs` dependencies installed in your environment (see local setup above).
You can then build the documentation locally by running:

```sh
make clean-doc
make doc
```

For each commit pushed to a pull request, the documentation is automatically built and deployed to a temporary URL that you can access from the Continuous Integration (CI) results. This allows you to verify that your changes are correctly documented and you can also look at the diff with the previous version.









