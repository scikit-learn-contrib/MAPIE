# Contribution guidelines

## How to contribute?

If you are new to open source, you might want to start with this guide: [Contributing to open source on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-open-source).

The typical workflow for contributing to `mapie` is:

1. [Choose what to work on](#choose-what-to-work-on)
2. Fork the `master` branch from the [GitHub repository](https://github.com/scikit-learn-contrib/MAPIE).
3. Clone your fork locally.
4. [Configure your development environment](#configure-your-development-environment).
5. Commit changes.
6. Push the changes to your fork.
7. Send a pull request from your fork back to the original `master` branch.
8. Ensure that all tests pass and documentation builds successfully (automatically checked by continuous integration).
9. `mapie` maintainers will review your contribution and provide feedback.

### Choose what to work on

Issues tagged "Good first issue" are perfect for open-source beginners.

For the more experienced, issues tagged "Contributors welcome" are recommended if you want to help.

You are also welcome to propose and contribute to new ideas, such as improving an existing implementation, adding new methods from the literature, or expanding the documentation.
We encourage you to [open an issue](https://github.com/scikit-learn-contrib/MAPIE/issues) so that we can align on the work to be done.
It is generally a good idea to have a quick discussion before opening a pull request that is potentially out-of-scope.


## Validation process

### Code quality (linting and typing)

The linter must pass:

```sh
make lint
```

The typing must pass.

```sh
make type-check
```
- docstring and doctests for any new public class or function

### Testing your change

See [the tests README.md](https://github.com/scikit-learn-contrib/MAPIE/blob/master/mapie/tests/README.md) for guidance.

The coverage should absolutely be 100%.

```sh
make coverage
```

The tests absolutely have to pass. You can run the test suite directly (optional) with:

```sh
make tests
```

### Documenting your change

If you're adding a public class or function, then you'll need to add a docstring with a doctest. We follow the [numpy docstring convention](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html), so please do too.
Any estimator should follow the [scikit-learn API](https://scikit-learn.org/stable/developers/develop.html), so please follow these guidelines.

In order to build the documentation locally, you need you need the `docs` dependencies installed in your environment (see local setup above).
Finally, once dependencies are installed, you can build the documentation locally by running:

```sh
make clean-doc
make doc
```

## Configure your development environment

### uv
We recommended to use [uv](https://docs.astral.sh/uv/), an extremely fast Python package and project manager, to create an environment for `mapie`.
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
### pip

Alternatively, using `pip`, you can install development dependencies with the following command:

```sh
python -m pip install -e '.[dev,docs,notebooks]'
```

If you don't have `pip` installed, you can install it by running:

```sh
python -m ensurepip --upgrade
```







## Updating changelog

You can make your contribution visible by:

1. Adding your name to the Contributors section of [AUTHORS.md](https://github.com/scikit-learn-contrib/MAPIE/blob/master/AUTHORS.md)
2. If your change is user-facing (bug fix, feature, ...), adding a line to describe it in [HISTORY.md](https://github.com/scikit-learn-contrib/MAPIE/blob/master/HISTORY.md)
