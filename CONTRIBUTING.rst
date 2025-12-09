=======================
Contribution guidelines
=======================

What to work on?
----------------

Issues tagged "Good first issue" are perfect for open-source beginners.

For the more experienced, issues tagged "Contributors welcome" are recommended if you want to help.

You are also welcome to propose and contribute to new ideas.
We encourage you to `open an issue <https://github.com/scikit-learn-contrib/MAPIE/issues>`_ so that we can align on the work to be done.
It is generally a good idea to have a quick discussion before opening a pull request that is potentially out-of-scope.

Fork/clone/pull
---------------

The typical workflow for contributing to `mapie` is:

1. Fork the ``master`` branch from the `GitHub repository <https://github.com/scikit-learn-contrib/MAPIE>`_.
2. Clone your fork locally.
3. Commit changes.
4. Push the changes to your fork.
5. Send a pull request from your fork back to the original ``master`` branch.


Local setup
-----------

We encourage you to use a virtual environment, with Python `3.10` and pip installed.
You'll want to activate it every time you want to work on `mapie`.

Here's how to create and activate a virtual environment called ``mapie_dev`` using `universal-virtualenv <https://pypi.org/project/universal-virtualenv/>`_.

.. code-block:: sh

    $ uv sync --python 3.10 --extra dev
    $ mv .venv mapie_dev
    $ source mapie_dev/bin/activate


Next, using ``pip``, you can install development dependencies with the following command:

.. code-block:: sh

    $ python -m pip install -e '.[dev]'

If you don't have ``pip`` installed, you can install it by running:

.. code-block:: sh

    $ python -m ensurepip --upgrade

Implementing your change
------------------------------------------

The linter must pass:

.. code-block:: sh

    $ make lint

The typing must pass.

.. code-block:: sh

    $ make type-check



Testing your change
---------------------

See `the tests README.md <https://github.com/scikit-learn-contrib/MAPIE/blob/master/mapie/tests/README.md>`_ for guidance.

The coverage should absolutely be 100%.

.. code-block:: sh

    $ make coverage

The tests absolutely have to pass. You can run the test suite directly (optional) with:

.. code-block:: sh

    $ make tests


Documenting your change
-----------------------

If you're adding a public class or function, then you'll need to add a docstring with a doctest. We follow the `numpy docstring convention <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_, so please do too.
Any estimator should follow the `scikit-learn API <https://scikit-learn.org/stable/developers/develop.html>`_, so please follow these guidelines.

In order to build the documentation locally, you first need to create a different virtual environment than the one used for development, and then install the documentation dependencies using ``pip`` with the following command. macOS users should install ``libomp`` beforehand if it is not already present (``brew install libomp``) because LightGBM depends on it.

Here's how to create and activate a virtual environment called ``mapie_docs`` using `universal-virtualenv <https://pypi.org/project/universal-virtualenv/>`_.

.. code-block:: sh

    $ uv sync --python 3.10 --extra docs
    $ mv .venv mapie_docs
    $ source mapie_docs/bin/activate


Next, using ``pip``, you can install documentation dependencies with the following command:

.. code-block:: sh

    $ python -m pip install -e '.[docs]'

Finally, once dependencies are installed, you can build the documentation locally by running:

.. code-block:: sh

    $ make clean-doc
    $ make doc


Running Jupyter notebooks
-------------------------

To run and edit the Jupyter notebooks located in the ``notebooks/`` folder, you first need to create a different virtual environment than the one used for development, and then install the notebook dependencies using ``pip`` with the following command.

Here's how to create and use a virtual environment called ``mapie_notebooks`` using `universal-virtualenv <https://pypi.org/project/universal-virtualenv/>`_.

.. code-block:: sh

    $ uv sync --python 3.10 --extra notebooks
    $ uv pip install --upgrade jsonschema referencing jupyter_server jupyterlab_server
    $ uv run jupyter lab


Updating changelog
------------------

You can make your contribution visible by:

1. Adding your name to the Contributors section of `AUTHORS.rst <https://github.com/scikit-learn-contrib/MAPIE/blob/master/AUTHORS.rst>`_
2. If your change is user-facing (bug fix, feature, ...), adding a line to describe it in `HISTORY.rst <https://github.com/scikit-learn-contrib/MAPIE/blob/master/HISTORY.rst>`_
