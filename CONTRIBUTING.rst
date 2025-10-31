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

We encourage you to use a virtual environment, with Python `3.10`. You'll want to activate it every time you want to work on `mapie`.

Using ``pip``, you can install dependencies with the following command:

.. code-block:: sh

    $ pip install -r requirements.dev.txt

If you work on Mac, you may have to install libomp manually in order to install LightGBM:

.. code-block:: sh

    $ brew install libomp

Finally, install ``mapie`` in development mode:

.. code-block:: sh

    $ pip install -e .

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

The tests absolutely have to pass.

.. code-block:: sh

    $ make tests

The coverage should absolutely be 100%.

.. code-block:: sh

    $ make coverage

If your modification affects notebook-based experiments in ``mapie/tests/notebooks/``,
you should also ensure that all notebooks run successfully.

.. code-block:: sh

    $ make notebook-tests

Documenting your change
-----------------------

If you're adding a public class or function, then you'll need to add a docstring with a doctest. We follow the `numpy docstring convention <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_, so please do too.
Any estimator should follow the `scikit-learn API <https://scikit-learn.org/stable/developers/develop.html>`_, so please follow these guidelines.

In order to build the documentation locally, you first need to create a different virtual environment than the one used for development, and then install some dependencies using ``pip`` with the following commands:

.. code-block:: sh

    $ pip install -r requirements.doc.txt
    $ pip install -e .

Finally, once dependencies are installed, you can build the documentation locally by running:

.. code-block:: sh

    $ make clean-doc
    $ make doc


Updating changelog
------------------

You can make your contribution visible by:

1. Adding your name to the Contributors section of `AUTHORS.rst <https://github.com/scikit-learn-contrib/MAPIE/blob/master/AUTHORS.rst>`_
2. If your change is user-facing (bug fix, feature, ...), adding a line to describe it in `HISTORY.rst <https://github.com/scikit-learn-contrib/MAPIE/blob/master/HISTORY.rst>`_
