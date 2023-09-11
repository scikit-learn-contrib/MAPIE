=======================
Contribution guidelines
=======================

What to work on?
----------------

You are welcome to propose and contribute new ideas.
We encourage you to `open an issue <https://github.com/simai-ml/MAPIE/issues>`_ so that we can align on the work to be done.
It is generally a good idea to have a quick discussion before opening a pull request that is potentially out-of-scope.

Fork/clone/pull
---------------

The typical workflow for contributing to `mapie` is:

1. Fork the `master` branch from the `GitHub repository <https://github.com/simai-ml/MAPIE>`_.
2. Clone your fork locally.
3. Commit changes.
4. Push the changes to your fork.
5. Send a pull request from your fork back to the original `master` branch.

Local setup
-----------

We encourage you to use a virtual environment. You'll want to activate it every time you want to work on `mapie`.

You can create a virtual environment via `conda`:

.. code:: sh

    $ conda env create -f environment.dev.yml
    $ conda activate mapie

Alternatively, you can install dependencies with `pip`:

.. code:: sh

    $ pip install -r requirements.dev.txt

Finally install `mapie` in development mode:

.. code:: sh

    pip install -e .


Documenting your change
-----------------------

If you're adding a class or a function, then you'll need to add a docstring with a doctest. We follow the `numpy docstring convention <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_, so please do too.
Any estimator should follow the [scikit-learn API](https://scikit-learn.org/stable/developers/develop.html), so please follow these guidelines.
In order to build the documentation locally, run :

.. code:: sh

    $ cd doc
    $ make clean-doc
    $ make doc


Updating changelog
------------------

You can make your contribution visible by :

1. adding your name to the Contributors sections of `AUTHORS.rst <https://github.com/simai-ml/MAPIE/blob/master/AUTHORS.rst>`_
2. adding a line describing your change into `HISTORY.rst <https://github.com/simai-ml/MAPIE/blob/master/HISTORY.rst>`_

Testing
-------

Linting
^^^^^^^

These tests absolutely have to pass.

.. code:: sh

    $ make lint


Static typing
^^^^^^^^^^^^^

These tests absolutely have to pass.

.. code:: sh

    $ make type-check


Unit tests
^^^^^^^^^^

These tests absolutely have to pass.

.. code:: sh

    $ make tests

Coverage
^^^^^^^^

The coverage should absolutely be 100%.

.. code:: sh

    $ make coverage
