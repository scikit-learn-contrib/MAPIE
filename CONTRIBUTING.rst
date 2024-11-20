=======================
Contribution guidelines
=======================

What to work on?
----------------

You are welcome to propose and contribute new ideas.
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

We encourage you to use a virtual environment. You'll want to activate it every time you want to work on `mapie`.

You can create a virtual environment via ``conda``:

.. code-block:: sh

    $ conda env create -f environment.dev.yml
    $ conda activate mapie

Alternatively, using ``pip``, create a virtual environment and install dependencies with the following command:

.. code-block:: sh

    $ pip install -r requirements.dev.txt

Finally, install `mapie` in development mode:

.. code-block:: sh

    $ pip install -e .


Documenting your change
-----------------------

If you're adding a public class or function, then you'll need to add a docstring with a doctest. We follow the `numpy docstring convention <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_, so please do too.
Any estimator should follow the `scikit-learn API <https://scikit-learn.org/stable/developers/develop.html>`_, so please follow these guidelines.

In order to build the documentation locally, you first need to install some dependencies:

Create a dedicated virtual environment via ``conda``:

.. code-block:: sh

    $ conda env create -f environment.doc.yml
    $ conda activate mapie-doc

Alternatively, using ``pip``, create a different virtual environment than the one used for development, and install the dependencies:

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
2. Adding a line describing your change into `HISTORY.rst <https://github.com/scikit-learn-contrib/MAPIE/blob/master/HISTORY.rst>`_

Testing
-------

Linting
^^^^^^^

These tests absolutely have to pass.

.. code-block:: sh

    $ make lint


Static typing
^^^^^^^^^^^^^

These tests absolutely have to pass.

.. code-block:: sh

    $ make type-check


Unit tests
^^^^^^^^^^

These tests absolutely have to pass.

.. code-block:: sh

    $ make tests

Coverage
^^^^^^^^

The coverage should absolutely be 100%.

.. code-block:: sh

    $ make coverage
