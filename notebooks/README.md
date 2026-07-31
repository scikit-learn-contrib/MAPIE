This folder contains a series of notebooks used for generating the tutorials for regression and classification as well as some other notebooks not represented in MAPIE examples.

# Create virtual environment

To run and convert the notebooks contained in this folder, create a dedicated virtual environment and install the notebook dependencies via the ``notebooks`` extra:

* `cd ../`
* `python -m venv .venv-mapie-notebooks`
* `source .venv-mapie-notebooks/bin/activate`
* `pip install -e '.[notebooks]'`
* `python -m ipykernel install --user --name=mapie_notebooks`

# Use notebooks in documentation

MAPIE's documentation is built with MkDocs. Prefer adding runnable documentation examples as Python scripts in the `examples/` folder so they are rendered by the gallery. Standalone notebooks can be linked from Markdown pages under `doc/` when a notebook format is more appropriate.
