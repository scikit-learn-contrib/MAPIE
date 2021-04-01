# MAPIE project: Model Agnostic Prediction Interval Estimator

This module allows you to easily estimate prediction intervals 
using your favourite sklearn-compatible regressor.


# Install Conda
* `curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh`
* `source Miniconda3-latest-MacOSX-x86_64.sh`
* `source ~/.bashrc`
* `rm Miniconda3-latest-MacOSX-x86_64.sh`

# Create virtual environment
* `conda env create -f environment_dev.yml`
* `conda activate mapie`
* `python -m ipykernel install --user --name=mapie`

# Create html documentation from rst files
* `cd doc/`
* `make clean`
* `make html`