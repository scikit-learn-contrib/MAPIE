This folder contains a series of notebooks used for generating the tutorials for regression and classification as well as some other notebooks not represented in MAPIE examples.

# Create virtual environment

To run and convert the notebooks contained in this folder, one needs first to create a separate conda environment containing additional packages from the
`environment.notebooks.yml`:

* `cd ../`
* `conda env create -f environment.notebooks.yml`
* `conda activate mapie_notebooks`
* `python -m ipykernel install --user --name=mapie_notebooks`

# Create notebooks

In order to make your notebook readable by the sphinx documentation, you need to convert your `.ipynb` file to a `.rst` format (along with the generated figures) and then copy the files in the `doc` directory. All these steps can be carried out at once with the following command:

* `make convert2rst dir="directory_name" file="file_name"`
