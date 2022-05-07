convert2rst:
	jupyter nbconvert --to rst $(dir)/$(file).ipynb
	gsed -i -e'/error/d' $(dir)/$(file).rst
	gsed -i -e'/warning/d' $(dir)/$(file).rst
	gsed -i -e'/import os/d' $(dir)/$(file).rst
	gsed -i -e'/os.environ/d' $(dir)/$(file).rst
	gsed -i -e'/UserWarning/d' $(dir)/$(file).rst
	gsed -i -e'/WARNING:tensorflow/d' $(dir)/$(file).rst
	gsed -i -e's/.. code:: ipython3/.. code-block:: python/g' $(dir)/$(file).rst
	gsed -i -e's/``/`/g' $(dir)/$(file).rst
	gsed -i -e's/`TensorflowToMapie`/``TensorflowToMapie``/g' $(dir)/$(file).rst
	gsed -i -e'/<matplotlib/d' $(dir)/$(file).rst
	gsed -i -e'/<Figure/d' $(dir)/$(file).rst
	cp $(dir)/$(file).rst ../doc/$(file).rst
	cp -r $(dir)/$(file)_files ../doc/
	rm -rf $(dir)/.ipynb_checkpoints
	rm -rf $(dir)/${file}_files
	rm -rf $(dir)/${file}.rst