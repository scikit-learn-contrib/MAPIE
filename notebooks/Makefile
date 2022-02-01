convert2rst:
	jupyter nbconvert --to rst $(dir)/$(file).ipynb
	sed -i '/error/d' $(dir)/$(file).rst
	sed -i '/warning/d' $(dir)/$(file).rst
	sed -i '/import os/d' $(dir)/$(file).rst
	sed -i '/os.environ/d' $(dir)/$(file).rst
	sed -i '/UserWarning/d' $(dir)/$(file).rst
	sed -i '/WARNING:tensorflow/d' $(dir)/$(file).rst
	sed -i 's/.. code:: ipython3/.. code-block:: python/g' $(dir)/$(file).rst
	cp $(dir)/$(file).rst ../doc/$(file).rst
	cp -r $(dir)/$(file)_files ../doc/