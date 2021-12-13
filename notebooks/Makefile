convert2rst:
	jupyter nbconvert --to rst $(dir)/$(file).ipynb
	sed -i '/error/!p' $(dir)/$(file).rst
	sed -i '/warning/!p' $(dir)/$(file).rst
	sed -i '/import os/!p' $(dir)/$(file).rst
	sed -i '/os.environ/!p' $(dir)/$(file).rst
	sed -i '/UserWarning/!p' $(dir)/$(file).rst
	sed -i '/WARNING:tensorflow/!p' $(dir)/$(file).rst
	sed -i 's/.. code:: ipython3/.. code-block:: python/g' $(dir)/$(file).rst
	cp $(dir)/$(file)7.rst ../doc/$(file).rst
	cp -r $(dir)/$(file)_files ../doc/