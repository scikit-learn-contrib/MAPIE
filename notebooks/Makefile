convert2rst:
	jupyter nbconvert --to rst $(dir)/$(file).ipynb
	perl -i -pe'/error/d' $(dir)/$(file).rst
	perl -i -pe'/warning/d' $(dir)/$(file).rst
	perl -i -pe'/import os/d' $(dir)/$(file).rst
	perl -i -pe'/os.environ/d' $(dir)/$(file).rst
	perl -i -pe'/UserWarning/d' $(dir)/$(file).rst
	perl -i -pe'/WARNING:tensorflow/d' $(dir)/$(file).rst
	perl -i -pe's/.. code:: ipython3/.. code-block:: python/g' $(dir)/$(file).rst
	cp $(dir)/$(file).rst ../doc/$(file).rst
	cp -r $(dir)/$(file)_files ../doc/