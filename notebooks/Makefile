convert2rst:
	jupyter nbconvert --to rst $(dir)/$(file).ipynb
	sed -n '/error/!p' $(dir)/$(file).rst > $(dir)/$(file)2.rst
	sed -n '/warning/!p' $(dir)/$(file)2.rst > $(dir)/$(file)3.rst
	sed -n '/import os/!p' $(dir)/$(file)3.rst > $(dir)/$(file)4.rst
	sed -n '/os.environ/!p' $(dir)/$(file)4.rst > $(dir)/$(file)5.rst
	sed -n '/UserWarning/!p' $(dir)/$(file)5.rst > $(dir)/$(file)6.rst
	sed -n '/WARNING:tensorflow/!p' $(file)6.rst > $(dir)/$(file)7.rst
	cp $(dir)/$(file)7.rst ../doc/$(file).rst
	cp -r $(dir)/$(file)_files ../doc/
    rm $(dir)/$(file)*.rst