.PHONY: install uninstall dev clean

PACKAGE_NAME=event_based_control

install:
	pip install .

dev:
	pip install -e .

uninstall:
	pip uninstall -y $(PACKAGE_NAME)

clean:
	python -c "import shutil, os, glob; [shutil.rmtree(p) for p in glob.glob('*.egg-info') + ['build', 'dist'] if os.path.exists(p)]"
