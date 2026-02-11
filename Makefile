.PHONY: install uninstall dev clean clean-cache

PACKAGE_NAME=event_based_control

install:
	pip install .

dev:
	pip install -e .

uninstall:
	pip uninstall -y $(PACKAGE_NAME)

clean:
	@python -c "import shutil, os, pathlib; \
	[shutil.rmtree(p) for p in ['build', 'dist'] if os.path.exists(p)]; \
	[shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('*.egg-info')]"

clean-cache:
	@python -c "import shutil, os, pathlib; \
	[shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; \
	[os.remove(p) for p in pathlib.Path('.').rglob('*.py[co]')]"