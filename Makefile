# TWINE_ARGS = --repository testpypi

PIP_PACKAGES = bump twine
K := $(foreach exec,$(PIP_PACKAGES),\
        $(if $(shell which $(exec)),some string,$(error ERROR: First run "pip install $(exec)")))

publish_patch:
	@python setup.py check
	$(eval NEW_RELEASE := $(shell bump))
	@$(MAKE) NEW_RELEASE=${NEW_RELEASE} publish

publish_minor:
	@python setup.py check
	$(eval NEW_RELEASE := $(shell bump --minor --reset))
	@$(MAKE) NEW_RELEASE=${NEW_RELEASE} publish

publish_major:
	@python setup.py check
	$(eval NEW_RELEASE := $(shell bump --major --reset))
	@$(MAKE) NEW_RELEASE=${NEW_RELEASE} publish

publish:
	@echo Updating to release ${NEW_RELEASE}
	@rm -i dist/*
	@python setup.py sdist
	twine upload ${TWINE_ARGS} dist/*
	@echo Release ${NEW_RELEASE} pushed to PIP
  # @conda skeleton pypi pyfvvdp
