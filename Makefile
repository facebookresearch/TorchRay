VER=$(shell head -n 1 torchray/VERSION)

define get_conda_env
$(shell which python | xargs dirname | xargs dirname)
endef

.PHONY: all
all:


.PHONY: dist
dist:
		rm -rf dist
		docs/make-docs
		python3 setup.py sdist
		tar -tzvf dist/torchray-*.tar.gz


.PHONY: conda
conda:
		mkdir -p dist
		VER=$(VER) conda build -c defaults -c pytorch -c conda-forge \
		  --no-anaconda-upload --python 3.7 packaging/meta.yaml


.PHONY: conda-install
conda-install:
		conda install --use-local torchray


.PHONY: docs
docs:
		rm -rf docs/html
		docs/make-docs


.PHONY: pub
pub: docs
		git push pub master
		git push -f pub gh-pages
		git push -f --tags pub

		touch docs/html/.nojekyll
		git -C docs/html init
		git -C docs/html remote add pub git@github.com:facebookresearch/TorchRay.git
		git -C docs/html add .
		git -C docs/html commit -m "add documentation"
		git -C docs/html push -f pub master:gh-pages


.PHONY: tag
tag:
		git tag -f v$(VER)


.PHONY: distclean
distclean:
		rm -rf dist
