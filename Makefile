install: ## [Local development] Upgrade pip, install requirements, install package.
	python -m pip install -U pip
	python -m pip install -e .

install-training:
	python -m pip install -r requirements-training.txt

install-test: ## [Local development] Install test requirements
	python -m pip install -r requirements-test.txt

lint: ## [Local development] Run mypy, pylint and black
	python -m black --check -l 120 open_ml

black: ## [Local development] Auto-format python code using black
	python -m black -l 120 open_ml

TEST_ARGS = tests ## set default to run all tests
test: ## [Local development] Run unit tests
	python -m pytest -x -s -v $(TEST_ARGS) -m "not gpu and not s3"

test-gpu: ## [Local development] Run unit tests
	python -m pytest -x -s -v $(TEST_ARGS) -m gpu

.PHONY: help

help: # Run `make help` to get help on the make commands
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
