#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = diabetes-mlops-v2
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install pinned dependencies (runtime + dev tools) and the project package in editable mode
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -r requirements-dev.txt
	$(PYTHON_INTERPRETER) -m pip install -e . --no-deps




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 diabetes_mlops_v2
	isort --check --diff diabetes_mlops_v2
	black --check diabetes_mlops_v2

## Format source code with black
.PHONY: format
format:
	isort diabetes_mlops_v2
	black diabetes_mlops_v2



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

MLFLOW_PORT ?= 5000

## Download the raw dataset from UCI and validate its schema
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m diabetes_mlops_v2.dataset

## Train the model and log the run to MLflow (e.g. make train ARGS="--C 0.1 --seed 7")
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m diabetes_mlops_v2.modeling.train $(ARGS)

## Start a local MLflow tracking server (SQLite backend + local artifact store)
.PHONY: mlflow-server
mlflow-server:
	mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts \
		--host 127.0.0.1 --port $(MLFLOW_PORT)


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
