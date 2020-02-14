DOCKER_RESEARCH_TAG := notgnoshi/research
DOCKER_BUILD_TRIGGER := .docker-build-trigger
REPO_INIT_TRIGGER := .repo-init-trigger
JUPYTER_PORT := 8888
# Use bash shell so that we use the `time` builtin rather than /usr/bin/time.
SHELL := bash
# The repository root directory inside the docker container
WORKSPACE := /workspaces/research
# The model configuration file. The path must be relative to the repository root.
CONFIG := data/models/markov-character.jsonc

## View this help message
.PHONY: help
help:
	@# Taken from https://gist.github.com/prwhite/8168133#gistcomment-2749866
	@awk '{ \
			if ($$0 ~ /^.PHONY: [a-zA-Z\-\_0-9]+$$/) { \
				helpCommand = substr($$0, index($$0, ":") + 2); \
				if (helpMessage) { \
					printf "\033[36m%-20s\033[0m %s\n", \
						helpCommand, helpMessage; \
					helpMessage = ""; \
				} \
			} else if ($$0 ~ /^[a-zA-Z\-\_0-9.]+:/) { \
				helpCommand = substr($$0, 0, index($$0, ":")); \
				if (helpMessage) { \
					printf "\033[36m%-20s\033[0m %s\n", \
						helpCommand, helpMessage; \
					helpMessage = ""; \
				} \
			# Handle multi-line comments \
			} else if ($$0 ~ /^##/) { \
				if (helpMessage) { \
					helpMessage = helpMessage"\n                     "substr($$0, 3); \
				} else { \
					helpMessage = substr($$0, 3); \
				} \
			# Handle section headings.\
			} else { \
				if (helpMessage) { \
					# Remove leading space \
					helpMessage = substr(helpMessage, 2); \
					print "\n"helpMessage \
				} \
				helpMessage = ""; \
			} \
		}' \
		$(MAKEFILE_LIST)

## Build the docker image necessary to run the deliverables

## Build the docker image for this repository. Takes about 3-4 CPU years and more than a few spare bytes.
.PHONY: docker-build
docker-build: $(DOCKER_BUILD_TRIGGER)

$(DOCKER_BUILD_TRIGGER): Dockerfile
	time docker build --tag $(DOCKER_RESEARCH_TAG) .
	touch $(DOCKER_BUILD_TRIGGER)

## Force a container rebuild. Docker can still choose to cache as many things as it wants. That's fine.
.PHONY: docker-rebuild
docker-rebuild:
	rm -f $(DOCKER_BUILD_TRIGGER)
	$(MAKE) build

## Run a shell in a fresh, new container.
## Useful for debugging, but since there are *no* bugs, I don't forsee this being used.
.PHONY: docker-shell
docker-shell: docker-build
	docker run \
		--user $(shell id -u):$(shell id -g) \
		--gpus all \
		--rm \
		--interactive \
		--tty \
		--mount "type=bind,source=$(shell pwd),target=$(WORKSPACE)" \
		$(DOCKER_RESEARCH_TAG) \
		bash

## Run the project deliverables

## Train and serialize the default Markov LM.
## Define CONFIG to specify which model to use.
## E.g., make CONFIG=data/models/markov-character.jsonc
.PHONY: train
train: $(REPO_INIT_TRIGGER)
	time docker run \
		--user $(shell id -u):$(shell id -g) \
		--gpus all \
		--rm \
		--interactive \
		--tty \
		--mount "type=bind,source=$(shell pwd),target=$(WORKSPACE)" \
		$(DOCKER_RESEARCH_TAG) \
		$(WORKSPACE)/haikulib/scripts/markov.py --train --config=$(WORKSPACE)/$(CONFIG)

## Generate haiku using the deserialized default Markov LM.
## Define CONFIG to specify which model to use.
## E.g., make CONFIG=data/models/markov-character.jsonc
.PHONY: generate
generate: $(REPO_INIT_TRIGGER)
	time docker run \
		--user $(shell id -u):$(shell id -g) \
		--gpus all \
		--rm \
		--interactive \
		--tty \
		--mount "type=bind,source=$(shell pwd),target=$(WORKSPACE)" \
		$(DOCKER_RESEARCH_TAG) \
		$(WORKSPACE)/haikulib/scripts/markov.py --generate --config=$(WORKSPACE)/$(CONFIG)

## Utilities

## Initialize the repository data
.PHONY: init-data
init-data: $(REPO_INIT_TRIGGER)

$(REPO_INIT_TRIGGER): $(DOCKER_BUILD_TRIGGER)
$(REPO_INIT_TRIGGER): haikulib/scripts/initialize.py
$(REPO_INIT_TRIGGER): haikulib/data/initialization.py
	time docker run \
		--user $(shell id -u):$(shell id -g) \
		--gpus all \
		--rm \
		--interactive \
		--tty \
		--mount "type=bind,source=$(shell pwd),target=$(WORKSPACE)" \
		$(DOCKER_RESEARCH_TAG) \
		$(WORKSPACE)/haikulib/scripts/initialize.py

	touch $(REPO_INIT_TRIGGER)

## Run Jupyter Lab from the Docker image.
## Uses actual witchcraft to open Jupyter Lab in a webbrowser.
.PHONY: jupyter
jupyter: $(REPO_INIT_TRIGGER)
	docker run \
		--user $(shell id -u):$(shell id -g) \
		--gpus all \
		--rm \
		--interactive \
		--tty \
		--mount "type=bind,source=$(shell pwd),target=$(WORKSPACE)" \
		--publish $(JUPYTER_PORT):$(JUPYTER_PORT) \
		$(DOCKER_RESEARCH_TAG) \
		jupyter lab --ip=0.0.0.0 --no-browser --port=$(JUPYTER_PORT) $(WORKSPACE) 2>&1 \
		| tee --output-error=warn /dev/tty \
		| grep --only-matching --max-count=1 "http://127\.0\.0\.1:$(JUPYTER_PORT)/?token=[0-9a-f]*" \
		| xargs xdg-open

## Run unit tests.
.PHONY: check
check: $(REPO_INIT_TRIGGER)
	docker run \
		--user $(shell id -u):$(shell id -g) \
		--gpus all \
		--rm \
		--interactive \
		--tty \
		--mount "type=bind,source=$(shell pwd),target=$(WORKSPACE)" \
		--workdir=$(WORKSPACE) \
		$(DOCKER_RESEARCH_TAG) \
		pytest
