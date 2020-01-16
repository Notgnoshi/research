DOCKER_RESEARCH_TAG := notgnoshi/research
DOCKER_BUILD_TRIGGER := .docker-build-trigger
JUPYTER_PORT := 8888

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

$(DOCKER_BUILD_TRIGGER):
	docker build --tag $(DOCKER_RESEARCH_TAG) .
	touch $(DOCKER_BUILD_TRIGGER)

## Force a container rebuild. Docker can still choose to cache as many things as it wants. That's fine.
.PHONY: docker-rebuild
docker-rebuild:
	rm -f $(DOCKER_BUILD_TRIGGER)
	$(MAKE) build

## Run the project deliverables

## TODO: Add target for training LM(s)
.PHONY: train
train:
	## Note that it will essentially be impossible (without pain) to pass arguments to the train and generate
	## targets from the commandline in a natural way. I think the best way will be to use a JSON config file.
	echo "TODO: Add target for training LM(s)"

## TODO: Add target for generating haiku
.PHONY: generate
generate:
	echo "TODO: Add target for generating haiku"

## Docker Utilities

## Run Jupyter Lab from the Docker image.
## Uses actual voodoo to open Jupyter Lab in a webbrowser.
.PHONY: jupyter
jupyter: docker-build
	docker run \
		--user $(shell id -u):$(shell id -g) \
		--gpus all \
		--rm \
		--interactive \
		--tty \
		--mount "type=bind,source=$(shell pwd),target=/home/nots/research" \
		--publish $(JUPYTER_PORT):$(JUPYTER_PORT) \
		$(DOCKER_RESEARCH_TAG) \
		jupyter lab --ip=0.0.0.0 --no-browser --port=$(JUPYTER_PORT) /home/nots/research 2>&1 \
		| tee --output-error=warn /dev/tty \
		| grep --only-matching --max-count=1 "http://127\.0\.0\.1:$(JUPYTER_PORT)/?token=[0-9a-f]*" \
		| xargs xdg-open

## Run a shell in a fresh, new container.
## Useful for debugging, but since there are *no* bugs, I don't forsee this being used.
.PHONY: shell
shell: docker-build
	docker run \
		--user $(shell id -u):$(shell id -g) \
		--gpus all \
		--rm \
		--interactive \
		--tty \
		$(DOCKER_RESEARCH_TAG) \
		bash
