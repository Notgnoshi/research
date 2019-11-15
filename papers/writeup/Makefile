TEX_ROOT := writeup.tex
# Where to find the bibliography.
PAPERS_DIR := $(realpath ..)
BIB_DIR := $(PAPERS_DIR)/bibliography
STYLE_FILE := $(PAPERS_DIR)/research.sty
BUILD_DIR := build
SECTIONS_DIR := sections
BIB_FILES := $(shell ls $(BIB_DIR)/*.bib)
SECTIONS := $(shell ls $(SECTIONS_DIR)/*.tex)
SECTION_TARGETS := $(SECTIONS:%.tex=$(BUILD_DIR)/%.pdf)
TEX_DIRS := $(PAPERS_DIR) $(PAPETS_DIR)/writeup
TEX_DIRS := $(shell readlink -f $(TEX_DIRS))
TEX_DIRS := $(shell echo $(TEX_DIRS) | sed "s/ \+/:/g")

WRITEUP_DOC := $(BUILD_DIR)/$(TEX_ROOT:%.tex=%.pdf)
SRC := $(shell find . -name '*.tex') $(BIB_FILES) $(STYLE_FILE)

.DEFAULT_GOAL := all

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

## Building the document.

## Build the entire writeup document and any supporting matter.
.PHONY: all
all: $(SRC) $(WRITEUP_DOC)

## Build all sections individually.
.PHONY: sections
sections: $(SECTION_TARGETS)

## Build the Introduction section.
.PHONY: introduction
introduction: $(BUILD_DIR)/$(SECTIONS_DIR)/introduction.pdf

## Build the Methodology section.
.PHONY: methodology
methodology: $(BUILD_DIR)/$(SECTIONS_DIR)/methodology.pdf

## Build the Results section.
.PHONY: results
results: $(BUILD_DIR)/$(SECTIONS_DIR)/results.pdf

## Build the Conclusions section.
.PHONY: conclusions
conclusions: $(BUILD_DIR)/$(SECTIONS_DIR)/conclusions.pdf

$(BUILD_DIR)/%.pdf: %.tex $(SRC)
	@# Use -outdir=$(@D) for nested directories, but then the subfile paths are relative to build output.
	BIBINPUTS="$(BIB_DIR)" TEXINPUTS="$(TEX_DIRS):$$TEXINPUTS" latexmk -pdf -outdir=$(@D) -bibtex --synctex=1 -file-line-error -interaction=batchmode -shell-escape $<
	@$(MAKE) --no-print-directory warn-todos

## LaTeX Utilities

## Clean any auxiliary files.
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)/_minted-*
	[ -d "$(BUILD_DIR)" ] && find $(BUILD_DIR) -type f -not -name '*.pdf' -delete || true
	@# Building the subfiles through VS Code places the build directory in the wrong location
	[ -d "$(SECTIONS_DIR)/build" ] && find $(SECTIONS_DIR)/build -type f -not -name '*.pdf' -delete || true

## Clean all generated files.
.PHONY: clean-all
clean-all:
	rm -rf $(BUILD_DIR)/*
	rm -rf $(SECTIONS_DIR)/build/*

## Parse LaTeX source code looking for TODO comments and macros.
.PHONY: warn-todos
warn-todos:
	@BOLD=$$(tput bold);                                                                           \
	RED=$$(tput setaf 1);                                                                          \
	RESET=$$(tput sgr0);                                                                           \
	grep --color=always -irn --include "*.tex" --include "*.sty" "\([%,#] TODO:\|\\\TODO{\|\\\FIXME{\)" && \
	echo "$${RED}===============================================================" &&               \
	echo "$${BOLD}Warning:$${RESET}$${RED} TODO comment(s) found." &&                              \
	echo "$${RED}===============================================================$${RESET}" || true

## Open the documents in a PDF viewer.

## Open the whole thesis in a PDF viewer.
.PHONY: view
view:
	1>/dev/null 2>/dev/null xdg-open $(WRITEUP_DOC) &

## Open the Introduction chapter individually in a PDF viewer.
.PHONY: view-introduction
view-introduction:
	1>/dev/null 2>/dev/null xdg-open $(BUILD_DIR)/$(SECTIONS_DIR)/introduction.pdf &

## Open the Methodology chapter individually in a PDF viewer.
.PHONY: view-methodology
view-methodology:
	1>/dev/null 2>/dev/null xdg-open $(BUILD_DIR)/$(SECTIONS_DIR)/methodology.pdf &

## Open the Results chapter individually in a PDF viewer.
.PHONY: view-results
view-results:
	1>/dev/null 2>/dev/null xdg-open $(BUILD_DIR)/$(SECTIONS_DIR)/results.pdf &

## Open the Conclusions chapter individually in a PDF viewer.
.PHONY: view-conclusions
view-conclusions:
	1>/dev/null 2>/dev/null xdg-open $(BUILD_DIR)/$(SECTIONS_DIR)/conclusions.pdf &
