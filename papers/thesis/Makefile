TEX_ROOT := thesis.tex
# Where to find the bibliography.
PAPERS_DIR := $(realpath ..)
BIB_FILE := $(PAPERS_DIR)/bibliography.bib
STYLE_FILE := $(PAPERS_DIR)/research.sty
BUILD_DIR := build
CHAPTERS_DIR := chapters
CHAPTERS := $(shell ls $(CHAPTERS_DIR)/*.tex)
CHAPTER_TARGETS := $(CHAPTERS:%.tex=$(BUILD_DIR)/%.pdf)

THESIS_DOC := $(BUILD_DIR)/$(TEX_ROOT:%.tex=%.pdf)
SRC := $(shell find . -name '*.tex') $(BIB_FILE) $(STYLE_FILE)

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

## Build the entire thesis document and any supporting matter.
.PHONY: all
all: warn-todos
all: $(SRC) $(THESIS_DOC)

## Build all chapters individually.
.PHONY: chapters
chapters: $(CHAPTER_TARGETS)

## Build the Introduction chapter.
.PHONY: introduction
introduction: $(BUILD_DIR)/$(CHAPTERS_DIR)/introduction.pdf

## Build the Literature Review chapter.
.PHONY: literature-review
literature-review: $(BUILD_DIR)/$(CHAPTERS_DIR)/literature-review.pdf

## Build the Methodology chapter.
.PHONY: methodology
methodology: $(BUILD_DIR)/$(CHAPTERS_DIR)/methodology.pdf

## Build the Results chapter.
.PHONY: results
results: $(BUILD_DIR)/$(CHAPTERS_DIR)/results.pdf

## Build the Conclusions chapter.
.PHONY: conclusions
conclusions: $(BUILD_DIR)/$(CHAPTERS_DIR)/conclusions.pdf

$(BUILD_DIR)/%.pdf: %.tex $(SRC)
	@# Use -outdir=$(@D) for nested directories, but then the subfile paths are relative to build output.
	BIBINPUTS=$(PAPERS_DIR) latexmk -pdf -outdir=$(@D) -bibtex --synctex=1 -file-line-error -interaction=batchmode -shell-escape $<

## LaTeX Utilities

## Attempt building the document without latexmk to help determine the source of build errors.
.PHONY: error
error:
	pdflatex -shell-escape -file-line-error $(TEX_ROOT)

## Clean all generated files.
.PHONY: clean-all
clean-all:
	rm -rf $(BUILD_DIR)/*

## Clean any auxiliary files.
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)/_minted-*
	find $(BUILD_DIR) -type f -not -name '*.pdf' -delete

## Open the document in a PDF viewer.
.PHONY: view
view:
	1>/dev/null 2>/dev/null xdg-open $(THESIS_DOC) &

## Parse LaTeX source code looking for TODO comments.
## TODO: Expand to include \todo checking too?
.PHONY: warn-todos
warn-todos:
	@BOLD=$$(tput bold);                                                                           \
	RED=$$(tput setaf 1);                                                                          \
	RESET=$$(tput sgr0);                                                                           \
	grep -irn --include "*.tex" --include "*.sty" --include "Makefile" "[%,#] TODO:" &&            \
	echo "$${RED}===============================================================" &&               \
	echo "$${BOLD}Warning:$${RESET}$${RED} TODO comment(s) found." &&                              \
	echo "$${RED}===============================================================$${RESET}" || true
