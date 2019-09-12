TEX_ROOT := thesis.tex
# Where to find the bibliography.
PAPERS_DIR := $(realpath ..)
BIB_FILE := $(PAPERS_DIR)/bibliography.bib
STYLE_FILE := $(PAPERS_DIR)/research.sty
BUILD_DIR := build

OUT_DOC := $(BUILD_DIR)/$(TEX_ROOT:.tex=.pdf)
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

## Build the thesis document and any supporting matter.
.PHONY: all
all: warn-todos
all: $(OUT_DOC)

$(OUT_DOC): $(SRC)
	@# bibtex is ran inside the build directory, so use an absolute path.
	BIBINPUTS=$(PAPERS_DIR) latexmk -pdf -outdir=$(BUILD_DIR) -bibtex --synctex=1 -file-line-error -interaction=batchmode -shell-escape  $(TEX_ROOT)

## Attempt building the document without latexmk to help determine the source of build errors.
.PHONY: error
error:
	pdflatex -shell-escape -file-line-error $(TEX_ROOT)

## Clean any auxiliary files.
.PHONY: clean
clean:
	latexmk -outdir=$(BUILD_DIR) -c
	rm -f *.synctex\(busy\)
	rm -f *.synctex.gz
	rm -rf _minted-* $(BUILD_DIR)/*.{bbl,nav,snm,vrb,tdo}

## Open the document in a PDF viewer.
.PHONY: view
view:
	1>/dev/null 2>/dev/null xdg-open $(OUT_DOC) &

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
