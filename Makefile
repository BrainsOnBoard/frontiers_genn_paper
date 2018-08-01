# Simple Xe/LaTeX Makefile
# (C) Andrew Mundy 2012

# Configuration
TEX=pdflatex
BIB=bibtex
TEXFLAGS=--shell-escape
BIBFLAGS=
texdoc=frontiers_genn

.PHONY: clean
.PHONY: cmyk
.PHONY: bib
.PHONY: complete
.PHONY: count

# Make all items
all : $(texdoc).pdf
	$(TEX) $(TEXFLAGS) $(texdoc)

$(texdoc).pdf : $(texdoc.tex)
	$(TEX) $(TEXFLAGS) $(texdoc)

# Complete (rather than quick build)
complete : clean bib all

# Generate reference requirements
$(texdoc).aux : $(texdoc).tex
	$(TEX) $(TEXFLAGS) $(texdoc)

# Generate the bibliography
bib : $(texdoc).aux
	$(BIB) $(BIBFLAGS) $(texdoc)
	$(TEX) $(TEXFLAGS) $(texdoc)
	$(TEX) $(TEXFLAGS) $(texdoc)

# Clean
clean :
	find . -type f -regex ".*$(texdoc).*\.\(aux\|bbl\|bcf\|blg\|log\|png\|out\|toc\|lof\|lot\|count\)" -delete
	rm -f $(texdoc).pdf $(texdoc).run.xml $(texdoc)-blx.bib
