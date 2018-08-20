# Simple Xe/LaTeX Makefile
# (C) Andrew Mundy 2012

# Configuration
TEX=pdflatex
BIB=bibtex
TEXFLAGS=--shell-escape
BIBFLAGS=
texdoc=frontiers_genn

TIKZ_PICTURES :=$(wildcard figures/*.tex)
TIKZ_PDF :=$(foreach fig,$(basename $(TIKZ_PICTURES)),$(fig).pdf)

.PHONY: clean
.PHONY: bib
.PHONY: count

# Make all items
all : $(texdoc).pdf $(TIKZ_PDF)
	$(TEX) $(TEXFLAGS) $(texdoc)

$(texdoc).pdf : $(texdoc.tex)
	$(TEX) $(TEXFLAGS) $(texdoc)

# Generate reference requirements
$(texdoc).aux : $(texdoc).tex
	$(TEX) $(TEXFLAGS) $(texdoc)

# Generate the bibliography
bib : $(texdoc).aux
	$(BIB) $(BIBFLAGS) $(texdoc)
	$(TEX) $(TEXFLAGS) $(texdoc)
	$(TEX) $(TEXFLAGS) $(texdoc)

# Build pdfs from Tikz diagrams
figures/%.pdf: figures/%.tex
	pdflatex -output-directory=figures/ $< 
	
# Clean
clean :
	find . -type f -regex ".*$(texdoc).*\.\(aux\|bbl\|bcf\|blg\|log\|png\|out\|toc\|lof\|lot\|count\)" -delete
	rm -f $(texdoc).pdf $(texdoc).run.xml $(texdoc)-blx.bib
