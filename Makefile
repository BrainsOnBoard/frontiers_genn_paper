# Simple Xe/LaTeX Makefile
# (C) Andrew Mundy 2012

# Configuration
TEX=pdflatex
BIB=bibtex
TEXFLAGS=--shell-escape
BIBFLAGS=
COMMIT?=5d8b8cdd51fd05a4699d24d5ff4afd7ed983eb95
texdoc=frontiers_genn

TIKZ_PICTURES :=$(wildcard figures/*.tex)
TIKZ_EPS :=$(foreach fig,$(basename $(TIKZ_PICTURES)),$(fig).eps)

.PHONY: clean bib count all

# Make all items
all : $(texdoc).pdf $(TIKZ_EPS)

markup : $(texdoc)_markup.pdf $(TIKZ_EPS)

$(texdoc)_markup.pdf : $(texdoc)_markup.tex
	-$(TEX) -interaction=nonstopmode $(TEXFLAGS) $(texdoc)_markup
	-$(BIB) $(BIBFLAGS) $(texdoc)_markup
	-$(TEX) -interaction=nonstopmode $(TEXFLAGS) $(texdoc)_markup
	-$(TEX) -interaction=nonstopmode $(TEXFLAGS) $(texdoc)_markup

$(texdoc).pdf : $(texdoc).tex
	$(TEX) $(TEXFLAGS) $(texdoc)

$(texdoc)_compare.tex : $(texdoc).tex
	git show $(COMMIT):$(texdoc).tex  > $(texdoc)_compare.tex

$(texdoc)_markup.tex : $(texdoc)_compare.tex
	latexdiff $(texdoc)_compare.tex $(texdoc).tex > $(texdoc)_markup.tex

# Generate reference requirements
$(texdoc).aux : $(texdoc).tex
	$(TEX) $(TEXFLAGS) $(texdoc)

# Generate the bibliography
bib : $(texdoc).aux
	$(BIB) $(BIBFLAGS) $(texdoc)
	$(TEX) $(TEXFLAGS) $(texdoc)
	$(TEX) $(TEXFLAGS) $(texdoc)

# Build PDFs from Tikz diagrams
figures/%.pdf: figures/%.tex
	pdflatex -output-directory=figures/ $< 

# Build EPSs from PDFs
figures/%.eps: figures/%.pdf
	gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile=$@ $<
# Clean
clean :
	find . -type f -regex ".*$(texdoc).*\.\(aux\|bbl\|bcf\|blg\|log\|png\|out\|toc\|lof\|lot\|count\)" -delete
	rm -f $(texdoc).pdf $(texdoc)_compare.* $(texdoc)_markup.* $(texdoc).run.xml $(texdoc)-blx.bib
