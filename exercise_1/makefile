REPORT_BASE_NAME := submission

$(REPORT_BASE_NAME).pdf:
	latexmk -pdf -use-make $(REPORT_BASE_NAME)

all: $(REPORT_BASE_NAME).pdf

clean:
	latexmk -CA
	rm -f $(REPORT_BASE_NAME).bbl
	rm -f $(REPORT_BASE_NAME).run.xml
	rm -f $(REPORT_BASE_NAME)-blx.bib

.PHONY: $(REPORT_BASE_NAME).pdf all clean

