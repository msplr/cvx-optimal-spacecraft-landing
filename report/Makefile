# inspired by https://github.com/matze/mtheme

CACHE_DIR = latex-cache
COMPILE_TEX := latexmk -xelatex -output-directory=$(CACHE_DIR)

all: $(CACHE_DIR)
	$(COMPILE_TEX) -halt-on-error main.tex

watch: $(CACHE_DIR)
	$(COMPILE_TEX) -pvc -halt-on-error -pdf main.tex

clean:
	rm -rf $(CACHE_DIR)

$(CACHE_DIR):
	mkdir -p $(CACHE_DIR)