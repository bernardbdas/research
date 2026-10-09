# .latexmkrc - High-performance build configuration for LaTeX documents
# Uses pdflatex with automatic dependency tracking, synctex, and multiple passes

$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';
$recorder = 1;
$bibtex_use = 2;

# Automatically build main.tex by default
@default_files = ('main.tex');

# File extensions to clean up with latexmk -c
$clean_ext = 'aux fdb_latexmk fls log out toc synctex.gz';
