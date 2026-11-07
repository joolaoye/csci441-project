# CSCI 441 Neural Networks Project Proposal

From this directory, run the following with a TeX distribution that includes
`latexmk`, pdfLaTeX, and the `geometry` package.

```sh
latexmk -pdf -outdir=build main.tex
```

The compiled proposal is at `build/main.pdf`.

Fill in the three sections in `main.tex`, replace the placeholder author names,
and enable the group contributions block when the details are final.
Project Update 1 has a 500-word limit.

When adding citations, put their BibTeX entries in `references.bib`, cite them
with `\cite{key}`, and uncomment the two bibliography commands in `main.tex`.
The same build command runs BibTeX when needed.
