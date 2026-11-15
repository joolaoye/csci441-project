# Reports

This directory contains the LaTeX source for the CSCI 441 project reports. Each report has its own directory and its own bibliography.

```text
reports/
├── proposal/
│   ├── proposal.tex
│   └── references.bib
├── experiment_design/
│   ├── experiment_design.tex
│   └── references.bib
└── README.md
```

## Editing and Previewing

You can use an online LaTeX editor or work locally.

### Option 1: Online Editors

No local LaTeX installation is required if you use a cloud-based editor with compilation support.

- [Overleaf](https://www.overleaf.com/) — Online LaTeX editor with PDF preview.
- [Prism](https://prism.openai.com/) — AI-assisted workspace for scientific writing and LaTeX.
- [Codex](https://chatgpt.com/codex) — Can help edit LaTeX files in the repository. Local PDF compilation requires a LaTeX toolchain.

When using an online editor, upload the report's `.tex` and `references.bib` files, along with any required figures or tables.

### Option 2: Local Development (macOS)

Install a LaTeX distribution such as [MacTeX](https://www.tug.org/mactex/).

Alternatively, install the smaller BasicTeX distribution if you prefer a minimal setup.

Verify the installation:

```bash
pdflatex --version
bibtex --version
```

### Compiling the Proposal

From the repository root:

```bash
cd reports/proposal
pdflatex proposal.tex
bibtex proposal
pdflatex proposal.tex
pdflatex proposal.tex
```

The compiled PDF is written to `reports/proposal/proposal.pdf`.

### Compiling the Experiment Design

From the repository root:

```bash
cd reports/experiment_design
pdflatex experiment_design.tex
bibtex experiment_design
pdflatex experiment_design.tex
pdflatex experiment_design.tex
```

The compiled PDF is written to `reports/experiment_design/experiment_design.pdf`.

Each report must be compiled from its respective directory so BibTeX can locate the corresponding `references.bib` file.

The first `pdflatex` run generates citation information, `bibtex` processes the bibliography, and the final two `pdflatex` runs resolve citations and references.

You can use VS Code with the LaTeX Workshop extension or any preferred editor to edit and preview the documents.

## Contributing

- Create a separate branch for report changes.
- Edit the LaTeX source file
- Compile and preview the document before submitting the change.
- Open a pull request for review.
- Keep the change focused, and coordinate edits to the same report so two people are not rewriting it at once.

## Generated files

Compiled PDFs and LaTeX build artifacts are excluded from Git.

Commit the source files, bibliographies, and any figures or tables required to compile the reports.

Selected comparison tables and figures should be copied manually into the corresponding report's directory. Original generated experimental outputs remain under `artifacts/`.

Do not commit temporary build artifacts or unselected experimental outputs.
