# Reports

This directory contains the LaTeX source files for our CSCI 441 Neural Networks project reports, including the project proposal, experimental design, and final report.

## Editing and Previewing

You can use an online LaTeX editor or work locally.

### Option 1: Online Editors

No local LaTeX installation is required if you use a cloud-based editor with compilation support.

- [Overleaf](https://www.overleaf.com/) — Online LaTeX editor with PDF preview.
- [Prism](https://prism.openai.com/) — AI-assisted workspace for scientific writing and LaTeX.
- [Codex](https://chatgpt.com/codex) — Can help edit LaTeX files in the repository. Local PDF compilation requires a LaTeX toolchain.

### Option 2: Local Development (macOS)

Install a LaTeX distribution such as [MacTeX](https://www.tug.org/mactex/).

Alternatively, install the smaller BasicTeX distribution if you prefer a minimal setup.

Verify the installation:

```bash
pdflatex --version
```

Compile a report from the repository root:

```bash
cd reports
pdflatex reports/proposal.tex
```

The compiled PDF will be available in `build/proposal.pdf`.

You can use VS Code with the LaTeX Workshop extension or any preferred editor to edit and preview the documents.

## Notes

- Edit the `.tex` source files rather than generated PDFs.
- LaTeX build artifacts are excluded from Git.
- Keep changes focused and coordinate edits to shared reports to avoid merge conflicts.
