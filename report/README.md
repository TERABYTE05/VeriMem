# The report

`verimem.tex` is written against the **ACL template**, which is not vendored here.
No LaTeX is installed in the project environment, so it has not been compiled.

## To build it

1. Create an Overleaf project from the official ACL template (it supplies `acl.sty`
   and `acl_natbib.bst`).
2. Upload `verimem.tex`, `verimem.bib`, and the whole `figures/` directory.
3. Compile. Switch `\usepackage[review]{acl}` to `\usepackage{acl}` for the camera-ready
   version, which removes the line numbers.

## Regenerating the figures

```bash
python -m report.make_figures
```

They are built from `results/` rather than drawn, so a re-run updates the paper. PNGs are
gitignored: they are build output.

## Where the numbers came from

Every figure in the paper traces to a results folder. The supporting notes under
`notes/` were written first and the paper draws on them:

| note | covers |
|---|---|
| `crag_reproduction.md` | P1.10 — what transferred from CRAG and what did not |
| `exprag_reproduction.md` | P2.12 — ExpRAG, and why the central claim did not replicate |
| `failure_analysis.md` | P4.5 — the error taxonomy over all 500 dev claims |
| `worked_traces.txt` | three traces, selected by rule rather than by hand |

Tables with bootstrap intervals are printed by:

```bash
python -m eval.report_tables --run results/2026-10-28_g1-rerun
python -m eval.report_tables --attack results/2026-10-28_poison-ablation-naive
```

## Checks before submitting

- The author block is a placeholder — fill in names and the contribution statement.
- `ferraz2026exprag` in the bibliography needs its real author list and venue.
- The ethics statement is written and should not be cut for space.


## If the venv misbehaves

It was created at an older path (`Desktop/VeriMem`) and the project has since moved. The
absolute paths baked into `.venv/bin/*` and `pyvenv.cfg` were repaired on 2026-11-04. If
`pip` reports a bad interpreter, or `python3` resolves outside the venv while `(.venv)`
shows in the prompt, the same staleness has returned:

```bash
grep -rl "Desktop/VeriMem" .venv/ | xargs sed -i '' \
  's|/Users/teesharamchandani/Desktop/VeriMem|/Users/teesharamchandani/Desktop/NLP-Assignment/VeriMem|g'
```

The installed packages are unaffected either way; only entry-point shebangs and the
activate scripts carry absolute paths.
