# The report

`verimem.tex` builds to **7 pages with 4 figures**, references on page 7. `acl.sty` and
`acl_natbib.bst` are vendored here, taken from
[acl-org/acl-style-files](https://github.com/acl-org/acl-style-files).

## To build it

```bash
tectonic -X compile report/verimem.tex
```

Tectonic is a single self-contained binary that fetches what it needs on demand — far
lighter than a full TeX install. Homebrew could not reach `ghcr.io` on this machine, so
it was installed from the GitHub release directly. Overleaf works too: upload
`verimem.tex`, `verimem.bib`, `acl.sty`, `acl_natbib.bst` and `figures/`.

Two things that will break the build if changed:

- **Do not add `\bibliographystyle`.** `acl.sty` already issues it, and a second one makes
  BibTeX skip the command and silently drop every reference — the PDF still compiles,
  just with no bibliography.
- **`\usepackage{acl}`, not `[review]`.** The review option anonymises the author block to
  "Anonymous ACL submission" and adds line numbers. Use it only for a blind submission.

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
