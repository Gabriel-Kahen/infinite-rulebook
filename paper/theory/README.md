# Theory paper build

The canonical preprint is [`paper.typ`](paper.typ). It uses only Typst's
standard library and the local [`references.bib`](references.bib).

From the repository root, build the paper with Typst 0.15.1 or newer:

```bash
mkdir -p output/pdf
typst compile \
  paper/theory/paper.typ \
  output/pdf/when-does-reward-require-information.pdf
```

The checked-in PDF is the reviewed rendering of the same source. The analytic
results do not depend on the executable checks; those checks are documented in
[`verification-report.md`](verification-report.md).
