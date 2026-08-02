"""Integrity checks for the canonical theory-paper artifact."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THEORY = ROOT / "paper" / "theory"
SOURCE = THEORY / "paper.typ"
PROOFS = THEORY / "proofs.typ"
REFERENCES = THEORY / "references.bib"
PDF = ROOT / "output" / "pdf" / "when-does-reward-require-information.pdf"


def test_canonical_paper_is_complete() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    proofs = PROOFS.read_text(encoding="utf-8")

    for title in (
        "Positive-affine conjugacy",
        "Affine uniqueness under universal frontier conjugacy",
        "Invertible monotone classification reversal",
        "Reward-sufficient source reduction",
        "Behavioral action quotient",
        "Bounded positive-gap certificate",
        "Tail escape is necessary for positive-gap collapse",
        "Independent finite composition",
        "Countable local-price law",
    ):
        assert title in source

    assert '#include "proofs.typ"' in source
    assert '#bibliography("references.bib"' in source
    assert len(proofs) > 5_000
    for marker in ("TODO", "TBD", "supplied below in the final manuscript"):
        assert marker not in source
        assert marker not in proofs


def test_every_citation_has_a_bibliography_entry() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    bibliography = REFERENCES.read_text(encoding="utf-8")
    citations = set(re.findall(r"(?<![\w.])@([a-z][a-z0-9]+)", source))
    entries = set(re.findall(r"@[a-zA-Z]+\{([^,]+),", bibliography))

    assert citations
    assert citations <= entries


def test_checked_in_preprint_is_a_pdf() -> None:
    payload = PDF.read_bytes()

    assert payload.startswith(b"%PDF-")
    assert payload.rstrip().endswith(b"%%EOF")
    assert len(payload) > 100_000
