"""docs/TECHNICAL_SUMMARY.md: the brief's concise technical document (a reviewer's entry point)."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = REPO_ROOT / "docs" / "TECHNICAL_SUMMARY.md"
TECHNICAL = REPO_ROOT / "docs" / "TECHNICAL.md"
WORD_LIMIT = 2000
BRIEF_SECTIONS = (
    "1. Problem formulation",
    "2. Architecture",
    "3. Data assumptions",
    "4. Feature engineering",
    "5. Model selection",
    "6. Training methodology",
    "7. Ranking methodology",
    "8. Scoring methodology",
    "9. Evaluation",
    "10. Cold-start strategy",
    "11. Production considerations",
)


def _slug(heading: str) -> str:
    """GitHub's heading anchor: lower-case, drop punctuation, spaces to hyphens."""
    text = heading.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s", "-", text)


def _sections(text: str) -> dict[str, str]:
    parts = re.split(r"^## ", text, flags=re.MULTILINE)[1:]
    return {p.split("\n", 1)[0].strip(): p for p in parts}


def test_summary_is_within_the_word_limit() -> None:
    words = len(SUMMARY.read_text(encoding="utf-8").split())
    assert words <= WORD_LIMIT, f"{words} words; the brief asks for a concise document"


def test_the_eleven_brief_sections_come_in_the_briefs_order_with_a_link_to_depth() -> None:
    sections = _sections(SUMMARY.read_text(encoding="utf-8"))
    headings = [h for h in sections if h[0].isdigit()]
    assert tuple(headings) == BRIEF_SECTIONS
    for heading in BRIEF_SECTIONS:
        body = sections[heading].rstrip()
        last_line = body.splitlines()[-1]
        assert last_line.startswith("Depth: ") and "](" in last_line, heading
        assert 60 <= len(body.split()) <= 200, f"{heading}: {len(body.split())} words"


def test_opens_with_five_lines_and_closes_with_at_most_six_limitations() -> None:
    sections = _sections(SUMMARY.read_text(encoding="utf-8"))
    five = [ln for ln in sections["In five lines"].splitlines() if re.match(r"\d\. ", ln)]
    assert len(five) == 5
    limits = [ln for ln in sections["Known limitations"].splitlines() if ln.startswith("- ")]
    assert 1 <= len(limits) <= 6


def test_every_technical_md_anchor_exists() -> None:
    headings = re.findall(r"^#{2,3} (.+)$", TECHNICAL.read_text(encoding="utf-8"), re.MULTILINE)
    anchors = {_slug(h) for h in headings}
    links = re.findall(r"\(TECHNICAL\.md#([^)]+)\)", SUMMARY.read_text(encoding="utf-8"))
    assert links
    missing = [a for a in links if a not in anchors]
    assert not missing, missing


def test_the_summary_states_decisions_not_their_history() -> None:
    text = SUMMARY.read_text(encoding="utf-8").lower()
    assert "amendment" not in text
    assert "pre-registered" in text  # the rule is stated; how it evolved lives in docs/experiments/
