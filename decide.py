"""Decision logic: reference portrait choice and the overall verdict. Pure functions, no I/O."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from compare import MATCH, MISMATCH, UNCERTAIN, ComparisonResult
from quality import QualityResult

PASS = "PASS"
FAIL = "FAIL"
REVIEW = "NEEDS HUMAN REVIEW"
RETAKE = "RETAKE NEEDED"


@dataclass
class Verdict:
    verdict: str
    reasons: List[str] = field(default_factory=list)
    retake_prompts: List[str] = field(default_factory=list)   # "<image>: <prompt>"
    special_flags: List[str] = field(default_factory=list)


def choose_reference(q1: QualityResult, q2: QualityResult, ids_agree: Optional[bool]) -> tuple[Optional[int], str]:
    """Pick the reference portrait (0 or 1) and return a sentence saying which and why.

    Only IDs that passed quality can be used. If both passed, the higher quality score wins.
    """
    ok = [i for i, q in enumerate((q1, q2)) if not q.needs_retake and q.face is not None]
    names = ["ID 1", "ID 2"]
    if not ok:
        return None, "No ID portrait passed the quality check, so there is no reference portrait yet."
    if len(ok) == 1:
        i = ok[0]
        other = names[1 - i]
        sent = f"{names[i]} portrait was used as the reference because {other} needs a retake."
        if ids_agree is False:
            sent += " The two IDs disagree, so this needs a person to look at it."
        return i, sent
    qs = (q1, q2)
    i = 0 if qs[0].score >= qs[1].score else 1
    sent = (f"{names[i]} portrait was used as the reference because it has the higher image quality score "
            f"({qs[i].score:.2f} vs {qs[1 - i].score:.2f}).")
    if ids_agree is False:
        sent += " The two IDs disagree, so the reference is the better photo, not a trusted identity."
    return i, sent


def webcam_id_special_case(webcam_pairs: Sequence[ComparisonResult], webcam_vs_id: Sequence[ComparisonResult]) -> bool:
    """Flag: the webcam images agree with each other, but none of them matches the ID.

    Every webcam pair is MATCH and no webcam-vs-ID comparison is MATCH. The flag is raised for
    MISMATCH and UNCERTAIN alike; the verdict itself still follows the normal rules (all mismatches
    give FAIL, anything uncertain gives NEEDS HUMAN REVIEW).
    """
    return (len(webcam_pairs) > 0 and all(c.result == MATCH for c in webcam_pairs)
            and len(webcam_vs_id) > 0 and not any(c.result == MATCH for c in webcam_vs_id))


def overall_verdict(qualities: Sequence[QualityResult], comparisons: Sequence[ComparisonResult],
                    special_case: bool = False) -> Verdict:
    """Order of precedence:
    1. RETAKE NEEDED if any image failed the quality gate (we cannot judge a bad image,
       and a bad image never counts as a mismatch).
    2. FAIL if any comparison is MISMATCH.
    3. NEEDS HUMAN REVIEW if any comparison is UNCERTAIN or too few comparisons exist.
    4. PASS only when every comparison is MATCH.
    Uncertain results are never auto-rejected.
    """
    v = Verdict(PASS)
    for q in qualities:
        for p in q.retake_prompts:
            v.retake_prompts.append(f"{q.image_id}: {p}")
    if special_case:
        v.special_flags.append("Webcam images match each other but not the ID.")
    if v.retake_prompts:
        v.verdict = RETAKE
        v.reasons.append("One or more images are too poor to assess. They were skipped, not marked as a mismatch.")
        return v
    mism = [c for c in comparisons if c.result == MISMATCH]
    unc = [c for c in comparisons if c.result == UNCERTAIN]
    if not comparisons:
        v.verdict = REVIEW
        v.reasons.append("No comparisons could be made.")
        return v
    if mism:
        v.verdict = FAIL
        v.reasons += [f"Mismatch: {c.pair}" for c in mism]
        return v
    if unc:
        v.verdict = REVIEW
        v.reasons += [f"Uncertain: {c.pair}" for c in unc]
        return v
    v.reasons.append("Every comparison matched.")
    return v
