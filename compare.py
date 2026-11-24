"""Steps 2-4: comparisons. Every comparison is MATCH, MISMATCH, or UNCERTAIN, with its score."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from rapidfuzz import fuzz

MATCH = "MATCH"
MISMATCH = "MISMATCH"
UNCERTAIN = "UNCERTAIN"


@dataclass
class ComparisonResult:
    pair: str                  # e.g. "Webcam A vs Webcam B"
    group: str                 # "webcam_webcam" | "id_id" | "id_webcam" | "text"
    kind: str                  # "face" | "text"
    score: Optional[float]
    result: str                # MATCH | MISMATCH | UNCERTAIN
    detail: str = ""
    match_at: Optional[float] = None
    mismatch_below: Optional[float] = None


def classify(score: Optional[float], match_at: float, mismatch_below: float) -> str:
    """High threshold = match, low threshold = mismatch, in between = uncertain."""
    if score is None:
        return UNCERTAIN
    if score >= match_at:
        return MATCH
    if score < mismatch_below:
        return MISMATCH
    return UNCERTAIN


def face_comparison(pair: str, group: str, score: float, match_at: float, mismatch_below: float) -> ComparisonResult:
    return ComparisonResult(pair, group, "face", score, classify(score, match_at, mismatch_below),
                            match_at=match_at, mismatch_below=mismatch_below)


# ---------------------------------------------------------------- text
def normalise_name(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", (s or "").casefold())
    return " ".join(s.split())


def normalise_id_number(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


_DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d", "%d %b %Y", "%d %B %Y",
                 "%b %d %Y", "%B %d %Y", "%Y%m%d"]


def normalise_date(s: str) -> str:
    """Return ISO yyyy-mm-dd, or '' if it cannot be parsed.

    Slash dates are tried day-first, then month-first. A date that parses neither way is
    never guessed. Caveat: 03/04/2000 is read day-first.
    """
    s = re.sub(r"\s+", " ", (s or "").replace(",", " ")).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def compare_text_fields(a: Dict[str, str], b: Dict[str, str], name_match: float, name_mismatch: float) -> List[ComparisonResult]:
    """Name = fuzzy; DOB and ID number = exact after normalising. A missing field is UNCERTAIN."""
    out: List[ComparisonResult] = []
    na, nb = normalise_name(a.get("name", "")), normalise_name(b.get("name", ""))
    if na and nb:
        sc = float(fuzz.token_sort_ratio(na, nb))
        out.append(ComparisonResult("ID 1 vs ID 2: name", "text", "text", sc, classify(sc, name_match, name_mismatch),
                                    f'"{a.get("name")}" vs "{b.get("name")}"', name_match, name_mismatch))
    else:
        out.append(ComparisonResult("ID 1 vs ID 2: name", "text", "text", None, UNCERTAIN, "Name not readable on one or both IDs"))
    da, db = normalise_date(a.get("date_of_birth", "")), normalise_date(b.get("date_of_birth", ""))
    if da and db:
        out.append(ComparisonResult("ID 1 vs ID 2: date of birth", "text", "text", 100.0 if da == db else 0.0,
                                    MATCH if da == db else MISMATCH, f"{da} vs {db}"))
    else:
        out.append(ComparisonResult("ID 1 vs ID 2: date of birth", "text", "text", None, UNCERTAIN, "Date of birth not readable on one or both IDs"))
    ia, ib = normalise_id_number(a.get("id_number", "")), normalise_id_number(b.get("id_number", ""))
    if ia and ib:
        out.append(ComparisonResult("ID 1 vs ID 2: ID number", "text", "text", 100.0 if ia == ib else 0.0,
                                    MATCH if ia == ib else MISMATCH, f"{ia} vs {ib}"))
    else:
        out.append(ComparisonResult("ID 1 vs ID 2: ID number", "text", "text", None, UNCERTAIN, "ID number not readable on one or both IDs"))
    return out
