"""Run the pipeline over labelled cases and print a confusion matrix and error rates.

Layout (folder name = expected outcome):

    cases/
      PASS/<case_name>/{id1,id2,a,b,c}.jpg     (+ optional id1.json, id2.json with the ID text fields)
      FAIL/<case_name>/...
      NEEDS_HUMAN_REVIEW/<case_name>/...
      RETAKE_NEEDED/<case_name>/...

    python scripts/evaluate.py cases/ --ocr sidecar     # offline: ID text from id1.json / id2.json
    python scripts/evaluate.py cases/ --ocr claude      # needs ANTHROPIC_API_KEY
    python scripts/evaluate.py cases/ --ocr tesseract

Rates:
  false-match rate  = FAIL-expected cases the app PASSED / FAIL-expected cases
  false-reject rate = PASS-expected cases the app FAILED / PASS-expected cases
  retake rate       = cases ending in RETAKE NEEDED / all cases
Cases sent to human review are counted separately, not as errors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faces import FaceEngine  # noqa: E402
from ocr import FixtureOcr, make_ocr  # noqa: E402
from pipeline import run_pipeline  # noqa: E402

LABELS = ["PASS", "FAIL", "NEEDS_HUMAN_REVIEW", "RETAKE_NEEDED"]
EXPECTED_BY_FOLDER = {l: l for l in LABELS}


def norm(label: str) -> str:
    return label.upper().replace(" ", "_")


def find_img(folder, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            return p
    return None


def load_case(folder):
    paths = [find_img(folder, s) for s in ("id1", "id2", "a", "b", "c")]
    imgs = [cv2.imread(p) if p else None for p in paths]
    sidecars = []
    for s in ("id1", "id2"):
        p = os.path.join(folder, s + ".json")
        sidecars.append(json.load(open(p)) if os.path.exists(p) else {})
    return imgs, sidecars


def sidecar_ocr(imgs, sidecars):
    def lookup(im):
        for i, side in zip(imgs[:2], sidecars):
            if i is not None and i.shape == im.shape and (i == im).all():
                return side
        return {}
    return FixtureOcr(lookup)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases")
    ap.add_argument("--ocr", choices=["sidecar", "claude", "tesseract"], default="sidecar")
    ap.add_argument("--verbose", "-v", action="store_true")
    a = ap.parse_args(argv)

    faces = FaceEngine()
    shared_ocr = None if a.ocr == "sidecar" else make_ocr(a.ocr)
    rows = []
    for expected in sorted(os.listdir(a.cases)):
        exp = norm(expected)
        d = os.path.join(a.cases, expected)
        if exp not in LABELS or not os.path.isdir(d):
            continue
        for case in sorted(os.listdir(d)):
            folder = os.path.join(d, case)
            if not os.path.isdir(folder):
                continue
            imgs, sidecars = load_case(folder)
            ocr = shared_ocr or sidecar_ocr(imgs, sidecars)
            r = run_pipeline(imgs[:2], imgs[2:], faces, ocr)
            got = norm(r.verdict.verdict)
            rows.append((exp, got, case))
            if a.verbose:
                print(f"{exp:20s} -> {got:20s} {case}")
    if not rows:
        sys.exit("no cases found")

    conf = Counter((e, g) for e, g, _ in rows)
    print("\nConfusion matrix (rows = expected, columns = got)")
    print(f"{'':22s}" + "".join(f"{l:>22s}" for l in LABELS))
    for e in LABELS:
        print(f"{e:22s}" + "".join(f"{conf[(e, g)]:>22d}" for g in LABELS))

    def rate(num, den):
        return f"{num}/{den} = {num / den:.1%}" if den else "n/a (no such cases)"

    fail_cases = [r for r in rows if r[0] == "FAIL"]
    pass_cases = [r for r in rows if r[0] == "PASS"]
    print()
    print("False-match rate  :", rate(sum(1 for r in fail_cases if r[1] == "PASS"), len(fail_cases)))
    print("False-reject rate :", rate(sum(1 for r in pass_cases if r[1] == "FAIL"), len(pass_cases)))
    print("Retake rate       :", rate(sum(1 for r in rows if r[1] == "RETAKE_NEEDED"), len(rows)))
    print("Sent to review    :", rate(sum(1 for r in rows if r[1] == "NEEDS_HUMAN_REVIEW"), len(rows)))
    wrong = [r for r in rows if r[0] != r[1]]
    if wrong:
        print("\nCases that did not match the expected outcome:")
        for e, g, c in wrong:
            print(f"  expected {e}, got {g}: {c}")


if __name__ == "__main__":
    main()
