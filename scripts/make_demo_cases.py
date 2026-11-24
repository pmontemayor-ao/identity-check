"""Build a small labelled case folder from two sample portraits (tests/data) for a smoke run of evaluate.py.

    python scripts/make_demo_cases.py demo_cases && python scripts/evaluate.py demo_cases -v

The IDs are synthetic cards (portrait pasted on a card with printed text). This checks the plumbing and the
case matrix; it says nothing about real-world accuracy. Use real, consented, labelled data for that.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from faces import FaceEngine  # noqa: E402
from scripts import degrade  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "data")
FIELDS_A = {"name": "MARY SMITH", "date_of_birth": "1990-03-07", "id_number": "D1234567", "expiry_date": "2030-01-01"}
FIELDS_B = {"name": "JOHN DOE", "date_of_birth": "1985-01-01", "id_number": "X9988776", "expiry_date": "2029-05-05"}


def face_close_up(path, fe, size=480):
    img = cv2.imread(path)
    f = max(fe.detect(img), key=lambda r: r[2] * r[3])
    cx, cy, s = f[0] + f[2] / 2, f[1] + f[3] / 2, max(f[2], f[3]) * 2.6
    x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    crop = img[y0:int(cy + s / 2), x0:int(cx + s / 2)]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)


def make_id(portrait, fields, seed=0):
    rng = np.random.default_rng(seed)
    table = np.full((600, 900, 3), 55, np.uint8)
    table = np.clip(table.astype(np.int16) + rng.integers(-5, 6, table.shape), 0, 255).astype(np.uint8)
    cv2.rectangle(table, (60, 60), (840, 540), (215, 222, 228), -1)
    p = cv2.resize(portrait, (230, 290))
    table[120:410, 90:320] = p
    y = 130
    for k, v in (("NAME", fields["name"]), ("DOB", fields["date_of_birth"]), ("ID NO", fields["id_number"]), ("EXPIRES", fields["expiry_date"])):
        cv2.putText(table, f"{k}: {v}", (350, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
        y += 70
    return table


def webcam_shot(face_img, seed):
    rng = np.random.default_rng(seed)
    bg = np.full((600, 800, 3), 135, np.uint8)
    bg = np.clip(bg.astype(np.int16) + rng.integers(-8, 9, bg.shape), 0, 255).astype(np.uint8)
    f = cv2.resize(face_img, (330, 330))
    bg[130:460, 235:565] = cv2.convertScaleAbs(f, alpha=0.95 + 0.05 * seed, beta=2 * seed)
    return bg


def write(case_dir, id1, id2, cams, f1, f2):
    os.makedirs(case_dir, exist_ok=True)
    for name, im in zip(("id1", "id2", "a", "b", "c"), [id1, id2] + cams):
        cv2.imwrite(os.path.join(case_dir, name + ".jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    json.dump(f1, open(os.path.join(case_dir, "id1.json"), "w"))
    json.dump(f2, open(os.path.join(case_dir, "id2.json"), "w"))


def main(out):
    fe = FaceEngine()
    A = face_close_up(os.path.join(DATA, "lena.jpg"), fe)
    B = face_close_up(os.path.join(DATA, "messi5.jpg"), fe)
    idA1, idA2 = make_id(A, FIELDS_A, 1), make_id(cv2.GaussianBlur(A, (0, 0), 0.8), {**FIELDS_A, "name": "Smith Mary"}, 2)
    idB = make_id(B, FIELDS_B, 3)
    camsA = [webcam_shot(A, s) for s in range(3)]
    camsB = [webcam_shot(B, s) for s in range(3)]
    write(f"{out}/PASS/same_person", idA1, idA2, camsA, FIELDS_A, {**FIELDS_A, "name": "Smith Mary"})
    write(f"{out}/FAIL/different_person", idA1, idA2, camsB, FIELDS_A, {**FIELDS_A, "name": "Smith Mary"})
    write(f"{out}/FAIL/mismatched_ids", idA1, idB, camsA, FIELDS_A, FIELDS_B)
    # Scores land in the wide uncertain band (about 0.2), so the designed outcome is human review, with the flag shown.
    write(f"{out}/NEEDS_HUMAN_REVIEW/webcam_consistent_id_different", idB, idB, camsA, FIELDS_B, FIELDS_B)
    blurred = [camsA[0], degrade.blur(camsA[1]), camsA[2]]
    write(f"{out}/RETAKE_NEEDED/blurry_webcam", idA1, idA2, blurred, FIELDS_A, {**FIELDS_A, "name": "Smith Mary"})
    write(f"{out}/RETAKE_NEEDED/glare_id", degrade.glare(idA1), idA2, camsA, FIELDS_A, {**FIELDS_A, "name": "Smith Mary"})
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo_cases")
