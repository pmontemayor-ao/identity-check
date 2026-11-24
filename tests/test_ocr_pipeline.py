import json
from types import SimpleNamespace

import numpy as np
import pytest

from compare import MATCH, MISMATCH, UNCERTAIN
from decide import FAIL, PASS, RETAKE, REVIEW
from ocr import ClaudeOcr, FixtureOcr, TesseractOcr, make_ocr, parse_json_object
from pipeline import plain_explanation, run_pipeline, template_explanation
from tests.synth import FakeFaces, id_card, webcam


class FakeClient:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0
        self.messages = self

    def create(self, **kw):
        self.calls += 1
        return SimpleNamespace(content=[SimpleNamespace(text=self.reply)])


def test_parse_json_tolerates_fences():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": 1} done') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_object("no json")


def test_claude_ocr_fields_obstruction_and_cache():
    reply = json.dumps({"name": "Mary Smith", "date_of_birth": "1990-03-07", "id_number": "D123", "expiry_date": None,
                        "obstruction": {"covered": True, "what": "thumb over name"}})
    c = FakeClient(reply)
    o = ClaudeOcr(client=c)
    img = id_card()
    f = o.extract_fields(img)
    assert f["name"] == "Mary Smith" and f["expiry_date"] == ""
    assert o.check_obstruction(img) == (True, "thumb over name")
    assert c.calls == 1   # cached by image hash


def test_claude_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        ClaudeOcr()


def test_make_ocr_unknown():
    with pytest.raises(ValueError):
        make_ocr("nope")


def test_tesseract_text_parser():
    f = TesseractOcr.parse_text("DRIVER LICENCE\nName: MARY SMITH\nDOB 1990-03-07\nLicence No: D1234567\nExpires 2030-01-01")
    assert f["name"] == "MARY SMITH" and f["date_of_birth"] == "1990-03-07" and f["expiry_date"] == "2030-01-01"
    assert f["id_number"].startswith("D1234567")


# ---- pipeline with a scripted face engine so no models or photos are needed
class ScriptedFaces(FakeFaces):
    """detect() returns a face; embed_* return vectors keyed by the image's mean pixel value."""

    def __init__(self, vec_by_tag):
        super().__init__([FakeFaces.row(300, 180, 200, 240)])
        self.vec_by_tag = vec_by_tag

    def _v(self, img):
        return self.vec_by_tag[int(round(float(img[0, 0, 0])))]

    def embed_face(self, img, face):
        return self._v(img)

    def embed_portrait(self, img, face=None):
        return self._v(img)

    @staticmethod
    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def tag(img, t):
    img = img.copy()
    img[0, 0] = (t, t, t)     # tag pixel identifies the "person" for the scripted embedder
    return img


def vec(angle):
    return np.array([np.cos(angle), np.sin(angle)], np.float32)


FIELDS = {"name": "MARY SMITH", "date_of_birth": "1990-03-07", "id_number": "D123", "expiry_date": "2030-01-01"}


def scene(person_ids=(10, 10), person_cams=(10, 10, 10), fields=(FIELDS, FIELDS), angles=None):
    angles = angles or {10: 0.0, 20: 1.5, 11: 0.2}
    vecs = {k: vec(v) for k, v in angles.items()}
    faces = ScriptedFaces(vecs)
    ids = [tag(id_card(), t) for t in person_ids]
    cams = [tag(webcam(), t) for t in person_cams]
    table = {int(i[0, 0, 0]): f for i, f in zip(ids, fields)}
    # the two IDs differ in a card pixel so lookup by object identity is used instead
    fx = FixtureOcr(lambda im: next(f for i, f in zip(ids, fields) if i is im or np.array_equal(i, im)))
    return ids, cams, faces, fx


def test_same_person_passes():
    ids, cams, faces, ocr = scene()
    r = run_pipeline(ids, cams, faces, ocr)
    assert r.verdict.verdict == PASS, [(c.pair, c.result, c.score) for c in r.comparisons]
    assert len(r.comparisons) == 3 + 4 + 3      # 3 webcam pairs, portrait + 3 text, 3 webcam-vs-ID
    assert "higher image quality" in r.reference_sentence


def test_different_person_fails():
    ids, cams, faces, ocr = scene(person_cams=(20, 20, 20))
    r = run_pipeline(ids, cams, faces, ocr)
    assert r.verdict.verdict == FAIL
    assert r.special_case and any("match each other but not the ID" in s for s in r.verdict.special_flags)


def test_mismatched_ids_fail_and_explain_reference():
    other = {"name": "JOHN DOE", "date_of_birth": "1985-01-01", "id_number": "X9", "expiry_date": ""}
    ids, cams, faces, ocr = scene(person_ids=(10, 20), fields=(FIELDS, other))
    r = run_pipeline(ids, cams, faces, ocr)
    assert r.verdict.verdict == FAIL
    assert "disagree" in r.reference_sentence


def test_borderline_goes_to_human_review():
    # ID-vs-webcam cosine ~0.3: inside the wide uncertain band
    ids, cams, faces, ocr = scene(person_cams=(11, 11, 11), angles={10: 0.0, 11: np.arccos(0.25), 20: 3})
    r = run_pipeline(ids, cams, faces, ocr)
    assert r.verdict.verdict == REVIEW
    assert not any(c.result == MISMATCH for c in r.comparisons)


def test_bad_image_is_skipped_and_asks_for_retake():
    ids, cams, faces, ocr = scene()
    import cv2
    cams[1] = tag(cv2.GaussianBlur(cams[1], (0, 0), 12), 20)   # blurry AND would be a different person
    r = run_pipeline(ids, cams, faces, ocr)
    assert r.verdict.verdict == RETAKE
    assert any("Webcam B" in p and "Hold still" in p for p in r.verdict.retake_prompts)
    assert not any("Webcam B" in c.pair for c in r.comparisons)       # skipped everywhere
    assert not any(c.result == MISMATCH for c in r.comparisons)


def test_missing_image_is_a_retake():
    ids, cams, faces, ocr = scene()
    r = run_pipeline(ids, [cams[0], None, cams[2]], faces, ocr)
    assert r.verdict.verdict == RETAKE


def test_ocr_failure_becomes_uncertain_not_mismatch():
    ids, cams, faces, _ = scene()

    class Broken(FixtureOcr):
        def extract_fields(self, img):
            raise RuntimeError("api down")
    r = run_pipeline(ids, cams, faces, Broken(lambda im: {}))
    assert r.verdict.verdict == REVIEW
    assert r.notes


def test_explanation_template_and_llm_fallback():
    ids, cams, faces, ocr = scene()
    r = run_pipeline(ids, cams, faces, ocr)
    assert "PASS" in template_explanation(r)

    class Boom(FixtureOcr):
        def explain(self, s):
            raise RuntimeError

    import config
    old = config.USE_LLM_EXPLANATION
    config.USE_LLM_EXPLANATION = True
    try:
        assert "PASS" in plain_explanation(r, Boom(lambda im: {}))
    finally:
        config.USE_LLM_EXPLANATION = old


def test_degrade_variants_trigger_the_matching_retake():
    """scripts/degrade.py output must be caught by the matching check."""
    from scripts import degrade
    from quality import check_image
    face = FakeFaces([FakeFaces.row(130, 150, 120, 150)])
    card = id_card()
    cases = {"blur": degrade.blur(card), "dark": degrade.darken(card), "glare": degrade.glare(card),
             "edge_crop": degrade.edge_crop(card, 0.18)}
    expect = {"blur": "blur_id", "dark": "dark", "glare": "glare", "edge_crop": "crop"}
    for name, img in cases.items():
        got = {m.failure_code for m in check_image("ID", "id", img, face).failures}
        assert expect[name] in got, (name, got)
    cam = webcam()
    assert "backlight" in {m.failure_code for m in check_image(
        "A", "webcam", degrade.backlight(cam, (300, 180, 200, 240)), FakeFaces()).failures}
    assert "partial_face" in {m.failure_code for m in check_image(
        "A", "webcam", cam, FakeFaces([FakeFaces.row(-90, 180, 200, 240)])).failures}
