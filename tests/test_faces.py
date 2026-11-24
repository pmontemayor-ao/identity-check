"""Integration tests with the real ONNX models. Skipped when the models cannot be downloaded."""
import os

import cv2
import pytest

DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="module")
def fe():
    try:
        from faces import FaceEngine
        return FaceEngine()
    except Exception as e:  # offline
        pytest.skip(f"models unavailable: {e}")


def load(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        pytest.skip("sample image missing")
    return cv2.imread(p)


def test_detects_one_face(fe):
    assert len(fe.detect(load("lena.jpg"))) == 1


def test_same_person_scores_high_different_person_low(fe):
    lena, messi = load("lena.jpg"), load("messi5.jpg")
    same = cv2.GaussianBlur(cv2.resize(lena, None, fx=0.7, fy=0.7), (0, 0), 1.2)
    same = cv2.convertScaleAbs(same, alpha=0.9, beta=-10)
    e1, e2, e3 = fe.embed_largest(lena), fe.embed_largest(same), fe.embed_largest(messi)
    assert fe.cosine(e1, e2) > 0.6
    assert fe.cosine(e1, e3) < 0.3


def test_portrait_crop_then_embed(fe):
    lena = load("lena.jpg")
    emb_full = fe.embed_largest(lena)
    emb_crop = fe.embed_portrait(lena)
    assert fe.cosine(emb_full, emb_crop) > 0.8


def test_real_quality_gate_on_real_face(fe):
    from quality import check_image
    qr = check_image("A", "webcam", load("lena.jpg"), fe)
    assert qr.face is not None
    qr_dark = check_image("A", "webcam", (load("lena.jpg") * 0.2).astype("uint8"), fe)
    assert "dark" in {m.failure_code for m in qr_dark.failures}
