import cv2
import numpy as np

from quality import (check_image, find_document_quad, glare_fraction, laplacian_variance,
                     mean_luminance, quad_inside_frame, backlight_ratio)
from tests.synth import FakeFaces, H, W, background, id_card, webcam


def codes(qr):
    return {m.failure_code for m in qr.failures}


def test_blur_metric_orders_sharp_above_blurred():
    sharp = webcam()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 8)
    assert laplacian_variance(sharp) > 5 * laplacian_variance(blurred)


def test_luminance():
    assert mean_luminance(np.zeros((10, 10, 3), np.uint8)) == 0
    assert mean_luminance(np.full((10, 10, 3), 255, np.uint8)) == 255


def test_clean_webcam_passes():
    qr = check_image("A", "webcam", webcam(), FakeFaces())
    assert not qr.needs_retake, [(m.name, m.value) for m in qr.failures]
    assert qr.status == "green"


def test_blurred_webcam_asks_for_hold_still():
    qr = check_image("A", "webcam", cv2.GaussianBlur(webcam(), (0, 0), 10), FakeFaces())
    assert "blur_face" in codes(qr)
    assert "Hold still and look at the camera" in qr.retake_prompts


def test_dark_and_bright_webcam():
    assert "dark" in codes(check_image("A", "webcam", (webcam() * 0.25).astype(np.uint8), FakeFaces()))
    bright = np.clip(webcam().astype(np.int16) + 140, 0, 255).astype(np.uint8)
    assert "bright" in codes(check_image("A", "webcam", bright, FakeFaces()))


def test_backlight_detected():
    img = background(200)
    x, y, w, h = 300, 180, 200, 240
    img[y:y + h, x:x + w] = 40
    for i in range(0, W, 16):
        cv2.line(img, (i, 0), (i, H), (230, 230, 230), 1)
    ratio, f, b = backlight_ratio(img, (x, y, w, h))
    assert ratio < 0.3
    qr = check_image("A", "webcam", img, FakeFaces())
    assert "backlight" in codes(qr)
    assert "Move so the light is in front of you, not behind you" in qr.retake_prompts


def test_no_face_multi_face_small_and_partial():
    assert "no_face" in codes(check_image("A", "webcam", webcam(), FakeFaces([])))
    two = FakeFaces([FakeFaces.row(100, 180, 200, 240), FakeFaces.row(450, 180, 200, 240)])
    assert "multi_face" in codes(check_image("A", "webcam", webcam(), two))
    assert "small_face" in codes(check_image("A", "webcam", webcam(), FakeFaces([FakeFaces.row(300, 250, 40, 50)])))
    cut = FakeFaces([FakeFaces.row(-90, 180, 200, 240)])   # left third of the face off-frame
    qr = check_image("A", "webcam", webcam(), cut)
    assert "partial_face" in codes(qr)
    assert "Make sure your whole face is visible to the camera" in qr.retake_prompts


def test_document_quad_and_crop():
    img = id_card()
    quad = find_document_quad(img)
    assert quad is not None and quad_inside_frame(quad, img.shape, 0.01)
    # card running off the left edge: corners no longer inside the frame
    cropped = id_card(margin=100)[:, 150:]
    qr = check_image("ID1", "id", cropped, FakeFaces([FakeFaces.row(150, 150, 120, 150)]))
    assert "crop" in codes(qr)
    assert "Fit all four corners of the ID in the frame" in qr.retake_prompts


def test_clean_id_passes_without_vision_model_but_is_amber():
    qr = check_image("ID1", "id", id_card(), FakeFaces([FakeFaces.row(130, 150, 120, 150)]))
    assert not qr.needs_retake, [(m.name, m.value, m.detail) for m in qr.failures]
    assert qr.status == "amber"   # obstruction check skipped


def test_glare_blob():
    img = id_card()
    cv2.circle(img, (450, 300), 150, (255, 255, 255), -1)
    quad = find_document_quad(img)
    assert glare_fraction(img, quad) > 0.04
    qr = check_image("ID1", "id", img, FakeFaces([FakeFaces.row(130, 150, 120, 150)]))
    assert "glare" in codes(qr)
    assert "Tilt the ID away from the light to remove glare" in qr.retake_prompts


def test_obstruction_uses_vision_model_and_failure_is_skipped_not_failed():
    face = FakeFaces([FakeFaces.row(130, 150, 120, 150)])
    qr = check_image("ID1", "id", id_card(), face, obstruction_fn=lambda im: (True, "thumb over name"))
    assert "thumb" in codes(qr) and "Move your thumb off the ID details" in qr.retake_prompts

    def boom(im):
        raise RuntimeError("api down")
    qr = check_image("ID1", "id", id_card(), face, obstruction_fn=boom)
    assert not qr.needs_retake and qr.status == "amber"


def test_unreadable_image_is_retake_not_exception():
    qr = check_image("A", "webcam", None, FakeFaces())
    assert qr.needs_retake and qr.retake_prompts
