"""All tunable settings. Everything here is a starting point, not a calibrated value.

Tune on your own labelled cases with scripts/evaluate.py before trusting any threshold.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# ---- OCR / privacy -------------------------------------------------------
# "claude" sends ID images to the Anthropic API. "tesseract" keeps everything local
# (no obstruction check in that mode). Override with env var IDCHECK_OCR.
OCR_BACKEND: str = os.environ.get("IDCHECK_OCR", "claude").lower()
CLAUDE_MODEL: str = os.environ.get("IDCHECK_CLAUDE_MODEL", "claude-sonnet-4-5")
# Plain-language verdict explanation also uses Claude when the backend is "claude".
USE_LLM_EXPLANATION: bool = OCR_BACKEND == "claude"

# ---- Models (downloaded on first run) ---------------------------------------
MODEL_DIR: str = os.environ.get("IDCHECK_MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"
DETECT_SCORE_THRESHOLD: float = 0.6   # YuNet confidence to count as a face
DETECT_MAX_SIDE: int = 960            # images are downscaled to this before detection


@dataclass
class QualityThresholds:
    # Blur: variance of the Laplacian on a copy resized to ANALYSIS_WIDTH. Lower = blurrier.
    blur_min: float = 60.0
    analysis_width: int = 640
    # Brightness: mean grey level 0-255.
    dark_max: float = 70.0     # mean below this = too dark
    bright_min: float = 215.0  # mean above this = too bright
    # Backlighting: face is much darker than the background.
    backlight_ratio_max: float = 0.55   # face_mean / background_mean below this ...
    backlight_bg_min: float = 120.0     # ... and the background is at least this bright
    # Glare (IDs): fraction of document pixels that are near-white and unsaturated.
    glare_max_fraction: float = 0.04
    glare_value_min: int = 245
    glare_saturation_max: int = 30
    # Cropping (IDs): document quad must be at least this much of the frame and every
    # corner must sit at least this fraction of the image size away from the border.
    doc_min_area_fraction: float = 0.15
    crop_margin_fraction: float = 0.01
    # Faces
    face_min_width_px_webcam: int = 90
    face_min_width_px_id: int = 50
    face_min_width_fraction_webcam: float = 0.10   # face width / image width
    # Landmarks visible: all 5 landmarks inside the frame by this margin, and at least
    # this fraction of the face box inside the frame.
    landmark_margin_fraction: float = 0.01
    face_box_inside_min: float = 0.92


@dataclass
class MatchThresholds:
    """(match_at_or_above, mismatch_below). Between the two = UNCERTAIN."""
    # SFace's recommended cosine threshold is 0.363 (about 0.1% false accept on LFW).
    # Webcam-vs-webcam: same session, same camera, so the bands are tighter.
    webcam_match: float = 0.40
    webcam_mismatch: float = 0.30
    # ID-vs-ID: two printed portraits, usually of different ages and print quality.
    id_match: float = 0.363
    id_mismatch: float = 0.25
    # Webcam-vs-ID: hardest pairing (print texture, age gap, lighting), so the uncertain
    # band is the widest. Scores in the band go to a human.
    id_webcam_match: float = 0.363
    id_webcam_mismatch: float = 0.15
    # Text: rapidfuzz token_sort_ratio on the name, 0-100.
    name_match: float = 90.0
    name_mismatch: float = 70.0


QUALITY = QualityThresholds()
MATCH = MatchThresholds()
