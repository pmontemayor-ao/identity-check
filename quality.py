"""Step 1: the quality gate. Quality problems never become mismatches: they become retakes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from config import QUALITY, QualityThresholds

# Failure code -> retake prompt. (ID vs webcam wording differs for blur.)
RETAKE_PROMPTS = {
    "blur_id": "Hold the ID steady and try again",
    "blur_face": "Hold still and look at the camera",
    "thumb": "Move your thumb off the ID details",
    "glare": "Tilt the ID away from the light to remove glare",
    "crop": "Fit all four corners of the ID in the frame",
    "dark": "Turn on a light in front of you and try again",
    "bright": "Move away from the bright light or lower the brightness, then try again",
    "backlight": "Move so the light is in front of you, not behind you",
    "partial_face": "Make sure your whole face is visible to the camera",
    "no_face_id": "We could not find the photo on the ID. Show the side with your portrait and try again",
    "no_face": "Make sure your whole face is visible to the camera and try again",
    "multi_face": "Make sure only one person is in the frame",
    "small_face": "Move closer to the camera",
    "small_face_id": "Move the ID closer so the portrait is larger",
}


@dataclass
class MetricResult:
    name: str
    value: Optional[float]           # measured value (None if not measurable / skipped)
    threshold: str                   # human-readable rule, e.g. ">= 60"
    passed: Optional[bool]           # None = skipped (could not be assessed)
    detail: str = ""
    failure_code: Optional[str] = None


@dataclass
class QualityResult:
    image_id: str
    kind: str                        # "id" or "webcam"
    metrics: List[MetricResult] = field(default_factory=list)
    score: float = 0.0               # 0-1 composite, used only to pick the better ID portrait
    face: Optional[np.ndarray] = field(default=None, repr=False)  # YuNet row (15,) or None
    error: Optional[str] = None      # image could not be decoded

    @property
    def failures(self) -> List[MetricResult]:
        return [m for m in self.metrics if m.passed is False]

    @property
    def retake_prompts(self) -> List[str]:
        out = []
        if self.error == "missing":
            out.append("No image was provided. Upload or capture one")
        elif self.error:
            out.append("We could not read that file. Upload a JPG or PNG image")
        for m in self.failures:
            p = RETAKE_PROMPTS.get(m.failure_code or "")
            if p and p not in out:
                out.append(p)
        return out

    @property
    def needs_retake(self) -> bool:
        return bool(self.error) or bool(self.failures)

    @property
    def status(self) -> str:
        """green = pass, amber = pass but a check was skipped, red = retake."""
        if self.needs_retake:
            return "red"
        if any(m.passed is None for m in self.metrics):
            return "amber"
        return "green"


# ---------------------------------------------------------------- pure metrics
def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def laplacian_variance(img: np.ndarray, width: int = 640) -> float:
    g = to_gray(img)
    if g.shape[1] != width:
        scale = width / g.shape[1]
        g = cv2.resize(g, (width, max(1, int(g.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    # Stretch contrast (1st to 99th percentile, gain capped at 4x) so a dim or low-contrast image is not called blurry
    # just because its pixel differences are small. Darkness has its own check.
    lo, hi = np.percentile(g, [1, 99])
    if hi - lo > 8:
        g = np.clip((g.astype(np.float32) - lo) * min(4.0, 255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def mean_luminance(img: np.ndarray) -> float:
    return float(to_gray(img).mean())


def find_document_quad(img: np.ndarray, min_area_fraction: float = 0.15) -> Optional[np.ndarray]:
    """Largest 4-corner contour that covers a plausible part of the frame, or None.

    Returns a (4,2) float array ordered tl, tr, br, bl. The image is padded by a few
    pixels first so a document that touches the border still closes into a contour.
    """
    h, w = img.shape[:2]
    scale = 640.0 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img.copy()
    sh, sw = small.shape[:2]
    pad = 6
    padded = cv2.copyMakeBorder(to_gray(small), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=int(np.median(to_gray(small))))
    blur = cv2.GaussianBlur(padded, (5, 5), 0)
    cands = []
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    cands.append(edges)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cands.append(otsu)
    cands.append(255 - otsu)
    best, best_area = None, 0.0
    for m in cands:
        cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            # skip tiny shapes, and shapes that are the whole frame (that is the
            # background, not a document; a document that fills the frame is a crop failure anyway)
            if area < min_area_fraction * sh * sw or area > 0.9 * sh * sw or area <= best_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                best, best_area = approx.reshape(4, 2).astype(np.float32), area
    if best is None:
        return None
    best = (best - pad) / scale
    # order tl, tr, br, bl
    s = best.sum(axis=1)
    d = np.diff(best, axis=1).ravel()
    return np.array([best[np.argmin(s)], best[np.argmin(d)], best[np.argmax(s)], best[np.argmax(d)]], dtype=np.float32)


def quad_inside_frame(quad: np.ndarray, shape: Tuple[int, int], margin_fraction: float) -> bool:
    h, w = shape[:2]
    mx, my = margin_fraction * w, margin_fraction * h
    return bool(np.all(quad[:, 0] >= mx) and np.all(quad[:, 0] <= w - 1 - mx)
                and np.all(quad[:, 1] >= my) and np.all(quad[:, 1] <= h - 1 - my))


def glare_fraction(img: np.ndarray, quad: Optional[np.ndarray], t: QualityThresholds = QUALITY) -> float:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hot = (hsv[..., 2] >= t.glare_value_min) & (hsv[..., 1] <= t.glare_saturation_max)
    if quad is not None:
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, quad.astype(np.int32), 1)
        area = int(mask.sum())
        if area > 0:
            return float((hot & (mask > 0)).sum() / area)
    return float(hot.mean())


def backlight_ratio(img: np.ndarray, face_box: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
    """Returns (face_mean / background_mean, face_mean, background_mean)."""
    g = to_gray(img).astype(np.float32)
    h, w = g.shape
    x, y, bw, bh = face_box
    x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
    mask = np.zeros((h, w), bool)
    mask[y0:y1, x0:x1] = True
    if mask.sum() == 0 or (~mask).sum() == 0:
        return 1.0, 0.0, 0.0
    f, b = float(g[mask].mean()), float(g[~mask].mean())
    return (f / b if b > 1 else 1.0), f, b


# ---------------------------------------------------------------- the gate
ObstructionFn = Callable[[np.ndarray], Optional[Tuple[bool, str]]]


def _fmt(v: float) -> str:
    return f"{v:.2f}"


def _exposure_score(mean: float) -> float:
    return float(max(0.0, 1.0 - abs(mean - 128.0) / 128.0))


def check_image(image_id: str, kind: str, img: Optional[np.ndarray], faces_engine,
                obstruction_fn: Optional[ObstructionFn] = None,
                t: QualityThresholds = QUALITY) -> QualityResult:
    """Run every applicable check. All metrics are recorded even after a failure."""
    qr = QualityResult(image_id=image_id, kind=kind)
    if img is None or img.size == 0:
        qr.error = "unreadable image"
        return qr
    h, w = img.shape[:2]
    is_id = kind == "id"

    # --- blur
    lap = laplacian_variance(img, t.analysis_width)
    ok = lap >= t.blur_min
    qr.metrics.append(MetricResult("Blur (Laplacian variance)", lap, f">= {t.blur_min:g}", ok,
                                   "" if ok else "Image looks blurry",
                                   None if ok else ("blur_id" if is_id else "blur_face")))
    sharp = min(1.0, lap / (3 * t.blur_min))

    # --- brightness
    mean = mean_luminance(img)
    if mean < t.dark_max:
        qr.metrics.append(MetricResult("Brightness (mean luminance)", mean, f"{t.dark_max:g} to {t.bright_min:g}", False, "Too dark", "dark"))
    elif mean > t.bright_min:
        qr.metrics.append(MetricResult("Brightness (mean luminance)", mean, f"{t.dark_max:g} to {t.bright_min:g}", False, "Too bright", "bright"))
    else:
        qr.metrics.append(MetricResult("Brightness (mean luminance)", mean, f"{t.dark_max:g} to {t.bright_min:g}", True))
    expo = _exposure_score(mean)

    # --- faces (detected on every image; on IDs this finds the portrait)
    faces = faces_engine.detect(img)
    face = max(faces, key=lambda f: f[2] * f[3]) if len(faces) else None
    qr.face = face

    glare_score = 1.0
    if is_id:
        quad = find_document_quad(img, t.doc_min_area_fraction)
        # --- cropping
        if quad is None:
            qr.metrics.append(MetricResult("Cropping (document corners)", None, "4 corners inside frame", False,
                                           "Could not find four document corners", "crop"))
        else:
            inside = quad_inside_frame(quad, img.shape, t.crop_margin_fraction)
            qr.metrics.append(MetricResult("Cropping (document corners)", 4.0 if inside else None,
                                           f"4 corners, margin >= {t.crop_margin_fraction:.0%}", inside,
                                           "" if inside else "A document corner is at or beyond the frame edge",
                                           None if inside else "crop"))
        # --- glare
        gf = glare_fraction(img, quad, t)
        ok = gf <= t.glare_max_fraction
        qr.metrics.append(MetricResult("Glare (near-white pixels in document)", gf, f"<= {t.glare_max_fraction:.1%}", ok,
                                       "" if ok else "Bright reflection over the document", None if ok else "glare"))
        glare_score = max(0.0, 1.0 - gf / (2 * t.glare_max_fraction))
        # --- obstruction (vision model)
        if obstruction_fn is None:
            qr.metrics.append(MetricResult("Obstruction (finger/thumb)", None, "no finger over text or portrait", None,
                                           "Skipped: no vision model available"))
        else:
            try:
                res = obstruction_fn(img)
            except Exception as e:  # network/API failure must not become a mismatch
                res = None
                err = f"Skipped: {type(e).__name__}"
            else:
                err = "Skipped: vision model gave no answer"
            if res is None:
                qr.metrics.append(MetricResult("Obstruction (finger/thumb)", None, "no finger over text or portrait", None, err))
            else:
                covered, why = res
                qr.metrics.append(MetricResult("Obstruction (finger/thumb)", 1.0 if covered else 0.0,
                                               "no finger over text or portrait", not covered,
                                               why, "thumb" if covered else None))

    # --- face checks
    min_px = t.face_min_width_px_id if is_id else t.face_min_width_px_webcam
    no_face_code = "no_face_id" if is_id else "no_face"
    if face is None:
        qr.metrics.append(MetricResult("Face present", 0.0, ">= 1 face", False, "No face detected", no_face_code))
        face_score = 0.0
    else:
        qr.metrics.append(MetricResult("Face present", float(len(faces)), ">= 1 face", True))
        if not is_id:
            ok = len(faces) == 1
            qr.metrics.append(MetricResult("Single face", float(len(faces)), "exactly 1", ok,
                                           "" if ok else f"{len(faces)} faces detected", None if ok else "multi_face"))
        fw = float(face[2])
        need = min_px if is_id else max(min_px, t.face_min_width_fraction_webcam * w)
        ok = fw >= need
        qr.metrics.append(MetricResult("Face size (width px)", fw, f">= {need:.0f}", ok,
                                       "" if ok else "Face is too small", None if ok else ("small_face_id" if is_id else "small_face")))
        # landmarks / partly visible
        lm = face[4:14].reshape(5, 2)
        mx, my = t.landmark_margin_fraction * w, t.landmark_margin_fraction * h
        lm_in = bool(np.all(lm[:, 0] >= mx) and np.all(lm[:, 0] <= w - mx) and np.all(lm[:, 1] >= my) and np.all(lm[:, 1] <= h - my))
        x, y, bw, bh = [float(v) for v in face[:4]]
        ix = max(0.0, min(w, x + bw) - max(0.0, x))
        iy = max(0.0, min(h, y + bh) - max(0.0, y))
        inside_frac = (ix * iy) / max(1.0, bw * bh)
        ok = lm_in and inside_frac >= t.face_box_inside_min
        qr.metrics.append(MetricResult("Landmarks visible / face inside frame", inside_frac,
                                       f"5 landmarks in frame, box inside >= {t.face_box_inside_min:.0%}", ok,
                                       "" if ok else "Face is cut off by the frame edge", None if ok else "partial_face"))
        # backlighting (webcams only: on an ID the portrait is a print)
        if not is_id:
            ratio, fm, bm = backlight_ratio(img, (int(x), int(y), int(bw), int(bh)))
            bad = ratio < t.backlight_ratio_max and bm >= t.backlight_bg_min
            qr.metrics.append(MetricResult("Backlighting (face / background luminance)", ratio,
                                           f">= {t.backlight_ratio_max:g} (when background >= {t.backlight_bg_min:g})", not bad,
                                           f"face {fm:.0f}, background {bm:.0f}", "backlight" if bad else None))
        face_score = min(1.0, fw / (min_px * 3)) * float(face[14])

    qr.score = float(0.45 * sharp + 0.15 * expo + 0.15 * glare_score + 0.25 * face_score)
    return qr
