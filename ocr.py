"""Step 3: ID text extraction and the finger/thumb check. Swappable behind OcrEngine."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import config

FIELDS = ("name", "date_of_birth", "id_number", "expiry_date")


class OcrEngine(ABC):
    """Interface. Implement these three methods to plug in another OCR/vision backend."""

    name = "base"

    @abstractmethod
    def extract_fields(self, img: np.ndarray) -> Dict[str, str]:
        """Return {name, date_of_birth, id_number, expiry_date}; '' for anything unreadable."""

    def check_obstruction(self, img: np.ndarray) -> Optional[Tuple[bool, str]]:
        """(covered, explanation) if a finger/thumb covers text or the portrait; None if unsupported."""
        return None

    def explain(self, structured: dict) -> Optional[str]:
        """Optional plain-language summary of structured results. None = use the template."""
        return None


def _empty() -> Dict[str, str]:
    return {k: "" for k in FIELDS}


def parse_json_object(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates code fences and chatter)."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


def _jpeg_b64(img: np.ndarray, max_side: int = 1600) -> Tuple[str, str]:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("could not encode image")
    raw = buf.tobytes()
    return base64.standard_b64encode(raw).decode(), hashlib.sha1(raw).hexdigest()


_ID_PROMPT = """You are reading a photo of an identity document. Reply with ONLY a JSON object, no other text:
{
  "name": "full name exactly as printed, or ''",
  "date_of_birth": "YYYY-MM-DD, or '' if not readable",
  "id_number": "document/licence/ID number as printed, or ''",
  "expiry_date": "YYYY-MM-DD, or '' if not readable",
  "obstruction": {"covered": true or false, "what": "short note"}
}
Rules: do not guess. If a field is hidden, blurred or cut off, return ''. "obstruction.covered" is true only if a
finger, thumb or hand covers any text field or the portrait photo."""

_EXPLAIN_SYSTEM = ("You write a short plain-language summary (3 to 5 sentences) of an identity check for a "
                   "supervisor. Use ONLY the structured results given. Do not add judgments, guesses or advice "
                   "beyond what the results state. Keep the verdict word exactly as given. Never say a poor-quality "
                   "image is a mismatch.")


class ClaudeOcr(OcrEngine):
    """Claude vision. One API call per image returns the text fields and the obstruction answer,
    cached by image hash so the quality gate and the text step do not pay twice."""

    name = "claude"

    def __init__(self, client=None, model: str = config.CLAUDE_MODEL):
        if client is None:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            import anthropic
            client = anthropic.Anthropic(api_key=key)
        self.client, self.model = client, model
        self._cache: Dict[str, dict] = {}

    def _analyse(self, img: np.ndarray) -> dict:
        b64, digest = _jpeg_b64(img)
        if digest in self._cache:
            return self._cache[digest]
        msg = self.client.messages.create(
            model=self.model, max_tokens=500,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": _ID_PROMPT}]}])
        text = "".join(getattr(b, "text", "") for b in msg.content)
        data = parse_json_object(text)
        self._cache[digest] = data
        return data

    def extract_fields(self, img: np.ndarray) -> Dict[str, str]:
        data = self._analyse(img)
        out = _empty()
        for k in FIELDS:
            v = data.get(k)
            out[k] = str(v).strip() if v else ""
        return out

    def check_obstruction(self, img: np.ndarray) -> Optional[Tuple[bool, str]]:
        ob = self._analyse(img).get("obstruction")
        if not isinstance(ob, dict) or "covered" not in ob:
            return None
        return bool(ob["covered"]), str(ob.get("what", ""))

    def explain(self, structured: dict) -> Optional[str]:
        msg = self.client.messages.create(
            model=self.model, max_tokens=400, system=_EXPLAIN_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(structured)}])
        return "".join(getattr(b, "text", "") for b in msg.content).strip() or None


class TesseractOcr(OcrEngine):
    """Local fallback. Heuristic regexes over raw OCR text: much weaker than Claude, and it
    cannot see fingers, so the obstruction check is skipped. Nothing leaves the machine."""

    name = "tesseract"

    def extract_fields(self, img: np.ndarray) -> Dict[str, str]:
        import pytesseract
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(g)
        return self.parse_text(text)

    @staticmethod
    def parse_text(text: str) -> Dict[str, str]:
        out = _empty()
        m = re.search(r"(?:name|nom)\s*[:\-]?\s*([A-Za-z][A-Za-z ,.'\-]{2,})", text, re.I)
        if m:
            out["name"] = m.group(1).strip()
        dates = re.findall(r"\b(\d{4}-\d{2}-\d{2}|\d{2}[/.\-]\d{2}[/.\-]\d{4})\b", text)
        m = re.search(r"(?:dob|birth|born)[^\d]{0,15}(\d{4}-\d{2}-\d{2}|\d{2}[/.\-]\d{2}[/.\-]\d{4})", text, re.I)
        if m:
            out["date_of_birth"] = m.group(1)
        elif dates:
            out["date_of_birth"] = dates[0]
        m = re.search(r"(?:exp|expiry|expires)[^\d]{0,15}(\d{4}-\d{2}-\d{2}|\d{2}[/.\-]\d{2}[/.\-]\d{4})", text, re.I)
        if m:
            out["expiry_date"] = m.group(1)
        m = re.search(r"(?:licen[cs]e|id|document|passport)\s*(?:no|number|#)?\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\- ]{5,})", text, re.I)
        if m:
            out["id_number"] = m.group(1).strip()
        return out


class FixtureOcr(OcrEngine):
    """Offline engine for tests and evaluate.py: reads '<image path>.json' sidecars through a callable
    that maps an image to its fields. Used so the harness runs without an API key."""

    name = "fixture"

    def __init__(self, lookup, obstruction=None):
        self._lookup, self._obstruction = lookup, obstruction

    def extract_fields(self, img):
        out = _empty()
        out.update(self._lookup(img) or {})
        return out

    def check_obstruction(self, img):
        return self._obstruction(img) if self._obstruction else None


def make_ocr(backend: str = config.OCR_BACKEND) -> OcrEngine:
    if backend == "tesseract":
        return TesseractOcr()
    if backend == "claude":
        return ClaudeOcr()
    raise ValueError(f"unknown OCR backend: {backend}")
