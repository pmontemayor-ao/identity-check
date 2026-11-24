"""Orchestration in the required order: quality gate, webcam pairs, ID vs ID, webcam vs ID, verdict."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np

import config
from compare import MATCH, MISMATCH, UNCERTAIN, ComparisonResult, compare_text_fields, face_comparison
from decide import Verdict, choose_reference, overall_verdict, webcam_id_special_case
from quality import QualityResult, check_image


@dataclass
class PipelineResult:
    qualities: List[QualityResult]
    comparisons: List[ComparisonResult]
    verdict: Verdict
    reference_sentence: str = ""
    reference_index: Optional[int] = None
    id_fields: Dict[str, Dict[str, str]] = field(default_factory=dict)
    special_case: bool = False
    notes: List[str] = field(default_factory=list)


def _missing(image_id: str, kind: str) -> QualityResult:
    return QualityResult(image_id=image_id, kind=kind, error="missing")


def run_pipeline(id_imgs: Sequence[Optional[np.ndarray]], webcam_imgs: Sequence[Optional[np.ndarray]],
                 faces, ocr, match: config.MatchThresholds = config.MATCH,
                 quality: config.QualityThresholds = config.QUALITY) -> PipelineResult:
    notes: List[str] = []
    id_names = ["ID 1", "ID 2"]
    cam_names = ["Webcam A", "Webcam B", "Webcam C"]

    # ---- 1. quality gate (every image, every metric recorded)
    obstruction_fn = ocr.check_obstruction if ocr is not None else None
    qs: List[QualityResult] = []
    for n, im in zip(id_names, id_imgs):
        qs.append(check_image(n, "id", im, faces, obstruction_fn, quality) if im is not None else _missing(n, "id"))
    for n, im in zip(cam_names, webcam_imgs):
        qs.append(check_image(n, "webcam", im, faces, None, quality) if im is not None else _missing(n, "webcam"))
    q_ids, q_cams = qs[:2], qs[2:]
    ok_cams = [i for i, q in enumerate(q_cams) if not q.needs_retake]
    ok_ids = [i for i, q in enumerate(q_ids) if not q.needs_retake]

    comps: List[ComparisonResult] = []
    emb_cache: Dict[str, Optional[np.ndarray]] = {}

    def cam_emb(i):
        key = f"cam{i}"
        if key not in emb_cache:
            emb_cache[key] = faces.embed_face(webcam_imgs[i], q_cams[i].face)
        return emb_cache[key]

    def id_emb(i):
        key = f"id{i}"
        if key not in emb_cache:
            emb_cache[key] = faces.embed_portrait(id_imgs[i], q_ids[i].face)
        return emb_cache[key]

    # ---- 2. webcam vs webcam
    cam_pairs: List[ComparisonResult] = []
    for i, j in combinations(ok_cams, 2):
        pair = f"{cam_names[i]} vs {cam_names[j]}"
        sc = faces.cosine(cam_emb(i), cam_emb(j))
        cam_pairs.append(face_comparison(pair, "webcam_webcam", sc, match.webcam_match, match.webcam_mismatch))
    comps += cam_pairs

    # ---- 3. ID vs ID (face + text)
    fields: Dict[str, Dict[str, str]] = {}
    for i in ok_ids:
        try:
            fields[id_names[i]] = ocr.extract_fields(id_imgs[i]) if ocr is not None else {}
        except Exception as e:
            fields[id_names[i]] = {}
            notes.append(f"Text extraction failed for {id_names[i]} ({type(e).__name__}).")
    id_comps: List[ComparisonResult] = []
    if len(ok_ids) == 2:
        e0, e1 = id_emb(0), id_emb(1)
        if e0 is None or e1 is None:
            id_comps.append(ComparisonResult("ID 1 vs ID 2: portrait", "id_id", "face", None, UNCERTAIN, "Portrait could not be embedded"))
        else:
            id_comps.append(face_comparison("ID 1 vs ID 2: portrait", "id_id", faces.cosine(e0, e1), match.id_match, match.id_mismatch))
        id_comps += compare_text_fields(fields.get("ID 1", {}), fields.get("ID 2", {}), match.name_match, match.name_mismatch)
    comps += id_comps
    ids_agree: Optional[bool] = None
    if id_comps:
        ids_agree = not any(c.result == MISMATCH for c in id_comps)

    ref_idx, ref_sentence = choose_reference(q_ids[0], q_ids[1], ids_agree)

    # ---- 4. webcam vs ID reference
    cam_id: List[ComparisonResult] = []
    if ref_idx is not None:
        r = id_emb(ref_idx)
        for i in ok_cams:
            pair = f"{cam_names[i]} vs {id_names[ref_idx]} portrait"
            if r is None:
                cam_id.append(ComparisonResult(pair, "id_webcam", "face", None, UNCERTAIN, "Portrait could not be embedded"))
            else:
                cam_id.append(face_comparison(pair, "id_webcam", faces.cosine(cam_emb(i), r),
                                              match.id_webcam_match, match.id_webcam_mismatch))
    comps += cam_id

    special = webcam_id_special_case(cam_pairs, cam_id)
    verdict = overall_verdict(qs, comps, special)
    return PipelineResult(qs, comps, verdict, ref_sentence, ref_idx, fields, special, notes)


# ---------------------------------------------------------------- explanation
def structured_summary(res: PipelineResult) -> dict:
    return {
        "verdict": res.verdict.verdict,
        "reasons": res.verdict.reasons,
        "special_flags": res.verdict.special_flags,
        "retakes": res.verdict.retake_prompts,
        "reference_portrait": res.reference_sentence,
        "comparisons": [{"pair": c.pair, "type": c.kind, "score": None if c.score is None else round(c.score, 3),
                         "result": c.result, "detail": c.detail} for c in res.comparisons],
        "notes": res.notes,
    }


def template_explanation(res: PipelineResult) -> str:
    v = res.verdict
    parts = [f"Verdict: {v.verdict}."]
    if v.retake_prompts:
        parts.append("These images are too poor to assess and were skipped: " + "; ".join(v.retake_prompts) + ".")
    parts += [r if r.endswith(".") else r + "." for r in v.reasons]
    parts += v.special_flags
    if res.reference_sentence:
        parts.append(res.reference_sentence)
    return " ".join(parts)


def plain_explanation(res: PipelineResult, ocr=None) -> str:
    """Claude writes it from the structured results only; falls back to a fixed template."""
    if config.USE_LLM_EXPLANATION and ocr is not None:
        try:
            text = ocr.explain(structured_summary(res))
            if text:
                return text
        except Exception:
            pass
    return template_explanation(res)
