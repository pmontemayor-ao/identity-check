from compare import (MATCH, MISMATCH, UNCERTAIN, ComparisonResult, classify, compare_text_fields,
                     normalise_date, normalise_id_number, normalise_name)
from decide import FAIL, PASS, RETAKE, REVIEW, choose_reference, overall_verdict, webcam_id_special_case
from quality import MetricResult, QualityResult


def cr(result, group="id_webcam", pair="x"):
    return ComparisonResult(pair, group, "face", 0.5, result)


def q(image_id, ok=True, score=0.8, kind="webcam"):
    r = QualityResult(image_id, kind, score=score, face=object())
    r.metrics.append(MetricResult("Blur", 100 if ok else 5, ">= 60", ok, failure_code=None if ok else "blur_face"))
    return r


def test_classify_two_thresholds():
    assert classify(0.5, 0.363, 0.15) == MATCH
    assert classify(0.363, 0.363, 0.15) == MATCH
    assert classify(0.2, 0.363, 0.15) == UNCERTAIN
    assert classify(0.1, 0.363, 0.15) == MISMATCH
    assert classify(None, 0.363, 0.15) == UNCERTAIN


def test_normalisers():
    assert normalise_date("1990-03-07") == normalise_date("07/03/1990") == normalise_date("March 7, 1990") == "1990-03-07"
    assert normalise_date("not a date") == ""
    assert normalise_id_number("d123-456 789") == "D123456789"
    assert normalise_name("O'Brien,  Mary-Jane") == "o brien mary jane"


def test_text_comparison():
    a = {"name": "MARY JANE SMITH", "date_of_birth": "1990-03-07", "id_number": "D1234-5678"}
    b = {"name": "Smith Mary Jane", "date_of_birth": "07/03/1990", "id_number": "d12345678"}
    res = {c.pair.split(": ")[1]: c.result for c in compare_text_fields(a, b, 90, 70)}
    assert res == {"name": MATCH, "date of birth": MATCH, "ID number": MATCH}
    c = {"name": "JOHN DOE", "date_of_birth": "1985-01-01", "id_number": "X999"}
    res = {x.pair.split(": ")[1]: x.result for x in compare_text_fields(a, c, 90, 70)}
    assert res == {"name": MISMATCH, "date of birth": MISMATCH, "ID number": MISMATCH}


def test_missing_text_is_uncertain_never_mismatch():
    res = compare_text_fields({"name": "A B"}, {"name": "A B"}, 90, 70)
    assert [c.result for c in res] == [MATCH, UNCERTAIN, UNCERTAIN]


def test_poor_quality_never_produces_mismatch_it_produces_retake():
    v = overall_verdict([q("Webcam A"), q("Webcam B", ok=False)], [cr(MISMATCH)])
    assert v.verdict == RETAKE
    assert any("Hold still" in p for p in v.retake_prompts)


def test_verdicts():
    good = [q("A"), q("B")]
    assert overall_verdict(good, [cr(MATCH), cr(MATCH)]).verdict == PASS
    assert overall_verdict(good, [cr(MATCH), cr(UNCERTAIN)]).verdict == REVIEW
    assert overall_verdict(good, [cr(MATCH), cr(UNCERTAIN), cr(MISMATCH)]).verdict == FAIL
    assert overall_verdict(good, []).verdict == REVIEW


def test_uncertain_never_auto_rejects():
    assert overall_verdict([q("A")], [cr(UNCERTAIN)] * 5).verdict == REVIEW


def test_special_case_flag():
    web = [cr(MATCH, "webcam_webcam")] * 3
    ref = [cr(MISMATCH)] * 3
    assert webcam_id_special_case(web, ref)
    assert webcam_id_special_case(web, [cr(MISMATCH), cr(UNCERTAIN)])      # still flagged, verdict stays REVIEW
    assert not webcam_id_special_case(web, [cr(MISMATCH), cr(MATCH)])
    assert not webcam_id_special_case([cr(MISMATCH, "webcam_webcam")], ref)
    v = overall_verdict([q("A")], web + ref, special_case=True)
    assert v.verdict == FAIL and v.special_flags


def test_reference_choice_uses_quality_and_explains():
    i, s = choose_reference(q("ID 1", score=0.6, kind="id"), q("ID 2", score=0.9, kind="id"), True)
    assert i == 1 and "ID 2" in s and "higher image quality" in s
    i, s = choose_reference(q("ID 1", ok=False, score=0.99, kind="id"), q("ID 2", score=0.5, kind="id"), None)
    assert i == 1 and "needs a retake" in s
    i, s = choose_reference(q("ID 1", ok=False, kind="id"), q("ID 2", ok=False, kind="id"), None)
    assert i is None
    i, s = choose_reference(q("ID 1", score=0.9, kind="id"), q("ID 2", score=0.5, kind="id"), False)
    assert i == 0 and "disagree" in s
