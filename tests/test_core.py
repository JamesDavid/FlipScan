"""Unit tests for FlipScan's pure-logic core."""

import numpy as np
import pytest

from flipscan.backends import TranscriptionError, needs_escalation, parse_result
from flipscan.build_epub import split_chapters
from flipscan.imaging import hamming, majority_hash, phash64
from flipscan.stages.assemble import _join_pages, _strip_repeated_lines
from flipscan.stages.cluster import _segments
from flipscan.stages.figures import snap_bbox


# ---------------- parse_result

def test_parse_result_plain():
    r = parse_result('{"markdown": "hello", "page_number_printed": 5, '
                     '"confidence": "high", "regions": [], "flags": []}')
    assert r["markdown"] == "hello"
    assert r["page_number_printed"] == 5


def test_parse_result_fenced_and_coerced():
    r = parse_result('```json\n{"markdown": "x", "page_number_printed": "n/a", '
                     '"confidence": "very sure", '
                     '"regions": [{"bbox_norm": [0, -0.5, 1, 1.5]}, {"bad": 1}], '
                     '"flags": ["blur", 42]}\n```')
    assert r["confidence"] == "low"          # invalid value coerced
    assert r["page_number_printed"] is None
    assert len(r["regions"]) == 1            # region without bbox dropped
    assert r["regions"][0]["bbox_norm"] == [0.0, 0.0, 1.0, 1.0]  # clamped
    assert r["flags"] == ["blur"]            # non-strings dropped


def test_parse_result_garbage_raises():
    with pytest.raises(TranscriptionError):
        parse_result("I could not read this page, sorry!")


def test_needs_escalation():
    esc = ["low_confidence", "malformed_json", "flags"]
    assert needs_escalation({"error": "boom"}, esc)
    assert needs_escalation({"confidence": "low", "flags": []}, esc)
    assert needs_escalation({"confidence": "high", "flags": ["blur"]}, esc)
    assert not needs_escalation({"confidence": "high", "flags": []}, esc)
    assert not needs_escalation({"confidence": "low", "flags": []}, ["flags"])


# ---------------- hashing

def test_hamming_and_majority():
    assert hamming(0b1010, 0b1001) == 2
    assert majority_hash([0b111, 0b101, 0b001]) == 0b101


def test_phash_similar_vs_different():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    noisy = np.clip(base.astype(int) + rng.integers(-8, 8, base.shape), 0, 255).astype(np.uint8)
    other = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    assert hamming(phash64(base), phash64(noisy)) < hamming(phash64(base), phash64(other))


# ---------------- clustering segments

def _rec(motion, flatness=0.6):
    return {"motion": motion, "flatness": flatness}


def test_segments_drop_turn_debris():
    # 8 rest frames, a turn with a brief sub-threshold dip, 8 more rest frames
    records = ([_rec(0.0)] + [_rec(2.0)] * 7
               + [_rec(20.0)] * 3 + [_rec(6.0)] * 2 + [_rec(20.0)] * 3
               + [_rec(2.0)] * 8)
    segs = _segments(records, spike_factor=2.5)
    assert len(segs) == 2
    assert all(len(s) == 8 for s in segs)


def test_segments_keep_genuinely_still_short_rest():
    records = ([_rec(0.0)] + [_rec(2.0)] * 9 + [_rec(20.0)] * 4
               + [_rec(2.0)] * 2                      # brief but truly still
               + [_rec(20.0)] * 4 + [_rec(2.0)] * 10)
    segs = _segments(records, spike_factor=2.5)
    assert len(segs) == 3


# ---------------- assembly

def test_join_hyphenation():
    assert _join_pages(["The quick-", "brown fox."]) == "The quickbrown fox."


def test_join_paragraph_continuation():
    out = _join_pages(["He walked to the", "store and bought milk."])
    assert out == "He walked to the store and bought milk."


def test_join_separate_paragraphs():
    out = _join_pages(["First page ends here.", "New paragraph starts."])
    assert out == "First page ends here.\n\nNew paragraph starts."


def test_join_heading_not_merged():
    out = _join_pages(["ended mid sentence", "# chapter two"])
    assert "\n\n# chapter two" in out


def test_strip_repeated_headers_and_page_numbers():
    pages = [f"MY BOOK TITLE\nContent of page {i}.\n{i + 1}" for i in range(10)]
    stripped = _strip_repeated_lines(pages)
    assert all("MY BOOK TITLE" not in t for t in stripped)
    assert stripped[0] == "Content of page 0."


def test_split_chapters():
    chs = split_chapters("intro text\n\n# One\nbody\n\n# Two\nmore")
    assert [t for t, _ in chs] == ["Front Matter", "One", "Two"]


# ---------------- figures

def test_snap_bbox_tightens_to_content():
    gray = np.full((200, 200), 240, np.uint8)
    gray[80:120, 90:150] = 40  # dark block
    x0, y0, x1, y1 = snap_bbox(gray, 50, 50, 180, 180)
    assert 80 <= x0 + 10 and x1 <= 165   # snapped near the block (+margin)
    assert abs(y0 - 72) <= 10 and abs(y1 - 128) <= 10


def test_snap_bbox_empty_returns_input():
    gray = np.full((100, 100), 240, np.uint8)
    assert snap_bbox(gray, 10, 10, 90, 90) == (10, 10, 90, 90)


# ---------------- dewarp

def test_dewarp_straightens_bowed_lines():
    import cv2
    from flipscan.stages.preprocess import dewarp_cylindrical

    h, w = 400, 600
    img = np.full((h, w, 3), 245, np.uint8)
    # draw text-like lines bowed downward in the middle (cylindrical curl)
    for row in range(60, 340, 40):
        for x in range(30, w - 30, 4):
            sag = int(20 * np.sin(np.pi * x / w))
            cv2.circle(img, (x, row + sag), 2, (30, 30, 30), -1)
    out = dewarp_cylindrical(img)

    def top_envelope_ptp(im):
        ink = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) < 128
        tops = [np.nonzero(ink[:, x])[0][0] for x in range(30, w - 30, 20)
                if ink[:, x].any()]
        return float(np.ptp(tops))

    assert top_envelope_ptp(out) < top_envelope_ptp(img) * 0.5
