"""Unit tests for FlipScan's pure-logic core."""

import numpy as np
import pytest

from flipscan.backends import TranscriptionError, needs_escalation, parse_result
from flipscan.build_epub import split_chapters
from flipscan.imaging import hamming, majority_hash, phash64
from flipscan.stages.assemble import _join_pages, _strip_repeated_lines
from flipscan.stages.cluster import segment_pages
from flipscan.stages.transcribe import (dedupe_by_printed, format_ranges,
                                        infer_missing_numbers, reorder_by_printed,
                                        sanitize_numbers_by_video)
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


# ---------------- page segmentation (turn events split, wobble doesn't)

def _rec(motion, flatness=0.6):
    return {"motion": motion, "flatness": flatness}


def test_turn_gap_splits_pages():
    records = ([_rec(0.0)] + [_rec(2.0)] * 7
               + [_rec(20.0)] * 6                     # sustained turn
               + [_rec(2.0)] * 8)
    pages = segment_pages(records, spike_factor=2.5)
    assert len(pages) == 2
    assert all(len(p) == 8 for p in pages)


def test_wobble_does_not_split_page():
    records = ([_rec(0.0)] + [_rec(2.0)] * 7
               + [_rec(20.0)] * 2                     # brief hand wobble
               + [_rec(2.0)] * 8)
    pages = segment_pages(records, spike_factor=2.5)
    assert len(pages) == 1                            # still the same page
    assert len(pages[0]) == 16


def test_mid_turn_debris_dropped():
    records = ([_rec(0.0)] + [_rec(2.0)] * 9
               + [_rec(20.0)] * 5 + [_rec(4.5)] * 2 + [_rec(20.0)] * 5
               + [_rec(2.0)] * 10)
    pages = segment_pages(records, spike_factor=2.5)
    assert len(pages) == 2                            # debris blip not a page


def test_brief_but_still_rest_is_a_page():
    records = ([_rec(0.0)] + [_rec(2.0)] * 9
               + [_rec(20.0)] * 5
               + [_rec(2.0)] * 3                      # quick flip, genuinely still
               + [_rec(20.0)] * 5 + [_rec(2.0)] * 10)
    pages = segment_pages(records, spike_factor=2.5)
    assert len(pages) == 3


# ---------------- deduplication by printed number

def _pg(pid, num, conf="high", score=0.5, **kw):
    return {"id": pid, "printed_number": num, "confidence": conf,
            "scores": {"composite": score}, "status": "ok", **kw}


def test_dedupe_keeps_best_capture():
    pages = [_pg("a", 5, "low", 0.3), _pg("b", 5, "high", 0.6), _pg("c", 6)]
    n = dedupe_by_printed(pages, log=lambda m: None)
    assert n == 1
    assert pages[0]["status"] == "duplicate"   # low-confidence capture loses
    assert pages[1]["status"] == "ok"
    assert pages[2]["status"] == "ok"


def test_dedupe_unnumbered_untouched():
    pages = [_pg("a", None), _pg("b", None), _pg("c", 3)]
    assert dedupe_by_printed(pages, log=lambda m: None) == 0
    assert all(p["status"] == "ok" for p in pages)


def test_dedupe_rerun_can_change_winner():
    pages = [_pg("a", 5, "high", 0.6), _pg("b", 5, "low", 0.3)]
    dedupe_by_printed(pages, log=lambda m: None)
    assert pages[1]["status"] == "duplicate"
    pages[1]["confidence"] = "high"
    pages[1]["scores"]["composite"] = 0.9      # better capture arrived
    dedupe_by_printed(pages, log=lambda m: None)
    assert pages[0]["status"] == "duplicate" and pages[1]["status"] == "ok"


def test_reorder_by_printed_numbers():
    pages = [
        {"id": "a", "printed_number": 3},
        {"id": "b", "printed_number": None},   # follows page 3
        {"id": "c", "printed_number": 1},
        {"id": "d", "printed_number": 2},
        {"id": "cov", "role": "cover", "pinned": "start"},
    ]
    out = reorder_by_printed(pages)
    assert [p["id"] for p in out] == ["cov", "c", "d", "a", "b"]


def test_reorder_no_numbers_is_stable():
    pages = [{"id": i, "printed_number": None} for i in range(5)]
    assert [p["id"] for p in reorder_by_printed(pages)] == [0, 1, 2, 3, 4]


# ---------------- number sanitizing + inference

def _cap(pid, frame, num, video="v0", **kw):
    return {"id": pid, "video": video, "canonical": f"{video}_f{frame:06d}",
            "printed_number": num, "status": "ok", "scores": {}, **kw}


def test_sanitize_rejects_misread_266_between_204_and_208():
    pages = [_cap("a", 100, 200), _cap("b", 200, 202), _cap("c", 300, 204),
             _cap("d", 400, 266),  # misread — really 206
             _cap("e", 500, 208), _cap("f", 600, 210)]
    assert sanitize_numbers_by_video(pages, log=lambda m: None) == 1
    assert pages[3]["printed_number"] is None and pages[3]["number_rejected"]
    infer_missing_numbers(pages, log=lambda m: None)
    assert pages[3]["printed_number"] == 206  # neighbors 204/208 pin it down


def test_sanitize_handles_descending_video():
    pages = [_cap("a", 100, 50), _cap("b", 200, 48), _cap("c", 300, 12),  # misread
             _cap("d", 400, 44), _cap("e", 500, 42)]
    assert sanitize_numbers_by_video(pages, log=lambda m: None) == 1
    assert pages[2]["printed_number"] is None


def test_sanitize_keeps_manual_numbers():
    pages = [_cap("a", 100, 10), _cap("b", 200, 99, number_manual=True),
             _cap("c", 300, 12), _cap("d", 400, 14), _cap("e", 500, 16)]
    sanitize_numbers_by_video(pages, log=lambda m: None)
    assert pages[1]["printed_number"] == 99  # user-entered survives


def test_infer_from_video_capture_order():
    from flipscan.stages.transcribe import infer_from_video_order
    # descending pass, step 2: 117, ?, 113 -> the unknown must be 115
    pages = [_cap("a", 100, 121), _cap("b", 200, 119), _cap("c", 300, 117),
             _cap("d", 400, None, number_rejected=True),
             _cap("e", 500, 113), _cap("f", 600, 111)]
    assert infer_from_video_order(pages, log=lambda m: None) == 1
    assert pages[3]["printed_number"] == 115
    assert pages[3]["number_inferred"] and "number_rejected" not in pages[3]


def test_front_matter_negative_numbers():
    from flipscan.stages.transcribe import compute_missing_pages
    pages = [_cap("toc", 100, -2), _cap("fwd", 200, -1),
             _cap("a", 300, 1), _cap("b", 400, 2), _cap("c", 500, 5)]
    out = reorder_by_printed(pages)
    assert [p["id"] for p in out] == ["toc", "fwd", "a", "b", "c"]
    assert compute_missing_pages(pages) == [3, 4]  # negatives excluded from range


def test_format_ranges():
    assert format_ranges([6, 7, 14, 22, 23, 24]) == "6-7, 14, 22-24"
    assert format_ranges([]) == ""


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


# ---------------- printed TOC + header cleanup

def test_parse_printed_toc():
    from flipscan.stages.assemble import parse_printed_toc
    toc_page = ("# CONTENTS\n\nIntroduction 1\nChapter 1 9\nChapter 2 45\n"
                "Notes 239\nIndex 253\n")
    assert parse_printed_toc(["body text", toc_page]) == [
        ("Introduction", 1), ("Chapter 1", 9), ("Chapter 2", 45),
        ("Notes", 239), ("Index", 253)]
    # a page that merely mentions the word is not a TOC
    assert parse_printed_toc(["contents of the gasbag spilled 12"]) == []


def test_strip_fuzzy_header_fragments():
    from flipscan.stages.assemble import _strip_repeated_lines
    pages = [f"## DIRIGIBLE DREAMS\nunique page text number {i}"
             for i in range(5)] + [
        "## BLE DREAMS\nthe expedition sailed north",   # cropped title fragment
        "## GIB DREAMS\nthe crew mutinied at dawn",     # OCR-mangled fragment
        "# ONE\nchapter opener stays",
    ]
    out = _strip_repeated_lines(pages, extra_refs={"dirigible dreams"})
    assert out[5] == "the expedition sailed north"
    assert out[6] == "the crew mutinied at dawn"
    assert out[7].startswith("# ONE")


def test_dedupe_headings_and_chapter_promotion():
    from flipscan.stages.assemble import (_dedupe_headings,
                                          _insert_chapter_breaks)
    texts = ["# INDEX\nAlpha", "# INDEX\nBeta", "## ONE\nchapter text",
             "plain start page"]
    texts = _dedupe_headings(texts)
    assert texts[1] == "Beta"                     # repeated heading dropped
    pages = [{"printed_number": n} for n in (250, 251, 9, 45)]
    texts = _insert_chapter_breaks(pages, texts,
                                   [("Chapter 1", 9), ("Chapter 2", 45)],
                                   log=lambda m: None)
    assert texts[2].splitlines()[0] == "# ONE"    # promoted to level 1
    assert texts[3].startswith("# Chapter 2\n")   # heading lost -> inserted


def test_split_chapters_merges_consecutive_duplicates():
    from flipscan.build_epub import split_chapters
    md = "# INDEX\nAlpha\n# INDEX\nBeta\n# NOTES\nGamma"
    got = split_chapters(md)
    assert [t for t, _ in got] == ["INDEX", "NOTES"]
    assert "Beta" in got[0][1]


# ---------------- proofread layer

def test_apply_edits_safety():
    from flipscan.proofread import apply_edits
    md = "The airship sailled on. The crew slept. The airship sailled on."
    finds = [
        {"quote": "crew slept", "replacement": "crew slept.", "note": ""},
        {"quote": "sailled", "replacement": "sailed", "note": ""},   # 2 hits
        {"quote": "not present", "replacement": "x", "note": ""},
        {"quote": "on.", "replacement": None, "note": "just a note"},
    ]
    out, applied = apply_edits(md, finds)
    assert applied == 1
    assert "crew slept." in out
    assert "sailled" in out               # ambiguous edit NOT applied
    assert finds[1]["applied"] is False and finds[1]["skip_reason"] == "ambiguous:2"
    assert finds[2]["applied"] is False and finds[2]["skip_reason"] == "not_found"
    # promoting an ambiguous fix applies it everywhere
    finds[1]["apply_all"] = True
    out2, applied2 = apply_edits(md, finds)
    assert applied2 == 2 and "sailled" not in out2


def test_lint_finds_duplicate_paragraph_and_garble():
    from flipscan.proofread import lint_chapter
    para = "A long paragraph about airships that repeats itself " * 3
    md = f"Andr�e went north.\n\n{para}\n\n{para}\n\n[[region-0]]"
    types = {f["type"] for f in lint_chapter(md)}
    assert {"ocr", "continuity", "formatting"} <= types


def test_gap_markers_inserted():
    from flipscan.stages.assemble import _with_gap_markers
    pages = [{"printed_number": 11}, {"printed_number": 14}, {"printed_number": 15}]
    texts = ["page eleven text", "page fourteen text", "page fifteen text"]
    out = _with_gap_markers(pages, texts)
    assert len(out) == 4
    assert "pages 12–13 missing" in out[1]


def test_build_uses_accepted_proof_only_when_hash_matches(tmp_path):
    from flipscan.proofread import chapter_hash
    md = "# ONE\n\nsome chapter text"
    assert chapter_hash(md) == chapter_hash(md + "  \n")   # strip-insensitive
    assert chapter_hash(md) != chapter_hash(md + " changed")


def test_apply_edits_tolerates_linebreaks_and_typography():
    from flipscan.proofread import apply_edits
    md = ("in less than fifty years dirigible became\na word for doomed\n"
          "futur- istic visions and the crew’s hope faded")
    finds = [
        {"quote": "years dirigible became a word for doomed futuristic",
         "replacement": "years, dirigible became a word for doomed futurism",
         "note": ""},
        {"quote": "the crew's hope", "replacement": "the crews' hope", "note": ""},
    ]
    out, applied = apply_edits(md, finds)
    assert applied == 2
    assert "doomed futurism visions" in out.replace("\n", " ")
    assert "crews' hope" in out


def test_apply_edits_empty_replacement_deletes():
    from flipscan.proofread import apply_edits
    md = "keep this 5 4 3 2 1 artifact gone"
    finds = [{"quote": "5 4 3 2 1 artifact ", "replacement": "", "note": ""}]
    out, applied = apply_edits(md, finds)
    assert applied == 1 and out == "keep this gone"


def test_dedupe_patched_retake_beats_video_capture():
    from flipscan.stages.transcribe import dedupe_by_printed
    pages = [_pg("vid", 24, "high", 0.9),
             _pg("shot", 24, "high", 0.0, patched_source="patches/shot.jpg",
                 status="patched")]
    n = dedupe_by_printed(pages, log=lambda m: None,
                          mtime_of=lambda p: 1000.0)
    assert n == 1
    assert pages[0]["status"] == "duplicate"      # old video frame hidden
    assert pages[1]["status"] == "patched"        # deliberate photo kept


def test_dedupe_newest_patch_wins():
    from flipscan.stages.transcribe import dedupe_by_printed
    times = {"old": 100.0, "new": 200.0}
    pages = [_pg("old", 7, "high", 0.5, patched_source="patches/old.jpg",
                 status="patched"),
             _pg("new", 7, "low", 0.1, patched_source="patches/new.jpg",
                 status="patched")]
    dedupe_by_printed(pages, log=lambda m: None,
                      mtime_of=lambda p: times[p["id"]])
    assert pages[0]["status"] == "duplicate"
    assert pages[1]["status"] == "patched"        # most recent retake wins


def test_dedupe_by_content_catches_unnumbered_twin():
    from flipscan.stages.transcribe import dedupe_by_content
    text = ("As enthusiastic supporters of the aeronaut's experiments, "
            "Parisians welcomed his newest creation Santos Dumont Number 5 "
            "and early one quiet Saturday morning July 13 1901 it started "
            "off from the Parc toward the Eiffel Tower at the controls stood "
            "the designer and sole occupant while the commission watched. " * 3)
    pages = [_pg("a", 9, "high", 0.2),
             _pg("b", None, "high", 0.0, patched_source="patches/b.jpg",
                 status="patched"),
             _pg("c", 11, "high", 0.5)]
    texts = {"a": text, "b": text + " tiny tail difference", "c": "totally other content " * 30}
    n = dedupe_by_content(pages, lambda p: texts[p["id"]],
                          log=lambda m: None, mtime_of=lambda p: 50.0)
    assert n == 1
    assert pages[0]["status"] == "duplicate"      # video copy loses
    assert pages[1]["status"] == "patched" and pages[1].get("content_duplicate") is None
    assert pages[2]["status"] == "ok"


def test_heal_hyphenation():
    from flipscan.stages.assemble import heal_hyphenation
    assert heal_hyphenation("the poten-\ntial of wing- less flight") == \
        "the potential of wingless flight"
    # suspended hyphens and true compounds survive
    assert heal_hyphenation("nineteenth- and twentieth-century craft") == \
        "nineteenth- and twentieth-century craft"
    assert heal_hyphenation("a well-known pre-war design") == \
        "a well-known pre-war design"
    assert heal_hyphenation("pages 12- 14 stay") == "pages 12- 14 stay"


def test_destructive_edits_held_back():
    from flipscan.proofread import apply_edits
    md = ("assurances were sent me\n\n![Santos](figures/p0012_a.png)\n\nmore text "
          "and the costly \"honorar-lum\" demanded by builders restraints.123 end")
    finds = [
        # deletes an image tag + invents words
        {"quote": "assurances were sent me\n\n![Santos](figures/p0012_a.png)",
         "replacement": "assurances were sent me that the", "note": ""},
        # deletes most of the quoted text
        {"quote": "costly \"honorar-lum\" demanded by builders",
         "replacement": "\"honorarium\"", "note": ""},
        # rewrites footnote digits
        {"quote": "restraints.123", "replacement": "restraints.¹²ⁱ", "note": ""},
    ]
    out, applied = apply_edits(md, finds)
    assert applied == 0
    assert all((f.get("skip_reason") or "").startswith("unsafe") for f in finds)
    assert "![Santos]" in out                     # nothing was harmed
    # a user-authored fix bypasses the guard
    finds[1]["user_edit"] = True
    finds[1]["replacement"] = 'costly "honorarium" demanded by builders'
    out2, applied2 = apply_edits(md, finds)
    assert applied2 == 1 and 'costly "honorarium" demanded' in out2


def test_inserted_chapter_titles_match_book_style():
    from flipscan.stages.assemble import _insert_chapter_breaks
    pages = [{"printed_number": n} for n in (9, 45, 74, 103)]
    texts = ["## ONE\nchapter one text", "## TWO\nchapter two text",
             "lost its heading entirely", "## FOUR\nchapter four text"]
    toc = [("Chapter 1", 9), ("Chapter 2", 45), ("Chapter 3", 74),
           ("Chapter 4", 103)]
    out = _insert_chapter_breaks(pages, texts, toc, log=lambda m: None)
    assert out[0].splitlines()[0] == "# ONE"
    assert out[2].splitlines()[0] == "# THREE"    # synthetic, restyled
    # a book with no clear convention keeps the TOC title
    texts2 = ["The Beginning\ntext", "no heading here", "also none"]
    pages2 = [{"printed_number": n} for n in (1, 20, 40)]
    out2 = _insert_chapter_breaks(pages2, texts2,
                                  [("Chapter 2", 20)], log=lambda m: None)
    assert out2[1].splitlines()[0] == "# Chapter 2"


def test_reference_findings_never_auto_apply():
    from flipscan.proofread import apply_edits
    md = "Cross, Wilbur. Zeppelins of World War I. New York, 1991."
    finds = [{"quote": "Cross, Wilbur", "replacement": "Croix, Wilbur",
              "note": "", "reference": True}]
    out, applied = apply_edits(md, finds)
    assert applied == 0 and finds[0]["skip_reason"] == "reference"
    assert "Cross" in out
    finds[0]["apply_all"] = True          # explicit user approval applies
    out2, applied2 = apply_edits(md, finds)
    assert applied2 == 1 and "Croix" in out2


def test_pdf_pages_never_deduped_by_repeating_number():
    from flipscan.stages.transcribe import dedupe_by_printed
    # three concatenated papers, each with its own 'page 1' — all distinct
    pages = [_pg("a", 1, source="pdf"), _pg("b", 2, source="pdf"),
             _pg("c", 1, source="pdf"), _pg("d", 2, source="pdf"),
             _pg("e", 1, source="pdf")]
    n = dedupe_by_printed(pages, log=lambda m: None)
    assert n == 0
    assert all(p["status"] == "ok" for p in pages)   # nothing collapsed
