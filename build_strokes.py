#!/usr/bin/env python3
"""Build per-category stroke data for the stroke-by-stroke drawing explorer.

Joins the *agreed* semantic part labels (one row per drawing-stroke, with
inter-annotator agreement) to the raw per-stroke SVG paths, and writes one
compact JSON file per category plus an index.json consumed by strokes.html.

Inputs
------
- AGREE  : agree_labels.csv  — exported from
           drawing_production_and_recognition/data/part_annotations_processed/
           merged_annotations.RData  (object `d_agree_labels`). Columns include
           filename, strokeIndex, roi_labelName (consensus part), category, age,
           count_participants, n.  Export once with R:

               load("merged_annotations.RData")
               write.csv(as.data.frame(d_agree_labels), "agree_labels.csv",
                         row.names = FALSE)

- RAW    : the 5 kiddraw_annotations_test_data_preprocessed_unnested_*.csv files
           in drawing_production_and_recognition/data/drawing_annotations_raw/ —
           these carry the per-stroke `svg` path (and arcLength).
- EMPH   : part_emphasis.csv — exported from part_emphasis.RData (optional;
           per drawing-part emphasis = part_area / total_arc).

Output
------
- strokes_data/<category>.json  — drawings with ordered strokes (svg path + part
  + agreement) and a per-drawing viewBox.
- strokes_data/index.json       — category list, counts, and part vocabulary.
"""
import csv
import io
import json
import os
import re
import sys
from collections import defaultdict

csv.field_size_limit(sys.maxsize)

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "drawing_production_and_recognition", "data")
RAW_DIR = os.path.join(SRC, "drawing_annotations_raw")
# CSVs exported from the RData (see module docstring). Override via env if needed.
SCRATCH = os.environ.get(
    "STROKE_SCRATCH",
    "/private/tmp/claude-501/-Users-brialong-Documents-GitHub-drawing-explorer/"
    "1f976fbd-0567-4a0d-b2e9-8a8ae8301dac/scratchpad",
)
AGREE = os.environ.get("AGREE_CSV", os.path.join(SCRATCH, "agree_labels.csv"))
EMPH = os.environ.get("EMPH_CSV", os.path.join(SCRATCH, "part_emphasis.csv"))
OUT_DIR = os.path.join(HERE, "strokes_data")

# Fixed museum-station tablet canvas (canvas = window.innerWidth*0.8 on the kiosk
# iPad, so constant across drawings). Confirmed empirically: per-drawing content
# centers cluster tightly at (~211, ~210), i.e. canvas side ~424. All drawings
# render in this single square frame so size/position are comparable.
CANVAS_SIZE = 424

NUM_RE = re.compile(r"-?\d+\.?\d*")
# the stroke paths use only M (absolute move), l/h/v (relative line/horiz/vert)
CMD_RE = re.compile(r"([Mlhv])([^Mlhv]*)")


def round_path(d, ndp=1):
    """Shrink an SVG path string by rounding every number to `ndp` decimals."""
    def repl(m):
        v = round(float(m.group(0)), ndp)
        return str(int(v)) if v == int(v) else str(v)
    return NUM_RE.sub(repl, d)


def path_bounds(d):
    """Accurate min/max of a polyline path, honoring relative l/h/v commands."""
    cx = cy = 0.0
    xs, ys = [], []
    for m in CMD_RE.finditer(d):
        c = m.group(1)
        nums = [float(x) for x in NUM_RE.findall(m.group(2))]
        if c == "M":  # absolute move (may carry extra pairs as implicit lines)
            for j in range(0, len(nums) - 1, 2):
                cx, cy = nums[j], nums[j + 1]; xs.append(cx); ys.append(cy)
        elif c == "l":  # relative line
            for j in range(0, len(nums) - 1, 2):
                cx += nums[j]; cy += nums[j + 1]; xs.append(cx); ys.append(cy)
        elif c == "h":  # relative horizontal
            for v in nums:
                cx += v; xs.append(cx); ys.append(cy)
        elif c == "v":  # relative vertical
            for v in nums:
                cy += v; xs.append(cx); ys.append(cy)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def load_agree():
    """(filename, strokeIndex) -> {part, agree:[count,n], age, category}."""
    out = {}
    with open(AGREE, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (r["filename"], str(int(float(r["strokeIndex"]))))
            try:
                total = int(float(r["count_participants"]))  # all annotators
                agreed = int(float(r["n"]))                   # # choosing winning label
            except (ValueError, KeyError):
                total, agreed = None, None
            part = (r["roi_labelName"] or "").strip()
            # single-character labels are stray free-text (e.g. a child who wrote
            # a word letter-by-letter, annotated t/r/a/i/n) — not object parts.
            if len(part) == 1:
                part = "other"
            out[key] = {
                "part": part,
                "agree": [agreed, total],   # [#agreed, #total]
                "age": r.get("age_numeric") or r.get("age", ""),
                "category": r["category"],
            }
    return out


def load_svgs(needed):
    """(filename, strokeIndex) -> svg path, for keys in `needed`."""
    out = {}
    for fn in sorted(os.listdir(RAW_DIR)):
        if not fn.endswith(".csv"):
            continue
        with open(os.path.join(RAW_DIR, fn), encoding="utf-8", errors="replace") as fh:
            for r in csv.DictReader(fh):
                key = (r["filename"], str(r["strokeIndex"]))
                if key in needed and key not in out:
                    out[key] = r["svg"]
    return out


def load_emphasis():
    """filename -> {part: emphasis}."""
    out = defaultdict(dict)
    if not os.path.exists(EMPH):
        return out
    with open(EMPH, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                out[r["filename"]][r["roi_labelName"].strip()] = round(float(r["emphasis"]), 3)
            except (ValueError, KeyError):
                pass
    return out


def short_age(a):
    m = re.search(r"\d+", str(a))
    return int(m.group(0)) if m else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    agree = load_agree()
    print(f"agreed strokes: {len(agree)}")
    svgs = load_svgs(set(agree))
    print(f"matched svg paths: {len(svgs)} / {len(agree)}")
    emph = load_emphasis()

    # group strokes by drawing
    by_draw = defaultdict(list)  # filename -> list of (strokeIndex, info, svg)
    for key, info in agree.items():
        fn, si = key
        svg = svgs.get(key)
        if not svg:
            continue
        by_draw[fn].append((int(si), info, svg))

    cats = defaultdict(list)         # category -> list of drawing dicts
    part_counts = defaultdict(lambda: defaultdict(int))  # category -> part -> n

    for fn, strokes in by_draw.items():
        strokes.sort(key=lambda t: t[0])
        cat = strokes[0][1]["category"]
        age = short_age(strokes[0][1]["age"])
        out_strokes = []
        for si, info, svg in strokes:
            part = info["part"] or "?"
            part_counts[cat][part] += 1
            out_strokes.append({
                "d": round_path(svg),
                "p": part,
                "a": info["agree"],
            })
        if not out_strokes:
            continue
        # Drawings are rendered in the FIXED original tablet canvas (see
        # CANVAS_SIZE) so relative position & size are preserved across drawings
        # and the scale is uniform; overflow is clipped like the real canvas.
        cats[cat].append({
            "id": fn,
            "age": age,
            "strokes": out_strokes,
            "emph": emph.get(fn, {}),
        })

    index = {"canvas": CANVAS_SIZE, "categories": [], "parts": {}}
    for cat in sorted(cats):
        draws = sorted(cats[cat], key=lambda d: (d["age"] or 0, d["id"]))
        with open(os.path.join(OUT_DIR, f"{cat}.json"), "w") as fh:
            json.dump({"category": cat, "drawings": draws}, fh, separators=(",", ":"))
        ages = sorted({d["age"] for d in draws if d["age"] is not None})
        # parts ranked by frequency for this category (drives the color legend)
        ranked = sorted(part_counts[cat].items(), key=lambda kv: -kv[1])
        index["categories"].append({
            "name": cat,
            "n": len(draws),
            "ages": ages,
            "parts": [p for p, _ in ranked],
        })
        index["parts"][cat] = ranked
        print(f"{cat:10s} {len(draws):4d} drawings  {len(ranked)} parts")

    with open(os.path.join(OUT_DIR, "index.json"), "w") as fh:
        json.dump(index, fh, separators=(",", ":"))
    total = sum(c["n"] for c in index["categories"])
    print(f"\nwrote {len(index['categories'])} categories, {total} drawings to {OUT_DIR}")


if __name__ == "__main__":
    main()
