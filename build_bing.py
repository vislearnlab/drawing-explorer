#!/usr/bin/env python3
"""Build the longitudinal, part-labeled Bing drawing data for `bing.html`.

The Bing study (Bing Nursery School, 2018–2020) is *longitudinal*: the same
children drew the same set of object categories across many sessions over ~15
months, so each child has a developmental trajectory of drawings you can watch
change with age. A crowd-sourced annotation pass then labeled the *semantic
part* each stroke belongs to (head, wheel, roof, ...), with inter-annotator
agreement — so strokes can be colored by meaning.

This script builds from the **anonymized gold-standard** tidy data in the
`bingdraw_annotations` repo — NOT from raw MongoDB (which still carries
children's names). Nothing here reads or writes a child's name; subjects are the
integer `active_sub_id` only.

Inputs (override the repo path via env BINGDRAW_REPO)
-----------------------------------------------------
- data/Bingdraw_svg_output_with_bbs_2022.csv : one row per drawn *stroke*, with
  session_id, active_sub_id, category, filename, date, stroke_count (1-based
  stroke order), svg (the stroke's path), and per-stroke bbox.
- data/sub_info/dob_by_anon_id.csv           : anon_sub_id -> DOB, for age.
- data/preprocessed_data/annotations_cleaned.RData (object `d_agree_labels`):
  per (filename, strokeIndex) the *consensus* part `roi_labelName`, with `n` of
  `count_participants` annotators agreeing. Joined by filename + stroke order
  (strokeIndex == stroke_count - 1). Requires `pyreadr`; if unavailable the
  build still runs, just without part colors.
- data/preprocessed_data/part_emphasis.RData (object `part_emphasis_all`):
  per (filename, part) emphasis = part_area / total_arc (optional).

Output (bing_data/)
-------------------
- index.json      : children (per-child session ages & categories), category
  summaries (with ranked part vocabulary), and dataset meta.
- child_<id>.json : one child's sessions ordered by age -> trajectory view.
- cat_<slug>.json : one category's sketches across children, age-sorted ->
  category-across-ages view.
  Each stroke is {d} or {d, p:<part>, a:[n,total]}; each sketch carries its
  fitted viewBox and (when present) per-part `emph`.
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

csv.field_size_limit(sys.maxsize)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("BINGDRAW_REPO", os.path.join(os.path.dirname(HERE), "bingdraw_annotations"))
STROKES_CSV = os.path.join(REPO, "data", "Bingdraw_svg_output_with_bbs_2022.csv")
DOB_CSV = os.path.join(REPO, "data", "sub_info", "dob_by_anon_id.csv")
PP = os.path.join(REPO, "data", "preprocessed_data")
AGREE_RDATA = os.path.join(PP, "annotations_cleaned.RData")
EMPH_RDATA = os.path.join(PP, "part_emphasis.RData")
OUT_DIR = os.path.join(HERE, "bing_data")

NUM_RE = re.compile(r"-?\d+\.?\d*")
MIN_STROKE_LEN = 2.5      # canvas units; shorter strokes are stray taps/dots
PAD_FRAC = 0.06           # viewBox padding as a fraction of the larger bbox side
# The square/shape *tracing* control trials are not object-drawing categories
# (kids traced a shown square/shape); the annotation analyses use only the 12
# object categories, so we drop these to match.
EXCLUDE_CATEGORIES = {"shape", "square", "this square"}
# labels that carry little semantic meaning (rendered gray / very faint)
LOW_MEANING = {"unintelligible", "other", "none", "?", "na", "n/a", "", "i cant tell"}


def round_path(d, ndp=1):
    def repl(m):
        v = round(float(m.group(0)), ndp)
        return str(int(v)) if v == int(v) else str(v)
    return NUM_RE.sub(repl, d)


def path_points(d):
    cx = cy = 0.0
    pts = []
    for m in re.finditer(r"([Mlhv])([^Mlhv]*)", d):
        c = m.group(1)
        nums = [float(x) for x in NUM_RE.findall(m.group(2))]
        if c == "M":
            for j in range(0, len(nums) - 1, 2):
                cx, cy = nums[j], nums[j + 1]; pts.append((cx, cy))
        elif c == "l":
            for j in range(0, len(nums) - 1, 2):
                cx += nums[j]; cy += nums[j + 1]; pts.append((cx, cy))
        elif c == "h":
            for v in nums:
                cx += v; pts.append((cx, cy))
        elif c == "v":
            for v in nums:
                cy += v; pts.append((cx, cy))
    return pts


def path_length(pts):
    return sum(((pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2) ** 0.5
               for i in range(1, len(pts)))


def slug(cat):
    return re.sub(r"[^a-z0-9]+", "_", cat.lower()).strip("_")


def clean_part(label):
    """Normalize a consensus part label; multi-part '[a, b]' -> first part."""
    s = str(label).strip()
    if s.startswith("["):
        inner = re.findall(r"'([^']+)'|\"([^\"]+)\"", s)
        s = (inner[0][0] or inner[0][1]) if inner else s.strip("[]")
    s = s.strip().lower()
    if s in LOW_MEANING or "cant tell" in s or "cannot tell" in s:
        return "unintelligible"
    return s


def load_dob():
    dob = {}
    with open(DOB_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            dob[int(r["anon_sub_id"])] = datetime.strptime(r["dob"], "%Y-%m-%d")
    return dob


def load_annotations():
    """(basename, strokeIndex:int) -> {'p':part, 'a':[n,total]} ; and
    basename -> {part: emphasis}. Returns ({}, {}) if pyreadr is unavailable."""
    try:
        import pyreadr
    except ImportError:
        print("!! pyreadr not installed; building WITHOUT part labels "
              "(`pip install pyreadr` to add them).", file=sys.stderr)
        return {}, {}
    parts = {}
    ag = pyreadr.read_r(AGREE_RDATA)["d_agree_labels"]
    for r in ag.itertuples(index=False):
        base = os.path.basename(str(r.filename))
        si = int(round(float(r.strokeIndex)))
        try:
            total = int(r.count_participants); agreed = int(r.n)
        except (ValueError, TypeError):
            total = agreed = None
        parts[(base, si)] = {"p": clean_part(r.roi_labelName), "a": [agreed, total]}
    emph = defaultdict(dict)
    if os.path.exists(EMPH_RDATA):
        em = pyreadr.read_r(EMPH_RDATA)["part_emphasis_all"]
        for r in em.itertuples(index=False):
            base = os.path.basename(str(r.filename))
            emph[base][clean_part(r.roi_labelName)] = round(float(r.emphasis), 3)
    print(f"annotated strokes: {len(parts)}  drawings with emphasis: {len(emph)}")
    return parts, emph


def parse_date(s):
    return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dob = load_dob()
    ann, emph = load_annotations()

    # group stroke rows -> sketch keyed by filename (one drawing = one filename)
    sk = {}
    with open(STROKES_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["category"] in EXCLUDE_CATEGORIES:
                continue
            fn = r["filename"]
            s = sk.get(fn)
            if s is None:
                s = sk[fn] = {"sub": int(r["active_sub_id"]), "sess": r["session_id"],
                              "cat": r["category"], "date": r["date"],
                              "base": os.path.basename(fn), "strokes": []}
            try:
                order = int(float(r["stroke_count"]))    # 1-based
            except ValueError:
                order = len(s["strokes"]) + 1
            s["strokes"].append((order, r["svg"]))

    children = defaultdict(lambda: defaultdict(lambda: {"sketches": {}}))
    cat_sketches = defaultdict(list)
    cat_ages = defaultdict(list)
    part_counts = defaultdict(lambda: defaultdict(int))   # cat -> part -> n
    n_sketches = n_annot = 0

    for fn, s in sk.items():
        sub, base, cat = s["sub"], s["base"], s["cat"]
        d0 = dob.get(sub)
        dt = parse_date(s["date"])
        age = round((dt - d0).days / 365.25, 2) if d0 else None

        s["strokes"].sort(key=lambda t: t[0])
        out_strokes = []
        minx = miny = 1e9; maxx = maxy = -1e9
        has_part = False
        for order, svg in s["strokes"]:
            pts = path_points(svg)
            if path_length(pts) < MIN_STROKE_LEN:
                continue
            for x, y in pts:
                minx = min(minx, x); maxx = max(maxx, x)
                miny = min(miny, y); maxy = max(maxy, y)
            stroke = {"d": round_path(svg)}
            info = ann.get((base, order - 1))     # strokeIndex == stroke_count - 1
            if info:
                stroke["p"] = info["p"]; stroke["a"] = info["a"]
                part_counts[cat][info["p"]] += 1
                has_part = True
            out_strokes.append(stroke)
        if not out_strokes:
            continue

        w = maxx - minx or 1.0
        h = maxy - miny or 1.0
        pad = PAD_FRAC * max(w, h)
        vb = [round(minx - pad, 1), round(miny - pad, 1),
              round(w + 2 * pad, 1), round(h + 2 * pad, 1)]

        rec = {"cat": cat, "age": age, "sub": sub, "date": s["date"][:10],
               "vb": vb, "strokes": out_strokes, "ann": has_part}
        if emph.get(base):
            rec["emph"] = emph[base]
        n_sketches += 1
        n_annot += has_part
        children[sub][s["sess"]].update({"date": s["date"][:10], "age": age})
        children[sub][s["sess"]]["sketches"][fn] = rec
        cat_sketches[cat].append(rec)
        if age is not None:
            cat_ages[cat].append(age)

    # ---- per-child files + index ----
    child_index = []
    for sub in sorted(children):
        sessions = []
        for sess, info in children[sub].items():
            recs = sorted(info["sketches"].values(), key=lambda r: r["cat"])
            sessions.append({"date": info["date"], "age": info["age"], "sketches": recs})
        sessions.sort(key=lambda s: (s["age"] if s["age"] is not None else 0, s["date"]))
        with open(os.path.join(OUT_DIR, f"child_{sub}.json"), "w") as fh:
            json.dump({"id": sub, "sessions": sessions}, fh, separators=(",", ":"))
        ages = [s["age"] for s in sessions if s["age"] is not None]
        child_index.append({
            "id": sub, "n_sessions": len(sessions),
            "n_sketches": sum(len(s["sketches"]) for s in sessions),
            "age_min": min(ages) if ages else None, "age_max": max(ages) if ages else None,
            "session_ages": [s["age"] for s in sessions],
            "categories": sorted({r["cat"] for s in sessions for r in s["sketches"]}),
        })

    # ---- per-category files (+ ranked part vocab) ----
    cat_index = []
    parts_by_cat = {}
    for cat in sorted(cat_sketches):
        recs = sorted(cat_sketches[cat], key=lambda r: (r["age"] if r["age"] is not None else 0, r["sub"]))
        with open(os.path.join(OUT_DIR, f"cat_{slug(cat)}.json"), "w") as fh:
            json.dump({"category": cat, "sketches": recs}, fh, separators=(",", ":"))
        ages = cat_ages[cat]
        ranked = sorted(part_counts[cat].items(), key=lambda kv: -kv[1])
        parts_by_cat[cat] = ranked
        cat_index.append({
            "name": cat, "slug": slug(cat), "n": len(recs),
            "n_children": len({r["sub"] for r in recs}),
            "n_annotated": sum(1 for r in recs if r["ann"]),
            "age_min": round(min(ages), 2) if ages else None,
            "age_max": round(max(ages), 2) if ages else None,
            "parts": [p for p, _ in ranked],
        })

    ages_all = [a for al in cat_ages.values() for a in al]
    dates_all = [r["date"] for recs in cat_sketches.values() for r in recs]
    index = {
        "children": sorted(child_index, key=lambda c: -c["n_sessions"]),
        "categories": cat_index,
        "parts": parts_by_cat,
        "meta": {
            "n_children": len(child_index), "n_sketches": n_sketches,
            "n_annotated": n_annot, "n_categories": len(cat_index),
            "age_min": round(min(ages_all), 2) if ages_all else None,
            "age_max": round(max(ages_all), 2) if ages_all else None,
            "date_min": min(dates_all) if dates_all else None,
            "date_max": max(dates_all) if dates_all else None,
        },
    }
    with open(os.path.join(OUT_DIR, "index.json"), "w") as fh:
        json.dump(index, fh, separators=(",", ":"))

    m = index["meta"]
    print(f"children: {m['n_children']}  sketches: {m['n_sketches']}  "
          f"annotated: {m['n_annotated']}  categories: {m['n_categories']}")
    print(f"ages {m['age_min']}-{m['age_max']}  dates {m['date_min']}..{m['date_max']}")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
