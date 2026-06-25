#!/usr/bin/env python3
"""Build the data file for the drawing explorer.

Loads the 48-d CLIP per-category probability vectors, joins each drawing to its
CLIP recognizability scores, computes a 2-D t-SNE layout, and writes a compact
JSON (parallel arrays) consumed by index.html.

Run from the repo root:  python3 explorer/build_data.py
"""
import csv
import json
import os
import time

import numpy as np
from sklearn.manifold import TSNE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRAW_REL = "data/drawings/stringent_cleaned_dataset_meta/stringent_cleaned_dataset"
NPY = os.path.join(ROOT, "data/clip_outputs/fullset/CLIP_FEATURES_kid_museumstation.npy")
META = os.path.join(ROOT, "data/clip_outputs/fullset/CLIP_METADATA_kid.csv")
MERGED = os.path.join(ROOT, "data/preprocessed_data/merged_clip_class_and_meta.csv")
RECOG_DIR = os.path.join(ROOT, "data/recognition_data/behavioral_data")
RECOG_GAMES = ["animalgame", "biganimalgame", "objectgame", "vehiclegame"]
OUT = os.path.join(ROOT, "explorer", "points.json")


def num(v):
    try:
        f = float(v)
        return None if (f != f) else f  # drop NaN
    except (TypeError, ValueError):
        return None


def load_recognition():
    """Per-drawing kid recognition from the 4 recognition games.

    Returns {basename: {"kid_c","kid_n"}} counting trials by child recognizers
    (recognizer_age != 'adult'); adult trials are dropped (too few per drawing
    to report reliably). Keyed by PNG basename, matching the explorer filenames.
    """
    from collections import defaultdict
    agg = defaultdict(lambda: {"kid_c": 0, "kid_n": 0})
    for g in RECOG_GAMES:
        path = os.path.join(RECOG_DIR, g + ".csv")
        for r in csv.DictReader(open(path)):
            if r["producer_age"] == "photo":
                continue  # skip photo-recognition trials; sketches only
            if r["recognizer_age"] == "adult":
                continue  # child recognizers only
            base = os.path.basename(r["sketch_path"].strip().strip("[]u'\""))
            a = agg[base]
            a["kid_n"] += 1
            a["kid_c"] += 1 if r["clicked_category"] == r["intended_category"] else 0
    return agg


# label shown in the UI -> recognition game file
GAME_LABELS = {
    "animalgame": "Animals", "biganimalgame": "Big animals",
    "objectgame": "Objects", "vehiclegame": "Vehicles",
}


def load_game_categories():
    """{game label: set(category)} of the categories tested in each game."""
    out = {}
    for g, label in GAME_LABELS.items():
        cats = set()
        for r in csv.DictReader(open(os.path.join(RECOG_DIR, g + ".csv"))):
            if r["producer_age"] != "photo":
                cats.add(r["intended_category"])
        out[label] = cats
    return out


def main():
    t0 = time.time()
    feats = np.load(NPY).astype(np.float32)            # (37770, 48)
    meta = list(csv.DictReader(open(META)))
    assert len(meta) == feats.shape[0], (len(meta), feats.shape)

    merged = {r["filename"]: r for r in csv.DictReader(open(MERGED))}
    recog = load_recognition()
    game_cats = load_game_categories()

    def fname(r):
        return f"{r['label']}_sketch_age{r['age']}_cdm_{r['session']}.png"

    # category vocabulary (sorted for stable indices)
    cats = sorted({r["label"] for r in meta})
    cat_idx = {c: i for i, c in enumerate(cats)}

    rows, keep = [], []
    for i, m in enumerate(meta):
        f = fname(m)
        mr = merged.get(f)
        if mr is None:
            continue
        keep.append(i)
        rows.append((f, m, mr))
    feats = feats[keep]
    print(f"joined {len(rows)} drawings in {time.time()-t0:.1f}s; running t-SNE on {feats.shape}...")

    # standardize features then t-SNE to 2-D
    fz = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    t1 = time.time()
    xy = TSNE(
        n_components=2, perplexity=30, init="pca",
        learning_rate="auto" if "auto" in TSNE.__init__.__code__.co_varnames else 200.0,
        random_state=0, verbose=1,
    ).fit_transform(fz)
    print(f"t-SNE done in {time.time()-t1:.1f}s")

    # normalize layout to [0, 1000] for compact ints
    xy -= xy.min(0)
    xy /= xy.max(0).max()
    xy *= 1000.0
    xy = np.round(xy, 1)

    def recog_fields(files):
        # child recognizers only; adult trials were too sparse per-drawing to report
        kid, kid_n = [], []
        for f in files:
            a = recog.get(f)
            if a and a["kid_n"]:
                kid.append(round(a["kid_c"] / a["kid_n"], 3)); kid_n.append(a["kid_n"])
            else:
                kid.append(None); kid_n.append(0)
        return kid, kid_n

    files = [r[0] for r in rows]
    kid_recog, kid_recog_n = recog_fields(files)
    print(f"kid-recognition coverage: {sum(1 for v in kid_recog if v is not None)} drawings")

    out = {
        "draw_dir": DRAW_REL,
        "categories": cats,
        "n": len(rows),
        # parallel arrays
        "file": files,
        "x": xy[:, 0].tolist(),
        "y": xy[:, 1].tolist(),
        "cat": [cat_idx[r[1]["label"]] for r in rows],
        "age": [int(r[1]["age"]) for r in rows],
        "correct": [1 if (num(r[2]["correct_or_not"]) or 0) >= 0.5 else 0 for r in rows],
        "target_prob": [round(num(r[2]["target_label_prob"]) or 0.0, 4) for r in rows],
        "log_odds": [round(num(r[2]["log_odds"]) or 0.0, 3) for r in rows],
        "max_prob": [round(num(r[2]["max_prob"]) or 0.0, 4) for r in rows],
        "guess": [cat_idx.get(r[2]["clip_category"], -1) for r in rows],
        "strokes": [num(r[2]["num_strokes"]) for r in rows],
        "duration": [round(num(r[2]["draw_duration"]) or 0.0, 1) for r in rows],
        "freq": [round(num(r[2]["drawing_frequency"]) or 0.0, 2) for r in rows],
        # kid recognition (subset of drawings used in the 4 recognition games)
        "kid_recog": kid_recog,       # prop. correct by child recognizers (null if none)
        "kid_recog_n": kid_recog_n,   # number of child-recognizer trials
        # quick-select category groups (the 4 recognition games)
        "groups": {label: sorted(cat_idx[c] for c in cats if c in cat_idx)
                   for label, cats in game_cats.items()},
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
