# Children's Drawing Explorer

An interactive web page for exploring **37,770 children's drawings** (48
categories, ages 2–10) in CLIP embedding space, colored by model
recognizability.

🔗 **Live page:** https://vislearnlab.github.io/drawing-explorer/

![overview](_overview.png)

## What it shows

- Drawings laid out by a 2-D **t-SNE** of the 48-d CLIP per-category probability
  vectors. Each point is one drawing.
- **Color by:** CLIP recognizability (probability of the intended category,
  default), correct/incorrect, log-odds, age, or category.
- **Filters:** category, age range, minimum recognizability, correct-only.
- **Hover** a point to see the drawing + its scores; **click** to pin it.
- The footer stat bar updates with the count / % CLIP-correct / mean
  recognizability of the currently visible subset.

## Run locally

```bash
python3 -m http.server 8000
# open http://127.0.0.1:8000/
```

(`index.html` is at the repo root; drawing PNGs live in `drawings/`.)

## Files

- `index.html` — the self-contained explorer (no dependencies, no build step).
- `points.json` — t-SNE layout + per-drawing CLIP scores consumed by the page.
- `drawings/` — the 37,770 drawing PNGs (150×150).
- `build_data.py` — regenerates `points.json` from the source CLIP embeddings
  and recognizability tables (see below).

## Rebuilding `points.json`

`build_data.py` expects the source data from the
[`drawing_production_and_recognition`](https://github.com/cogtoolslab/drawing_production_and_recognition)
repository (CLIP feature `.npy`, metadata, and `merged_clip_class_and_meta.csv`).
Point the paths at that checkout and run:

```bash
python3 build_data.py
```

## Data & paper

Drawings and recognizability scores come from:

> Long, B. et al. *Parallel developmental changes in children's production and
> recognition of line drawings of visual concepts.*
> [PsyArXiv](https://psyarxiv.com/5yv7x/) · [OSF](https://osf.io/qymjr/)

## License

Drawings and data: CC BY-NC-SA 4.0 (per the source dataset).
