# datasets/

YOLO-format datasets assembled from `../spectrograms/`. One folder per dataset, named
`<source>_sf<sf>_bw<bw>_<date>` (e.g. `synth_sf7_bw125_20260916`).

## Starting a new dataset

```bash
cp -r ResearchData/datasets/_template ResearchData/datasets/synth_sf7_bw125_20260916
```

Then fill in `data.yaml` and `DATASET_CARD.md`, and lay the files out like this:

```
synth_sf7_bw125_20260916/
├── data.yaml
├── DATASET_CARD.md
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

The `images/` ↔ `labels/` split is not cosmetic: Ultralytics finds a label by swapping the last `images` path segment
for `labels` and the extension for `.txt`. A folder called `imgs/` or `annotations/` will silently yield a dataset with
zero labels.

## Before training

```bash
python ResearchData/tools/verify_dataset.py ResearchData/datasets/synth_sf7_bw125_20260916/data.yaml
```

This checks image/label pairing, class ids against `names`, coordinate ranges and degenerate boxes, and prints a
per-class instance count. Run it every time you regenerate labels.

## Splitting

Split by **capture session**, not by random shuffle. Consecutive spectrogram frames from one capture overlap heavily in
content; a random split puts near-duplicates on both sides and inflates val mAP by a wide margin. Note the split rule in
`DATASET_CARD.md` so the number means something.

Dataset images and labels are gitignored — `data.yaml` and `DATASET_CARD.md` are not.
