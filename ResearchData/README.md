# ResearchData

Data-side companion to the LoRa signal-detection experiments in this fork. Everything a run *consumes* (captures,
spectrograms, YOLO datasets) and everything a run *claims* (experiment records, metrics, provenance) lives here.

This folder is **research scaffolding, not part of the `ultralytics` package**. Nothing in `ultralytics/` imports from
it, and nothing here is published to PyPI.

## Layout

```
ResearchData/
├── raw/                    # Untouched captures as they came off the radio / simulator
├── spectrograms/           # Rendered images, one subfolder per rendering recipe
├── datasets/               # YOLO-format datasets assembled from spectrograms/
│   └── _template/          # Copy this to start a new dataset
├── experiments/            # One folder per training run: config in, numbers out
│   └── _template/          # Copy this to start a new experiment
└── tools/                  # Small utilities for keeping the above honest
    └── verify_dataset.py   # Pre-flight check for a YOLO dataset
```

The four data folders are tracked but effectively empty: `ResearchData/.gitignore` keeps captures, images and archives
out of git while letting every `README.md`, `*.yaml`, `*.csv` and `*.json` through. That is deliberate — the repo should
describe the data, not carry it.

## Naming conventions

Consistent names are what make results comparable six months from now.

| Thing | Pattern | Example |
| --- | --- | --- |
| Capture set | `<source>_<date>` | `usrp-b210_20260914` |
| Spectrogram recipe | `<nfft>-<hop>-<window>-<scale>` | `1024-256-hann-db` |
| Dataset | `<source>_sf<sf>_bw<bw>_<date>` | `synth_sf7_bw125_20260916` |
| Experiment | `EXP<nnn>_<slug>` | `EXP007_height-loss-w2` |

Pick the number for an experiment by taking the highest existing one and adding 1. Never renumber or reuse a retired
experiment folder — a stale reference in a notebook pointing at the wrong numbers is worse than a gap in the sequence.

## Workflow

1. **Land a capture.** Drop it in `raw/<source>_<date>/` and write the `README.md` there (hardware, SF, BW, coding rate,
   gain, duration, anything you would otherwise forget).
2. **Render spectrograms.** Output to `spectrograms/<recipe>/`, and record the exact parameters in that folder's
   `README.md`. If two experiments disagree, the rendering recipe is the first suspect — so write it down.
3. **Assemble a dataset.** Copy `datasets/_template/` to `datasets/<dataset-id>/`, fill in `data.yaml` and
   `DATASET_CARD.md`, then lay out `images/{train,val}` and `labels/{train,val}`.
4. **Verify before you train.** `python ResearchData/tools/verify_dataset.py ResearchData/datasets/<dataset-id>/data.yaml`
   catches unpaired labels, out-of-range class ids and denormalized boxes in seconds. A training run that dies on
   epoch 40 because of a bad label costs a lot more.
5. **Run the experiment.** Copy `experiments/_template/` to `experiments/EXP<nnn>_<slug>/`, keep `hyper_param.yaml`
   next to the record, and fill in `EXPERIMENT.md` as you go — hypothesis *before* the run, results after.

## What does not belong here

- Model weights (`*.pt`) — already ignored repo-wide, and they are outputs, not data.
- Anything you cannot regenerate or re-download. If a capture is irreplaceable, back it up outside git; this folder is
  a description of your data, not its only copy.
- Code that the main package needs. Custom datasets, losses and trainers belong in `ultralytics/` or in the `YOLO-LoRa/`
  experiment directory on `lora-dev`, not in `ResearchData/tools/`.
