# Dataset: TODO-dataset-id

One paragraph: what this dataset is for, and what question it was built to answer.

## Provenance

| Field | Value |
| --- | --- |
| Source captures | TODO — `raw/...` folders |
| Rendering recipe | TODO — `spectrograms/...` folder |
| Renderer commit | TODO — short SHA |
| Built on | TODO — YYYY-MM-DD |
| Built by | TODO |

## Contents

| Split | Images | Instances | Capture sessions |
| --- | --- | --- | --- |
| train | TODO | TODO | TODO |
| val | TODO | TODO | TODO |

Fill these from `python ResearchData/tools/verify_dataset.py <path>/data.yaml`, which prints per-split image counts and
a per-class instance histogram.

## Classes

| id | name | Definition — what counts as this class, and what does not |
| --- | --- | --- |
| 0 | lora_signal | TODO |

Be precise here. "Where does the box end when the chirp runs off the edge of the frame?" is the kind of question that
costs a week of confused results if two labelling sessions answered it differently.

## Split rule

TODO — e.g. "sessions 1–4 train, session 5 val; no session appears in both."

Random frame-level splits leak: consecutive spectrogram frames from one capture are near-duplicates, so a shuffled split
scores high and generalizes badly. If you did split randomly, say so — the val number is then a training-set number.

## Labelling

- **Tool**: TODO
- **Annotator(s)**: TODO
- **Guidelines**: TODO — tight boxes vs. padded, treatment of overlapping transmissions, minimum visible chirp length
- **Review**: TODO — was any split double-checked, and by whom

## Known issues

- TODO — class imbalance, mislabelled ranges, frames with clipping, anything a future reader should distrust.
