# spectrograms/

Images rendered from `raw/`, ready to be annotated and assembled into a dataset.

One subfolder per **rendering recipe**, not per capture — the same capture rendered two ways is two recipes, and that
distinction is usually what explains a surprising mAP delta. Name recipes `<nfft>-<hop>-<window>-<scale>`, e.g.
`1024-256-hann-db`.

Each recipe folder carries a `README.md` pinning down the parameters:

```markdown
# 1024-256-hann-db

- **Source captures**: raw/usrp-b210_20260914, raw/gnuradio-sim_20260901
- **STFT**: n_fft 1024, hop 256, Hann window
- **Scale**: power in dB, clipped to [-90, -30] dBFS, then linearly mapped to [0, 255]
- **Channels**: magnitude replicated to 3 channels (R=G=B)
- **Output**: 1024x1024 PNG, one image per 0.26 s of signal
- **Script**: YOLO-LoRa/convert.py @ a1b2c3d
```

Two things worth being strict about:

- **Record the clip range.** Per-image min/max normalization makes every image look good and quietly destroys absolute
  power information across the dataset. If you use it, say so here — it changes what the model can learn.
- **Record the script commit.** "The renderer" changes; `a1b2c3d` does not.

Image files are gitignored. Recipe READMEs are not.
