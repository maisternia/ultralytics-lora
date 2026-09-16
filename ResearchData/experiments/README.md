# experiments/

One folder per training run: `EXP<nnn>_<slug>`, e.g. `EXP007_height-loss-w2`.

```bash
cp -r ResearchData/experiments/_template ResearchData/experiments/EXP007_height-loss-w2
```

Each folder holds the config that went in and the numbers that came out:

```
EXP007_height-loss-w2/
├── EXPERIMENT.md      # hypothesis, setup, results, conclusion
├── hyper_param.yaml   # the exact config passed to model.train()
└── results.csv        # copied out of runs/detect/<name>/ after the run
```

Weights and plots stay in `runs/` (gitignored). Copy out only `results.csv` and any figure you actually reference — the
point is that `EXPERIMENT.md` plus `hyper_param.yaml` is enough to re-run and to argue about the outcome.

## Rules that pay for themselves

**Write the hypothesis before the run.** A prediction recorded afterwards is a description, and it will always sound
like it was expected. The template puts the hypothesis above the results for exactly this reason.

**Change one thing.** If EXP007 changes the loss weight *and* the image size, it cannot tell you which one moved mAP.
Reference the baseline experiment id explicitly in `EXPERIMENT.md`.

**Record failures.** A run that made things worse is a result, and it is the one you are most likely to accidentally
repeat. Keep the folder, set the conclusion to "rejected", and say why.

**Pin what varies.** `seed: 0`, `deterministic: true` and `workers: 0` are already in the template. Without them, a
2-point mAP difference between two runs may be nothing but run-to-run noise.

## Numbering

Highest existing number plus one. Never reuse or renumber — notebooks, notes and chat logs point at these ids, and a
recycled `EXP007` makes every one of those references wrong.
