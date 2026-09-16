# EXP<nnn>_<slug>

| Field | Value |
| --- | --- |
| Date | TODO — YYYY-MM-DD |
| Baseline | TODO — the experiment id this is compared against |
| Dataset | TODO — `ResearchData/datasets/<dataset-id>` |
| Base weights | TODO — e.g. `yolo11n.pt`, or the best.pt of an earlier experiment |
| Code commit | TODO — short SHA |
| Hardware | TODO — e.g. M3 Max / mps, or 1x A100 |

## Hypothesis

*Write this before launching the run.*

TODO — what you expect to change, in which direction, and roughly by how much. State the mechanism: "weighting the
height term should tighten bandwidth estimates on narrow SF12 chirps, which the baseline consistently over-boxes."

## What changed vs. the baseline

TODO — ideally one line. If it is more than one line, you will not be able to attribute the result.

```diff
- ph: 0.0
+ ph: 0.2
```

## Results

| Metric | Baseline | This run | Δ |
| --- | --- | --- | --- |
| mAP50 | TODO | TODO | TODO |
| mAP50-95 | TODO | TODO | TODO |
| precision | TODO | TODO | TODO |
| recall | TODO | TODO | TODO |
| epochs to best | TODO | TODO | TODO |

## Observations

TODO — training curve shape, where it plateaued, classes that moved, failure modes in the predictions. Note anything
that looked wrong even if the headline metric improved.

## Conclusion

**TODO: confirmed / rejected / inconclusive**

TODO — one paragraph. If inconclusive, say what would settle it; if rejected, say so plainly and keep the folder. A
recorded negative result is the cheapest way to avoid running the same idea again in three months.

## Follow-ups

- [ ] TODO
