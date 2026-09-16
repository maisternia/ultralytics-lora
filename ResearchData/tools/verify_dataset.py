#!/usr/bin/env python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Pre-flight check for a YOLO-format dataset in ResearchData/datasets/.

Validates the things that silently ruin a training run: images with no label file, label files with no image, class ids
that fall outside `names`, coordinates that were never normalized, and degenerate zero-area boxes. Also prints a
per-class instance histogram, which is what you copy into DATASET_CARD.md.

Depends only on the standard library plus PyYAML (already an Ultralytics core dependency).

Usage:
    python ResearchData/tools/verify_dataset.py ResearchData/datasets/<dataset-id>/data.yaml
    python ResearchData/tools/verify_dataset.py <path>/data.yaml --splits train
    python ResearchData/tools/verify_dataset.py <path>/data.yaml --max-report 50

Exits 0 when the dataset is clean, 1 when any error is found.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_SUFFIXES = {".bmp", ".dng", ".heic", ".jpeg", ".jpg", ".mpo", ".pfm", ".png", ".tif", ".tiff", ".webp"}

# Coordinates are compared against [0, 1] with a small tolerance, so a value of 1.0000001 from a float round-trip is not
# reported as an error while a genuinely unnormalized 640 still is.
COORD_TOLERANCE = 1e-6


@dataclass
class SplitReport:
    """
    Findings for a single dataset split.

    Attributes:
        name (str): Split name, such as 'train' or 'val'.
        image_dirs (list[Path]): Image directories that were scanned for this split.
        n_images (int): Number of image files found.
        n_labelled (int): Number of images with a non-empty label file.
        n_background (int): Number of images with an empty label file (valid negatives).
        class_counts (Counter): Instance count per class id.
        errors (list[str]): Problems that make the split unusable.
        warnings (list[str]): Problems worth a look that do not block training.
    """

    name: str
    image_dirs: list[Path] = field(default_factory=list)
    n_images: int = 0
    n_labelled: int = 0
    n_background: int = 0
    class_counts: Counter = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def label_path_for(image_path: Path) -> Path:
    """
    Return the label path Ultralytics would look for, given an image path.

    Mirrors `ultralytics.data.utils.img2label_paths`: the *last* 'images' path segment becomes 'labels' and the suffix
    becomes '.txt'. Reproduced here so a dataset can be checked without importing (or installing) the package.

    Args:
        image_path (Path): Path to an image file.

    Returns:
        (Path): Expected path of the corresponding label file.
    """
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def parse_label_file(path: Path, n_classes: int) -> tuple[Counter, list[str]]:
    """
    Parse one YOLO label file and report per-line problems.

    Accepts detection rows (class + 4 box values) and segmentation rows (class + an even number of polygon values).

    Args:
        path (Path): Label file to parse.
        n_classes (int): Number of classes declared in data.yaml, used to range-check class ids.

    Returns:
        counts (Counter): Instance count per class id found in this file.
        problems (list[str]): Human-readable description of each problem, prefixed with the line number.
    """
    counts: Counter = Counter()
    problems: list[str] = []
    seen: set[str] = set()

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as e:
        return counts, [f"unreadable ({e})"]

    for n, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue

        if line in seen:
            problems.append(f"line {n}: duplicate row")
        seen.add(line)

        tokens = line.split()
        if len(tokens) < 5:
            problems.append(f"line {n}: expected at least 5 fields, got {len(tokens)}")
            continue

        try:
            values = [float(t) for t in tokens]
        except ValueError:
            problems.append(f"line {n}: non-numeric field in {line!r}")
            continue

        cls_f, coords = values[0], values[1:]
        if cls_f != int(cls_f):
            problems.append(f"line {n}: class id {tokens[0]} is not an integer")
            continue
        cls = int(cls_f)
        if not 0 <= cls < n_classes:
            problems.append(f"line {n}: class id {cls} outside declared range 0..{n_classes - 1}")
            continue

        if len(coords) == 4:
            _, _, w, h = coords
            if w <= 0 or h <= 0:
                problems.append(f"line {n}: zero-area box (w={w}, h={h})")
                continue
        elif len(coords) < 6 or len(coords) % 2:
            problems.append(f"line {n}: {len(coords)} coordinates is neither a box (4) nor a polygon (even, >= 6)")
            continue

        out_of_range = [c for c in coords if c < -COORD_TOLERANCE or c > 1 + COORD_TOLERANCE]
        if out_of_range:
            hint = " — looks like pixel coordinates" if max(out_of_range) > 1.5 else ""
            problems.append(f"line {n}: coordinates outside [0, 1]: {out_of_range[:4]}{hint}")
            continue

        counts[cls] += 1

    return counts, problems


def scan_split(name: str, image_dirs: list[Path], n_classes: int, max_report: int) -> SplitReport:
    """
    Scan one split for image/label pairing and label content problems.

    Args:
        name (str): Split name, such as 'train' or 'val'.
        image_dirs (list[Path]): Directories to scan recursively for images.
        n_classes (int): Number of classes declared in data.yaml.
        max_report (int): Maximum number of problems to list per category.

    Returns:
        (SplitReport): Populated report for this split.
    """
    report = SplitReport(name=name, image_dirs=image_dirs)
    missing: list[str] = []
    bad_labels: list[str] = []
    label_files_seen: set[Path] = set()

    for image_dir in image_dirs:
        if not image_dir.is_dir():
            report.errors.append(f"image directory not found: {image_dir}")
            continue

        for image in sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
            report.n_images += 1
            label = label_path_for(image)
            if not label.is_file():
                missing.append(str(image))
                continue

            label_files_seen.add(label.resolve())
            counts, problems = parse_label_file(label, n_classes)
            if counts:
                report.n_labelled += 1
                report.class_counts.update(counts)
            elif not problems:
                report.n_background += 1
            bad_labels.extend(f"{label}: {p}" for p in problems)

    if report.n_images == 0 and not report.errors:
        report.errors.append(f"no images found under {', '.join(str(d) for d in image_dirs)}")

    if missing:
        report.errors.append(f"{len(missing)} image(s) without a label file:")
        report.errors.extend(f"    {m}" for m in missing[:max_report])
        if len(missing) > max_report:
            report.errors.append(f"    ... and {len(missing) - max_report} more")

    if bad_labels:
        report.errors.append(f"{len(bad_labels)} label problem(s):")
        report.errors.extend(f"    {b}" for b in bad_labels[:max_report])
        if len(bad_labels) > max_report:
            report.errors.append(f"    ... and {len(bad_labels) - max_report} more")

    orphans = [
        label
        for image_dir in image_dirs
        for label in label_path_for(image_dir / "x.png").parent.rglob("*.txt")
        if label.resolve() not in label_files_seen
    ]
    if orphans:
        report.warnings.append(f"{len(orphans)} label file(s) with no matching image:")
        report.warnings.extend(f"    {o}" for o in sorted(str(o) for o in orphans)[:max_report])
        if len(orphans) > max_report:
            report.warnings.append(f"    ... and {len(orphans) - max_report} more")

    return report


def resolve_root(cfg: dict, yaml_path: Path) -> Path:
    """
    Resolve the dataset root declared by the `path` key.

    An absolute `path` is used as-is. A relative one is tried against the data.yaml's own directory first, then the
    current working directory, which covers both "self-contained dataset folder" and "path relative to the repo root".

    Args:
        cfg (dict): Parsed data.yaml contents.
        yaml_path (Path): Path to the data.yaml file itself.

    Returns:
        (Path): Resolved dataset root directory.
    """
    raw = cfg.get("path")
    if not raw:
        return yaml_path.parent
    root = Path(raw)
    if root.is_absolute():
        return root
    for candidate in (yaml_path.parent / root, Path.cwd() / root):
        if candidate.is_dir():
            return candidate
    return yaml_path.parent / root


def split_dirs(cfg: dict, root: Path, split: str) -> list[Path]:
    """
    Return the image directories declared for a split.

    Args:
        cfg (dict): Parsed data.yaml contents.
        root (Path): Resolved dataset root.
        split (str): Split key to look up, such as 'train' or 'val'.

    Returns:
        (list[Path]): Image directories for the split, empty if the key is absent.
    """
    entry = cfg.get(split)
    if not entry:
        return []
    entries = entry if isinstance(entry, list) else [entry]
    return [p if (p := Path(e)).is_absolute() else root / e for e in entries]


def main(argv: list[str] | None = None) -> int:
    """
    Run the dataset check and print a report.

    Args:
        argv (list[str], optional): Command-line arguments; defaults to sys.argv[1:].

    Returns:
        (int): 0 if every checked split is clean, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_yaml", type=Path, help="path to the dataset's data.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="splits to check (default: train val)")
    parser.add_argument("--max-report", type=int, default=20, help="max problems listed per category (default: 20)")
    args = parser.parse_args(argv)

    try:
        import yaml
    except ImportError:
        print("error: PyYAML is required — pip install pyyaml", file=sys.stderr)
        return 1

    if not args.data_yaml.is_file():
        print(f"error: no such file: {args.data_yaml}", file=sys.stderr)
        return 1

    cfg = yaml.safe_load(args.data_yaml.read_text(encoding="utf-8")) or {}
    names = cfg.get("names")
    if not names:
        print(f"error: {args.data_yaml} declares no 'names'", file=sys.stderr)
        return 1
    names = dict(enumerate(names)) if isinstance(names, list) else names
    n_classes = max(names) + 1 if names else 0

    root = resolve_root(cfg, args.data_yaml)
    print(f"dataset: {args.data_yaml}")
    print(f"root:    {root}")
    print(f"classes: {n_classes} — {', '.join(f'{k}:{v}' for k, v in sorted(names.items()))}")

    failed = False
    for split in args.splits:
        dirs = split_dirs(cfg, root, split)
        print(f"\n--- {split} ---")
        if not dirs:
            print(f"  not declared in {args.data_yaml.name}, skipped")
            continue
        if any(d.suffix == ".txt" for d in dirs):
            print("  declared as a .txt image list, which this tool does not read — skipped")
            continue

        report = scan_split(split, dirs, n_classes, args.max_report)
        print(f"  images:     {report.n_images}")
        print(f"  labelled:   {report.n_labelled}")
        print(f"  background: {report.n_background}")
        if report.class_counts:
            total = sum(report.class_counts.values())
            print(f"  instances:  {total}")
            for cls in sorted(report.class_counts):
                count = report.class_counts[cls]
                print(f"    {cls} {names.get(cls, '?'):<20} {count:>8}  ({100 * count / total:.1f}%)")

        for warning in report.warnings:
            print(f"  WARN  {warning}")
        for error in report.errors:
            print(f"  ERROR {error}")
        failed |= bool(report.errors)

    print("\nFAILED — fix the errors above before training." if failed else "\nOK — dataset looks consistent.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
