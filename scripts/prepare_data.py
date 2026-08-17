"""
scripts/prepare_data.py — audit and restructure the VIPER image corpus.

Two jobs:

  audit      Walk DATA_DIR, report exactly how many images exist, where, and
             how many of them carry a usable class label.

  restructure Build the layout the training pipeline expects —
                 <root>/train/AI_GENERATED/
                 <root>/train/NON_AI_GENERATED/
             — using **symlinks**, never copies. Nothing is duplicated on disk
             and the originals are left untouched.

Usage:
    python scripts/prepare_data.py audit
    python scripts/prepare_data.py restructure [--labels labels.csv] [--force]

A labels CSV is only needed for flat, unlabelled directories (e.g. the numbered
`test/` split). It must have an id/filename column and a label column; labels
may be 0/1, REAL/AI_GENERATED, or human/ai in any casing.
"""

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import AI_ART_DIR, DATA_DIR, REAL_ART_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# Directory-name conventions that already imply a class, checked case-folded
# against each path component.
AI_DIR_HINTS = {
    "ai_generated", "aiartdata", "ai_art", "ai", "fake", "generated",
    "ai-generated", "aiart",
}
REAL_DIR_HINTS = {
    "non_ai_generated", "realart", "real_art", "real", "human", "nonai",
    "non-ai-generated", "humanart",
}

AI_LABELS = {"1", "ai", "ai_generated", "aigenerated", "fake", "generated"}
REAL_LABELS = {"0", "real", "non_ai_generated", "nonai", "human", "authentic"}


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def infer_label(path: Path) -> Optional[int]:
    """Return 1 (AI), 0 (real), or None when no directory component says."""
    for part in path.parts:
        key = part.lower().replace(" ", "_").replace("-", "_")
        if key in AI_DIR_HINTS:
            return 1
        if key in REAL_DIR_HINTS:
            return 0
    return None


def load_labels_csv(csv_path: Path) -> Dict[str, int]:
    """Map bare filename (and stem) -> label from a two-column CSV."""
    mapping: Dict[str, int] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"[prep] {csv_path} has no header row.")

        fields = {name.lower().strip(): name for name in reader.fieldnames}
        id_field = next(
            (fields[k] for k in ("id", "filename", "file_name", "image", "image_id", "path")
             if k in fields),
            None,
        )
        label_field = next(
            (fields[k] for k in ("label", "class", "target", "y", "is_ai")
             if k in fields),
            None,
        )
        if id_field is None or label_field is None:
            raise SystemExit(
                f"[prep] Could not find id/label columns in {csv_path}. "
                f"Saw: {reader.fieldnames}"
            )

        for row in reader:
            raw_id = (row[id_field] or "").strip()
            raw_label = (row[label_field] or "").strip().lower()
            if not raw_id:
                continue
            if raw_label in AI_LABELS:
                label = 1
            elif raw_label in REAL_LABELS:
                label = 0
            else:
                continue
            name = Path(raw_id).name
            mapping[name] = label
            mapping[Path(name).stem] = label
    return mapping


def audit(root: Path) -> None:
    if not root.exists():
        raise SystemExit(f"[prep] DATA_DIR does not exist: {root}")

    per_dir: Counter = Counter()
    labelled = Counter()
    unlabelled: List[Path] = []
    total = 0

    for path in iter_images(root):
        total += 1
        per_dir[path.parent] += 1
        label = infer_label(path.relative_to(root))
        if label is None:
            unlabelled.append(path)
        else:
            labelled[label] += 1

    print(f"\n{'=' * 68}")
    print(f"DATA AUDIT — {root}")
    print(f"{'=' * 68}")
    print(f"\nTotal image files found: {total:,}\n")

    print("Per-directory breakdown:")
    for directory, count in sorted(per_dir.items(), key=lambda kv: -kv[1]):
        rel = directory.relative_to(root) if directory != root else Path(".")
        hint = infer_label(rel)
        tag = {1: "AI_GENERATED", 0: "REAL", None: "UNLABELLED"}[hint]
        print(f"  {count:>8,}  {str(rel):<48}  [{tag}]")

    print(f"\nLabelled by directory convention: {sum(labelled.values()):,}")
    print(f"    AI_GENERATED : {labelled[1]:,}")
    print(f"    REAL         : {labelled[0]:,}")
    print(f"Unlabelled (no class in path)  : {len(unlabelled):,}")

    if unlabelled:
        print("\n  Unlabelled images cannot be used for supervised training")
        print("  without a labels CSV. Sample:")
        for path in unlabelled[:3]:
            print(f"    {path.relative_to(root)}")

    print(f"\nTraining pipeline expects:")
    print(f"    AI_ART_DIR   = {AI_ART_DIR}   [{'OK' if AI_ART_DIR.exists() else 'MISSING'}]")
    print(f"    REAL_ART_DIR = {REAL_ART_DIR}   [{'OK' if REAL_ART_DIR.exists() else 'MISSING'}]")
    print()


def _link(src: Path, dest_dir: Path, seen: Dict[str, int]) -> bool:
    """Symlink src into dest_dir, de-duplicating colliding basenames."""
    name = src.name
    if name in seen:
        seen[name] += 1
        name = f"{src.stem}__{seen[src.name]}{src.suffix}"
    else:
        seen[name] = 0

    dest = dest_dir / name
    if dest.is_symlink() or dest.exists():
        return False
    dest.symlink_to(os.path.relpath(src.resolve(), dest_dir.resolve()))
    return True


def restructure(
    root: Path,
    labels_csv: Optional[Path],
    force: bool,
    ai_sources: List[Path],
    real_sources: List[Path],
) -> None:
    ai_dir, real_dir = AI_ART_DIR, REAL_ART_DIR

    for target in (ai_dir, real_dir):
        if target.exists() and any(target.iterdir()) and not force:
            raise SystemExit(
                f"[prep] {target} already exists and is non-empty. "
                f"Re-run with --force to add to it."
            )
        target.mkdir(parents=True, exist_ok=True)

    label_map = load_labels_csv(labels_csv) if labels_csv else {}
    if labels_csv:
        print(f"[prep] Loaded {len(label_map) // 2:,} labels from {labels_csv}")

    # Explicit directory->class assignments always win over the name heuristic
    # and over the labels CSV. Resolved so parent checks work on real paths.
    explicit: List[Tuple[Path, int]] = (
        [(p.resolve(), 1) for p in ai_sources] + [(p.resolve(), 0) for p in real_sources]
    )
    for source, label in explicit:
        if not source.exists():
            raise SystemExit(f"[prep] Declared source does not exist: {source}")
        print(f"[prep] Declared {'AI_GENERATED' if label else 'REAL'}: {source}")

    def explicit_label(path: Path) -> Optional[int]:
        resolved = path.resolve()
        for source, label in explicit:
            if source == resolved or source in resolved.parents:
                return label
        return None

    counts = Counter()
    seen_ai: Dict[str, int] = {}
    seen_real: Dict[str, int] = {}

    for path in iter_images(root):
        # Never re-link images that already live in the target layout.
        if ai_dir in path.parents or real_dir in path.parents:
            continue

        label = explicit_label(path)
        if label is None and not explicit:
            label = infer_label(path.relative_to(root))
        if label is None:
            label = label_map.get(path.name, label_map.get(path.stem))
        if label is None:
            counts["skipped"] += 1
            continue

        dest_dir, seen = (ai_dir, seen_ai) if label == 1 else (real_dir, seen_real)
        if _link(path, dest_dir, seen):
            counts["ai" if label == 1 else "real"] += 1

    print(f"\n[prep] Symlinked {counts['ai']:,} -> {ai_dir}")
    print(f"[prep] Symlinked {counts['real']:,} -> {real_dir}")
    print(f"[prep] Skipped {counts['skipped']:,} unlabelled images")
    total = counts["ai"] + counts["real"]
    print(f"[prep] Trainable corpus: {total:,} images (0 bytes copied)\n")

    if total == 0:
        raise SystemExit("[prep] Nothing was linked — training cannot proceed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "restructure"])
    parser.add_argument("--root", type=Path, default=DATA_DIR)
    parser.add_argument("--labels", type=Path, default=None,
                        help="CSV mapping filenames to labels, for flat directories.")
    parser.add_argument("--force", action="store_true",
                        help="Link into existing non-empty target directories.")
    parser.add_argument("--ai-dir", type=Path, action="append", default=[],
                        help="Directory whose images are all AI_GENERATED. Repeatable.")
    parser.add_argument("--real-dir", type=Path, action="append", default=[],
                        help="Directory whose images are all REAL. Repeatable.")
    args = parser.parse_args()

    if args.command == "audit":
        audit(args.root)
    else:
        restructure(args.root, args.labels, args.force, args.ai_dir, args.real_dir)


if __name__ == "__main__":
    main()
