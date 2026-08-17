"""
Hugging Face dataset acquisition.

Downloads a dataset repo to /data/raw/<name>/ and writes ACQUIRED.json on
success so reruns skip finished datasets. Every dataset is tagged train or
eval at download time; canonicalization asserts that no eval-tagged source reaches
a training manifest.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download  # noqa: E402

RAW = Path("/data/raw")
LOGS = Path("/data/logs")

# Registry: every entry must be backed by a peer-reviewed / archival publication.
REGISTRY = {
    "community_forensics_small": {
        "repo_id": "OwensLab/CommunityForensics-Small",
        "repo_type": "dataset",
        "tag": "train",
        "citation": "Park & Owens, Community Forensics, CVPR 2025 (arXiv:2411.04125)",
        "license": "cc-by-nc-sa-4.0 (non-commercial research only)",
    },
    "community_forensics_eval": {
        "repo_id": "OwensLab/CommunityForensics-Eval",
        "repo_type": "dataset",
        "tag": "eval",
        "citation": "Park & Owens, Community Forensics, CVPR 2025 (arXiv:2411.04125)",
        "license": "cc-by-nc-sa-4.0 (non-commercial research only)",
    },
}


def acquire(name: str, allow_patterns=None, max_files=None) -> dict:
    if name not in REGISTRY:
        raise SystemExit(f"[acquire] Unknown dataset '{name}'. Known: {sorted(REGISTRY)}")

    spec = REGISTRY[name]
    dest = RAW / name
    marker = dest / "ACQUIRED.json"

    if marker.exists():
        rec = json.loads(marker.read_text())
        print(f"[acquire] {name}: already acquired ({rec['n_files']} files, "
              f"{rec['bytes'] / 1e9:.1f} GB) — skipping.")
        return rec

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[acquire] {name}: downloading {spec['repo_id']} -> {dest}")
    print(f"[acquire]   tag={spec['tag']}  license={spec['license']}")

    t0 = time.time()
    failed = []
    try:
        snapshot_download(
            repo_id=spec["repo_id"],
            repo_type=spec["repo_type"],
            local_dir=str(dest),
            allow_patterns=allow_patterns,
            max_workers=16,
        )
    except Exception as exc:
        # Partial downloads stay on disk; rerunning resumes rather than restarts.
        failed.append(str(exc))
        print(f"[acquire] {name}: ERROR {exc}", file=sys.stderr)

    elapsed = time.time() - t0
    files = [p for p in dest.rglob("*") if p.is_file() and p.name != "ACQUIRED.json"]
    total = sum(p.stat().st_size for p in files)

    rec = {
        "dataset": name,
        "repo_id": spec["repo_id"],
        "tag": spec["tag"],
        "citation": spec["citation"],
        "license": spec["license"],
        "n_files": len(files),
        "bytes": total,
        "gb": round(total / 1e9, 2),
        "wall_seconds": round(elapsed, 1),
        "mb_per_s": round(total / 1e6 / max(elapsed, 1e-9), 1),
        "failed": failed,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"acquire_{name}.json").write_text(json.dumps(rec, indent=2))

    if not failed:
        marker.write_text(json.dumps(rec, indent=2))
        print(f"[acquire] {name}: DONE {rec['n_files']} files, {rec['gb']} GB, "
              f"{elapsed / 60:.1f} min, {rec['mb_per_s']} MB/s")
    else:
        print(f"[acquire] {name}: INCOMPLETE — rerun to resume.")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--allow", nargs="*", default=None,
                    help="glob patterns to restrict the download")
    args = ap.parse_args()
    acquire(args.name, allow_patterns=args.allow)
