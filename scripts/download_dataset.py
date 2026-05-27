"""Download nlqtsbench/ts_data/ CSVs from the HuggingFace dataset."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TARGET_DIR = _REPO_ROOT / "nlqtsbench" / "ts_data"

HF_REPO_ID   = "mrtan/NLQTSBench"
HF_REPO_TYPE = "dataset"
HF_PATTERNS  = ["ts_data/**"]


def main() -> None:
    p = argparse.ArgumentParser(
        prog="download_dataset",
        description="Pull nlqtsbench/ts_data/ from huggingface.co/"
                    f"datasets/{HF_REPO_ID}.")
    p.add_argument("--rebuild", action="store_true",
                   help="Delete any existing local ts_data/ before download.")
    args = p.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is required. Install with:\n"
            "    pip install huggingface_hub"
        )

    if args.rebuild and _TARGET_DIR.exists():
        print(f"Removing existing {_TARGET_DIR} …")
        shutil.rmtree(_TARGET_DIR)
    if _TARGET_DIR.exists() and any(_TARGET_DIR.iterdir()):
        n = sum(1 for _ in _TARGET_DIR.glob("*.csv"))
        print(f"{_TARGET_DIR} already populated ({n} CSVs). "
              f"Use --rebuild to re-download.")
        return

    _TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HF_REPO_ID} → {_TARGET_DIR} …")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        allow_patterns=HF_PATTERNS,
        local_dir=str(_REPO_ROOT / "nlqtsbench"),
        local_dir_use_symlinks=False,
    )
    n = sum(1 for _ in _TARGET_DIR.glob("*.csv"))
    print(f"\nDone — {n} CSVs in {_TARGET_DIR}.")
    print("Next: run  python -m scripts.load_benchmark  to build SQLite dbs.")


if __name__ == "__main__":
    main()
