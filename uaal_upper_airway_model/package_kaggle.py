#!/usr/bin/env python3
"""Bundle the prepared dataset and the training code into one zip for a Kaggle dataset upload.

    python package_kaggle.py            -> dist/uaal_det_kaggle.zip

Zip layout (Kaggle extracts it to /kaggle/input/<dataset-slug>/):
    uaal_det/   data.yaml, data_clinical.yaml, data_phantom.yaml, images/, labels/, lists/,
                manifest.csv, dropped.csv, config_used.json, reports/
    code/       train.py, preprocessing.py, prepare_data.py, README.md, requirements.txt
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_ITEMS = ["data.yaml", "data_clinical.yaml", "data_phantom.yaml", "images", "labels", "lists",
              "manifest.csv", "dropped.csv", "config_used.json", "reports"]
CODE_ITEMS = ["train.py", "preprocessing.py", "prepare_data.py", "README.md", "requirements.txt"]


def add(z: zipfile.ZipFile, src: Path, arc: str) -> int:
    files = sorted(p for p in src.rglob("*") if p.is_file()) if src.is_dir() else [src]
    for f in files:
        name = arc if f == src else f"{arc}/{f.relative_to(src).as_posix()}"
        # PNGs are already compressed: store them, deflate the text files
        z.write(f, name, compress_type=zipfile.ZIP_STORED if f.suffix.lower() == ".png" else zipfile.ZIP_DEFLATED)
    return len(files)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=HERE / "data" / "uaal_det")
    p.add_argument("--out", type=Path, default=HERE / "dist" / "uaal_det_kaggle.zip")
    args = p.parse_args()

    missing = [i for i in DATA_ITEMS if not (args.data_dir / i).exists()] + [i for i in CODE_ITEMS if not (HERE / i).exists()]
    if missing:
        raise SystemExit(f"missing: {missing} - run prepare_data.py first")
    if "ERRORS (0)" not in (args.data_dir / "reports" / "verify.txt").read_text():
        raise SystemExit("dataset did not pass verification - run prepare_data.py verify")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".zip.part")
    n = 0
    with zipfile.ZipFile(tmp, "w") as z:
        for item in DATA_ITEMS:
            n += add(z, args.data_dir / item, f"uaal_det/{item}")
        for item in CODE_ITEMS:
            n += add(z, HERE / item, f"code/{item}")
    with zipfile.ZipFile(tmp) as z:  # integrity check before replacing the old zip
        bad = z.testzip()
        if bad:
            raise SystemExit(f"corrupt entry in zip: {bad}")
    tmp.replace(args.out)
    print(f"{args.out}  ({n} files, {args.out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
