#!/usr/bin/env python3
"""Validate and install a measured MoE fragment; never touches manifest.json."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--slug", default="glm-5.3-flash-nvfp4")
    ap.add_argument("--public-root", default="public/models")
    args = ap.parse_args()
    src = Path(args.src)
    payload = json.loads(src.read_text(encoding="utf-8"))
    required = {"schema", "checkpoint", "dataset", "mode", "domains", "layers"}
    missing = required - payload.keys()
    if missing or not payload.get("layers"):
        raise ValueError(f"invalid MoE fragment, missing={sorted(missing)}, layers={len(payload.get('layers', []))}")
    if payload["mode"] != "reap" or not payload.get("crosscheck"):
        raise ValueError("Atlas publication requires exact REAP mode and replay crosscheck evidence")
    dest = Path(args.public_root) / args.slug / "moe.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
