#!/usr/bin/env python3
"""Explicit online preparation: download public model weights, never case media.

Run this before moving to the offline competition environment. The inference
application does not import or call this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


MODELS = {
    "qwen3-vl-2b": ("Qwen/Qwen3-VL-2B-Instruct", "models/Qwen3-VL-2B-Instruct"),
    "dinov2-small": ("facebook/dinov2-small", "models/dinov2-small"),
}


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="qwen3-vl-2b")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--revision", default="main", help="A branch or commit; its resolved commit is recorded.")
    args = parser.parse_args()
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface-hub before downloading public weights.") from exc
    # token=False avoids sending a cached account token for this public model.
    model_id, default_output = MODELS[args.model]
    info = HfApi().model_info(model_id, revision=args.revision, token=False)
    license_name = getattr(info.card_data, "license", None)
    if license_name != "apache-2.0":
        raise SystemExit("The model card did not confirm Apache-2.0. Review its current license manually.")
    if not info.sha:
        raise SystemExit("Could not resolve an immutable model revision.")
    output = (args.output or Path(default_output)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id, revision=info.sha, local_dir=str(output), token=False,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE*", "README.md"],
        max_workers=4,
    )
    files = {}
    for path in sorted(output.rglob("*")):
        relative = path.relative_to(output)
        if not path.is_file() or any(part.startswith(".") for part in relative.parts):
            continue
        if path.name == "cashwatch_model_manifest.json":
            continue
        files[str(relative)] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    if not any(name.endswith(".safetensors") for name in files):
        raise SystemExit("Download did not produce weight files.")
    manifest = {
        "model_id": model_id, "resolved_revision": info.sha, "license": license_name,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "files": files,
        "bank_case_media_processed": False,
    }
    (output / "cashwatch_model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"downloaded": True, "path": str(output), "model_id": model_id,
                      "revision": info.sha, "license": license_name}, indent=2))


if __name__ == "__main__":
    main()
