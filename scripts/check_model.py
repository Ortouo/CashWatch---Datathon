#!/usr/bin/env python3
"""Offline model verification; --smoke uses only an artificial blank image."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cashwatch.vision.contracts import EvidenceFrame, VisionError
from cashwatch.vision.model import LocalQwenVisionModel, inspect_model_directory
from cashwatch.vision.parsing import parse_observations


def verify_hashes(model_dir: Path) -> int:
    manifest_file = model_dir / "cashwatch_model_manifest.json"
    if not manifest_file.is_file():
        raise ValueError("No SHA-256 provenance manifest. Use model_download.py or omit --verify-hashes.")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    checked = 0
    for relative, expected in manifest["files"].items():
        path = (model_dir / relative).resolve()
        if not path.is_relative_to(model_dir.resolve()) or not path.is_file():
            raise ValueError("Manifest file path is invalid or missing.")
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                checksum.update(chunk)
        if path.stat().st_size != expected["bytes"] or checksum.hexdigest() != expected["sha256"]:
            raise ValueError("A local model file failed integrity verification.")
        checked += 1
    if not checked:
        raise ValueError("Manifest contains no files.")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--load", action="store_true", help="Actually initialize the local model.")
    parser.add_argument("--smoke", action="store_true", help="Load and run real generation on an artificial image.")
    parser.add_argument("--seconds", type=float, default=120.0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    try:
        info = inspect_model_directory(args.model_dir)
        info["structural_check"] = "passed"
        info["weights_loaded"] = False
        info["real_generation_tested"] = False
        if args.verify_hashes:
            info["files_hash_verified"] = verify_hashes(args.model_dir)
        if args.load or args.smoke:
            started = time.monotonic()
            model = LocalQwenVisionModel(args.model_dir, device=args.device, max_new_tokens=512)
            model.load()
            info["weights_loaded"] = True
            info["device"] = model.device
            info["load_seconds"] = round(time.monotonic() - started, 3)
            if args.smoke:
                from PIL import Image
                with tempfile.TemporaryDirectory(prefix="cashwatch-model-smoke-") as temporary:
                    path = Path(temporary) / "synthetic_blank.jpg"
                    Image.new("RGB", (320, 240), color=(125, 125, 125)).save(path, format="JPEG")
                    frame = EvidenceFrame("f000000", 0.0, path)
                    raw = model.observe((frame,), (), deadline_monotonic=time.monotonic() + args.seconds)
                    observations = parse_observations(raw, (frame,), (), window_id="synthetic")
                    info["real_generation_tested"] = True
                    info["synthetic_observation_count"] = len(observations)
                    info["real_competition_video_tested"] = False
        print(json.dumps(info, ensure_ascii=False, indent=2))
    except (VisionError, ValueError, OSError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
