#!/usr/bin/env python3
"""Experimental local DINOv2 features. Not used by the default jury analyzer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cashwatch.vision.video import sample_video


def sha256_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def pooled_windows(embeddings, timestamps, duration, window_seconds, stride_seconds):
    """Order-aware frozen features: mean, std, last-minus-first, absolute change."""
    import numpy as np
    centers = list(np.arange(0.0, duration, stride_seconds, dtype=np.float64))
    if duration - centers[-1] > stride_seconds / 2:
        centers.append(float(duration))
    rows, kept_centers, starts, ends = [], [], [], []
    for center in centers:
        start, end = max(0.0, center - window_seconds / 2), min(duration, center + window_seconds / 2)
        selected = embeddings[(timestamps >= start) & (timestamps <= end)]
        if len(selected) == 0:
            raise ValueError("A feature window contains no frames; increase sampling or window size.")
        absolute_change = np.abs(np.diff(selected, axis=0)).mean(axis=0) if len(selected) > 1 else np.zeros(selected.shape[1])
        rows.append(np.concatenate([
            selected.mean(axis=0), selected.std(axis=0), selected[-1] - selected[0], absolute_change,
        ]).astype(np.float32))
        kept_centers.append(center)
        starts.append(start)
        ends.append(end)
    return np.stack(rows), np.array(kept_centers), np.array(starts), np.array(ends)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--window-seconds", type=float, default=6.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, help="Explicit local Torch CPU thread count.")
    args = parser.parse_args()
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.sample_fps, args.window_seconds, args.stride_seconds, args.batch_size)):
        parser.error("Sampling, window, stride, and batch size must be positive.")
    if args.threads is not None and not 1 <= args.threads <= 32:
        parser.error("Use a CPU thread count in 1..32.")
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if args.output.exists():
        parser.error("Output already exists. Use a new feature artifact path.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    from cashwatch.network_guard import enforce_network_guard
    network_guard = enforce_network_guard()  # Linux fails closed before ML imports.
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoImageProcessor, AutoModel
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("videos"):
        raise ValueError("Expected a version 1 nonempty dataset manifest.")
    videos = manifest["videos"]
    input_root_value = manifest.get("metadata", {}).get("input_dir")
    if input_root_value is not None and not Path(input_root_value).is_absolute():
        raise ValueError("metadata.input_dir must be an absolute directory.")
    input_root = Path(input_root_value).resolve() if input_root_value else manifest_path.parent
    if len({video["video_id"] for video in videos}) != len(videos):
        raise ValueError("video_id must be unique.")
    model_dir = args.model_dir.resolve(strict=True)
    config_file = model_dir / "config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("model_type") != "dinov2":
        raise ValueError("Provide the local facebook/dinov2-small checkpoint.")
    if config.get("hidden_size") != 384:
        raise ValueError("This reproducible feature configuration expects DINOv2-small (384 dimensions).")
    weight_files = sorted(model_dir.glob("*.safetensors"))
    if not weight_files:
        raise ValueError("Local DINOv2 safetensors weights are missing.")
    fingerprint = hashlib.sha256()
    model_files = {"config.json": sha256_file(config_file)}
    for path in weight_files:
        model_files[path.name] = sha256_file(path)
    for name, digest in sorted(model_files.items()):
        fingerprint.update((name + digest).encode())
    revision = "local-sha256:" + fingerprint.hexdigest()
    model_manifest = model_dir / "cashwatch_model_manifest.json"
    if model_manifest.is_file():
        provenance = json.loads(model_manifest.read_text(encoding="utf-8"))
        for name, actual_digest in model_files.items():
            if provenance.get("files", {}).get(name, {}).get("sha256") != actual_digest:
                raise ValueError("Local DINOv2 files differ from their downloaded provenance manifest.")
        revision = provenance.get("resolved_revision") or revision
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable.")
    processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    all_rows, all_ids, all_centers, all_starts, all_ends = [], [], [], [], []
    source_hashes, source_frame_counts = {}, {}
    for ordinal, video in enumerate(videos, start=1):
        identifier = video["video_id"]
        path = Path(video["path"])
        relative_path = not path.is_absolute()
        if relative_path:
            path = input_root / path
        path = path.resolve(strict=True)
        if relative_path and not path.is_relative_to(input_root):
            raise ValueError("Relative video path escapes metadata.input_dir.")
        actual_sha = sha256_file(path)
        if actual_sha != video["sha256"]:
            raise ValueError("Input file hash differs from the manifest.")
        source_hashes[identifier] = actual_sha
        with tempfile.TemporaryDirectory(prefix="cashwatch-local-features-") as temporary:
            sampled = sample_video(path, Path(temporary), sample_fps=args.sample_fps, max_image_edge=448,
                                   deadline_monotonic=time.monotonic() + 300)
            if abs(sampled.duration_sec - float(video["duration_sec"])) > 0.5:
                raise ValueError("Decoded video duration differs from the labeled manifest.")
            source_frame_counts[identifier] = len(sampled.frames)
            vectors = []
            with torch.inference_mode():
                for start in range(0, len(sampled.frames), args.batch_size):
                    images = []
                    try:
                        for frame in sampled.frames[start:start + args.batch_size]:
                            with Image.open(frame.image_path) as image:
                                images.append(ImageOps.pad(image.convert("RGB"), (224, 224),
                                                           method=Image.Resampling.BICUBIC, color=(114, 114, 114)))
                        batch = processor(images=images, do_resize=False, do_center_crop=False, return_tensors="pt").to(device)
                        output = model(**batch).last_hidden_state[:, 0, :]
                        output = torch.nn.functional.normalize(output, dim=-1)
                        vectors.append(output.float().cpu().numpy())
                    finally:
                        for image in images:
                            image.close()
            embeddings = np.concatenate(vectors)
            timestamps = np.array([frame.timestamp_sec for frame in sampled.frames])
            rows, centers, starts, ends = pooled_windows(
                embeddings, timestamps, sampled.duration_sec, args.window_seconds, args.stride_seconds,
            )
        all_rows.append(rows)
        all_ids.extend([identifier] * len(rows))
        all_centers.append(centers)
        all_starts.append(starts)
        all_ends.append(ends)
        print(json.dumps({"video_ordinal": ordinal, "video_count": len(videos), "windows_extracted": len(rows)}), flush=True)
    metadata = {
        "schema_version": 1, "model_id": "facebook/dinov2-small", "model_revision": revision,
        "model_files_sha256": model_files, "feature_recipe": "cls_l2_mean_std_last_minus_first_mean_abs_change_v1",
        "feature_dimension": int(all_rows[0].shape[1]), "sample_fps": args.sample_fps,
        "window_seconds": args.window_seconds, "stride_seconds": args.stride_seconds,
        "image_transform": "letterbox_224_rgb114_no_center_crop",
        "source_sha256": source_hashes,
        "torch_version": torch.__version__, "device": device,
        "torch_num_threads": torch.get_num_threads(),
        "source_frame_count": source_frame_counts, "total_sampled_frames": sum(source_frame_counts.values()),
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
        "network_isolation": network_guard,
        "bank_media_sent_to_external_inference": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        np.savez_compressed(
            output, embeddings=np.concatenate(all_rows).astype(np.float32),
            video_id=np.array(all_ids, dtype=str), timestamp_sec=np.concatenate(all_centers),
            window_start_sec=np.concatenate(all_starts), window_end_sec=np.concatenate(all_ends),
            metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
        )
    print(json.dumps({"features_written": True, "windows": len(all_ids), "experimental": True}))


if __name__ == "__main__":
    main()
