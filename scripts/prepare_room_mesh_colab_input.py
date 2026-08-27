#!/usr/bin/env python3
"""Prepare compact, coverage-balanced keyframe packs from long room videos.

The output zip is designed for the companion VGGT/PyCOLMAP Colab notebook.
Source videos are never modified or copied into the output.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrameRecord:
    video_index: int
    video_name: str
    time_sec: float
    candidate_name: str
    blur_score: float
    phash: str


def perceptual_hash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coeffs = cv2.dct(small)[:8, :8]
    median = float(np.median(coeffs[1:]))
    bits = coeffs > median
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def hash_distance(left: int, right: int) -> int:
    return bin(left ^ right).count("1")


def resize_to_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    return cv2.resize(frame, (max_width, max(2, round(height * scale))), interpolation=cv2.INTER_AREA)


def inspect_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0
    metadata = {
        "name": path.name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration,
    }
    capture.release()
    if duration <= 0:
        raise RuntimeError(f"Could not determine duration for {path}")
    return metadata


def extract_candidates(
    video: Path,
    video_index: int,
    metadata: dict,
    output_dir: Path,
    interval_sec: float,
    max_width: int,
    jpeg_quality: int,
) -> list[FrameRecord]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")
    records: list[FrameRecord] = []
    duration = float(metadata["duration_sec"])
    sample_count = max(1, int(math.floor(duration / interval_sec)) + 1)
    prefix = f"v{video_index}"
    for sample_index in range(sample_count):
        timestamp = min(duration - 0.001, sample_index * interval_sec)
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        frame = resize_to_width(frame, max_width)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        phash_value = perceptual_hash(gray)
        name = f"{prefix}_{sample_index:06d}.jpg"
        if not cv2.imwrite(str(output_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
            raise RuntimeError(f"Could not write {name}")
        records.append(
            FrameRecord(
                video_index=video_index,
                video_name=video.name,
                time_sec=round(timestamp, 3),
                candidate_name=name,
                blur_score=round(blur, 3),
                phash=f"{phash_value:016x}",
            )
        )
        if sample_index % 25 == 0 or sample_index + 1 == sample_count:
            print(f"[{video.name}] sampled {sample_index + 1}/{sample_count}", flush=True)
    capture.release()
    return records


def quality_filter(
    records: list[FrameRecord],
    absolute_blur_floor: float,
    blur_percentile: float,
    duplicate_distance: int,
) -> tuple[list[FrameRecord], dict]:
    if not records:
        return [], {"blur_threshold": None, "blur_rejected": 0, "duplicate_rejected": 0}
    scores = np.asarray([record.blur_score for record in records], dtype=np.float64)
    adaptive = float(np.percentile(scores, blur_percentile))
    threshold = min(max(0.0, absolute_blur_floor), adaptive) if adaptive > 0 else absolute_blur_floor
    sharp = [record for record in records if record.blur_score >= threshold]
    accepted: list[FrameRecord] = []
    previous_hash: int | None = None
    duplicate_rejected = 0
    for record in sharp:
        value = int(record.phash, 16)
        if previous_hash is not None and hash_distance(previous_hash, value) < duplicate_distance:
            duplicate_rejected += 1
            continue
        accepted.append(record)
        previous_hash = value
    return accepted, {
        "blur_threshold": round(threshold, 3),
        "adaptive_blur_percentile": round(adaptive, 3),
        "blur_rejected": len(records) - len(sharp),
        "duplicate_rejected": duplicate_rejected,
    }


def allocate_targets(groups: dict[int, list[FrameRecord]], target_total: int) -> dict[int, int]:
    available = {key: len(value) for key, value in groups.items()}
    total_available = sum(available.values())
    if target_total >= total_available:
        return available
    allocations = {
        key: min(count, max(1, round(target_total * count / total_available)))
        for key, count in available.items()
    }
    while sum(allocations.values()) > target_total:
        key = max(allocations, key=lambda item: allocations[item])
        allocations[key] -= 1
    while sum(allocations.values()) < target_total:
        candidates = [key for key in allocations if allocations[key] < available[key]]
        if not candidates:
            break
        key = max(candidates, key=lambda item: available[item] - allocations[item])
        allocations[key] += 1
    return allocations


def coverage_select(records: list[FrameRecord], target_total: int) -> list[FrameRecord]:
    groups: dict[int, list[FrameRecord]] = {}
    for record in records:
        groups.setdefault(record.video_index, []).append(record)
    allocations = allocate_targets(groups, target_total)
    selected: list[FrameRecord] = []
    for video_index, group in sorted(groups.items()):
        group = sorted(group, key=lambda record: record.time_sec)
        target = allocations[video_index]
        if target >= len(group):
            selected.extend(group)
            continue
        edges = np.linspace(0, len(group), target + 1, dtype=int)
        for start, stop in zip(edges[:-1], edges[1:]):
            window = group[start:max(start + 1, stop)]
            selected.append(max(window, key=lambda record: record.blur_score))
    return sorted(selected, key=lambda record: (record.video_index, record.time_sec))


def materialize_scene(name: str, records: list[FrameRecord], candidates: Path, root: Path) -> None:
    image_dir = root / name / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, record in enumerate(records):
        output_name = f"{record.video_index:02d}_{index:04d}_{record.candidate_name}"
        shutil.copy2(candidates / record.candidate_name, image_dir / output_name)
        row = asdict(record)
        row["scene_image"] = output_name
        manifest.append(row)
    (root / name / "frames.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def make_contact_sheet(records: list[FrameRecord], candidates: Path, output: Path, columns: int = 6) -> None:
    if not records:
        return
    thumb_width, thumb_height = 220, 150
    rows = math.ceil(len(records) / columns)
    sheet = np.full((rows * thumb_height, columns * thumb_width, 3), 245, dtype=np.uint8)
    for index, record in enumerate(records):
        image = cv2.imread(str(candidates / record.candidate_name))
        if image is None:
            continue
        scale = min(thumb_width / image.shape[1], (thumb_height - 22) / image.shape[0])
        resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
        y = (index // columns) * thumb_height
        x = (index % columns) * thumb_width
        ox = x + (thumb_width - resized.shape[1]) // 2
        sheet[y : y + resized.shape[0], ox : ox + resized.shape[1]] = resized
        label = f"v{record.video_index} {record.time_sec:6.1f}s blur {record.blur_score:.0f}"
        cv2.putText(sheet, label, (x + 5, y + thumb_height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (28, 70, 92), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def zip_package(root: Path, output_zip: Path) -> None:
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--blur-floor", type=float, default=60.0)
    parser.add_argument("--blur-percentile", type=float, default=15.0)
    parser.add_argument("--duplicate-distance", type=int, default=6)
    parser.add_argument("--pilot-frames", type=int, default=48)
    parser.add_argument("--medium-frames", type=int, default=96)
    parser.add_argument("--full-frames", type=int, default=220)
    parser.add_argument("--keep-candidates", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    videos = [path.expanduser().resolve() for path in args.videos]
    missing = [str(path) for path in videos if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source videos: {missing}")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    candidates = output / "candidates"
    package = output / "package"
    candidates.mkdir(parents=True)
    package.mkdir(parents=True)

    metadata = [inspect_video(path) for path in videos]
    all_records: list[FrameRecord] = []
    filters = []
    for video_index, (video, video_metadata) in enumerate(zip(videos, metadata), start=1):
        records = extract_candidates(
            video,
            video_index,
            video_metadata,
            candidates,
            args.interval,
            args.max_width,
            args.jpeg_quality,
        )
        filtered, filter_summary = quality_filter(
            records,
            args.blur_floor,
            args.blur_percentile,
            args.duplicate_distance,
        )
        all_records.extend(filtered)
        filters.append({"video_index": video_index, "candidates": len(records), "accepted": len(filtered), **filter_summary})

    if len(all_records) < args.pilot_frames:
        raise RuntimeError(f"Only {len(all_records)} usable frames survived filtering")
    pilot = coverage_select(all_records, min(args.pilot_frames, len(all_records)))
    medium = coverage_select(all_records, min(args.medium_frames, len(all_records)))
    full = coverage_select(all_records, min(args.full_frames, len(all_records)))
    materialize_scene("scene_pilot", pilot, candidates, package)
    materialize_scene("scene_medium", medium, candidates, package)
    materialize_scene("scene_full", full, candidates, package)
    make_contact_sheet(pilot, candidates, package / "pilot_contact_sheet.jpg")
    summary = {
        "created_at_unix": time.time(),
        "source_videos": metadata,
        "settings": vars(args) | {"videos": [str(path) for path in videos], "output": str(output)},
        "filtering": filters,
        "usable_frames": len(all_records),
        "scenes": {"scene_pilot": len(pilot), "scene_medium": len(medium), "scene_full": len(full)},
        "elapsed_sec": round(time.time() - started, 2),
    }
    (package / "manifest.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    output_zip = output / "room_mesh_colab_input.zip"
    zip_package(package, output_zip)
    if not args.keep_candidates:
        shutil.rmtree(candidates)
    print(json.dumps(summary, indent=2, default=str))
    print(f"Colab package: {output_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
