#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from colmap_io import read_images_model, read_images_text, read_points3d_text


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def image_prefix(name: str) -> str:
    stem = Path(name).stem
    match = re.match(r"(.+?)_frame_\d+$", stem)
    if match:
        return match.group(1)
    match = re.match(r"(webcam_map)_\d+$", stem)
    if match:
        return match.group(1)
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_bank_report(frames: Path) -> dict:
    files = sorted(p for p in frames.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    prefix_counts: dict[str, int] = {}
    digest_to_names: dict[str, list[str]] = {}
    for path in files:
        prefix = image_prefix(path.name)
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        digest_to_names.setdefault(sha1_file(path), []).append(path.name)

    duplicate_groups = [names for names in digest_to_names.values() if len(names) > 1]
    return {
        "frame_file_count": len(files),
        "unique_frame_hash_count": len(digest_to_names),
        "duplicate_file_count": sum(len(names) - 1 for names in duplicate_groups),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_examples": duplicate_groups[:8],
        "prefix_counts": dict(sorted(prefix_counts.items())),
    }


def chosen_model_report(colmap_root: Path) -> dict:
    text = colmap_root / "sparse_text"
    if not (text / "images.txt").exists():
        return {}
    images = read_images_text(text / "images.txt")
    points = read_points3d_text(text / "points3D.txt")
    prefix_stats: dict[str, dict] = {}
    valid_observations = []
    for image in images.values():
        prefix = image_prefix(image.name)
        valid = int((image.point3d_ids >= 0).sum())
        valid_observations.append(valid)
        stat = prefix_stats.setdefault(prefix, {"registered_images": 0, "points2d_to_3d": []})
        stat["registered_images"] += 1
        stat["points2d_to_3d"].append(valid)

    compact_prefix_stats = {}
    for prefix, stat in sorted(prefix_stats.items()):
        values = sorted(stat["points2d_to_3d"])
        mid = len(values) // 2
        median = 0 if not values else (values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid]))
        compact_prefix_stats[prefix] = {
            "registered_images": stat["registered_images"],
            "median_points2d_to_3d": median,
        }

    values = sorted(valid_observations)
    mid = len(values) // 2
    median = 0 if not values else (values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid]))
    return {
        "registered_images": len(images),
        "points3D": len(points),
        "median_points2d_to_3d": median,
        "registered_prefixes": compact_prefix_stats,
    }


def sparse_components_report(colmap_root: Path) -> list[dict]:
    sparse = colmap_root / "sparse"
    if not sparse.exists():
        return []
    rows = []
    for component in sorted(p for p in sparse.iterdir() if p.is_dir()):
        if not (component / "images.bin").exists() and not (component / "images.txt").exists():
            continue
        try:
            images = read_images_model(component)
            point_bytes = (component / "points3D.bin").stat().st_size if (component / "points3D.bin").exists() else 0
            rows.append(
                {
                    "component": component.name,
                    "registered_images": len(images),
                    "points3D_bin_bytes": point_bytes,
                }
            )
        except Exception as exc:
            rows.append({"component": component.name, "error": repr(exc)})
    rows.sort(key=lambda r: (int(r.get("registered_images") or 0), int(r.get("points3D_bin_bytes") or 0)), reverse=True)
    return rows


def validation_notes(frame_report: dict, model_report: dict, components: list[dict]) -> list[str]:
    notes = []
    duplicate_count = int(frame_report.get("duplicate_file_count") or 0)
    if duplicate_count:
        notes.append(f"{duplicate_count} duplicate frame files detected; duplicates can bias COLMAP.")
    if len(components) > 1:
        notes.append(f"{len(components)} sparse components were created; ATLAS uses the largest component.")

    frame_prefix_counts = frame_report.get("prefix_counts") or {}
    registered_prefixes = model_report.get("registered_prefixes") or {}
    for prefix, total in frame_prefix_counts.items():
        registered = int((registered_prefixes.get(prefix) or {}).get("registered_images") or 0)
        if total and registered / total < 0.65:
            notes.append(f"Video group {prefix} registered weakly: {registered}/{total} images.")
    if not notes:
        notes.append("No obvious frame-bank or COLMAP component warning detected.")
    return notes


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate an ATLAS COLMAP map build.")
    ap.add_argument("--frames", required=True, type=Path)
    ap.add_argument("--colmap-root", required=True, type=Path)
    ap.add_argument("--asset-dir", required=True, type=Path)
    ap.add_argument("--matcher", default="unknown")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    frame_report = frame_bank_report(args.frames)
    model_report = chosen_model_report(args.colmap_root)
    components = sparse_components_report(args.colmap_root)
    report = {
        "frames": str(args.frames),
        "colmap_root": str(args.colmap_root),
        "asset_dir": str(args.asset_dir),
        "matcher": args.matcher,
        "frame_bank": frame_report,
        "chosen_model": model_report,
        "sparse_components": components,
        "notes": validation_notes(frame_report, model_report, components),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
