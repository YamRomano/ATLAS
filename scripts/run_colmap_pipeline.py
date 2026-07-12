#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[object]) -> None:
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "minimal")
    subprocess.run(cmd, check=True, env=env)


def copy_tree_images(src: Path, dst: Path, prefix: str) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    names = []
    for p in sorted(src.glob("*.jpg")):
        name = f"{prefix}/{p.name}"
        out = dst / p.name
        shutil.copy2(p, out)
        names.append(name)
    return names


def choose_largest_sparse_model(sparse_root: Path) -> Path:
    candidates = [p for p in sparse_root.iterdir() if p.is_dir() and (p / "images.bin").exists()]
    if not candidates:
        raise RuntimeError(f"No sparse models found in {sparse_root}")
    return max(candidates, key=lambda p: (p / "points3D.bin").stat().st_size if (p / "points3D.bin").exists() else 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap", required=True, type=Path)
    ap.add_argument("--map-frames", required=True, type=Path)
    ap.add_argument("--query-frames", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-image-size", type=int, default=1600)
    ap.add_argument("--map-camera-model", default="SIMPLE_RADIAL")
    ap.add_argument("--query-camera-model", default="SIMPLE_RADIAL")
    ap.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    args = ap.parse_args()

    if not args.colmap.exists():
        raise FileNotFoundError(f"COLMAP binary not found: {args.colmap}")

    out = args.out_dir
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    all_images = out / "all_images"
    map_root = all_images / "map"
    query_root = all_images / "query"
    map_names = copy_tree_images(args.map_frames, map_root, "map")
    query_names = copy_tree_images(args.query_frames, query_root, "query")

    if len(map_names) < 3:
        raise RuntimeError("Need at least 3 map frames for COLMAP mapping.")
    if len(query_names) < 1:
        raise RuntimeError("Need at least 1 query frame for localization.")

    (out / "map_image_list.txt").write_text("\n".join(map_names) + "\n", encoding="utf-8")
    (out / "query_image_list.txt").write_text("\n".join(query_names) + "\n", encoding="utf-8")

    db = out / "database.db"
    sparse = out / "sparse_map"
    localized = out / "localized_model"
    sparse.mkdir()
    localized.mkdir()

    run(
        [
            args.colmap,
            "feature_extractor",
            "--database_path",
            db,
            "--image_path",
            all_images,
            "--ImageReader.camera_model",
            args.map_camera_model,
            "--ImageReader.single_camera_per_folder",
            "1",
            "--SiftExtraction.max_image_size",
            args.max_image_size,
            "--SiftExtraction.use_gpu",
            "0",
        ]
    )
    if args.matcher == "sequential":
        run([
            args.colmap,
            "sequential_matcher",
            "--database_path",
            db,
            "--SiftMatching.guided_matching",
            "1",
            "--SiftMatching.use_gpu",
            "0",
        ])
    else:
        run([
            args.colmap,
            "exhaustive_matcher",
            "--database_path",
            db,
            "--SiftMatching.guided_matching",
            "1",
            "--SiftMatching.use_gpu",
            "0",
        ])

    run(
        [
            args.colmap,
            "mapper",
            "--database_path",
            db,
            "--image_path",
            all_images,
            "--output_path",
            sparse,
            "--image_list_path",
            out / "map_image_list.txt",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.init_min_num_inliers",
            "30",
            "--Mapper.abs_pose_min_num_inliers",
            "15",
            "--Mapper.abs_pose_min_inlier_ratio",
            "0.10",
            "--Mapper.min_model_size",
            "2",
        ]
    )

    sparse0 = choose_largest_sparse_model(sparse)

    run(
        [
            args.colmap,
            "image_registrator",
            "--database_path",
            db,
            "--input_path",
            sparse0,
            "--output_path",
            localized,
        ]
    )

    sparse_text = out / "sparse_map_text"
    localized_text = out / "localized_model_text"
    sparse_text.mkdir()
    localized_text.mkdir()
    run([args.colmap, "model_converter", "--input_path", sparse0, "--output_path", sparse_text, "--output_type", "TXT"])
    run([args.colmap, "model_converter", "--input_path", localized, "--output_path", localized_text, "--output_type", "TXT"])

    (out / "colmap_summary.json").write_text(
        json.dumps(
            {
                "map_frames": len(map_names),
                "query_frames": len(query_names),
                "database": str(db),
                "sparse_model": str(sparse0),
                "localized_model": str(localized),
                "sparse_text": str(sparse_text),
                "localized_text": str(localized_text),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("COLMAP pipeline complete:", out)


if __name__ == "__main__":
    main()
