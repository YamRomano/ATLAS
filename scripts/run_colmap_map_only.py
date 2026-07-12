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


def copy_images(src: Path, dst: Path) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    names = []
    for p in sorted(src.glob("*.jpg")):
        out = dst / p.name
        shutil.copy2(p, out)
        names.append(p.name)
    return names


def choose_largest_sparse_model(sparse_root: Path) -> Path:
    candidates = [p for p in sparse_root.iterdir() if p.is_dir() and (p / "images.bin").exists()]
    if not candidates:
        raise RuntimeError(f"No sparse models found in {sparse_root}")
    return max(candidates, key=lambda p: (p / "points3D.bin").stat().st_size if (p / "points3D.bin").exists() else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a COLMAP sparse map from webcam/image frames only.")
    ap.add_argument("--colmap", required=True, type=Path)
    ap.add_argument("--frames", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-image-size", type=int, default=1200)
    ap.add_argument("--dense", action="store_true", help="Also attempt COLMAP dense stereo/fusion for viewer-only display points.")
    ap.add_argument("--dense-max-image-size", type=int, default=900)
    ap.add_argument("--camera-model", default="SIMPLE_RADIAL")
    ap.add_argument("--matcher", choices=["sequential", "exhaustive"], default="exhaustive")
    args = ap.parse_args()

    if not args.colmap.exists():
        raise FileNotFoundError(f"COLMAP binary not found: {args.colmap}")

    images = sorted(args.frames.glob("*.jpg"))
    if len(images) < 12:
        raise RuntimeError(f"Need at least 12 frames for a useful map; found {len(images)} in {args.frames}")

    out = args.out_dir
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    image_dir = out / "images"
    names = copy_images(args.frames, image_dir)
    (out / "image_list.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    db = out / "database.db"
    sparse = out / "sparse"
    sparse.mkdir()

    run(
        [
            args.colmap,
            "feature_extractor",
            "--database_path",
            db,
            "--image_path",
            image_dir,
            "--ImageReader.camera_model",
            args.camera_model,
            "--ImageReader.single_camera",
            "1",
            "--SiftExtraction.max_image_size",
            args.max_image_size,
            "--SiftExtraction.use_gpu",
            "0",
        ]
    )

    if args.matcher == "exhaustive":
        run(
            [
                args.colmap,
                "exhaustive_matcher",
                "--database_path",
                db,
                "--SiftMatching.guided_matching",
                "1",
                "--SiftMatching.use_gpu",
                "0",
            ]
        )
    else:
        run(
            [
                args.colmap,
                "sequential_matcher",
                "--database_path",
                db,
                "--SiftMatching.guided_matching",
                "1",
                "--SiftMatching.use_gpu",
                "0",
            ]
        )

    run(
        [
            args.colmap,
            "mapper",
            "--database_path",
            db,
            "--image_path",
            image_dir,
            "--output_path",
            sparse,
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

    model = choose_largest_sparse_model(sparse)
    text = out / "sparse_text"
    text.mkdir()
    run([args.colmap, "model_converter", "--input_path", model, "--output_path", text, "--output_type", "TXT"])

    dense_ply = None
    dense_error = None
    if args.dense:
        dense = out / "dense"
        try:
            if dense.exists():
                shutil.rmtree(dense)
            dense.mkdir(parents=True, exist_ok=True)
            run(
                [
                    args.colmap,
                    "image_undistorter",
                    "--image_path",
                    image_dir,
                    "--input_path",
                    model,
                    "--output_path",
                    dense,
                    "--output_type",
                    "COLMAP",
                    "--max_image_size",
                    args.dense_max_image_size,
                ]
            )
            run(
                [
                    args.colmap,
                    "patch_match_stereo",
                    "--workspace_path",
                    dense,
                    "--workspace_format",
                    "COLMAP",
                    "--PatchMatchStereo.geom_consistency",
                    "true",
                    "--PatchMatchStereo.gpu_index",
                    "-1",
                ]
            )
            fused = dense / "fused.ply"
            run(
                [
                    args.colmap,
                    "stereo_fusion",
                    "--workspace_path",
                    dense,
                    "--workspace_format",
                    "COLMAP",
                    "--input_type",
                    "geometric",
                    "--output_path",
                    fused,
                ]
            )
            if fused.exists():
                dense_ply = str(fused)
        except Exception as exc:
            dense_error = str(exc)
            print(f"WARNING: dense COLMAP viewer reconstruction failed; keeping sparse map. {dense_error}", flush=True)

    summary = {
        "frames": len(names),
        "matcher": args.matcher,
        "camera_model": args.camera_model,
        "max_image_size": args.max_image_size,
        "database": str(db),
        "sparse_model": str(model),
        "sparse_text": str(text),
        "dense_ply": dense_ply,
        "dense_error": dense_error,
        "out_dir": str(out),
    }
    (out / "colmap_map_only_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
