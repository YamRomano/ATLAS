#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np


def rotmat_to_qvec(rotation: np.ndarray) -> np.ndarray:
    # The stable eigen decomposition below yields the quaternion for the
    # transpose convention. Transpose the requested COLMAP world-to-camera
    # rotation up front so qvec_to_rotmat(qvec) round-trips to `rotation`.
    matrix = np.asarray(rotation, dtype=float).reshape(3, 3).T
    values, vectors = np.linalg.eigh(
        np.array(
            [
                [
                    matrix[0, 0] - matrix[1, 1] - matrix[2, 2],
                    matrix[1, 0] + matrix[0, 1],
                    matrix[2, 0] + matrix[0, 2],
                    matrix[1, 2] - matrix[2, 1],
                ],
                [
                    matrix[1, 0] + matrix[0, 1],
                    matrix[1, 1] - matrix[0, 0] - matrix[2, 2],
                    matrix[2, 1] + matrix[1, 2],
                    matrix[2, 0] - matrix[0, 2],
                ],
                [
                    matrix[2, 0] + matrix[0, 2],
                    matrix[2, 1] + matrix[1, 2],
                    matrix[2, 2] - matrix[0, 0] - matrix[1, 1],
                    matrix[0, 1] - matrix[1, 0],
                ],
                [
                    matrix[1, 2] - matrix[2, 1],
                    matrix[2, 0] - matrix[0, 2],
                    matrix[0, 1] - matrix[1, 0],
                    matrix[0, 0] + matrix[1, 1] + matrix[2, 2],
                ],
            ],
            dtype=float,
        )
        / 3.0
    )
    qvec = vectors[[3, 0, 1, 2], int(np.argmax(values))]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def load_keypoints(database: sqlite3.Connection, image_id: int) -> np.ndarray:
    row = database.execute(
        "SELECT rows, cols, data FROM keypoints WHERE image_id = ?",
        (image_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"No COLMAP keypoints for database image {image_id}")
    rows, cols, raw = int(row[0]), int(row[1]), row[2]
    values = np.frombuffer(raw, dtype=np.float32).reshape(rows, cols)
    if values.shape[1] < 2:
        raise RuntimeError(f"Invalid keypoint table for image {image_id}: {values.shape}")
    return values[:, :2].astype(float)


def read_camera_lines(path: Path) -> dict[int, str]:
    cameras: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        camera_id = int(stripped.split()[0])
        if camera_id in cameras:
            raise RuntimeError(f"Duplicate camera {camera_id} in {path}")
        cameras[camera_id] = stripped
    return cameras


def read_image_headers(path: Path) -> tuple[set[int], set[str]]:
    image_ids: set[int] = set()
    image_names: set[str] = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    data_lines = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    if len(data_lines) % 2:
        raise RuntimeError(f"Invalid COLMAP text image model: {path}")
    for header in data_lines[::2]:
        fields = header.split()
        if len(fields) < 10:
            raise RuntimeError(f"Invalid COLMAP image header in {path}: {header}")
        image_id = int(fields[0])
        image_name = fields[9]
        if image_id in image_ids:
            raise RuntimeError(f"Duplicate image id {image_id} in {path}")
        if image_name in image_names:
            raise RuntimeError(f"Duplicate image name {image_name} in {path}")
        image_ids.add(image_id)
        image_names.add(image_name)
    return image_ids, image_names


def camera_lines_match(existing: str, requested: str) -> bool:
    existing_fields = existing.split()
    requested_fields = requested.split()
    if len(existing_fields) != len(requested_fields):
        return False
    if existing_fields[:4] != requested_fields[:4]:
        return False
    try:
        return bool(
            np.allclose(
                np.asarray(existing_fields[4:], dtype=float),
                np.asarray(requested_fields[4:], dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
        )
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add accepted live TSolve poses to a fixed COLMAP text model as locked triangulation anchors."
    )
    parser.add_argument("--reference-text", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--poses-json", type=Path, required=True)
    parser.add_argument("--selected-frames", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--new-prefix", default="enhancement/")
    parser.add_argument("--camera-id", type=int, default=2)
    parser.add_argument(
        "--camera-line",
        default="2 SIMPLE_RADIAL 1200 675 884.32333453521096 600 337.5 0",
    )
    args = parser.parse_args()

    reference = args.reference_text.resolve()
    for filename in ("cameras.txt", "images.txt", "points3D.txt"):
        if not (reference / filename).exists():
            raise RuntimeError(f"Missing reference model file: {reference / filename}")

    camera_line = args.camera_line.strip()
    camera_fields = camera_line.split()
    if not camera_fields or int(camera_fields[0]) != args.camera_id:
        raise RuntimeError(
            f"--camera-line must define camera id {args.camera_id}: {camera_line}"
        )
    reference_cameras = read_camera_lines(reference / "cameras.txt")
    if args.camera_id in reference_cameras and not camera_lines_match(
        reference_cameras[args.camera_id], camera_line
    ):
        raise RuntimeError(
            f"Reference camera {args.camera_id} differs from --camera-line: "
            f"{reference_cameras[args.camera_id]}"
        )
    reference_image_ids, reference_image_names = read_image_headers(
        reference / "images.txt"
    )

    selected = {path.name for path in args.selected_frames.glob("*.jpg")}
    payload = json.loads(args.poses_json.read_text(encoding="utf-8"))
    accepted_by_name: dict[str, dict[str, object]] = {}
    for pose in payload.get("poses", []):
        image_name = Path(str(pose.get("image_name", ""))).name
        if (
            image_name in selected
            and pose.get("success")
            and not pose.get("held_pose")
            and not pose.get("output_rejected")
            and pose.get("R") is not None
            and pose.get("t") is not None
        ):
            accepted_by_name[image_name] = pose
    if len(accepted_by_name) < 12:
        raise RuntimeError(
            f"Only {len(accepted_by_name)} selected frames have trusted live poses; need at least 12."
        )
    already_registered = {
        basename
        for basename in accepted_by_name
        if f"{args.new_prefix}{basename}" in reference_image_names
    }
    new_by_name = {
        basename: pose
        for basename, pose in accepted_by_name.items()
        if basename not in already_registered
    }
    if not new_by_name:
        raise RuntimeError("All trusted selected frames are already registered in the reference.")

    out = args.out_model.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(reference / "cameras.txt", out / "cameras.txt")
    shutil.copy2(reference / "images.txt", out / "images.txt")
    shutil.copy2(reference / "points3D.txt", out / "points3D.txt")

    if args.camera_id not in reference_cameras:
        with (out / "cameras.txt").open("a", encoding="utf-8") as handle:
            handle.write(f"\n{camera_line}\n")

    seeded: list[dict[str, object]] = []
    with sqlite3.connect(
        f"file:{args.database.resolve()}?mode=ro&immutable=1", uri=True
    ) as database:
        with (out / "images.txt").open("a", encoding="utf-8") as handle:
            for basename in sorted(new_by_name):
                name = f"{args.new_prefix}{basename}"
                database_row = database.execute(
                    "SELECT image_id, camera_id FROM images WHERE name = ?",
                    (name,),
                ).fetchone()
                if database_row is None:
                    raise RuntimeError(f"Live anchor is missing from the staging database: {name}")
                image_id, camera_id = int(database_row[0]), int(database_row[1])
                if camera_id != args.camera_id:
                    raise RuntimeError(
                        f"Unexpected camera for {name}: {camera_id}, expected {args.camera_id}"
                    )
                if image_id in reference_image_ids:
                    raise RuntimeError(
                        f"Database image id {image_id} for {name} is already used by the reference."
                    )
                keypoints = load_keypoints(database, image_id)
                pose = new_by_name[basename]
                rotation = np.asarray(pose["R"], dtype=float).reshape(3, 3)
                translation = np.asarray(pose["t"], dtype=float).reshape(3)
                orthogonality_error = float(
                    np.linalg.norm(rotation @ rotation.T - np.eye(3))
                )
                determinant = float(np.linalg.det(rotation))
                if orthogonality_error > 1e-5 or abs(determinant - 1.0) > 1e-5:
                    raise RuntimeError(
                        f"Invalid live rotation for {name}: "
                        f"orthogonality={orthogonality_error}, determinant={determinant}"
                    )
                qvec = rotmat_to_qvec(rotation)
                header = " ".join(
                    [
                        str(image_id),
                        *(f"{value:.17g}" for value in qvec),
                        *(f"{value:.17g}" for value in translation),
                        str(camera_id),
                        name,
                    ]
                )
                observations = " ".join(
                    f"{xy[0]:.9g} {xy[1]:.9g} -1" for xy in keypoints
                )
                handle.write(f"\n{header}\n{observations}\n")
                seeded.append(
                    {
                        "image_id": image_id,
                        "name": name,
                        "frame_index": int(Path(basename).stem.rsplit("_", 1)[-1]),
                        "keypoints": len(keypoints),
                        "center": [
                            float(value)
                            for value in (-rotation.T @ translation).reshape(3)
                        ],
                    }
                )

    summary = {
        "mode": "fixed_reference_live_pose_seed",
        "reference_text": str(reference),
        "database": str(args.database.resolve()),
        "poses_json": str(args.poses_json.resolve()),
        "selected_frame_count": len(selected),
        "trusted_selected_images": len(accepted_by_name),
        "already_registered_images": len(already_registered),
        "seeded_live_images": len(seeded),
        "first_seed_frame": min(item["frame_index"] for item in seeded),
        "last_seed_frame": max(item["frame_index"] for item in seeded),
        "camera_id": args.camera_id,
        "seeds": seeded,
    }
    (out / "live_pose_seed_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "seeds"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
