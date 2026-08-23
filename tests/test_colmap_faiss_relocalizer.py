from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from colmap_faiss_relocalizer import (
    FaissIVF3DRelocalizer,
    _rotmat_to_qvec,
    build_faiss_index,
    index_is_current,
    source_geometry_tsolve_pool,
    unique_point_matches,
)
from colmap_io import Camera, qvec_to_rotmat


def _blob(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array).tobytes()


class FaissRelocalizerTest(unittest.TestCase):
    def test_ratio_test_compares_different_3d_points(self) -> None:
        matches = unique_point_matches(
            distances=np.array([[1.0, 2.0, 9.0], [1.5, 4.0, 16.0]], dtype=np.float32),
            neighbor_ids=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
            point3d_ids=np.array([10, 10, 20, 30], dtype=np.int64),
            source_image_ids=np.array([1, 2, 3, 4], dtype=np.int32),
            ratio=0.8,
        )
        self.assertEqual([(row["query_index"], row["point3d_id"]) for row in matches], [(0, 10)])

    def test_rotation_quaternion_round_trip(self) -> None:
        rotation, _ = cv2.Rodrigues(np.array([0.31, -0.47, 1.08], dtype=np.float64))
        restored = qvec_to_rotmat(_rotmat_to_qvec(rotation))
        np.testing.assert_allclose(restored, rotation, atol=1.0e-10)

    def test_source_geometry_returns_tsolve_pool_without_pnp_pose(self) -> None:
        rng = np.random.default_rng(20260823)
        count = 72
        point_ids = np.arange(1, count + 1, dtype=np.int64)
        p3d = np.column_stack(
            [
                rng.uniform(-1.4, 1.4, count),
                rng.uniform(-0.8, 0.8, count),
                rng.uniform(3.5, 7.0, count),
            ]
        )
        focal, width, height = 820.0, 1200, 675

        def project(rotation: np.ndarray, center: np.ndarray) -> np.ndarray:
            camera_points = (rotation @ (p3d - center).T).T
            return np.column_stack(
                [
                    focal * camera_points[:, 0] / camera_points[:, 2] + width / 2.0,
                    focal * camera_points[:, 1] / camera_points[:, 2] + height / 2.0,
                ]
            ).astype(np.float32)

        query_xy = project(np.eye(3), np.zeros(3))
        map_images: dict[int, SimpleNamespace] = {}
        source_count = 8
        vector_point_ids: list[int] = []
        vector_source_ids: list[int] = []
        vector_feature_indices: list[int] = []
        neighbor_ids = np.empty((count, source_count), dtype=np.int64)
        for source_offset in range(source_count):
            angle = 0.025 * (source_offset - 3.5)
            rotation, _ = cv2.Rodrigues(
                np.array([0.01 * source_offset, angle, -0.006 * source_offset])
            )
            center = np.array(
                [0.05 * (source_offset - 3.5), 0.015 * source_offset, 0.025 * source_offset]
            )
            source_id = 100 + source_offset
            map_images[source_id] = SimpleNamespace(
                xys=project(rotation, center),
                point3d_ids=point_ids.copy(),
            )
            for feature_index, point_id in enumerate(point_ids):
                vector_id = len(vector_point_ids)
                vector_point_ids.append(int(point_id))
                vector_source_ids.append(source_id)
                vector_feature_indices.append(feature_index)
                neighbor_ids[feature_index, source_offset] = vector_id

        matches = [
            {
                "query_index": index,
                "point3d_id": int(point_id),
                "vector_id": int(neighbor_ids[index, 0]),
                "source_image_id": 100,
                "distance": 1.0,
                "second_point_distance": 4.0,
            }
            for index, point_id in enumerate(point_ids)
        ]
        map_points = {
            int(point_id): SimpleNamespace(xyz=point.copy())
            for point_id, point in zip(point_ids, p3d)
        }
        pool = source_geometry_tsolve_pool(
            image_name="query.jpg",
            image_id=-1,
            camera=Camera(
                camera_id=-1,
                model="SIMPLE_PINHOLE",
                width=width,
                height=height,
                params=[focal, width / 2.0, height / 2.0],
            ),
            keypoints=query_xy,
            matches=matches,
            neighbor_ids=neighbor_ids,
            point3d_ids=np.asarray(vector_point_ids, dtype=np.int64),
            source_image_ids=np.asarray(vector_source_ids, dtype=np.int32),
            source_feature_indices=np.asarray(vector_feature_indices, dtype=np.int32),
            map_images=map_images,
            map_points=map_points,
            min_points=40,
        )
        self.assertTrue(pool["accepted"], pool)
        self.assertGreaterEqual(pool["valid_2d3d"], 40)
        self.assertTrue(pool["tsolve_only_correspondences"])
        self.assertNotIn("colmap_qvec_world_to_camera", pool)
        self.assertNotIn("colmap_tvec_world_to_camera", pool)

    def test_build_and_localize_synthetic_colmap_map(self) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss-cpu is not installed")

        with tempfile.TemporaryDirectory(prefix="atlas-faiss-test-") as temporary:
            root = Path(temporary)
            database = root / "database.db"
            sparse = root / "sparse"
            index_dir = root / "faiss"
            sparse.mkdir()

            width, height = 960, 540
            focal = 720.0
            rng = np.random.default_rng(2903)
            points = []
            for row in range(10):
                for column in range(14):
                    z = 4.0 + 0.08 * ((row + column) % 9)
                    points.append(
                        [
                            (column - 6.5) * 0.19,
                            (row - 4.5) * 0.16,
                            z,
                        ]
                    )
            p3d = np.asarray(points, dtype=np.float64)
            point_ids = np.arange(1, len(p3d) + 1, dtype=np.int64)
            xy = np.column_stack(
                [
                    focal * p3d[:, 0] / p3d[:, 2] + width / 2.0,
                    focal * p3d[:, 1] / p3d[:, 2] + height / 2.0,
                ]
            ).astype(np.float32)
            descriptors = rng.integers(0, 256, size=(len(p3d), 128), dtype=np.uint8)
            query_descriptors = descriptors.copy()

            with sqlite3.connect(database) as conn:
                conn.executescript(
                    """
                    CREATE TABLE cameras (
                        camera_id INTEGER PRIMARY KEY, model INTEGER NOT NULL,
                        width INTEGER NOT NULL, height INTEGER NOT NULL,
                        params BLOB, prior_focal_length INTEGER NOT NULL);
                    CREATE TABLE images (
                        image_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
                        camera_id INTEGER NOT NULL);
                    CREATE TABLE keypoints (
                        image_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL,
                        cols INTEGER NOT NULL, data BLOB);
                    CREATE TABLE descriptors (
                        image_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL,
                        cols INTEGER NOT NULL, data BLOB);
                    """
                )
                camera_params = np.array([focal, width / 2.0, height / 2.0], dtype=np.float64)
                conn.execute(
                    "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
                    (1, 0, width, height, _blob(camera_params), 1),
                )
                conn.executemany(
                    "INSERT INTO images VALUES (?, ?, ?)",
                    [(1, "map.jpg", 1), (2, "query.jpg", 1)],
                )
                conn.executemany(
                    "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
                    [
                        (1, len(xy), 2, _blob(xy)),
                        (2, len(xy), 2, _blob(xy)),
                    ],
                )
                conn.executemany(
                    "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
                    [
                        (1, len(descriptors), 128, _blob(descriptors)),
                        (2, len(query_descriptors), 128, _blob(query_descriptors)),
                    ],
                )

            (sparse / "images.txt").write_text(
                "1 1 0 0 0 0 0 0 1 map.jpg\n"
                + " ".join(
                    f"{float(pixel[0])} {float(pixel[1])} {int(point_id)}"
                    for pixel, point_id in zip(xy, point_ids)
                )
                + "\n",
                encoding="utf-8",
            )
            (sparse / "points3D.txt").write_text(
                "".join(
                    f"{int(point_id)} {point[0]} {point[1]} {point[2]} 255 255 255 0.1\n"
                    for point_id, point in zip(point_ids, p3d)
                ),
                encoding="utf-8",
            )
            (sparse / "cameras.txt").write_text(
                f"1 SIMPLE_PINHOLE {width} {height} {focal} {width / 2.0} {height / 2.0}\n",
                encoding="utf-8",
            )

            manifest = build_faiss_index(
                database=database,
                sparse_model=sparse,
                out_dir=index_dir,
                nlist=4,
                nprobe=4,
                training_sample_size=4096,
            )
            self.assertEqual(manifest["indexed_observations"], len(p3d))
            self.assertTrue(index_is_current(index_dir, database=database, sparse_model=sparse))
            self.assertEqual(
                json.loads((index_dir / "manifest.json").read_text())["algorithm"],
                "faiss_ivf_sq8_colmap_sift_2d3d",
            )

            relocalizer = FaissIVF3DRelocalizer(
                index_dir,
                nprobe=4,
                top_k=8,
                ratio=0.8,
                min_points=40,
                reprojection_error=2.0,
            )
            map_points = {
                int(point_id): SimpleNamespace(xyz=point.copy())
                for point_id, point in zip(point_ids, p3d)
            }
            pool, diagnostic = relocalizer.localize(
                database_path=database,
                image_name="query.jpg",
                map_points=map_points,
                expected_center=np.zeros(3, dtype=np.float64),
            )
            self.assertIsNotNone(pool, diagnostic)
            assert pool is not None
            self.assertGreaterEqual(pool["faiss_pnp_inliers"], 100)
            recovered_center = -qvec_to_rotmat(
                np.asarray(pool["colmap_qvec_world_to_camera"])
            ).T @ np.asarray(pool["colmap_tvec_world_to_camera"])
            np.testing.assert_allclose(recovered_center, np.zeros(3), atol=1.0e-4)


if __name__ == "__main__":
    unittest.main()
