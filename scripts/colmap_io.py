from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Iterable

import numpy as np


@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]

    def K(self) -> np.ndarray:
        model = self.model.upper()
        p = self.params
        if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            f, cx, cy = p[:3]
            fx = fy = f
        elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
            fx, fy, cx, cy = p[:4]
        else:
            if len(p) >= 4:
                fx, fy, cx, cy = p[:4]
            else:
                f, cx, cy = p[:3]
                fx = fy = f
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


@dataclass
class Image:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3d_ids: np.ndarray


@dataclass
class Point3D:
    point3d_id: int
    xyz: np.ndarray
    rgb: tuple[int, int, int]
    error: float


def _data_lines(path: Path) -> Iterable[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line


def read_cameras_text(path: Path) -> dict[int, Camera]:
    cams = {}
    for line in _data_lines(path):
        toks = line.split()
        camera_id = int(toks[0])
        model = toks[1]
        width = int(toks[2])
        height = int(toks[3])
        params = [float(x) for x in toks[4:]]
        cams[camera_id] = Camera(camera_id, model, width, height, params)
    return cams


def read_points3d_text(path: Path) -> dict[int, Point3D]:
    pts = {}
    for line in _data_lines(path):
        toks = line.split()
        point_id = int(toks[0])
        xyz = np.array([float(toks[1]), float(toks[2]), float(toks[3])], dtype=float)
        rgb = (int(toks[4]), int(toks[5]), int(toks[6]))
        error = float(toks[7])
        pts[point_id] = Point3D(point_id, xyz, rgb, error)
    return pts


def read_images_text(path: Path) -> dict[int, Image]:
    raw = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    images = {}
    i = 0
    while i + 1 < len(raw):
        header = raw[i].strip()
        points = raw[i + 1].strip()
        toks = header.split()
        image_id = int(toks[0])
        qvec = np.array([float(x) for x in toks[1:5]], dtype=float)
        tvec = np.array([float(x) for x in toks[5:8]], dtype=float)
        camera_id = int(toks[8])
        name = " ".join(toks[9:])

        pt_toks = points.split()
        triples = len(pt_toks) // 3
        xys = np.zeros((triples, 2), dtype=float)
        pids = np.zeros((triples,), dtype=np.int64)
        for j in range(triples):
            xys[j, 0] = float(pt_toks[3 * j])
            xys[j, 1] = float(pt_toks[3 * j + 1])
            pids[j] = int(pt_toks[3 * j + 2])
        images[image_id] = Image(image_id, qvec, tvec, camera_id, name, xys, pids)
        i += 2
    return images


CAMERA_MODEL_BY_ID = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def read_cameras_binary(path: Path) -> dict[int, Camera]:
    data = memoryview(path.read_bytes())
    off = 0

    def unpack(fmt: str):
        nonlocal off
        size = struct.calcsize(fmt)
        value = struct.unpack_from(fmt, data, off)
        off += size
        return value

    (count,) = unpack("<Q")
    cams = {}
    for _ in range(count):
        camera_id, model_id = unpack("<ii")
        width, height = unpack("<QQ")
        model, param_count = CAMERA_MODEL_BY_ID.get(model_id, (f"MODEL_{model_id}", 0))
        params = list(unpack("<" + "d" * param_count)) if param_count else []
        cams[int(camera_id)] = Camera(int(camera_id), model, int(width), int(height), params)
    return cams


def read_images_binary(path: Path) -> dict[int, Image]:
    data = memoryview(path.read_bytes())
    off = 0

    def unpack(fmt: str):
        nonlocal off
        size = struct.calcsize(fmt)
        value = struct.unpack_from(fmt, data, off)
        off += size
        return value

    (count,) = unpack("<Q")
    images = {}
    for _ in range(count):
        (image_id,) = unpack("<i")
        qvec = np.array(unpack("<dddd"), dtype=float)
        tvec = np.array(unpack("<ddd"), dtype=float)
        (camera_id,) = unpack("<i")
        end = off
        while data[end] != 0:
            end += 1
        name = bytes(data[off:end]).decode("utf-8", errors="replace")
        off = end + 1
        (num_points2d,) = unpack("<Q")
        xys = np.zeros((num_points2d, 2), dtype=float)
        pids = np.zeros((num_points2d,), dtype=np.int64)
        for j in range(num_points2d):
            x, y, pid = unpack("<ddq")
            xys[j, 0] = x
            xys[j, 1] = y
            pids[j] = pid
        images[int(image_id)] = Image(int(image_id), qvec, tvec, int(camera_id), name, xys, pids)
    return images


def read_cameras_model(path: Path) -> dict[int, Camera]:
    text = path / "cameras.txt"
    binary = path / "cameras.bin"
    if text.exists():
        return read_cameras_text(text)
    return read_cameras_binary(binary)


def read_images_model(path: Path) -> dict[int, Image]:
    text = path / "images.txt"
    binary = path / "images.bin"
    if text.exists():
        return read_images_text(text)
    return read_images_binary(binary)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    q = np.asarray(qvec, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=float,
    )


def camera_center(image: Image) -> np.ndarray:
    R = qvec_to_rotmat(image.qvec)
    return -R.T @ image.tvec
