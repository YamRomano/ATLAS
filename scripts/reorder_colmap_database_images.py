#!/usr/bin/env python3
"""Rewrite image IDs in a fresh COLMAP database to a manifest's time order.

Run this only before matching.  COLMAP's parallel feature extractor may assign
image IDs in worker-completion order, while sequential matching expects IDs to
follow capture time.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    database = args.database.resolve()
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    ordered_names = [str(item["relative_name"]) for item in manifest["mapping"]]
    desired = {name: index + 1 for index, name in enumerate(ordered_names)}

    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys=OFF")
        rows = connection.execute("SELECT image_id, name FROM images").fetchall()
        current = {str(name): int(image_id) for image_id, name in rows}
        if set(current) != set(desired):
            missing = sorted(set(desired) - set(current))
            extra = sorted(set(current) - set(desired))
            raise RuntimeError(f"Database/manifest mismatch: missing={missing[:5]} extra={extra[:5]}")
        match_count = connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        geometry_count = connection.execute(
            "SELECT COUNT(*) FROM two_view_geometries"
        ).fetchone()[0]
        if match_count or geometry_count:
            raise RuntimeError("Refusing to reorder a database after matching")

        offset = 1_000_000_000
        with connection:
            for table in ("images", "keypoints", "descriptors"):
                connection.execute(f"UPDATE {table} SET image_id = image_id + ?", (offset,))
            for name, old_id in current.items():
                temporary_id = old_id + offset
                new_id = desired[name]
                for table in ("images", "keypoints", "descriptors"):
                    connection.execute(
                        f"UPDATE {table} SET image_id = ? WHERE image_id = ?",
                        (new_id, temporary_id),
                    )

        verified = connection.execute(
            "SELECT image_id, name FROM images ORDER BY image_id"
        ).fetchall()
        if [str(name) for _, name in verified] != ordered_names:
            raise RuntimeError("Image-ID reorder verification failed")
        print(
            json.dumps(
                {
                    "database": str(database),
                    "images": len(verified),
                    "first": {"image_id": verified[0][0], "name": verified[0][1]},
                    "last": {"image_id": verified[-1][0], "name": verified[-1][1]},
                },
                indent=2,
            )
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
