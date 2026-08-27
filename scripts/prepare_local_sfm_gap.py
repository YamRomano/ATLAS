#!/usr/bin/env python3
"""Prepare a chronological dense-video slice for an isolated local SfM model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-frames", required=True, type=Path)
    parser.add_argument("--start-index", required=True, type=int)
    parser.add_argument("--end-index", required=True, type=int)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--selected-offset", type=int, default=260)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be positive")
    if args.end_index <= args.start_index:
        raise ValueError("--end-index must be greater than --start-index")

    source_dir = args.source_frames.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing SfM input: {out_dir}")
    image_dir = out_dir / "images" / "enhancement"
    image_dir.mkdir(parents=True)

    with (source_dir / "frames.csv").open(newline="", encoding="utf-8") as handle:
        source_rows = {
            Path(str(row["image_name"])).name: row for row in csv.DictReader(handle)
        }

    mapping: list[dict[str, Any]] = []
    image_list: list[str] = []
    indices = list(range(args.start_index, args.end_index + 1, args.stride))
    if indices[-1] != args.end_index:
        indices.append(args.end_index)
    for raw_index in indices:
        source_name = f"manual_patrol_{raw_index:06d}.jpg"
        source_path = source_dir / source_name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_row = source_rows[source_name]
        if raw_index % 10 == 0:
            selected_index = args.selected_offset + raw_index // 10
            local_name = f"manual_patrol_{selected_index:06d}.jpg"
            common_reference_name = f"enhancement/{local_name}"
        else:
            selected_index = None
            local_name = f"dense_raw_{raw_index:06d}.jpg"
            common_reference_name = None
        relative_name = f"enhancement/{local_name}"
        (image_dir / local_name).symlink_to(source_path)
        image_list.append(relative_name)
        mapping.append(
            {
                "relative_name": relative_name,
                "source_image_name": source_name,
                "raw_10fps_index": raw_index,
                "selected_1fps_index": selected_index,
                "common_reference_name": common_reference_name,
                "source_frame": int(source_row["source_frame"]),
                "time_sec": float(source_row["time_sec"]),
            }
        )

    (out_dir / "image_list.txt").write_text("\n".join(image_list) + "\n", encoding="utf-8")
    atomic_write_json(
        out_dir / "manifest.json",
        {
            "source_frames": str(source_dir),
            "start_index": args.start_index,
            "end_index": args.end_index,
            "stride": args.stride,
            "selected_offset": args.selected_offset,
            "image_count": len(mapping),
            "common_reference_image_count": sum(
                item["common_reference_name"] is not None for item in mapping
            ),
            "mapping": mapping,
        },
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "image_count": len(mapping),
                "common_reference_image_count": sum(
                    item["common_reference_name"] is not None for item in mapping
                ),
                "first": mapping[0],
                "last": mapping[-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
