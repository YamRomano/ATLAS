#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def metric(results, name: str) -> float | None:
    value = getattr(results, name, None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate an ATLAS enemy-drone YOLO detector.")
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="enemy_drone_detector")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(f"Ultralytics is required for Enemy Lab training: {exc}") from exc

    dataset_yaml = args.dataset_yaml.resolve()
    if not dataset_yaml.exists():
        raise SystemExit(f"Dataset YAML does not exist: {dataset_yaml}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.project_dir.mkdir(parents=True, exist_ok=True)
    device = None if args.device.lower() == "auto" else args.device
    started = time.time()
    model = YOLO(args.model)
    train_result = model.train(
        data=str(dataset_yaml),
        epochs=max(1, args.epochs),
        imgsz=max(160, args.imgsz),
        batch=max(1, args.batch),
        device=device,
        project=str(args.project_dir),
        name=args.run_name,
        exist_ok=False,
        plots=True,
    )
    save_dir = Path(getattr(train_result, "save_dir", args.project_dir / args.run_name))
    best_source = save_dir / "weights" / "best.pt"
    if not best_source.exists():
        raise SystemExit(f"Training did not produce best.pt under {save_dir}")
    best_target = args.output_dir / "best.pt"
    shutil.copy2(best_source, best_target)

    best_model = YOLO(str(best_target))
    validation = best_model.val(data=str(dataset_yaml), split="val", imgsz=max(160, args.imgsz), device=device)
    box_metrics = getattr(validation, "box", None)
    summary = {
        "version": 1,
        "status": "trained_not_activated",
        "dataset_yaml": str(dataset_yaml),
        "base_model": args.model,
        "best_model": str(best_target),
        "save_dir": str(save_dir),
        "duration_seconds": time.time() - started,
        "metrics": {
            "precision": metric(box_metrics, "mp"),
            "recall": metric(box_metrics, "mr"),
            "map50": metric(box_metrics, "map50"),
            "map50_95": metric(box_metrics, "map"),
        },
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
