#!/bin/zsh
set -euo pipefail

out="outputs/spy_demo/capture_review"
mkdir -p "$out"

files=(
  "/var/folders/yj/0rnc6hws72n_zrcq_z42nzgr0000gn/T/TemporaryItems/NSIRD_screencaptureui_5onHT6/Screen Recording 2026-08-16 at 20.11.00.mov"
  "/var/folders/yj/0rnc6hws72n_zrcq_z42nzgr0000gn/T/TemporaryItems/NSIRD_screencaptureui_hwDyJ8/Screen Recording 2026-08-16 at 20.18.12.mov"
  "/var/folders/yj/0rnc6hws72n_zrcq_z42nzgr0000gn/T/TemporaryItems/NSIRD_screencaptureui_2NjbZT/Screen Recording 2026-08-16 at 20.19.37.mov"
)

intervals=(8 2 5)
for i in 1 2 3; do
  ffmpeg -y -v error -i "${files[$i]}" \
    -vf "fps=1/${intervals[$i]},scale=640:-2,tile=4x5:padding=8:margin=8:color=0x071019" \
    -frames:v 1 "$out/capture_${i}_sheet.jpg"
done
