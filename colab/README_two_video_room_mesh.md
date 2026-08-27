# Two-video room mesh handoff

## Local preparation

Run from the project root:

```bash
/Users/yamromano/Documents/PythonF4/PythonF4/.venv-metis/bin/python \
  scripts/prepare_room_mesh_colab_input.py \
  /Users/yamromano/Downloads/IMG_9961.MOV \
  /Users/yamromano/Downloads/IMG_9515.MOV \
  --output runtime/room_mesh_colab_20260802
```

This samples one frame every two seconds, performs adaptive blur and near-duplicate filtering, and creates three coverage-balanced scenes:

- `scene_pilot`: 48 frames, both videos, intended for T4 validation.
- `scene_medium`: 96 frames, intended for a 24 GB L4.
- `scene_full`: 220 frames, intended for a 40+ GB A100.

The resulting upload file is:

`runtime/room_mesh_colab_20260802/room_mesh_colab_input.zip`

## Colab

1. Create `MyDrive/room_reconstruction/` in Google Drive.
2. Upload `room_mesh_colab_input.zip` there.
3. Upload and open `colab/two_video_room_mesh_vggt.ipynb` in Colab.
4. Select a GPU runtime and run through the pilot validation.
5. Continue only if both `01` and `02` video prefixes register into one coherent trajectory.
6. Enable the larger solve and dense mesh cells only after the pilot passes.

The original videos and the working localization map are not modified.
