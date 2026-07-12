# TSolve Drone Phase-1 Replay Demo

This folder builds the first real replay demo:

```text
iPhone room video
  -> extracted mapping frames
  -> COLMAP sparse 3D map

DJI Mini 3 Pro video
  -> extracted query frames
  -> COLMAP image registration against the iPhone map
  -> 2D-3D PnP correspondences
  -> FARES-format TSolve inputs
  -> static-C TSolve action matrices + numeric roots + FARES scoring
  -> R,t per drone frame
  -> local HTML replay viewer
```

The demo is intentionally a replay first. It proves the vision-to-TSolve localization path before we attach live video/control to the DJI drone.

## 0. Install COLMAP Locally

This Mac currently has no Homebrew, no conda, and no COLMAP. The included installer creates a local micromamba environment inside this folder.

```bash
cd /Users/yamromano/Documents/PythonF4/PythonF4/drone_phase1_replay_demo_20260706
bash scripts/install_colmap_micromamba.sh
```

After install, `config.json` points to:

```text
tools/colmap-env/bin/colmap
```

## 1. Run the Full Replay Pipeline

```bash
cd /Users/yamromano/Documents/PythonF4/PythonF4/drone_phase1_replay_demo_20260706
/Users/yamromano/Documents/PythonF4/PythonF4/.venv-metis/bin/python scripts/run_phase1.py --config config.json
```

Main outputs:

- `data/map_frames`: iPhone frames used for mapping.
- `data/query_frames`: drone frames used for localization.
- `results/colmap/sparse_map_text`: iPhone COLMAP map.
- `results/colmap/localized_model_text`: map plus registered drone query frames.
- `results/tsolve_inputs`: FARES/TSolve `input.json`, `p3d.csv`, `p2d.csv` cases.
- `results/tsolve_runtime`: packaged TSolve code used by the replay.
- `results/tsolve_replay_*`: TSolve R,t JSON outputs and timing CSV.
- `viewer/public`: browser-ready point cloud, pose track, and video symlink.

## 2. Open the Local Viewer

For the full ATLAS app with map creation, video upload, and TSolve replay controls, run:

```bash
cd /Users/yamromano/Documents/PythonF4/PythonF4/drone_phase1_replay_demo_20260706
/Users/yamromano/Documents/PythonF4/PythonF4/.venv-metis/bin/python scripts/atlas_app_server.py --port 8767
```

Then open:

```text
http://127.0.0.1:8767
```

From the start screen:

1. Press `Create Map`.
2. Choose `Upload Video` to build a COLMAP map from a video, or `Create Live` to capture from the PC/Mac camera.
3. During live capture, ATLAS shows the camera feed, the number of saved map frames, and a live 3D build preview. Press `Stop Mapping` when the room has enough coverage; COLMAP reconstruction starts from the saved frames.
4. The finished point cloud is added to the `3D Map Library` as a new map. It does not replace the original drone-demo map.
5. Select any map in the library. User-created maps can be deleted; the built-in demo map is protected.
6. After selecting a map, press `Upload Drone Video`.
7. The backend extracts drone frames, localizes them against the selected map, builds TSolve PnP inputs, runs TSolve, and refreshes that map with live-style route, \(R\), and \(t\).

The live mapping preview is immediate, but the true COLMAP sparse 3D reconstruction still starts after capture stops. A real-time dense mesh would require a SLAM or live reconstruction engine; this app keeps COLMAP as the reliable offline mapper and TSolve as the online localization solver.

Map-library files:

- `viewer/public/maps/manifest.json`: persistent list of available maps.
- `viewer/public/maps/default_demo`: protected original TSolve drone replay.
- `viewer/public/maps/<generated-map-id>`: each uploaded/live-created COLMAP map.

The older static-only server is still useful when you only want to view already generated data:

```bash
cd /Users/yamromano/Documents/PythonF4/PythonF4/drone_phase1_replay_demo_20260706
python3 scripts/serve_viewer.py
```

Then open:

```text
http://127.0.0.1:8765
```

The viewer now shows the sparse COLMAP map in a PCA-aligned room frame with Top, Side, and 3D views, a floor grid, the room footprint, map-camera positions, and the TSolve drone path. This makes localization understandable, but it is still a sparse SfM map, not a textured room mesh.

The current drone marker uses the uploaded DJI Mini 3 Pro GLB from `/Users/yamromano/Downloads/dji-mini-3-pro.zip`. The GLB is converted into a lightweight dependency-free wireframe asset at `viewer/public/models/dji-mini-3-pro-wire.json`, while the original GLB is also copied to `viewer/public/models/dji-mini-3-pro.glb`.

## Notes

- Mapping frames are extracted sparsely from the iPhone video because the source is about 120 fps.
- Query frames are extracted from the drone video at lower frequency first; later we can raise the rate.
- COLMAP supplies the geometric 2D-3D correspondences.
- TSolve receives only the PnP problem: `K`, `p3d.csv`, `p2d.csv`.
- Proof checks and msolve comparison remain separate from the timed demo runtime.
- For a visually realistic room in the final demo, add a LiDAR/dense mesh layer (`.ply`, `.obj`, or `.glb`) and overlay the TSolve trajectory on top of it.

## 3. Live Webcam COLMAP Map Test

This path tests whether COLMAP alone can build a good enough room map from the PC/Mac camera. It captures sparse frames live, runs COLMAP map-only reconstruction, and exports the resulting map to the ATLAS viewer. This does not localize the DJI drone yet; it is for verifying the map quality first.

```bash
cd /Users/yamromano/Documents/PythonF4/PythonF4/drone_phase1_replay_demo_20260706
/Users/yamromano/Documents/PythonF4/PythonF4/.venv-metis/bin/python scripts/run_webcam_map_demo.py \
  --config config.json \
  --duration 75 \
  --fps 1.5 \
  --camera-index 0
```

Walk or rotate the laptop/USB camera slowly around the room with strong overlap. For a good model, prefer:

- slow motion;
- many side-looking views, not only forward motion;
- textured surfaces;
- avoid shiny/blank walls;
- at least 60-120 saved frames.

If the preview window is not useful or camera GUI access is awkward, add:

```bash
  --no-preview
```

After it finishes:

```bash
python3 scripts/serve_viewer.py
```

Open:

```text
http://127.0.0.1:8765
```

Outputs:

- `data/webcam_map_frames`: captured frames.
- `results/webcam_colmap_map`: COLMAP database and sparse model.
- `viewer/public/scene.json`: browser-ready COLMAP map.
- `viewer/public/poses.json`: empty for this map-only test.

If this COLMAP map looks good, the next step is to use it as the persistent ATLAS patrol map and localize DJI query frames against it. That gives TSolve inputs in the same COLMAP coordinate frame, avoiding the external LiDAR-mesh alignment problem.

## Current Successful Run

The current local run used:

- iPhone map video: `/Users/yamromano/Downloads/IMG_9452.MOV`
- DJI Mini 3 Pro query video: `/Users/yamromano/Downloads/2026-07-06-9-08-01-460 AM.MP4`
- iPhone map extraction: 1.5 fps, max image size 1200 px
- Drone query extraction: 0.5 fps, max image size 1200 px

Measured outputs:

- COLMAP map/query model: 10,813 sparse points, 47 map camera poses.
- Localized drone frames: 37 frames with at least 40 2D-3D correspondences.
- TSolve online replay: 36/36 successful after one offline training frame.
- TSolve branch count: 1 branch, 0 online branch forks.
- Median TSolve total per localized frame: 57.96 ms.
- Median TSolve polynomial stage: 46.90 ms.
- Median static-C action stage: 3.87 ms.
- Median root/scoring stage: 43.51 ms.

Current viewer data is in `viewer/public`, generated from `results/colmap/localized_model_retry_text2` and `results/tsolve_replay_retry2`.
