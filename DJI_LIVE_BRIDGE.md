# ATLAS DJI Live Bridge

This is the current live-drone integration layer for ATLAS. For the complete
architecture, safety state, and partner setup, read
[`docs/DAVID_HANDOFF.md`](docs/DAVID_HANDOFF.md).

It connects to the Android `MSDKRemote` app from
`DJI-MSDK-to-PC`, receives the DJI Mini camera stream, and writes frames in the
same format that the ATLAS TSolve localization pipeline already uses.

The bridge can receive video and telemetry **and can send guarded takeoff,
translation, yaw, hover, and landing commands**. Use it only with the physical
emergency controls available and after confirming the selected map, patrol,
fresh localization, and command status.

## Phone And Drone Setup

1. Connect the Android phone to the DJI RC/drone setup and open `MSDKRemote`.
2. Confirm the phone app shows live drone video.
3. Confirm the phone and Mac are on the same reachable network.
   A phone hotspot or home Wi-Fi is usually safer than university/public Wi-Fi.
4. Read the phone IP shown by the `MSDKRemote` app.

The app exposes:

- video on TCP `9999`
- control on TCP `9998`
- query/telemetry on TCP `9997`

The bridge connects through `OpenDJI.py`. Read-only frame capture is available,
but patrol mode also uses the OpenDJI control channel.

## Run The Frame Bridge

From the workspace root:

```bash
cd /Users/yamromano/Documents/PythonF4/PythonF4
.venv-metis/bin/python drone_phase1_replay_demo_20260706/scripts/atlas_dji_live_bridge.py \
  --phone-ip PHONE_IP_FROM_ANDROID_APP \
  --fps 2 \
  --show
```

If the OpenDJI repo is somewhere else:

```bash
.venv-metis/bin/python drone_phase1_replay_demo_20260706/scripts/atlas_dji_live_bridge.py \
  --phone-ip PHONE_IP_FROM_ANDROID_APP \
  --opendji-root /path/to/DJI-MSDK-to-PC-main \
  --fps 2 \
  --show
```

## Outputs

The session frame bank is written to:

```text
drone_phase1_replay_demo_20260706/data/dji_live/<session>/query_frames/
```

with:

```text
query_000000.jpg
query_000001.jpg
...
frames.csv
```

The ATLAS browser-visible live preview is written to:

```text
drone_phase1_replay_demo_20260706/viewer/public/live_dji/latest.jpg
drone_phase1_replay_demo_20260706/viewer/public/live_dji/status.json
```

This gives us a clean live source for the next layer:

```text
existing COLMAP map
    + incoming DJI frame bank
    -> 2D/3D correspondences
    -> TSolve R,t
    -> live ATLAS pose stream
```

## Run TSolve On The Captured Live Frames

After the bridge has saved some frames, run:

```bash
.venv-metis/bin/python drone_phase1_replay_demo_20260706/scripts/atlas_localize_dji_frame_bank.py \
  --map-id MAP_ID_FROM_ATLAS \
  --query-frames drone_phase1_replay_demo_20260706/data/dji_live/<session>/query_frames \
  --title "DJI Live Test"
```

For a quick first test, limit the frame bank:

```bash
.venv-metis/bin/python drone_phase1_replay_demo_20260706/scripts/atlas_localize_dji_frame_bank.py \
  --map-id MAP_ID_FROM_ATLAS \
  --query-frames drone_phase1_replay_demo_20260706/data/dji_live/<session>/query_frames \
  --max-frames 20 \
  --title "DJI Live Smoke Test"
```

This adds a new drone path to the selected ATLAS map. The frame-bank command is
offline and sends no movement commands; live patrol is a separate guarded mode.

## Next Layer: True Watch Mode

The current replay-localization pipeline already has the right fast design:

1. First frame: global COLMAP registration.
2. Following frames: local tracking by optical flow.
3. Every accepted frame: build PnP equations and run TSolve.

The next patch should attach the bridge's `query_frames` folder to that bounded
localizer in watch mode, so incoming live frames produce `poses_partial.json`
without requiring an uploaded MP4.
