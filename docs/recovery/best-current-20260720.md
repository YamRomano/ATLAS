# ATLAS Best Current Snapshot - 2026-07-20

Branch: `best-current-20260720`

This branch is a restore point for the current best ATLAS demo state before the next round of patrol, localization, and drone-control changes.

## Included

- ATLAS app/server source code.
- Viewer UI, styling, DJI GLB overlay code, and branding/model assets.
- Live DJI bridge and localization scripts.
- Launch/keepalive helper scripts.
- Current lightweight runtime config/state.
- Current map metadata, scene JSON, patrol/path/wall/obstacle JSON state, and replay pose data under `viewer/public/maps`.
- Enemy-drone calibration UI/data folder structure.

## Intentionally Not Included

These are kept out of git because they are too large or generated live:

- `data/`
- `results/`
- raw uploaded videos and MP4/MOV media files
- `viewer/public/media/`
- `viewer/public/live_dji_sessions/`
- saved live DJI frame banks
- local Python/Conda/tool environments

To recreate the current app from this branch, check out the branch, run the ATLAS server, and re-upload any raw videos if a replay media file is missing. The map geometry and saved pose/path metadata are stored in git so the app can recover the current map library structure.
