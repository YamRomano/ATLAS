# ATLAS partner handoff — `David`

This branch is a self-contained restore snapshot of the ATLAS indoor patrol
system as used on 2026-08-23. It is intended for code review, exact saved-run
diagnostics, UI work, and continuing the physical DJI Mini 3 Pro integration.

## What is in this branch

- The ATLAS local web application and map/patrol UI (`viewer/` and
  `scripts/atlas_app_server.py`).
- The live DJI bridge and guarded patrol controller
  (`scripts/atlas_dji_live_bridge.py`).
- The bounded live localizer, in-process OpenCV SIFT/direct Faiss recovery,
  optical-flow tracker, and TSolve input/output path.
- Taught-route and multi-run visual route-recovery logic.
- The current safety gates, patrol audit tools, and regression tests.
- The current enemy-drone detector UI and the trained Neo detector weights.
- The TSolve/FARES sources previously referenced from directories outside this
  repository, copied under `vendor/tsolve/` so the algorithm can be inspected
  and its runtime rebuilt.
- Compact active-map visualization, Patrol 1 geometry/locks, and the precision
  route-recovery banks. Large binary assets are stored with Git LFS.

The `best 1 live lap` Git LFS restore bundle includes the full active COLMAP
working map, the exact saved live frames plus pose/control trace, and the exact
macOS arm64 COLMAP environment. Generated movie outputs and the Python virtual
environment remain excluded; Python versions are locked in `requirements.txt`.

## Current architecture

```text
DJI Mini 3 Pro camera
    -> Android MSDKRemote / OpenDJI TCP video
    -> newest decoded frame + capture timestamp
    -> bounded localizer
         global acquisition/relocalization:
           COLMAP SIFT registration against the fixed room map
         inter-frame tracking:
           pyramidal optical flow of mapped 2D/3D anchors
         pose solve:
           2D/3D correspondences -> TSolve/FARES PnP -> R,t
         route recovery:
           recorded multi-run visual anchors constrained to current leg/progress
    -> pose validity, freshness, route, heading and arrival gates
    -> guarded short OpenDJI motion/yaw commands
    -> stop/hover/relocalize on stale or unverified state
```

COLMAP is the persistent mapper because the room map and query registration use
SIFT. ORB is only used as a compact route-image matching layer; this code does
not run ORB-SLAM and therefore has no ORB-SLAM keyframe graph, map management,
loop closure, or inertial fusion. GPS is not used for room patrol.

The important design choice is that route recovery is a tracking prior, not
arrival evidence. A visual match may reconcile the displayed position and
search window, but movement/turn completion still requires current, fresh,
geometrically valid evidence. This prevents an old or aliased frame from
declaring a checkpoint reached and issuing the next command.

## Main code entry points

| Area | File |
|---|---|
| Local app/API and process orchestration | `scripts/atlas_app_server.py` |
| DJI stream, flight state machine and safety commands | `scripts/atlas_dji_live_bridge.py` |
| Bounded COLMAP/flow/TSolve localizer | `scripts/run_bounded_tsolve_video_stream.py` |
| Existing-map TSolve stream | `scripts/run_live_tsolve_existing_map_stream.py` |
| Route-constrained visual recovery | `scripts/patrol_visual_route_recovery.py` |
| Taught patrol recovery bank | `scripts/taught_patrol_recovery.py` |
| Real-run comparison | `scripts/audit_live_patrol_runs.py` |
| Browser client | `viewer/app.js`, `viewer/index.html`, `viewer/style.css` |

## Current selected patrol data

- Map: `map_copy_20260730_114851_cfefdc` / **Video Map 20:07:46 Copy**
- Patrol: `patrol_ms4br5xr_4xclts` / **Patrol 1**
- Precision baseline: `patrol_baseline_precision_20260813`
- Preserved full-lap session: `atlas_dji_live_20260823_154451_94903f`
- Preserved pose replay: `dji_live_20260823_154451_94903f`
- Map scale: 3,235 registered cameras and 440,249 sparse points

The branch includes the browser scene, map validation summary, route reference,
route/geometry locks, active taught recovery bank, precision poses, and both
single-run and multi-run visual recovery banks. The large physical map and
saved full-lap evidence are split into sub-2-GB Git LFS archive parts with
SHA-256 checksums.

## Known live limitation

The system can calculate good poses and has physical runs through checkpoints
1–4, but it does not yet complete two physical circles consistently. The
remaining problem is continuity and controller/localizer agreement, especially
after rotations and on the visually weak/repetitive 3→4 and 4→1 legs.

When tracking loses mapped anchors, a physically valid motion can occur during
the previous bounded command while the published model remains at its last
trusted position. A later global or route match can be delayed, rejected by a
step/route gate, or reconcile progress differently. Safety then withholds the
next command. This visible “stuck” state is usually a deliberate hover/abort,
not proof that TSolve cannot produce a pose.

Do not solve this by simply lowering every threshold. The recorded regressions
include both false progress (model reaches a checkpoint before the drone) and
stale progress (drone moves while model is held). Threshold changes must be
validated separately for acquisition, translation, turn completion, arrival,
and relocalization.

## Setup on another Mac

1. Install Git LFS before cloning/pulling this branch.
2. Create a Python environment and install `requirements.txt`.
3. Fetch LFS objects and verify the restore bundle:

   ```bash
   git lfs pull
   bash scripts/restore_best_1_live_lap.sh --verify-only
   ```

4. Restore the active map, saved run, and macOS arm64 COLMAP environment:

   ```bash
   bash scripts/restore_best_1_live_lap.sh --restore
   ```

5. Copy `config.example.json` to `config.json` if needed. The app now resolves
   missing workstation-specific TSolve/COLMAP paths to the vendored/restored
   paths automatically.
6. The PC-side `OpenDJI.py` used by this snapshot is vendored. The Android
   MSDKRemote application remains a separate deployment and must use the
   operator's own DJI developer key. No key or key-bearing APK is stored here.

Typical app startup from the repository root:

```bash
python3 scripts/atlas_app_server.py --port 8765
```

Open `http://127.0.0.1:8765/`.

For a first review, keep the aircraft landed and inspect saved replays and audit
outputs. Before any physical autonomous run, verify the phone IP, video
freshness, emergency landing access, coordinate alignment, selected map and
patrol, current recovery bank, and command status. Never treat an offline replay
as proof that a live control change is safe.

## Validation commands

The test suite uses the standard library `unittest` runner:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Audit saved live runs against the locked route:

```bash
python3 scripts/audit_live_patrol_runs.py \
  --replays viewer/public/maps/map_copy_20260730_114851_cfefdc/replays \
  --reference viewer/public/maps/map_copy_20260730_114851_cfefdc/taught_patrols/patrol_ms4br5xr_4xclts/reference.json \
  --from-date 2026-08-16 \
  --out /tmp/atlas_live_patrol_audit.json
```

## Suggested investigation order

1. Build one command/frame/pose timeline using capture time—not UI receipt
   time—and verify no command is evaluated against a pre-command frame.
2. Separate flight truth, estimated pose, accepted/published pose, and route
   progress in logs and UI. They are different state variables.
3. Reproduce failures from actual saved live frames; do not validate only by
   replaying the same baseline images used to construct the route bank.
4. Check the turn state machine before changing localization thresholds:
   yaw start/end, heading convention, endpoint evidence, and any fallback turn.
5. Add a motion budget tied to each issued command and frame freshness so a
   held pose can never cause repeated forward commands.
6. Only then evaluate a predictive filter or a continuous VIO/SLAM tracker,
   periodically anchored by COLMAP+TSolve. A Kalman filter can bridge short
   delays but cannot repair a wrong data association by itself.

## Restore bundle contents

`restore_assets/best_1_live_lap/manifest.json` is the machine-readable source
of truth. The bundle restores:

- `results/maps/map_copy_20260730_114851_cfefdc/` — database, source images,
  sparse models, text models, and FAISS index used by global relocalization.
- `viewer/public/live_dji_sessions/atlas_dji_live_20260823_154451_94903f/` —
  4,052 live frames and the complete command trace.
- `viewer/public/maps/map_copy_20260730_114851_cfefdc/replays/dji_live_20260823_154451_94903f/`
  — accepted/held pose stream and events.
- `tools/colmap-env/` — COLMAP 3.11.1 for macOS arm64.

Every archive part is tracked by Git LFS and verified against
`restore_assets/best_1_live_lap/SHA256SUMS` before extraction. On non-arm64 or
non-macOS systems, install COLMAP 3.11.1 separately instead of using the bundled
runtime.
