import * as THREE from "./vendor/three.module.js";
import { GLTFLoader } from "./vendor/GLTFLoader.js";

const REFERENCE_MAP_ID = "map_copy_20260730_114851_cfefdc";
const MESH_GLB_URL = "./public/camera_path_lab/room_scan_textured.glb";
const MESH_GLB_VERSION = "20260803-wide-ceiling-cut";
const MESH_FALLBACK_URL = "./public/camera_path_lab/good_copy_mesh.json";
const CAMERA_MODEL_URL = "./public/camera_path_lab/analog_camera.glb";
const ROOM_MESH_OPACITY = 0.82;
const POSITION_ONLY_CAMERA_MARKER = true;
// The textured scan is visual-only. Its floor slab and horizontal room axis
// were measured against the copy map's four-wall footprint, so correcting it
// here cannot alter localization, patrol coordinates, or the production map.
const ROOM_SCAN_SOURCE_FLOOR_Y = -1.362;
const ROOM_SCAN_SOURCE_AXIS_DEG = 174.616;
const ROOM_SCAN_SOURCE_CENTER_XZ = new THREE.Vector2(-0.929, 0.952);
const ROOM_SCAN_SOURCE_LONG_M = 16.237;
// Final horizontal ICP refinement measured against the copy map's colored
// reference surface after applying the room-footprint scale below. Keeping it
// explicit makes the scan/path registration deterministic in every browser.
const ROOM_SCAN_REFINEMENT_YAW_DEG = 0.420824;
const ROOM_SCAN_REFINEMENT_XZ = new THREE.Vector2(-0.012658, 0.158903);
const FLOOR_GRID_CLEARANCE_M = 0.025;
const LIVE_START_BUFFER_SECONDS = 4.0;
const LIVE_RESUME_BUFFER_SECONDS = 2.0;
const LIVE_STALL_GUARD_SECONDS = 0.22;
const PREVIEW_TIMING_BLEND_FRAMES = 40;
const PREVIEW_TIMING_MIN_SECTION_FRAMES = PREVIEW_TIMING_BLEND_FRAMES + 1;
const PREVIEW_TIMING_MAX_SECTIONS = 12;
const DEFAULT_PREVIEW_TIMING_SEGMENTS = Object.freeze([
  Object.freeze({ start_frame: 0, end_frame: 333, offset_sec: -4.2 }),
  Object.freeze({ start_frame: 334, end_frame: 599, offset_sec: 2.0 }),
  Object.freeze({ start_frame: 600, end_frame: 749, offset_sec: -1.5 }),
  Object.freeze({ start_frame: 750, end_frame: null, offset_sec: 1.0 }),
]);

const el = (id) => document.getElementById(id);
const stage = document.querySelector(".lab-stage");
const container = el("lab-canvas");
const videoCard = document.querySelector(".video-card");
const videoInput = el("video-input");
const sourceVideo = el("source-video");
const videoPanelSizeInput = el("video-panel-size");
const videoPanelSizeValue = el("video-panel-size-value");
const statusDot = el("status-dot");
const statusText = el("status-text");
const cameraLabel = el("camera-label");
const coordinates = el("camera-coordinates");
const coordinateKicker = el("coordinate-kicker");
const coordinateLink = el("coordinate-link");
const coordinateLinkPath = el("coordinate-link-path");
const coordinateCameraRing = el("coordinate-camera-ring");
const coordinateCameraDot = el("coordinate-camera-dot");
const coordinateLabelDot = el("coordinate-label-dot");
const meshBadge = el("mesh-badge");
const startButton = el("start-button");
const stopButton = el("stop-button");
const replayButton = el("replay-button");
const playbackSpeedControl = el("playback-speed-control");
const playbackSpeedButtons = [...playbackSpeedControl.querySelectorAll("[data-playback-rate]")];
const adjustPreviewPathButton = el("adjust-preview-path");
const placePreviewStartButton = el("place-preview-start");
const previewScaleControl = el("preview-scale-control");
const previewMotionScaleInput = el("preview-motion-scale");
const previewMotionValue = el("preview-motion-value");
const lockPreviewPathButton = el("lock-preview-path");
const previewTimingSegmentsPanel = el("preview-timing-segments");
const previewTimingSegmentList = el("preview-timing-segment-list");
const togglePreviewTimingButton = el("toggle-preview-timing");
const previewTransformControls = el("preview-transform-controls");
const previewRotationInput = el("preview-rotation");
const previewRotationValue = el("preview-rotation-value");
const previewScaleXInput = el("preview-scale-x");
const previewScaleXValue = el("preview-scale-x-value");
const previewScaleYInput = el("preview-scale-y");
const previewScaleYValue = el("preview-scale-y-value");

let renderer = null;
let fallbackCanvas = null;
let fallbackContext = null;
let fallbackVoxels = [];
let fallbackTriangles = [];
let fallbackWalls = [];
let fallbackPath = [];
let fallbackHeading = null;
let fallbackDirty = true;
let targetHeading = null;
let displayedHeading = null;
let previousAnimationTime = performance.now();
let videoPanelWidth = 400;
let videoAspectRatio = 9 / 16;
let canvasResizeObserver = null;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x071b2c, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.localClippingEnabled = true;
  container.appendChild(renderer.domElement);
} catch (_) {
  container.classList.add("canvas-fallback");
  fallbackCanvas = document.createElement("canvas");
  fallbackCanvas.className = "fallback-canvas";
  fallbackCanvas.setAttribute("aria-label", "Camera path map");
  fallbackContext = fallbackCanvas.getContext("2d", { alpha: false });
  container.appendChild(fallbackCanvas);
  if (!fallbackContext) {
    const notice = document.createElement("div");
    notice.className = "webgl-notice";
    notice.textContent = "Map preview unavailable";
    container.appendChild(notice);
  }
}

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a2d49, 0.012);
// Keep the tracked camera in a dedicated foreground scene. The room mesh is
// intentionally dense, so rendering both into the same depth buffer lets the
// map hide the camera as soon as the user orbits behind a wall.
const cameraOverlayScene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(44, 1, 0.02, 180);
const orbit = { target: new THREE.Vector3(-1.7, 0.4, 0.2), yaw: -0.84, pitch: 0.52, distance: 17.5 };
const wallsGroup = new THREE.Group();
const pathGroup = new THREE.Group();
const pathGlowGroup = new THREE.Group();
scene.add(wallsGroup, pathGroup);
cameraOverlayScene.add(pathGlowGroup);

scene.add(new THREE.HemisphereLight(0xffffff, 0xd9eef7, 1.45));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.05);
keyLight.position.set(5, 11, 4);
scene.add(keyLight);
cameraOverlayScene.add(new THREE.HemisphereLight(0xffffff, 0xd9eef7, 1.45));
const cameraOverlayLight = new THREE.DirectionalLight(0xffffff, 1.05);
cameraOverlayLight.position.copy(keyLight.position);
cameraOverlayScene.add(cameraOverlayLight);
const floorGrid = new THREE.GridHelper(24, 24, 0x69b7d8, 0x225a78);
floorGrid.position.y = -1.105;
floorGrid.material.opacity = 0.32;
floorGrid.material.transparent = true;
scene.add(floorGrid);

let mapEntry = null;
let roomMatrix = null;
let selectedFile = null;
let selectedVideoSubmitted = false;
let uploadInFlight = false;
let videoObjectUrl = null;
let latestPoseUrl = null;
let latestPathSignature = "";
let latestRenderedPoseKey = "";
let pathLine = null;
let pathGlowLine = null;
let cameraRig = null;
let cameraPosePosition = null;
let wallsVisible = true;
let ceilingVisible = true;
let ceilingCutY = 2.08;
const roomMeshMaterials = [];
let localizedPoses = [];
let displayPoses = [];
let firstLocalizationReady = false;
let currentJobStatus = "idle";
let replayActive = false;
let replayPoseIndex = -1;
let replayPathIndex = -1;
let livePlaybackActive = false;
let livePlaybackBuffering = false;
let livePlaybackUserPaused = false;
let livePlaybackSyntheticClock = false;
let livePlaybackClockStartMs = 0;
let livePlaybackClockStartTime = 0;
let livePlaybackLastMediaTime = 0;
let livePlaybackLastMediaProgressMs = 0;
let livePlaybackLastSeekMs = 0;
let livePlaybackPoseIndex = -1;
let livePlaybackPathIndex = -1;
let streamComplete = false;
let loadedStreamMediaUrl = "";
let roomCeilingY = 2.42;
let roomFloorY = -1.080262;
let roomFootprintCenter = new THREE.Vector3(-1.658511, roomFloorY, 0.300419);
let roomLongAxisDeg = 172.381;
let roomLongAxisLength = 12.0;
let currentInputFrameIndex = null;
let latestPoseFrameIndex = null;
let latestDisplayHeld = false;
let streamExpectedCount = 0;
let streamOfflineValidated = false;
let streamValidationPreview = false;
let previewCalibrationKey = "";
let previewSourceStart = null;
let previewTargetStart = null;
let previewMovementScale = 0.35;
let previewScaleX = 0.35;
let previewScaleY = 0.35;
let previewRotationDeg = 0;
let previewPlaceStartMode = false;
let previewAdjustMode = false;
let previewAdjustDrag = null;
let previewCalibrationLocked = false;
let previewCalibrationSaving = false;
let activePreviewReplayId = "";
let pendingPreviewCalibration = null;
let previewTimingSegments = DEFAULT_PREVIEW_TIMING_SEGMENTS.map((segment) => ({ ...segment }));
let previewTimingSaveTimer = 0;
let previewTimingCollapsed = false;
let playbackRate = 1;

function setOrbit(top = false) {
  if (top) {
    orbit.target.set(-1.7, 0.2, 0.2);
    orbit.yaw = 0;
    orbit.pitch = Math.PI / 2 - 0.025;
    orbit.distance = 18.5;
  } else {
    orbit.target.set(-1.7, 0.45, 0.2);
    orbit.yaw = -0.84;
    orbit.pitch = 0.52;
    orbit.distance = 17.5;
  }
  fallbackDirty = true;
}

function updateOrbitCamera() {
  const cp = Math.cos(orbit.pitch);
  camera.position.set(
    orbit.target.x + orbit.distance * cp * Math.sin(orbit.yaw),
    orbit.target.y + orbit.distance * Math.sin(orbit.pitch),
    orbit.target.z + orbit.distance * cp * Math.cos(orbit.yaw),
  );
  camera.up.set(0, 1, 0);
  camera.lookAt(orbit.target);
  camera.updateMatrixWorld(true);
}

function resize() {
  const width = Math.max(1, container.clientWidth);
  const height = Math.max(1, container.clientHeight);
  if (renderer) renderer.setSize(width, height, false);
  if (fallbackCanvas && fallbackContext) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    fallbackCanvas.width = Math.round(width * dpr);
    fallbackCanvas.height = Math.round(height * dpr);
    fallbackCanvas.style.width = `${width}px`;
    fallbackCanvas.style.height = `${height}px`;
    fallbackContext.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  fallbackDirty = true;
}

function videoPanelWidthBounds() {
  const minimum = window.innerWidth <= 760 ? 220 : 260;
  const captionAndMargins = window.innerWidth <= 760 ? 116 : 130;
  const maxByHeight = (Math.max(320, window.innerHeight - captionAndMargins)) * videoAspectRatio;
  const maximum = Math.max(
    minimum,
    Math.floor(Math.min(520, window.innerWidth * 0.52, maxByHeight) / 10) * 10,
  );
  return { minimum, maximum };
}

function setVideoPanelWidth(value, { persist = true } = {}) {
  const { minimum, maximum } = videoPanelWidthBounds();
  videoPanelSizeInput.min = String(Math.round(minimum));
  videoPanelSizeInput.max = String(Math.round(maximum));
  videoPanelWidth = Math.round(
    THREE.MathUtils.clamp(Number(value) || 400, minimum, maximum) / 10,
  ) * 10;
  stage.style.setProperty("--video-panel-width", `${videoPanelWidth}px`);
  videoPanelSizeInput.value = String(Math.round(videoPanelWidth));
  videoPanelSizeValue.value = `${Math.round(videoPanelWidth)} px`;
  if (persist) {
    try {
      window.localStorage.setItem("camera-path-video-panel-width", String(videoPanelWidth));
    } catch (_) { /* The layout still works when browser storage is unavailable. */ }
  }
  window.requestAnimationFrame(resize);
}

function initializeVideoPanelLayout() {
  let savedWidth = 400;
  try {
    savedWidth = Number(window.localStorage.getItem("camera-path-video-panel-width")) || savedWidth;
  } catch (_) { /* Use the default size. */ }
  setVideoPanelWidth(savedWidth, { persist: false });
}

function updateVideoAspectRatio() {
  const width = Number(sourceVideo.videoWidth);
  const height = Number(sourceVideo.videoHeight);
  if (width > 0 && height > 0) {
    videoAspectRatio = width / height;
    videoCard.style.setProperty("--phone-video-aspect", `${width} / ${height}`);
  } else {
    videoAspectRatio = 9 / 16;
    videoCard.style.setProperty("--phone-video-aspect", "9 / 16");
  }
  setVideoPanelWidth(videoPanelWidth, { persist: false });
}

function projectFallback(point) {
  const projected = point.clone().project(camera);
  if (!Number.isFinite(projected.x) || projected.z < -1 || projected.z > 1) return null;
  return {
    x: (projected.x * 0.5 + 0.5) * container.clientWidth,
    y: (-projected.y * 0.5 + 0.5) * container.clientHeight,
    z: projected.z,
  };
}

function drawFallbackMesh(context) {
  if (fallbackTriangles.length) {
    const triangles = fallbackTriangles
      .filter((triangle) => ceilingVisible || (triangle.a.y + triangle.b.y + triangle.c.y) / 3 <= ceilingCutY)
      .map((triangle) => {
      const a = projectFallback(triangle.a);
      const b = projectFallback(triangle.b);
      const c = projectFallback(triangle.c);
      return a && b && c ? { a, b, c, z: (a.z + b.z + c.z) / 3 } : null;
    }).filter(Boolean).sort((a, b) => b.z - a.z);
    context.lineWidth = 0.45;
    for (const triangle of triangles) {
      context.beginPath();
      context.moveTo(triangle.a.x, triangle.a.y);
      context.lineTo(triangle.b.x, triangle.b.y);
      context.lineTo(triangle.c.x, triangle.c.y);
      context.closePath();
      context.fillStyle = "rgba(174, 220, 239, 0.32)";
      context.strokeStyle = "rgba(82, 155, 187, 0.14)";
      context.fill();
      context.stroke();
    }
    return;
  }
  const points = fallbackVoxels
    .filter((voxel) => ceilingVisible || voxel.point.y <= ceilingCutY)
    .map((voxel) => {
    const projected = projectFallback(voxel.point);
    return projected ? { ...projected, color: voxel.color, weight: voxel.weight } : null;
  }).filter(Boolean).sort((a, b) => b.z - a.z);
  for (const point of points) {
    const radius = 1.15 + Math.min(1.15, Number(point.weight || 1) * 0.04);
    context.globalAlpha = 0.5;
    context.fillStyle = point.color;
    context.fillRect(point.x - radius, point.y - radius, radius * 2, radius * 2);
  }
  context.globalAlpha = 1;
}

function drawFallbackWalls(context) {
  if (!wallsVisible) return;
  context.lineWidth = 1;
  for (const wall of fallbackWalls) {
    const points = wall.map(projectFallback).filter(Boolean);
    if (points.length < 3) continue;
    context.beginPath();
    context.moveTo(points[0].x, points[0].y);
    points.slice(1).forEach((point) => context.lineTo(point.x, point.y));
    context.closePath();
    context.fillStyle = "rgba(111, 190, 224, 0.045)";
    context.strokeStyle = "rgba(71, 147, 181, 0.32)";
    context.fill();
    context.stroke();
  }
}

function drawFallbackPath(context) {
  const points = fallbackPath.map(projectFallback).filter(Boolean);
  if (points.length < 2) return;
  context.save();
  context.lineCap = "round";
  context.lineJoin = "round";
  context.beginPath();
  context.moveTo(points[0].x, points[0].y);
  points.slice(1).forEach((point) => context.lineTo(point.x, point.y));
  context.strokeStyle = "rgba(77, 187, 234, 0.20)";
  context.lineWidth = 8;
  context.shadowColor = "rgba(91, 204, 248, 0.55)";
  context.shadowBlur = 14;
  context.stroke();
  context.shadowBlur = 7;
  context.strokeStyle = "#a8e8ff";
  context.lineWidth = 3.4;
  context.stroke();
  context.restore();
}

function drawFallbackCamera(context) {
  if (!cameraPosePosition) return;
  const position = projectFallback(cameraPosePosition);
  if (!position) return;
  context.save();
  context.translate(position.x, position.y);
  context.lineCap = "round";
  context.shadowColor = "rgba(88, 200, 243, 0.72)";
  context.shadowBlur = 13;
  context.strokeStyle = "rgba(88, 200, 243, 0.28)";
  context.lineWidth = 10;
  context.beginPath();
  context.moveTo(-12, -12);
  context.lineTo(12, 12);
  context.moveTo(12, -12);
  context.lineTo(-12, 12);
  context.stroke();
  context.shadowBlur = 4;
  context.strokeStyle = "#effbff";
  context.lineWidth = 3;
  context.stroke();
  context.restore();
}

function drawFallbackScene() {
  if (!fallbackContext || !fallbackCanvas) return;
  const width = container.clientWidth;
  const height = container.clientHeight;
  const context = fallbackContext;
  context.clearRect(0, 0, width, height);
  const background = context.createLinearGradient(0, 0, width, height);
  background.addColorStop(0, "#061725");
  background.addColorStop(0.52, "#0d3454");
  background.addColorStop(1, "#17587d");
  context.fillStyle = background;
  context.fillRect(0, 0, width, height);
  const glow = context.createRadialGradient(width * 0.48, height * 0.44, 0, width * 0.48, height * 0.44, Math.max(width, height) * 0.62);
  glow.addColorStop(0, "rgba(105, 188, 224, 0.24)");
  glow.addColorStop(1, "rgba(6, 23, 37, 0)");
  context.fillStyle = glow;
  context.fillRect(0, 0, width, height);
  drawFallbackMesh(context);
  drawFallbackWalls(context);
  drawFallbackPath(context);
  drawFallbackCamera(context);
  fallbackDirty = false;
}

function applyDisplayedHeading(heading) {
  if (!heading || !cameraRig || !cameraPosePosition) return;
  if (POSITION_ONLY_CAMERA_MARKER) {
    fallbackHeading = null;
    cameraRig.quaternion.identity();
    return;
  }
  fallbackHeading = heading.clone();
  const target = cameraPosePosition.clone().add(heading);
  cameraRig.up.set(0, 1, 0);
  cameraRig.lookAt(target);
  cameraRig.rotateY(Math.PI);
}

function animate(now = performance.now()) {
  requestAnimationFrame(animate);
  const deltaSeconds = Math.min(0.1, Math.max(0, (now - previousAnimationTime) / 1000));
  previousAnimationTime = now;
  if (replayActive) updateReplayFrame();
  else if (livePlaybackActive) updateLivePlaybackFrame();
  if (targetHeading) {
    if (!displayedHeading) displayedHeading = targetHeading.clone();
    const angleBefore = displayedHeading.angleTo(targetHeading);
    if (angleBefore > 0.0005) {
      const blend = 1 - Math.exp(-5.2 * deltaSeconds);
      displayedHeading.lerp(targetHeading, blend).normalize();
      fallbackDirty = true;
    } else {
      displayedHeading.copy(targetHeading);
    }
    applyDisplayedHeading(displayedHeading);
  }
  updateOrbitCamera();
  updateCameraLabel();
  if (renderer) {
    renderer.render(scene, camera);
    // Erase only the room's depth values, then draw the real 3D camera pose.
    // Its position and rotation remain world-correct, but walls cannot occlude it.
    renderer.autoClear = false;
    renderer.clearDepth();
    renderer.render(cameraOverlayScene, camera);
    renderer.autoClear = true;
  }
  else if (fallbackDirty) drawFallbackScene();
}

function updateCameraLabel() {
  if (!cameraPosePosition || !cameraRig?.visible) {
    cameraLabel.hidden = true;
    coordinateLink.hidden = true;
    return;
  }
  const cameraProjected = cameraPosePosition.clone().project(camera);
  const elevatedPosition = cameraPosePosition.clone();
  elevatedPosition.y = Math.max(roomCeilingY + 0.68, cameraPosePosition.y + 1.5);
  const labelProjected = elevatedPosition.project(camera);
  if (
    cameraProjected.z < -1 || cameraProjected.z > 1
    || labelProjected.z < -1 || labelProjected.z > 1
  ) {
    cameraLabel.hidden = true;
    coordinateLink.hidden = true;
    return;
  }
  const width = container.clientWidth;
  const height = container.clientHeight;
  const cameraX = (cameraProjected.x * 0.5 + 0.5) * width;
  const cameraY = (-cameraProjected.y * 0.5 + 0.5) * height;
  const rawLabelX = (labelProjected.x * 0.5 + 0.5) * width;
  const rawLabelY = (-labelProjected.y * 0.5 + 0.5) * height;
  const labelX = THREE.MathUtils.clamp(rawLabelX, 132, Math.max(132, width - 132));
  const labelY = THREE.MathUtils.clamp(rawLabelY, 112, Math.max(112, height - 72));
  cameraLabel.hidden = false;
  coordinateLink.hidden = false;
  cameraLabel.style.left = `${labelX}px`;
  cameraLabel.style.top = `${labelY}px`;
  const bendY = labelY + Math.max(18, (cameraY - labelY) * 0.28);
  coordinateLinkPath.setAttribute("d", `M ${cameraX.toFixed(1)} ${cameraY.toFixed(1)} L ${cameraX.toFixed(1)} ${bendY.toFixed(1)} Q ${cameraX.toFixed(1)} ${labelY.toFixed(1)} ${labelX.toFixed(1)} ${labelY.toFixed(1)}`);
  for (const marker of [coordinateCameraRing, coordinateCameraDot]) {
    marker.setAttribute("cx", cameraX.toFixed(1));
    marker.setAttribute("cy", cameraY.toFixed(1));
  }
  coordinateLabelDot.setAttribute("cx", labelX.toFixed(1));
  coordinateLabelDot.setAttribute("cy", labelY.toFixed(1));
}

function previewPlanePoint(event, surface, height = previewSourceStart?.y) {
  if (!Number.isFinite(Number(height))) return null;
  const rect = surface.getBoundingClientRect();
  const pointer = new THREE.Vector2(
    ((event.clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1,
    -((event.clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1,
  );
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(pointer, camera);
  const target = new THREE.Vector3();
  const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -Number(height));
  return raycaster.ray.intersectPlane(plane, target) ? target : null;
}

function beginPreviewPathDrag(event, surface) {
  if (!previewAdjustMode || !previewTargetStart || !previewSourceStart) return false;
  const startPoint = previewPlanePoint(event, surface);
  if (!startPoint) return false;
  previewAdjustDrag = {
    pointerId: event.pointerId,
    surface,
    startPoint,
    startTarget: previewTargetStart.clone(),
  };
  surface.setPointerCapture(event.pointerId);
  return true;
}

function movePreviewPathDrag(event) {
  const drag = previewAdjustDrag;
  if (!drag || drag.pointerId !== event.pointerId || !previewSourceStart) return false;
  const point = previewPlanePoint(event, drag.surface);
  if (!point) return true;
  const dx = point.x - drag.startPoint.x;
  const dz = point.z - drag.startPoint.z;
  previewTargetStart = drag.startTarget.clone().add(new THREE.Vector3(dx, 0, dz));
  previewTargetStart.y = previewSourceStart.y;
  if (renderer) {
    pathGroup.position.set(dx, 0, dz);
    pathGlowGroup.position.set(dx, 0, dz);
  } else {
    renderPath(localizedPoses.map(posePosition).filter(Boolean));
  }
  const first = localizedPoses[0];
  if (first) applyCameraPose(first);
  fallbackDirty = true;
  return true;
}

function endPreviewPathDrag(event) {
  const drag = previewAdjustDrag;
  if (!drag || drag.pointerId !== event.pointerId) return false;
  previewAdjustDrag = null;
  pathGroup.position.set(0, 0, 0);
  pathGlowGroup.position.set(0, 0, 0);
  refreshPreviewCalibration({ showStart: true });
  setStatus(
    "preview",
    `Path moved · X size ${Math.round(previewScaleX * 100)}% · Y size ${Math.round(previewScaleY * 100)}% · rotate ${Math.round(previewRotationDeg)}°.`,
  );
  el("detail-text").textContent = "Path moved. Rotate or resize either floor axis if needed, then press Lock.";
  return true;
}

function installPointerControls() {
  const surface = renderer?.domElement || fallbackCanvas;
  if (!surface) return;
  let pointer = null;
  let lastX = 0;
  let lastY = 0;
  surface.addEventListener("pointerdown", (event) => {
    if (placePreviewStart(event, surface)) return;
    if (beginPreviewPathDrag(event, surface)) return;
    pointer = event.pointerId;
    lastX = event.clientX;
    lastY = event.clientY;
    surface.setPointerCapture(pointer);
  });
  surface.addEventListener("pointermove", (event) => {
    if (movePreviewPathDrag(event)) return;
    if (pointer !== event.pointerId) return;
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
    orbit.yaw -= dx * 0.006;
    orbit.pitch = THREE.MathUtils.clamp(orbit.pitch + dy * 0.005, -1.25, 1.52);
    fallbackDirty = true;
  });
  const release = (event) => {
    if (endPreviewPathDrag(event)) return;
    if (pointer === event.pointerId) pointer = null;
  };
  surface.addEventListener("pointerup", release);
  surface.addEventListener("pointercancel", release);
  surface.addEventListener("wheel", (event) => {
    event.preventDefault();
    orbit.distance = THREE.MathUtils.clamp(orbit.distance * Math.exp(event.deltaY * 0.001), 2.2, 46);
    fallbackDirty = true;
  }, { passive: false });
}

function setStatus(status, message) {
  currentJobStatus = status || "idle";
  statusDot.className = `status-dot ${status || "idle"}`;
  statusText.textContent = message || "Ready";
  const active = ["queued", "running", "stopping"].includes(status);
  startButton.disabled = active || !selectedFile;
  stopButton.disabled = !active;
  // Replay is also the Pause/Resume control for the live presentation. Keep
  // it usable while the localizer is active so autoplay restrictions can
  // never leave the video and tracked camera pinned at frame zero.
  replayButton.disabled = localizedPoses.length < 2;
}

function formatTime(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  return `${String(minutes).padStart(2, "0")}:${(value % 60).toFixed(1).padStart(4, "0")}`;
}

function formatTimingOffset(seconds) {
  const value = Math.abs(Number(seconds) || 0) < 0.05 ? 0 : Number(seconds) || 0;
  return `${value > 0 ? "+" : ""}${value.toFixed(1)}s`;
}

function previewTimingLastFrame() {
  const poseFrames = [localizedPoses.at(-1), displayPoses.at(-1)]
    .map(poseFrameIndex)
    .filter(Number.isFinite);
  return Math.max(899, streamExpectedCount - 1, ...poseFrames);
}

function normalizedPreviewTimingSegments(segments) {
  const source = Array.isArray(segments) && segments.length
    ? segments.slice(0, PREVIEW_TIMING_MAX_SECTIONS)
    : DEFAULT_PREVIEW_TIMING_SEGMENTS;
  const cleaned = source
    .map((segment, index) => ({
      start_frame: Math.max(0, Math.round(Number(segment?.start_frame) || 0)),
      end_frame: segment?.end_frame === null || segment?.end_frame === undefined
        ? null
        : Math.max(0, Math.round(Number(segment.end_frame) || 0)),
      offset_sec: THREE.MathUtils.clamp(
        Number.isFinite(Number(segment?.offset_sec))
          ? Number(segment.offset_sec)
          : Number(DEFAULT_PREVIEW_TIMING_SEGMENTS[index]?.offset_sec || 0),
        -8,
        8,
      ),
    }))
    .sort((a, b) => a.start_frame - b.start_frame);
  const normalized = [];
  cleaned.forEach((segment, index) => {
    const start = index === 0 ? 0 : normalized[index - 1].end_frame + 1;
    if (index === cleaned.length - 1) {
      normalized.push({ start_frame: start, end_frame: null, offset_sec: segment.offset_sec });
      return;
    }
    const nextStart = cleaned[index + 1].start_frame;
    const requestedEnd = Number.isFinite(segment.end_frame) ? segment.end_frame : nextStart - 1;
    const end = Math.max(start + PREVIEW_TIMING_MIN_SECTION_FRAMES - 1, requestedEnd);
    normalized.push({ start_frame: start, end_frame: end, offset_sec: segment.offset_sec });
  });
  return normalized;
}

function timingElement(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function timingNumberInput(value, action, index, { disabled = false, label = "" } = {}) {
  const input = timingElement("input");
  input.type = "number";
  input.min = "0";
  input.step = "1";
  input.value = String(value);
  input.disabled = disabled;
  input.dataset.timingAction = action;
  input.dataset.timingIndex = String(index);
  input.setAttribute("aria-label", label);
  return input;
}

function updatePreviewTimingControls() {
  previewTimingSegmentList.replaceChildren();
  const lastFrame = previewTimingLastFrame();
  previewTimingSegments.forEach((segment, index) => {
    const row = timingElement("div", "timing-segment-row");
    row.dataset.timingIndex = String(index);
    row.append(timingElement("span", "timing-segment-index", `S${index + 1}`));

    const startLabel = timingElement("label");
    startLabel.append(timingElement("span", "", "From frame"));
    startLabel.append(timingNumberInput(segment.start_frame, "start", index, {
      disabled: index === 0,
      label: `Section ${index + 1} start frame`,
    }));
    row.append(startLabel);

    const endLabel = timingElement("label");
    endLabel.append(timingElement("span", "", index === previewTimingSegments.length - 1 ? "To video end" : "To frame"));
    endLabel.append(timingNumberInput(segment.end_frame ?? lastFrame, "end", index, {
      disabled: index === previewTimingSegments.length - 1,
      label: `Section ${index + 1} end frame`,
    }));
    row.append(endLabel);

    const offsetLabel = timingElement("label", "timing-segment-offset");
    offsetLabel.append(timingElement("span", "", "Wanted path offset"));
    const range = timingElement("input");
    range.type = "range";
    range.min = "-8";
    range.max = "8";
    range.step = "0.1";
    range.value = String(segment.offset_sec);
    range.dataset.timingAction = "offset";
    range.dataset.timingIndex = String(index);
    range.setAttribute("aria-label", `Section ${index + 1} path offset`);
    offsetLabel.append(range);
    const offsetNumber = timingNumberInput(segment.offset_sec, "offset-number", index, {
      label: `Section ${index + 1} path offset seconds`,
    });
    offsetNumber.min = "-8";
    offsetNumber.max = "8";
    offsetNumber.step = "0.1";
    offsetLabel.append(offsetNumber);
    row.append(offsetLabel);

    const offsetOutput = timingElement("output", "", formatTimingOffset(segment.offset_sec));
    row.append(offsetOutput);

    const actions = timingElement("div", "timing-segment-actions");
    for (const [action, text] of [["add-before", "+ before"], ["add-after", "+ after"], ["remove", "×"]]) {
      const button = timingElement("button", "", text);
      button.type = "button";
      button.dataset.timingAction = action;
      button.dataset.timingIndex = String(index);
      button.disabled = (action === "remove" && previewTimingSegments.length === 1)
        || (action.startsWith("add") && previewTimingSegments.length >= PREVIEW_TIMING_MAX_SECTIONS);
      actions.append(button);
    }
    row.append(actions);
    previewTimingSegmentList.append(row);
  });
}

function highlightPreviewTimingSegment(segmentIndex) {
  previewTimingSegmentList.querySelectorAll(".timing-segment-row").forEach((row, index) => {
    row.classList.toggle("active", index === segmentIndex);
  });
}

function previewTimingOffsetForFrame(frameIndex) {
  const frame = Math.max(0, Number(frameIndex) || 0);
  let segmentIndex = previewTimingSegments.findLastIndex((segment) => frame >= segment.start_frame);
  if (segmentIndex < 0) segmentIndex = 0;
  const segment = previewTimingSegments[segmentIndex];
  let offsetSec = segment.offset_sec;
  if (frame < segment.start_frame + PREVIEW_TIMING_BLEND_FRAMES) {
    const progress = THREE.MathUtils.clamp(
      (frame - segment.start_frame) / PREVIEW_TIMING_BLEND_FRAMES,
      0,
      1,
    );
    const smoothProgress = progress * progress * (3 - 2 * progress);
    offsetSec = THREE.MathUtils.lerp(
      segmentIndex > 0 ? previewTimingSegments[segmentIndex - 1].offset_sec : 0,
      segment.offset_sec,
      smoothProgress,
    );
  }
  return { offsetSec, segmentIndex, frameIndex: frame };
}

function previewTimingForPlaybackTime(poses, playbackTime) {
  const index = Math.max(0, poseIndexForTime(poses, playbackTime));
  const current = poses[index];
  const next = poses[Math.min(index + 1, poses.length - 1)];
  const currentFrame = Number(poseFrameIndex(current));
  const nextFrame = Number(poseFrameIndex(next));
  const currentTime = Number(current?.time_sec || 0);
  const nextTime = Number(next?.time_sec || currentTime);
  const alpha = nextTime > currentTime
    ? THREE.MathUtils.clamp((playbackTime - currentTime) / (nextTime - currentTime), 0, 1)
    : 0;
  const interpolatedFrame = Number.isFinite(currentFrame)
    ? THREE.MathUtils.lerp(currentFrame, Number.isFinite(nextFrame) ? nextFrame : currentFrame, alpha)
    : index;
  return previewTimingOffsetForFrame(interpolatedFrame);
}

function setPlaybackRate(rate) {
  playbackRate = [1, 2, 3].includes(Number(rate)) ? Number(rate) : 1;
  sourceVideo.defaultPlaybackRate = playbackRate;
  sourceVideo.playbackRate = playbackRate;
  for (const button of playbackSpeedButtons) {
    const active = Number(button.dataset.playbackRate) === playbackRate;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  if (livePlaybackSyntheticClock && !livePlaybackUserPaused) {
    beginSyntheticPlaybackClock(Number(sourceVideo.currentTime) || 0);
  }
}

function roomPoint(raw) {
  if (!Array.isArray(raw) || raw.length < 3) return null;
  if (!roomMatrix) return new THREE.Vector3(...raw.slice(0, 3).map(Number));
  const x = Number(raw[0]);
  const y = Number(raw[1]);
  const z = Number(raw[2]);
  return new THREE.Vector3(
    roomMatrix[0][0] * x + roomMatrix[0][1] * y + roomMatrix[0][2] * z + roomMatrix[0][3],
    roomMatrix[1][0] * x + roomMatrix[1][1] * y + roomMatrix[1][2] * z + roomMatrix[1][3],
    roomMatrix[2][0] * x + roomMatrix[2][1] * y + roomMatrix[2][2] * z + roomMatrix[2][3],
  );
}

function roomDirection(raw) {
  if (!Array.isArray(raw) || raw.length < 3) return null;
  const source = raw.map(Number);
  const out = roomMatrix
    ? new THREE.Vector3(
        roomMatrix[0][0] * source[0] + roomMatrix[0][1] * source[1] + roomMatrix[0][2] * source[2],
        roomMatrix[1][0] * source[0] + roomMatrix[1][1] * source[1] + roomMatrix[1][2] * source[2],
        roomMatrix[2][0] * source[0] + roomMatrix[2][1] * source[1] + roomMatrix[2][2] * source[2],
      )
    : new THREE.Vector3(...source);
  out.y = 0;
  return out.lengthSq() > 1e-8 ? out.normalize() : null;
}

function acceptedPose(pose) {
  return pose && pose.success !== false && !pose.held_pose && (pose.rcenter || pose.center);
}

function poseIndexForTime(poses, time) {
  if (!poses.length) return -1;
  let low = 0;
  let high = poses.length - 1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (Number(poses[middle]?.time_sec || 0) <= time + 0.002) low = middle;
    else high = middle - 1;
  }
  return low;
}

function displayablePose(pose) {
  return pose && (pose.success !== false || pose.held_pose) && (pose.rcenter || pose.center);
}

function rawPosePosition(pose) {
  const center = streamValidationPreview && Array.isArray(pose.raw_preview_rcenter)
    ? pose.raw_preview_rcenter
    : pose.rcenter;
  return center ? new THREE.Vector3(...center.map(Number)) : roomPoint(pose.center);
}

function posePosition(pose) {
  const position = rawPosePosition(pose);
  if (
    !position
    || pose.preview_calibrated
    || !streamValidationPreview
    || !previewSourceStart
    || !previewTargetStart
  ) return position;
  const dx = position.x - previewSourceStart.x;
  const dz = position.z - previewSourceStart.z;
  const radians = THREE.MathUtils.degToRad(previewRotationDeg);
  const rotatedX = dx * Math.cos(radians) - dz * Math.sin(radians);
  const rotatedY = dx * Math.sin(radians) + dz * Math.cos(radians);
  return new THREE.Vector3(
    previewTargetStart.x + rotatedX * previewScaleX,
    position.y,
    previewTargetStart.z + rotatedY * previewScaleY,
  );
}

function previewCalibrationSignature(calibration) {
  if (!calibration || calibration.replay_id !== activePreviewReplayId) return "none";
  const target = Array.isArray(calibration.target_start) ? calibration.target_start : [];
  const timing = normalizedPreviewTimingSegments(calibration.timing_segments)
    .map((segment) => segment.offset_sec)
    .join(":");
  return `${calibration.updated_at || "saved"}:${target.map(Number).join(":")}:${Number(calibration.movement_scale)}:${Number(calibration.movement_scale_x || calibration.movement_scale)}:${Number(calibration.movement_scale_y || calibration.movement_scale)}:${Number(calibration.rotation_deg || 0)}:${timing}`;
}

function updatePreviewTransformValues() {
  const uniformPercent = Math.round(previewMovementScale * 100);
  const xPercent = Math.round(previewScaleX * 100);
  const yPercent = Math.round(previewScaleY * 100);
  previewMotionScaleInput.value = String(uniformPercent);
  previewMotionValue.value = Math.abs(previewScaleX - previewScaleY) < 0.001 ? `${uniformPercent}%` : "mixed";
  previewScaleXInput.value = String(xPercent);
  previewScaleXValue.value = `${xPercent}%`;
  previewScaleYInput.value = String(yPercent);
  previewScaleYValue.value = `${yPercent}%`;
  previewRotationInput.value = String(Math.round(previewRotationDeg));
  previewRotationValue.value = `${Math.round(previewRotationDeg)}°`;
}

function updatePreviewControlState() {
  const editable = !previewCalibrationLocked && !previewCalibrationSaving;
  adjustPreviewPathButton.classList.toggle("armed", previewAdjustMode);
  adjustPreviewPathButton.textContent = previewAdjustMode ? "Drag path…" : "Adjust path";
  adjustPreviewPathButton.disabled = previewCalibrationSaving || !previewSourceStart;
  placePreviewStartButton.disabled = !editable || !previewSourceStart;
  previewMotionScaleInput.disabled = !editable || !previewSourceStart;
  previewRotationInput.disabled = !editable || !previewSourceStart;
  previewScaleXInput.disabled = !editable || !previewSourceStart;
  previewScaleYInput.disabled = !editable || !previewSourceStart;
  lockPreviewPathButton.disabled = !editable || !previewTargetStart;
  lockPreviewPathButton.textContent = previewCalibrationSaving
    ? "Saving…"
    : previewCalibrationLocked
    ? "Locked"
    : "Lock";
  lockPreviewPathButton.classList.toggle("locked", previewCalibrationLocked);
}

function initializePreviewCalibration(poses) {
  if (!streamValidationPreview || !poses.length) return;
  const first = poses[0];
  const source = rawPosePosition(first);
  const published = Array.isArray(first.rcenter)
    ? new THREE.Vector3(...first.rcenter.map(Number))
    : source?.clone();
  if (!source || !published) return;
  const savedSignature = previewCalibrationSignature(pendingPreviewCalibration);
  const key = `${poseKey(first)}:${source.toArray().map((value) => value.toFixed(5)).join(":")}:${savedSignature}`;
  if (key === previewCalibrationKey) return;
  previewCalibrationKey = key;
  previewSourceStart = source;
  previewTargetStart = published;
  previewMovementScale = THREE.MathUtils.clamp(Number(first.preview_movement_scale) || 0.35, 0.1, 1.5);
  previewScaleX = previewMovementScale;
  previewScaleY = previewMovementScale;
  previewRotationDeg = 0;
  previewTimingSegments = normalizedPreviewTimingSegments();
  const savedTarget = pendingPreviewCalibration?.target_start;
  if (
    pendingPreviewCalibration?.replay_id === activePreviewReplayId
    && Array.isArray(savedTarget)
    && savedTarget.length >= 3
    && savedTarget.slice(0, 3).every((value) => Number.isFinite(Number(value)))
    && Number.isFinite(Number(pendingPreviewCalibration.movement_scale))
  ) {
    previewTargetStart = new THREE.Vector3(...savedTarget.slice(0, 3).map(Number));
    previewTargetStart.y = previewSourceStart.y;
    previewMovementScale = THREE.MathUtils.clamp(Number(pendingPreviewCalibration.movement_scale), 0.1, 1.5);
    previewScaleX = THREE.MathUtils.clamp(
      Number(pendingPreviewCalibration.movement_scale_x ?? previewMovementScale),
      0.1,
      1.5,
    );
    previewScaleY = THREE.MathUtils.clamp(
      Number(pendingPreviewCalibration.movement_scale_y ?? previewMovementScale),
      0.1,
      1.5,
    );
    previewMovementScale = (previewScaleX + previewScaleY) / 2;
    previewRotationDeg = THREE.MathUtils.clamp(
      Number(pendingPreviewCalibration.rotation_deg) || 0,
      -180,
      180,
    );
    previewTimingSegments = normalizedPreviewTimingSegments(pendingPreviewCalibration.timing_segments);
    previewCalibrationLocked = Boolean(pendingPreviewCalibration.locked);
  } else {
    previewCalibrationLocked = false;
  }
  updatePreviewTransformValues();
  updatePreviewTimingControls();
  updatePreviewControlState();
}

function refreshPreviewCalibration({ showStart = true } = {}) {
  if (!streamValidationPreview || !localizedPoses.length) return;
  stopLivePlayback(false);
  stopReplay(false);
  latestPathSignature = "";
  renderPath(localizedPoses.map(posePosition).filter(Boolean));
  if (showStart) {
    const first = localizedPoses[0];
    latestRenderedPoseKey = poseKey(first);
    applyCameraPose(first, { syncVideo: true });
    latestPoseFrameIndex = poseFrameIndex(first);
    updateCoordinateKicker();
    el("frame-chip").textContent = `FRAME ${latestPoseFrameIndex ?? 0}`;
  }
}

function setPreviewPlacementMode(enabled) {
  if (enabled && previewAdjustMode) setPreviewAdjustMode(false);
  previewPlaceStartMode = Boolean(enabled && streamValidationPreview && previewSourceStart);
  container.classList.toggle("place-preview-start", previewPlaceStartMode);
  placePreviewStartButton.classList.toggle("armed", previewPlaceStartMode);
  placePreviewStartButton.textContent = previewPlaceStartMode ? "Click floor…" : "Place start";
  if (previewPlaceStartMode) {
    stopLivePlayback(true);
    stopReplay(true);
    el("detail-text").textContent = "Click the room floor at the exact camera starting point.";
  }
  updatePreviewControlState();
}

function setPreviewAdjustMode(enabled) {
  previewAdjustMode = Boolean(enabled && streamValidationPreview && previewSourceStart);
  if (previewAdjustMode) {
    previewCalibrationLocked = false;
    updatePreviewTransformValues();
    setPreviewPlacementMode(false);
    stopLivePlayback(true);
    stopReplay(true);
    refreshPreviewCalibration({ showStart: true });
    el("detail-text").textContent = "Drag to move. Rotate or resize X/Y on the floor; path height is unchanged.";
  } else if (previewAdjustDrag) {
    previewAdjustDrag = null;
    pathGroup.position.set(0, 0, 0);
    pathGlowGroup.position.set(0, 0, 0);
    refreshPreviewCalibration({ showStart: true });
  }
  container.classList.toggle("adjust-preview-path", previewAdjustMode);
  previewTransformControls.hidden = !previewAdjustMode;
  updatePreviewControlState();
}

function placePreviewStart(event, surface) {
  if (!previewPlaceStartMode || !previewSourceStart) return false;
  const target = previewPlanePoint(event, surface);
  if (!target) return false;
  previewTargetStart = target;
  previewTargetStart.y = previewSourceStart.y;
  setPreviewPlacementMode(false);
  refreshPreviewCalibration({ showStart: true });
  setStatus(
    "preview",
    `Start placed at X ${target.x.toFixed(3)}, Z ${target.z.toFixed(3)} · motion ${Math.round(previewMovementScale * 100)}%.`,
  );
  el("detail-text").textContent = "Start placed. Press Replay to inspect the shortened motion and turns.";
  return true;
}

async function savePreviewCalibration({ announce = true } = {}) {
  if (!activePreviewReplayId || !previewTargetStart || previewCalibrationSaving) return;
  if (announce) {
    setPreviewAdjustMode(false);
    setPreviewPlacementMode(false);
  }
  previewCalibrationSaving = true;
  updatePreviewControlState();
  try {
    const response = await fetch("/api/camera-path-lab/preview-calibration", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        replay_id: activePreviewReplayId,
        target_start: previewTargetStart.toArray(),
        movement_scale: previewMovementScale,
        movement_scale_x: previewScaleX,
        movement_scale_y: previewScaleY,
        rotation_deg: previewRotationDeg,
        timing_offset_sec: previewTimingSegments[0].offset_sec,
        timing_segments: previewTimingSegments,
      }),
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.error || "Could not save path calibration.");
    pendingPreviewCalibration = payload.calibration;
    previewCalibrationLocked = true;
    if (announce) {
      setStatus(
        "done",
        `Path locked · X ${Math.round(previewScaleX * 100)}% · Y ${Math.round(previewScaleY * 100)}% · rotate ${Math.round(previewRotationDeg)}°.`,
      );
      el("detail-text").textContent = "Path calibration and timing are locked and will be restored after refresh.";
    }
  } catch (error) {
    if (announce) previewCalibrationLocked = false;
    setStatus("error", error.message);
    el("detail-text").textContent = announce
      ? "The path remains editable because locking did not complete."
      : "The timing adjustment could not be saved; the current preview still uses it.";
  } finally {
    previewCalibrationSaving = false;
    updatePreviewControlState();
  }
}

function poseHeading(pose) {
  if (pose.rotation_heading) return new THREE.Vector3(...pose.rotation_heading.map(Number)).normalize();
  if (pose.rheading) return new THREE.Vector3(...pose.rheading.map(Number)).normalize();
  if (Array.isArray(pose.R) && pose.R.length >= 3) return roomDirection(pose.R[2]);
  return null;
}

function makeCameraRig() {
  const rig = new THREE.Group();
  const shell = new THREE.BoxGeometry(0.34, 0.22, 0.20);
  const body = new THREE.Mesh(shell, new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.9,
  }));
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(shell),
    new THREE.LineBasicMaterial({ color: 0x477f99, transparent: true, opacity: 0.92 }),
  );
  const lens = new THREE.Mesh(
    new THREE.CylinderGeometry(0.065, 0.085, 0.13, 18),
    new THREE.MeshStandardMaterial({
      color: 0xb9ddeb,
      emissive: 0x6eb7d5,
      emissiveIntensity: 0.28,
      roughness: 0.35,
    }),
  );
  lens.rotation.x = Math.PI / 2;
  lens.position.z = -0.15;
  const frontDot = new THREE.Mesh(
    new THREE.SphereGeometry(0.027, 12, 8),
    new THREE.MeshBasicMaterial({ color: 0x347fa2 }),
  );
  frontDot.position.z = -0.225;
  const halo = new THREE.Mesh(
    new THREE.TorusGeometry(0.28, 0.006, 6, 40),
    new THREE.MeshBasicMaterial({ color: 0x65b7d9, transparent: true, opacity: 0.42 }),
  );
  halo.position.z = 0.015;
  rig.add(body, edges, lens, frontDot, halo);
  rig.userData.proceduralBody = [body, edges, lens];
  const points = [
    [0, 0, -0.18], [-0.32, -0.20, -0.75], [0, 0, -0.18], [0.32, -0.20, -0.75],
    [0, 0, -0.18], [-0.32, 0.20, -0.75], [0, 0, -0.18], [0.32, 0.20, -0.75],
    [-0.32, -0.20, -0.75], [0.32, -0.20, -0.75], [0.32, -0.20, -0.75], [0.32, 0.20, -0.75],
    [0.32, 0.20, -0.75], [-0.32, 0.20, -0.75], [-0.32, 0.20, -0.75], [-0.32, -0.20, -0.75],
  ].flat();
  const frustumGeometry = new THREE.BufferGeometry();
  frustumGeometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
  const frustumEdges = new THREE.LineSegments(frustumGeometry, new THREE.LineBasicMaterial({
    color: 0x9ddff7,
    transparent: true,
    opacity: 0.82,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  }));
  const apex = [0, 0, -0.18];
  const corners = [
    [-0.32, -0.20, -0.75], [0.32, -0.20, -0.75],
    [0.32, 0.20, -0.75], [-0.32, 0.20, -0.75],
  ];
  const fieldVertices = [];
  for (let index = 0; index < corners.length; index += 1) {
    fieldVertices.push(...apex, ...corners[index], ...corners[(index + 1) % corners.length]);
  }
  const fieldGeometry = new THREE.BufferGeometry();
  fieldGeometry.setAttribute("position", new THREE.Float32BufferAttribute(fieldVertices, 3));
  const viewField = new THREE.Mesh(fieldGeometry, new THREE.MeshBasicMaterial({
    color: 0x57bde8,
    transparent: true,
    opacity: 0.045,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
  }));
  const centerRay = new THREE.Mesh(
    new THREE.CylinderGeometry(0.006, 0.006, 0.82, 8),
    new THREE.MeshBasicMaterial({
      color: 0xc4f1ff,
      transparent: true,
      opacity: 0.86,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }),
  );
  centerRay.rotation.x = Math.PI / 2;
  centerRay.position.z = -0.59;
  const focusPoint = new THREE.Mesh(
    new THREE.SphereGeometry(0.016, 10, 8),
    new THREE.MeshBasicMaterial({ color: 0xd8f7ff, transparent: true, opacity: 0.92 }),
  );
  focusPoint.position.z = -1.0;
  rig.add(viewField, frustumEdges, centerRay, focusPoint);
  const positionMarker = new THREE.Group();
  const markerMaterial = new THREE.MeshBasicMaterial({
    color: 0xe9faff,
    transparent: true,
    opacity: 0.98,
    depthTest: false,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const markerGlowMaterial = new THREE.MeshBasicMaterial({
    color: 0x58c8f3,
    transparent: true,
    opacity: 0.22,
    depthTest: false,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  for (const angle of [-Math.PI / 4, Math.PI / 4]) {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.055, 0.035, 0.43), markerMaterial);
    const glow = new THREE.Mesh(new THREE.BoxGeometry(0.13, 0.025, 0.52), markerGlowMaterial);
    bar.rotation.y = angle;
    glow.rotation.y = angle;
    positionMarker.add(glow, bar);
  }
  positionMarker.position.y = 0.045;
  positionMarker.renderOrder = 20;
  rig.add(positionMarker);
  const directionalParts = [body, edges, lens, frontDot, halo, viewField, frustumEdges, centerRay, focusPoint];
  if (POSITION_ONLY_CAMERA_MARKER) directionalParts.forEach((part) => { part.visible = false; });
  rig.userData.positionOnlyMarker = POSITION_ONLY_CAMERA_MARKER;
  rig.userData.positionMarker = positionMarker;
  rig.scale.setScalar(2.1);
  rig.visible = false;
  cameraOverlayScene.add(rig);
  return rig;
}

function loadAnalogCameraModel() {
  if (!renderer) return Promise.resolve(null);
  return new Promise((resolve, reject) => {
    new GLTFLoader().load(
      CAMERA_MODEL_URL,
      (gltf) => {
        const model = gltf.scene;
        model.name = "Camara Analogica byquuimey";
        model.traverse((node) => {
          if (!node.isMesh) return;
          node.material = new THREE.MeshStandardMaterial({
            color: 0xffffff,
            vertexColors: Boolean(node.geometry.getAttribute("color")),
            roughness: 0.56,
            metalness: 0.16,
          });
        });
        model.visible = !POSITION_ONLY_CAMERA_MARKER;
        cameraRig.add(model);
        for (const part of cameraRig.userData.proceduralBody || []) part.visible = false;
        resolve(model);
      },
      undefined,
      reject,
    );
  });
}

function poseKey(pose) {
  return `${pose?.image_name || pose?.instance_id || "pose"}:${Number(pose?.time_sec || 0).toFixed(6)}`;
}

function poseFrameIndex(pose) {
  const match = String(pose?.image_name || "").match(/_(\d+)\.[^.]+$/);
  return match ? Number(match[1]) : null;
}

function updateCoordinateKicker({ replayFrame = null } = {}) {
  if (replayFrame !== null) {
    cameraLabel.classList.remove("localizing");
    coordinateKicker.textContent = `REPLAY · FRAME ${replayFrame}`;
    return;
  }
  if (previewAdjustMode || previewPlaceStartMode) {
    cameraLabel.classList.remove("localizing");
    coordinateKicker.textContent = previewAdjustMode ? "PATH ADJUSTMENT · START" : "PLACE PATH START";
    return;
  }
  const localizing = Number.isFinite(currentInputFrameIndex)
    && Number.isFinite(latestPoseFrameIndex)
    && currentInputFrameIndex > latestPoseFrameIndex + 1;
  cameraLabel.classList.toggle("localizing", localizing || latestDisplayHeld);
  coordinateKicker.textContent = latestDisplayHeld
    ? `RECOVERING FRAME ${currentInputFrameIndex} · POSITION HELD AT ${latestPoseFrameIndex}`
    : localizing
    ? `LOCALIZING FRAME ${currentInputFrameIndex} · LAST POSE ${latestPoseFrameIndex}`
    : `${streamValidationPreview ? "ALIGNMENT PREVIEW" : streamOfflineValidated ? "OFFLINE CAMERA POSITION" : "LIVE CAMERA POSITION"}`
      + `${Number.isFinite(latestPoseFrameIndex) ? ` · FRAME ${latestPoseFrameIndex}` : ""}`;
}

function renderPath(positions) {
  pathGroup.position.set(0, 0, 0);
  pathGlowGroup.position.set(0, 0, 0);
  fallbackPath = positions;
  if (pathLine) {
    pathGroup.remove(pathLine);
    pathLine.geometry.dispose();
    pathLine.material.dispose();
    pathLine = null;
  }
  if (pathGlowLine) {
    pathGlowGroup.remove(pathGlowLine);
    pathGlowLine.geometry.dispose();
    pathGlowLine.material.dispose();
    pathGlowLine = null;
  }
  const distinct = positions.filter((point, index) => index === 0 || point.distanceToSquared(positions[index - 1]) > 1e-7);
  if (distinct.length >= 2) {
    const stride = Math.max(1, Math.ceil(distinct.length / 520));
    const visualPoints = distinct.filter((_, index) => index % stride === 0 || index === distinct.length - 1);
    const curve = new THREE.CatmullRomCurve3(visualPoints, false, "centripetal", 0.32);
    const segments = Math.min(720, Math.max(16, visualPoints.length * 2));
    pathLine = new THREE.Mesh(
      new THREE.TubeGeometry(curve, segments, 0.022, 7, false),
      new THREE.MeshBasicMaterial({ color: 0xa8e8ff, transparent: true, opacity: 0.94 }),
    );
    pathLine.renderOrder = 5;
    pathGroup.add(pathLine);
    pathGlowLine = new THREE.Mesh(
      new THREE.TubeGeometry(curve, segments, 0.067, 7, false),
      new THREE.MeshBasicMaterial({
        color: 0x4ebbea,
        transparent: true,
        opacity: 0.17,
        depthTest: false,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    pathGlowLine.renderOrder = 2;
    pathGlowGroup.add(pathGlowLine);
  }
  fallbackDirty = true;
}

function syncVideoToLocalizedFrame(time) {
  const target = Math.max(0, Number(time) || 0);
  sourceVideo.pause();
  if (sourceVideo.readyState >= HTMLMediaElement.HAVE_METADATA) {
    const duration = Number.isFinite(sourceVideo.duration) ? sourceVideo.duration : target;
    const safeTarget = Math.min(target, Math.max(0, duration - 0.001));
    if (Math.abs(sourceVideo.currentTime - safeTarget) > 0.015) sourceVideo.currentTime = safeTarget;
  }
  el("video-time").textContent = formatTime(target);
}

function applyCameraPose(pose, { syncVideo = false } = {}) {
  if (!pose) return;
  const position = posePosition(pose);
  if (!position) return;
  cameraPosePosition = position;
  cameraRig.position.copy(position);
  const heading = poseHeading(pose);
  if (heading) {
    targetHeading = heading.clone();
    if (!displayedHeading) displayedHeading = heading.clone();
    applyDisplayedHeading(displayedHeading);
  }
  cameraRig.visible = true;
  const yaw = !POSITION_ONLY_CAMERA_MARKER && targetHeading
    ? THREE.MathUtils.radToDeg(Math.atan2(targetHeading.x, targetHeading.z))
    : null;
  coordinates.textContent = `X ${position.x.toFixed(3)} · Y ${position.y.toFixed(3)} · Z ${position.z.toFixed(3)}`
    + (yaw === null ? "" : ` · YAW ${yaw.toFixed(1)}°`);
  if (syncVideo) syncVideoToLocalizedFrame(pose.time_sec);
  fallbackDirty = true;
}

function updatePath(payload) {
  const all = Array.isArray(payload?.poses) ? payload.poses : [];
  const poses = all.filter(acceptedPose);
  localizedPoses = poses;
  initializePreviewCalibration(poses);
  // Held poses are useful recovery telemetry, but they are not new camera
  // positions.  Driving the video clock from them made the video continue for
  // minutes while the camera remained at its last trusted location.  Playback
  // advances only on independently accepted map poses.
  displayPoses = poses;
  streamComplete = Boolean(payload?.complete);
  const latest = poses.at(-1);
  const latestDisplay = [...all].reverse().find(displayablePose) || latest;
  latestDisplayHeld = Boolean(latestDisplay?.held_pose);
  const incomingFrame = Number(payload?.current_frame?.frame_index);
  currentInputFrameIndex = Number.isFinite(incomingFrame) ? incomingFrame : null;
  latestPoseFrameIndex = latest ? poseFrameIndex(latest) : null;
  const activeJob = ["queued", "running", "stopping"].includes(currentJobStatus);
  if (!replayActive && !livePlaybackActive) updateCoordinateKicker();
  const pathSignature = `${poses.length}:${latest ? poseKey(latest) : "none"}:${Boolean(payload?.complete)}`;
  if (!replayActive && !livePlaybackActive && !activeJob && pathSignature !== latestPathSignature) {
    latestPathSignature = pathSignature;
    renderPath(poses.map(posePosition).filter(Boolean));
  }
  if (!replayActive && !livePlaybackActive && !previewAdjustMode && !previewPlaceStartMode && latestDisplay) {
    if (!firstLocalizationReady) {
      const first = poses[0] || latestDisplay;
      latestRenderedPoseKey = poseKey(first);
      firstLocalizationReady = true;
      // Show the first trusted camera pose without seeking the phone video to
      // that pose's timestamp. The video must remain visibly at frame zero
      // until the pose buffer is ready and synchronized playback begins.
      applyCameraPose(first);
    } else if (!activeJob) {
      const key = poseKey(latestDisplay);
      if (key !== latestRenderedPoseKey) {
        latestRenderedPoseKey = key;
        applyCameraPose(latestDisplay);
      }
    }
  }
  const accepted = Number(payload?.accepted_count ?? poses.length);
  const processed = Number(payload?.processed_count ?? all.length);
  const expected = Number(payload?.expected_count ?? 0);
  streamExpectedCount = expected;
  el("accepted-count").textContent = String(accepted);
  el("processed-count").textContent = String(processed);
  if (expected) el("target-count").textContent = String(expected);
  if (!replayActive && !livePlaybackActive && !previewAdjustMode && !previewPlaceStartMode) {
    el("frame-chip").textContent = processed ? `FRAME ${processed}` : "FRAME —";
  }
  replayButton.disabled = poses.length < 2;
  maybeStartLivePlayback();
}

function resetLivePresentation() {
  stopLivePlayback(false);
  stopReplay(false);
  localizedPoses = [];
  displayPoses = [];
  latestPathSignature = "";
  latestRenderedPoseKey = "";
  firstLocalizationReady = false;
  replayPoseIndex = -1;
  replayPathIndex = -1;
  cameraRig.visible = false;
  cameraPosePosition = null;
  targetHeading = null;
  displayedHeading = null;
  currentInputFrameIndex = null;
  latestPoseFrameIndex = null;
  latestDisplayHeld = false;
  streamExpectedCount = 0;
  streamComplete = false;
  setPreviewAdjustMode(false);
  setPreviewPlacementMode(false);
  previewCalibrationKey = "";
  previewSourceStart = null;
  previewTargetStart = null;
  previewMovementScale = 0.35;
  previewScaleX = 0.35;
  previewScaleY = 0.35;
  previewRotationDeg = 0;
  previewCalibrationLocked = false;
  previewCalibrationSaving = false;
  pendingPreviewCalibration = null;
  previewTimingSegments = normalizedPreviewTimingSegments();
  updatePreviewTimingControls();
  highlightPreviewTimingSegment(-1);
  previewTransformControls.hidden = true;
  updatePreviewTransformValues();
  updatePreviewControlState();
  livePlaybackSyntheticClock = false;
  livePlaybackClockStartMs = 0;
  livePlaybackClockStartTime = 0;
  livePlaybackLastMediaTime = 0;
  livePlaybackLastMediaProgressMs = 0;
  livePlaybackLastSeekMs = 0;
  updateCoordinateKicker();
  coordinateLink.hidden = true;
  renderPath([]);
  sourceVideo.pause();
  if (sourceVideo.readyState >= HTMLMediaElement.HAVE_METADATA) sourceVideo.currentTime = 0;
  el("video-time").textContent = "00:00.0";
  el("frame-chip").textContent = "FRAME —";
  coordinates.textContent = "X 0.000 · Y 0.000 · Z 0.000";
}

function beginSyntheticPlaybackClock(time = sourceVideo.currentTime) {
  const now = performance.now();
  livePlaybackSyntheticClock = true;
  livePlaybackClockStartMs = now;
  livePlaybackClockStartTime = Math.max(0, Number(time) || 0);
  livePlaybackLastMediaTime = livePlaybackClockStartTime;
  livePlaybackLastMediaProgressMs = now;
}

async function resumeLiveMediaPlayback() {
  const mediaTime = Math.max(0, Number(sourceVideo.currentTime) || 0);
  const now = performance.now();
  livePlaybackClockStartMs = now;
  livePlaybackClockStartTime = mediaTime;
  livePlaybackLastMediaTime = mediaTime;
  livePlaybackLastMediaProgressMs = now;
  // Start the deterministic presentation clock immediately. A successful
  // native play() call replaces it; a blocked/stalled player keeps using it
  // and is advanced with guarded seeks so the visible frame still changes.
  livePlaybackSyntheticClock = true;
  sourceVideo.playbackRate = playbackRate;
  try {
    await sourceVideo.play();
    if (!sourceVideo.paused) {
      livePlaybackSyntheticClock = false;
      livePlaybackLastMediaTime = Number(sourceVideo.currentTime) || mediaTime;
      livePlaybackLastMediaProgressMs = performance.now();
    }
  } catch (_) {
    beginSyntheticPlaybackClock(mediaTime);
  }
}

function livePresentationTime(latestTime) {
  const now = performance.now();
  let mediaTime = Math.max(0, Number(sourceVideo.currentTime) || 0);
  // Native media playback is not automatically bounded by the pose buffer.
  // Never let the visible frame run beyond the latest localized camera pose.
  if (mediaTime > latestTime + 0.015) {
    sourceVideo.pause();
    sourceVideo.currentTime = Math.max(0, latestTime);
    mediaTime = Math.max(0, latestTime);
    livePlaybackSyntheticClock = false;
  }
  if (!sourceVideo.paused && mediaTime > livePlaybackLastMediaTime + 0.002) {
    livePlaybackLastMediaTime = mediaTime;
    livePlaybackLastMediaProgressMs = now;
  }
  if (
    !livePlaybackSyntheticClock
    && !livePlaybackUserPaused
    && !livePlaybackBuffering
    && (sourceVideo.paused || now - livePlaybackLastMediaProgressMs > 650)
  ) {
    beginSyntheticPlaybackClock(mediaTime);
  }
  if (!livePlaybackSyntheticClock || livePlaybackUserPaused || livePlaybackBuffering) {
    return Math.min(mediaTime, latestTime);
  }
  const syntheticTime = Math.min(
    latestTime,
    livePlaybackClockStartTime + Math.max(0, now - livePlaybackClockStartMs) / 1000 * playbackRate,
  );
  if (now - livePlaybackLastSeekMs >= 90 && Math.abs(mediaTime - syntheticTime) > 0.035) {
    sourceVideo.currentTime = syntheticTime;
    livePlaybackLastSeekMs = now;
    mediaTime = syntheticTime;
  }
  return syntheticTime;
}

async function maybeStartLivePlayback() {
  if (
    replayActive || livePlaybackActive || displayPoses.length < 2
    || previewAdjustMode || previewPlaceStartMode
    || sourceVideo.readyState < HTMLMediaElement.HAVE_METADATA
  ) return;
  const firstTime = Number(displayPoses[0]?.time_sec || 0);
  const latestTime = Number(displayPoses.at(-1)?.time_sec || firstTime);
  const activeJob = ["queued", "running", "stopping"].includes(currentJobStatus) && !streamComplete;
  if (activeJob && latestTime - firstTime < LIVE_START_BUFFER_SECONDS) return;

  livePlaybackActive = true;
  livePlaybackBuffering = false;
  livePlaybackUserPaused = false;
  livePlaybackSyntheticClock = false;
  livePlaybackPoseIndex = -1;
  livePlaybackPathIndex = -1;
  renderPath([]);
  applyCameraPose(displayPoses[0]);
  sourceVideo.autoplay = true;
  sourceVideo.defaultMuted = true;
  sourceVideo.muted = true;
  // Always begin the presentation at the first video frame. If the earliest
  // accepted localization is later, the marker holds its first trusted pose
  // until that timestamp rather than making the video appear to have started
  // before the path.
  sourceVideo.currentTime = 0;
  el("detail-text").textContent = activeJob
    ? "Playing live while localization runs ahead in a pose buffer."
    : streamValidationPreview
    ? "Playing a short alignment preview. Check placement, turning, and synchronization."
    : streamOfflineValidated
    ? "Playing the validated offline path in sync with the source video."
    : "Playing the freshly localized camera path.";
  replayButton.textContent = "Pause";
  replayButton.disabled = false;
  await resumeLiveMediaPlayback();
}

function updateLivePlaybackFrame() {
  if (!livePlaybackActive || displayPoses.length < 2) return;
  const latestTime = Number(displayPoses.at(-1)?.time_sec || 0);
  const playbackTime = livePresentationTime(latestTime);
  const activeJob = ["queued", "running", "stopping"].includes(currentJobStatus) && !streamComplete;
  const bufferedSeconds = latestTime - playbackTime;

  if (
    activeJob
    && bufferedSeconds <= LIVE_STALL_GUARD_SECONDS
    && !livePlaybackBuffering
    && !livePlaybackUserPaused
  ) {
    if (!sourceVideo.paused) sourceVideo.pause();
    livePlaybackBuffering = true;
    livePlaybackSyntheticClock = false;
    el("detail-text").textContent = "Localization is catching up… playback will resume automatically.";
  } else if (
    livePlaybackBuffering
    && !livePlaybackUserPaused
    && bufferedSeconds >= LIVE_RESUME_BUFFER_SECONDS
  ) {
    livePlaybackBuffering = false;
    resumeLiveMediaPlayback();
    el("detail-text").textContent = "Playing live while localization runs ahead in a pose buffer.";
  }

  const firstTime = Number(displayPoses[0]?.time_sec || 0);
  const previewTiming = streamValidationPreview
    ? previewTimingForPlaybackTime(displayPoses, playbackTime)
    : { offsetSec: 0, segmentIndex: -1 };
  const pathPlaybackTime = THREE.MathUtils.clamp(
    playbackTime + previewTiming.offsetSec,
    firstTime,
    latestTime,
  );
  livePlaybackPoseIndex = poseIndexForTime(displayPoses, pathPlaybackTime);
  const index = Math.max(0, livePlaybackPoseIndex);
  const current = displayPoses[index];
  const next = displayPoses[Math.min(index + 1, displayPoses.length - 1)];
  const currentTime = Number(current.time_sec || 0);
  const nextTime = Number(next.time_sec || currentTime);
  const alpha = nextTime > currentTime
    ? THREE.MathUtils.clamp((pathPlaybackTime - currentTime) / (nextTime - currentTime), 0, 1)
    : 0;
  const currentPosition = posePosition(current);
  const nextPosition = posePosition(next);
  const position = currentPosition.clone().lerp(nextPosition, alpha);
  const currentDirection = poseHeading(current);
  const nextDirection = poseHeading(next);
  const direction = currentDirection && nextDirection
    ? currentDirection.clone().lerp(nextDirection, alpha).normalize()
    : currentDirection || nextDirection;
  applyCameraPose({
    rcenter: position.toArray(),
    rheading: direction?.toArray(),
    time_sec: pathPlaybackTime,
    preview_calibrated: true,
  });
  if (index !== livePlaybackPathIndex) {
    livePlaybackPathIndex = index;
    const pathPoints = localizedPoses
      .filter((pose) => Number(pose.time_sec || 0) <= pathPlaybackTime + 0.002)
      .map(posePosition)
      .filter(Boolean);
    pathPoints.push(position);
    renderPath(pathPoints);
  }
  const frameIndex = poseFrameIndex(current);
  const videoPoseIndex = poseIndexForTime(displayPoses, playbackTime);
  const videoFrameIndex = poseFrameIndex(displayPoses[Math.max(0, videoPoseIndex)]);
  el("frame-chip").textContent = `FRAME ${Number.isFinite(videoFrameIndex) ? videoFrameIndex : videoPoseIndex + 1}`;
  if (streamValidationPreview) highlightPreviewTimingSegment(previewTiming.segmentIndex);
  coordinateKicker.textContent = livePlaybackBuffering
    ? `BUFFERING · FRAME ${Number.isFinite(frameIndex) ? frameIndex : index + 1}`
    : `${streamValidationPreview ? `PATH ${formatTimingOffset(previewTiming.offsetSec)} · S${previewTiming.segmentIndex + 1}` : streamOfflineValidated ? "OFFLINE CAMERA POSITION" : "LIVE CAMERA POSITION"} · FRAME ${Number.isFinite(frameIndex) ? frameIndex : index + 1}`;
  cameraLabel.classList.toggle("localizing", livePlaybackBuffering);
  el("video-time").textContent = formatTime(playbackTime);
  if (streamComplete && (playbackTime >= latestTime || sourceVideo.ended)) {
    const lastFrame = poseFrameIndex(displayPoses.at(-1));
    const incomplete = streamExpectedCount > 0
      && (!Number.isFinite(lastFrame) || lastFrame < streamExpectedCount - 2);
    stopLivePlayback(true);
    if (incomplete) {
      el("detail-text").textContent =
        `Localization ended at ${formatTime(latestTime)}; later video frames were not played without synchronized camera poses.`;
    }
  }
}

function stopLivePlayback(restoreFullPath = true) {
  if (livePlaybackActive) sourceVideo.pause();
  livePlaybackActive = false;
  highlightPreviewTimingSegment(-1);
  livePlaybackBuffering = false;
  livePlaybackUserPaused = false;
  livePlaybackSyntheticClock = false;
  livePlaybackPoseIndex = -1;
  livePlaybackPathIndex = -1;
  replayButton.textContent = "Replay";
  if (restoreFullPath && displayPoses.length) {
    renderPath(localizedPoses.map(posePosition).filter(Boolean));
    const latest = displayPoses.at(-1);
    latestRenderedPoseKey = poseKey(latest);
    applyCameraPose(latest);
    updateCoordinateKicker();
  }
}

function updateReplayFrame() {
  if (!replayActive || localizedPoses.length < 2) return;
  const playbackTime = Number(sourceVideo.currentTime) || 0;
  const firstTime = Number(localizedPoses[0]?.time_sec || 0);
  const lastTime = Number(localizedPoses.at(-1)?.time_sec || firstTime);
  const previewTiming = streamValidationPreview
    ? previewTimingForPlaybackTime(localizedPoses, playbackTime)
    : { offsetSec: 0, segmentIndex: -1 };
  const pathPlaybackTime = THREE.MathUtils.clamp(
    playbackTime + previewTiming.offsetSec,
    firstTime,
    lastTime,
  );
  replayPoseIndex = poseIndexForTime(localizedPoses, pathPlaybackTime);
  const index = Math.max(0, replayPoseIndex);
  const current = localizedPoses[index];
  const next = localizedPoses[Math.min(index + 1, localizedPoses.length - 1)];
  const currentTime = Number(current.time_sec || 0);
  const nextTime = Number(next.time_sec || currentTime);
  const alpha = nextTime > currentTime
    ? THREE.MathUtils.clamp((pathPlaybackTime - currentTime) / (nextTime - currentTime), 0, 1)
    : 0;
  const currentPosition = posePosition(current);
  const nextPosition = posePosition(next);
  const position = currentPosition.clone().lerp(nextPosition, alpha);
  const currentDirection = poseHeading(current);
  const nextDirection = poseHeading(next);
  const direction = currentDirection && nextDirection
    ? currentDirection.clone().lerp(nextDirection, alpha).normalize()
    : currentDirection || nextDirection;
  applyCameraPose({
    rcenter: position.toArray(),
    rheading: direction?.toArray(),
    time_sec: pathPlaybackTime,
    preview_calibrated: true,
  });
  if (index !== replayPathIndex) {
    replayPathIndex = index;
    const replayPoints = localizedPoses.slice(0, index + 1).map(posePosition).filter(Boolean);
    replayPoints.push(position);
    renderPath(replayPoints);
    const videoIndex = poseIndexForTime(localizedPoses, playbackTime);
    const videoFrame = poseFrameIndex(localizedPoses[Math.max(0, videoIndex)]);
    el("frame-chip").textContent = `FRAME ${Number.isFinite(videoFrame) ? videoFrame : videoIndex + 1}`;
  }
  if (streamValidationPreview) highlightPreviewTimingSegment(previewTiming.segmentIndex);
  coordinateKicker.textContent = streamValidationPreview
    ? `PATH ${formatTimingOffset(previewTiming.offsetSec)} · S${previewTiming.segmentIndex + 1} · FRAME ${poseFrameIndex(current) ?? index + 1}`
    : `REPLAY · FRAME ${index + 1}`;
  el("video-time").textContent = formatTime(playbackTime);
  if (playbackTime >= lastTime || sourceVideo.ended) stopReplay(true);
}

function stopReplay(restoreFullPath = true) {
  if (replayActive) sourceVideo.pause();
  replayActive = false;
  highlightPreviewTimingSegment(-1);
  replayButton.textContent = "Replay";
  replayPoseIndex = -1;
  replayPathIndex = -1;
  if (restoreFullPath && localizedPoses.length) {
    renderPath(localizedPoses.map(posePosition).filter(Boolean));
    const latest = localizedPoses.at(-1);
    latestRenderedPoseKey = poseKey(latest);
    applyCameraPose(latest, { syncVideo: true });
    updateCoordinateKicker();
  }
}

async function startReplay() {
  if (localizedPoses.length < 2) return;
  stopLivePlayback(false);
  replayActive = true;
  replayPoseIndex = -1;
  replayPathIndex = -1;
  replayButton.textContent = "Pause";
  renderPath([]);
  const first = localizedPoses[0];
  applyCameraPose(first);
  sourceVideo.currentTime = 0;
  sourceVideo.playbackRate = playbackRate;
  try {
    await sourceVideo.play();
  } catch (error) {
    stopReplay(false);
    el("detail-text").textContent = `Replay could not start: ${error.message}`;
  }
}

function addWalls(entry) {
  wallsGroup.clear();
  const walls = entry?.safety_barriers || entry?.barriers || [];
  fallbackWalls = walls
    .map((wall) => (wall.corners || []).map((point) => new THREE.Vector3(...point.map(Number))))
    .filter((corners) => corners.length >= 4);
  const wallHeights = fallbackWalls.flatMap((corners) => corners.map((point) => point.y));
  if (wallHeights.length) {
    roomFloorY = Math.min(...wallHeights);
    roomCeilingY = Math.max(...wallHeights);
    floorGrid.position.y = roomFloorY - FLOOR_GRID_CLEARANCE_M;
  }
  const floorCorners = fallbackWalls
    .flatMap((corners) => corners)
    .filter((point) => Math.abs(point.y - roomFloorY) < 0.01);
  if (floorCorners.length) {
    roomFootprintCenter = floorCorners
      .reduce((sum, point) => sum.add(point), new THREE.Vector3())
      .multiplyScalar(1 / floorCorners.length);
  }
  const longestWall = [...fallbackWalls]
    .filter((corners) => corners.length >= 2)
    .sort((left, right) => right[0].distanceToSquared(right[1]) - left[0].distanceToSquared(left[1]))[0];
  if (longestWall) {
    const direction = longestWall[1].clone().sub(longestWall[0]);
    roomLongAxisLength = direction.length();
    roomLongAxisDeg = THREE.MathUtils.radToDeg(Math.atan2(direction.z, direction.x));
    if (roomLongAxisDeg < 90) roomLongAxisDeg += 180;
  }
  for (const wall of walls) {
    const corners = wall.corners || [];
    if (corners.length < 4) continue;
    const points = corners.map((point) => new THREE.Vector3(...point.map(Number)));
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute([
      ...points[0].toArray(), ...points[1].toArray(), ...points[2].toArray(),
      ...points[0].toArray(), ...points[2].toArray(), ...points[3].toArray(),
    ], 3));
    geometry.computeVertexNormals();
    const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
      color: 0x9bd8ef,
      transparent: true,
      opacity: Math.min(0.045, Math.max(0.025, (Number(wall.opacity) || 0.1) * 0.4)),
      depthWrite: false,
      side: THREE.DoubleSide,
    }));
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), new THREE.LineBasicMaterial({ color: 0x65afd0, transparent: true, opacity: 0.28 }));
    wallsGroup.add(mesh, edges);
  }
  fallbackDirty = true;
}

function alignRoomScanForDisplay(roomScan) {
  const targetFloorOffset = roomFloorY - ROOM_SCAN_SOURCE_FLOOR_Y;
  const roomYawCorrectionDeg = roomLongAxisDeg - ROOM_SCAN_SOURCE_AXIS_DEG;
  const horizontalScale = THREE.MathUtils.clamp(
    roomLongAxisLength / ROOM_SCAN_SOURCE_LONG_M,
    0.6,
    1.0,
  );
  // THREE's positive Y rotation is the opposite sign of the room-frame X/Z
  // yaw convention used by the alignment audit.
  const rotation = new THREE.Matrix4().makeRotationY(
    THREE.MathUtils.degToRad(-roomYawCorrectionDeg),
  );
  const correction = new THREE.Matrix4()
    .makeTranslation(
      roomFootprintCenter.x,
      targetFloorOffset,
      roomFootprintCenter.z,
    )
    .multiply(rotation)
    .multiply(new THREE.Matrix4().makeScale(horizontalScale, 1, horizontalScale))
    .multiply(new THREE.Matrix4().makeTranslation(
      -ROOM_SCAN_SOURCE_CENTER_XZ.x,
      0,
      -ROOM_SCAN_SOURCE_CENTER_XZ.y,
    ));
  const refinement = new THREE.Matrix4()
    .makeTranslation(
      ROOM_SCAN_REFINEMENT_XZ.x,
      0,
      ROOM_SCAN_REFINEMENT_XZ.y,
    )
    .multiply(new THREE.Matrix4().makeRotationY(
      THREE.MathUtils.degToRad(-ROOM_SCAN_REFINEMENT_YAW_DEG),
    ));
  roomScan.applyMatrix4(refinement.multiply(correction));
  roomScan.updateMatrixWorld(true);
}

function updateCeilingVisibility() {
  const clippingPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), ceilingCutY);
  for (const material of roomMeshMaterials) {
    material.clippingPlanes = ceilingVisible ? [] : [clippingPlane];
    material.needsUpdate = true;
  }
  fallbackDirty = true;
}

async function loadReferenceMap() {
  const response = await fetch("./public/maps/manifest.json", { cache: "no-store" });
  const manifest = await response.json();
  mapEntry = (manifest.maps || []).find((entry) => entry.id === REFERENCE_MAP_ID);
  if (!mapEntry) throw new Error(`Reference map ${REFERENCE_MAP_ID} is missing.`);
  roomMatrix = mapEntry.room_alignment?.matrix || null;
  el("map-name").textContent = mapEntry.title || mapEntry.id;
  addWalls(mapEntry);
}

function loadGlbMesh() {
  return new Promise((resolve, reject) => {
    new GLTFLoader().load(
      `${MESH_GLB_URL}?v=${MESH_GLB_VERSION}`,
      (gltf) => {
        alignRoomScanForDisplay(gltf.scene);
        const faceLimit = 12000;
        gltf.scene.traverse((node) => {
          if (!node.isMesh) return;
          const positions = node.geometry.getAttribute("position");
          const indices = node.geometry.index;
          const faceCount = indices ? Math.floor(indices.count / 3) : Math.floor(positions.count / 3);
          const remaining = Math.max(1, faceLimit - fallbackTriangles.length);
          const stride = Math.max(1, Math.ceil(faceCount / remaining));
          for (let face = 0; face < faceCount && fallbackTriangles.length < faceLimit; face += stride) {
            const vertexIndex = (offset) => indices ? indices.getX(face * 3 + offset) : face * 3 + offset;
            const vertex = (offset) => new THREE.Vector3()
              .fromBufferAttribute(positions, vertexIndex(offset))
              .applyMatrix4(node.matrixWorld);
            fallbackTriangles.push({ a: vertex(0), b: vertex(1), c: vertex(2) });
          }
          const sourceMaterials = Array.isArray(node.material) ? node.material : [node.material];
          const displayMaterials = sourceMaterials.map((sourceMaterial) => {
            const material = sourceMaterial?.clone?.() || new THREE.MeshBasicMaterial({ color: 0xffffff });
            if (material.map) material.map.colorSpace = THREE.SRGBColorSpace;
            material.transparent = true;
            material.opacity = ROOM_MESH_OPACITY;
            material.depthWrite = true;
            material.side = THREE.DoubleSide;
            material.toneMapped = false;
            material.needsUpdate = true;
            roomMeshMaterials.push(material);
            return material;
          });
          node.material = Array.isArray(node.material) ? displayMaterials : displayMaterials[0];
        });
        scene.add(gltf.scene);
        const bounds = new THREE.Box3().setFromObject(gltf.scene);
        if (!bounds.isEmpty()) {
          const center = bounds.getCenter(new THREE.Vector3());
          const size = bounds.getSize(new THREE.Vector3());
          orbit.target.copy(center);
          orbit.target.y = THREE.MathUtils.clamp(center.y, -0.1, 0.8);
          orbit.distance = THREE.MathUtils.clamp(Math.max(size.x, size.z) * 1.12, 12, 26);
          roomCeilingY = Math.max(roomCeilingY, bounds.max.y);
          // The scan contains pipes, beams and hanging ceiling fragments at
          // several heights. Removing only its top skin leaves most of that
          // geometry in the way, so Ceiling off opens the complete upper 36%
          // of the room while the camera/path remain in the overlay scene.
          ceilingCutY = bounds.min.y + size.y * 0.64;
        }
        updateCeilingVisibility();
        fallbackDirty = true;
        resolve(gltf.scene);
      },
      (event) => {
        if (!event.total) return;
        const percent = Math.min(100, Math.round((event.loaded / event.total) * 100));
        meshBadge.textContent = `Loading textured room mesh · ${percent}%`;
      },
      reject,
    );
  });
}

async function loadVoxelFallback() {
  const response = await fetch(MESH_FALLBACK_URL, { cache: "no-store" });
  if (!response.ok) throw new Error("display mesh is still being prepared");
  const asset = await response.json();
  const voxels = asset.voxels || [];
  const size = Number(asset.voxel_size) || 0.1;
  const displayStride = Math.max(1, Math.ceil(voxels.length / 14000));
  fallbackVoxels = voxels
    .filter((_, index) => index % displayStride === 0)
    .map((voxel) => ({
      point: new THREE.Vector3(Number(voxel[0]), Number(voxel[1]), Number(voxel[2])),
      color: `rgb(${Number(voxel[3])}, ${Number(voxel[4])}, ${Number(voxel[5])})`,
      weight: Number(voxel[6]) || 1,
    }));
  const positions = new Float32Array(voxels.length * 3);
  const colors = new Float32Array(voxels.length * 3);
  const color = new THREE.Color();
  voxels.forEach((voxel, index) => {
    const offset = index * 3;
    positions[offset] = Number(voxel[0]);
    positions[offset + 1] = Number(voxel[1]);
    positions[offset + 2] = Number(voxel[2]);
    color.setRGB(
      Number(voxel[3]) / 255,
      Number(voxel[4]) / 255,
      Number(voxel[5]) / 255,
      THREE.SRGBColorSpace,
    );
    colors[offset] = color.r;
    colors[offset + 1] = color.g;
    colors[offset + 2] = color.b;
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.computeBoundingSphere();
  const material = new THREE.PointsMaterial({
    color: 0xffffff,
    vertexColors: true,
    size: Math.max(0.065, size * 0.82),
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.5,
    depthWrite: false,
    toneMapped: false,
  });
  const points = new THREE.Points(geometry, material);
  scene.add(points);
  fallbackDirty = true;
  return { count: voxels.length };
}

async function loadDisplayMesh() {
  try {
    await loadGlbMesh();
    meshBadge.textContent = renderer ? "Textured room scan · GPU display" : "Textured room scan";
  } catch (_) {
    try {
      const fallback = await loadVoxelFallback();
      meshBadge.textContent = renderer
        ? `${fallback.count.toLocaleString()} colored surface points · GPU display`
        : `${fallback.count.toLocaleString()} colored surface points`;
    } catch (error) {
      meshBadge.textContent = "Room shell only · mesh pending";
      el("detail-text").textContent = `The page is ready; ${error.message}.`;
    }
  }
}

async function fetchPoseStream(url) {
  if (!url) return;
  const response = await fetch(url.startsWith("/") ? `.${url}` : `./${url}`, { cache: "no-store" });
  if (!response.ok) return;
  updatePath(await response.json());
}

function loadStoredStreamVideo(mediaUrl, title = "Recorded camera run") {
  if (!mediaUrl || selectedFile || mediaUrl === loadedStreamMediaUrl) return;
  loadedStreamMediaUrl = mediaUrl;
  const resolved = mediaUrl.startsWith("/") ? `.${mediaUrl}` : `./${mediaUrl}`;
  sourceVideo.src = resolved;
  sourceVideo.pause();
  el("video-empty").hidden = true;
  el("file-name").textContent = title;
}

async function pollStatus() {
  try {
    const response = await fetch("/api/camera-path-lab/status", { cache: "no-store" });
    const payload = await response.json();
    const stream = payload.stream || {};
    const nextPreviewReplayId = String(stream.replay_id || "");
    if (nextPreviewReplayId !== activePreviewReplayId) {
      setPreviewAdjustMode(false);
      setPreviewPlacementMode(false);
      activePreviewReplayId = nextPreviewReplayId;
      previewCalibrationKey = "";
    }
    pendingPreviewCalibration = stream.preview_calibration || null;
    streamOfflineValidated = Boolean(stream.offline_validated);
    streamValidationPreview = Boolean(stream.validation_preview);
    loadStoredStreamVideo(stream.media_url, stream.title || "Recorded camera run");
    // Parsing a large multipart upload happens before the server can replace
    // the previous terminal job state.  Do not flash that old cancelled/error
    // state after the operator has already pressed Run.
    const active = uploadInFlight || ["queued", "running"].includes(payload.status);
    const newVideoReady = Boolean(selectedFile && !selectedVideoSubmitted && !active);
    const previewControlsVisible = streamValidationPreview && !newVideoReady && !uploadInFlight;
    adjustPreviewPathButton.hidden = !previewControlsVisible;
    placePreviewStartButton.hidden = !previewControlsVisible;
    previewScaleControl.hidden = !previewControlsVisible;
    lockPreviewPathButton.hidden = !previewControlsVisible;
    previewTimingSegmentsPanel.hidden = !previewControlsVisible;
    playbackSpeedControl.hidden = !previewControlsVisible;
    previewTransformControls.hidden = !(previewControlsVisible && previewAdjustMode);
    if (!previewControlsVisible) {
      if (previewAdjustMode) setPreviewAdjustMode(false);
      if (previewPlaceStartMode) setPreviewPlacementMode(false);
    }
    const polledStatus = uploadInFlight ? "queued" : newVideoReady ? "idle" : payload.status || "idle";
    const polledMessage = uploadInFlight ? "Uploading video…" : newVideoReady ? "New video ready" : payload.message;
    setStatus(
      previewAdjustMode || previewPlaceStartMode ? "preview" : polledStatus,
      previewAdjustMode
        ? "Adjusting floor path · drag to move, rotate, resize X/Y, then Lock."
        : previewPlaceStartMode
        ? "Click the room floor to place the path start."
        : polledMessage,
    );
    if (active && replayActive) stopReplay(false);
    el("detail-text").textContent = previewAdjustMode
      ? "Drag to move. Rotate or resize X/Y on the floor; path height is unchanged."
      : previewPlaceStartMode
      ? "Click the room floor at the exact camera starting point."
      : newVideoReady
      ? "Press Run to localize this video live inside the room mesh."
      : stream.error
      || (active && !firstLocalizationReady
        ? "Initializing the first camera pose… video is held until localization is ready."
        : livePlaybackBuffering
        ? "Localization is catching up… playback will resume automatically."
        : livePlaybackActive
        ? streamValidationPreview
          ? "Playing the alignment preview. Check placement and video synchronization."
          : streamOfflineValidated
          ? "Playing the validated offline path in sync with the source video."
          : "Playing live while fresh localization runs ahead in a pose buffer."
        : payload.message || "Ready.");
    const processed = uploadInFlight ? 0 : Number(stream.pose_count || 0);
    const accepted = uploadInFlight ? 0 : Number(stream.accepted_pose_count || 0);
    const expected = uploadInFlight ? 0 : Number(stream.expected_count || 0);
    el("accepted-count").textContent = String(accepted);
    el("processed-count").textContent = String(processed);
    el("target-count").textContent = String(expected);
    el("progress-fill").style.width = `${expected ? Math.min(100, (processed / expected) * 100) : 0}%`;
    const completedPoseUrl = stream.asset_base ? `${stream.asset_base}/poses.json` : null;
    const poseUrl = stream.final_pose_url
      || ((stream.complete || stream.failed) ? completedPoseUrl : null)
      || stream.partial_pose_url;
    if (poseUrl && !newVideoReady && !uploadInFlight) {
      latestPoseUrl = poseUrl;
      await fetchPoseStream(poseUrl);
    }
  } catch (_) {
    setStatus("error", "Local processing service is unavailable");
    el("detail-text").textContent = "Start the local processing service, then reopen this page.";
  } finally {
    window.setTimeout(pollStatus, 300);
  }
}

videoInput.addEventListener("change", () => {
  selectedFile = videoInput.files?.[0] || null;
  selectedVideoSubmitted = false;
  loadedStreamMediaUrl = "";
  if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
  videoObjectUrl = selectedFile ? URL.createObjectURL(selectedFile) : null;
  sourceVideo.src = videoObjectUrl || "";
  resetLivePresentation();
  el("video-empty").hidden = Boolean(selectedFile);
  el("file-name").textContent = selectedFile?.name || "Choose a lab video";
  videoInput.closest(".file-button").classList.toggle("has-file", Boolean(selectedFile));
  startButton.disabled = !selectedFile;
});

startButton.addEventListener("click", async () => {
  if (!selectedFile) return;
  selectedVideoSubmitted = true;
  resetLivePresentation();
  const form = new FormData();
  form.append("video", selectedFile, selectedFile.name);
  form.append("map_id", REFERENCE_MAP_ID);
  uploadInFlight = true;
  setStatus("queued", "Uploading video…");
  try {
    const response = await fetch("/api/camera-path-lab/upload", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.error || "Upload failed");
    sourceVideo.currentTime = 0;
    sourceVideo.pause();
    el("detail-text").textContent = "Initializing the first camera pose… video is held until localization is ready.";
  } catch (error) {
    setStatus("error", error.message);
  } finally {
    uploadInFlight = false;
  }
});

stopButton.addEventListener("click", async () => {
  setStatus("stopping", "Stopping after the active localization step…");
  stopLivePlayback(false);
  try { await fetch("/api/drone/stop", { method: "POST" }); } catch (_) { /* status poll will report it */ }
});

replayButton.addEventListener("click", () => {
  if (livePlaybackActive) {
    if (livePlaybackUserPaused) {
      livePlaybackUserPaused = false;
      resumeLiveMediaPlayback();
      replayButton.textContent = "Pause";
    } else {
      livePlaybackUserPaused = true;
      livePlaybackSyntheticClock = false;
      sourceVideo.pause();
      replayButton.textContent = "Resume";
      el("detail-text").textContent = "Camera-path playback paused.";
    }
  } else if (replayActive) stopReplay(true);
  else startReplay();
});
for (const button of playbackSpeedButtons) {
  button.addEventListener("click", () => {
    setPlaybackRate(Number(button.dataset.playbackRate));
    el("detail-text").textContent = `Video and path playback set to ${playbackRate}×.`;
  });
}
placePreviewStartButton.addEventListener("click", () => {
  setPreviewPlacementMode(!previewPlaceStartMode);
});
adjustPreviewPathButton.addEventListener("click", () => {
  setPreviewAdjustMode(!previewAdjustMode);
});
previewMotionScaleInput.addEventListener("input", () => {
  previewCalibrationLocked = false;
  previewMovementScale = THREE.MathUtils.clamp(Number(previewMotionScaleInput.value) / 100, 0.1, 1.5);
  previewScaleX = previewMovementScale;
  previewScaleY = previewMovementScale;
  updatePreviewTransformValues();
  refreshPreviewCalibration({ showStart: true });
  setStatus(
    "preview",
    `Path resized to ${Math.round(previewMovementScale * 100)}% around the selected start.`,
  );
  el("detail-text").textContent = "Path scale changed. Move it if needed, then press Lock.";
  updatePreviewControlState();
});
function updatePreviewFloorTransform() {
  previewCalibrationLocked = false;
  previewScaleX = THREE.MathUtils.clamp(Number(previewScaleXInput.value) / 100, 0.1, 1.5);
  previewScaleY = THREE.MathUtils.clamp(Number(previewScaleYInput.value) / 100, 0.1, 1.5);
  previewRotationDeg = THREE.MathUtils.clamp(Number(previewRotationInput.value) || 0, -180, 180);
  previewMovementScale = (previewScaleX + previewScaleY) / 2;
  updatePreviewTransformValues();
  refreshPreviewCalibration({ showStart: true });
  setStatus(
    "preview",
    `Floor path · X ${Math.round(previewScaleX * 100)}% · Y ${Math.round(previewScaleY * 100)}% · rotate ${Math.round(previewRotationDeg)}°.`,
  );
  el("detail-text").textContent = "Only the floor footprint changed; vertical camera height is untouched.";
  updatePreviewControlState();
}
previewScaleXInput.addEventListener("input", updatePreviewFloorTransform);
previewScaleYInput.addEventListener("input", updatePreviewFloorTransform);
previewRotationInput.addEventListener("input", updatePreviewFloorTransform);
function schedulePreviewTimingSave() {
  window.clearTimeout(previewTimingSaveTimer);
  if (previewCalibrationLocked) {
    previewTimingSaveTimer = window.setTimeout(
      () => savePreviewCalibration({ announce: false }),
      180,
    );
  }
}

function announcePreviewTimingChange(segment, index) {
  livePlaybackPathIndex = -1;
  replayPathIndex = -1;
  el("detail-text").textContent =
    `Section ${index + 1} · frames ${segment.start_frame}–${segment.end_frame ?? "end"} · path ${formatTimingOffset(segment.offset_sec)}. Transitions are eased without jumps.`;
  schedulePreviewTimingSave();
}

function splitPreviewTimingSegment(index, side) {
  const segment = previewTimingSegments[index];
  if (!segment) return;
  const end = segment.end_frame ?? previewTimingLastFrame();
  const length = end - segment.start_frame + 1;
  if (length < PREVIEW_TIMING_MIN_SECTION_FRAMES * 2) {
    el("detail-text").textContent =
      `This section needs at least ${PREVIEW_TIMING_MIN_SECTION_FRAMES * 2} frames before it can be split smoothly.`;
    return;
  }
  const firstEnd = segment.start_frame + Math.floor(length / 2) - 1;
  const originalEnd = segment.end_frame;
  if (side === "before") {
    const inserted = {
      start_frame: segment.start_frame,
      end_frame: firstEnd,
      offset_sec: segment.offset_sec,
    };
    segment.start_frame = firstEnd + 1;
    previewTimingSegments.splice(index, 0, inserted);
  } else {
    segment.end_frame = firstEnd;
    previewTimingSegments.splice(index + 1, 0, {
      start_frame: firstEnd + 1,
      end_frame: originalEnd,
      offset_sec: segment.offset_sec,
    });
  }
  updatePreviewTimingControls();
  const changedIndex = side === "before" ? index : index + 1;
  announcePreviewTimingChange(previewTimingSegments[changedIndex], changedIndex);
}

function removePreviewTimingSegment(index) {
  if (previewTimingSegments.length <= 1 || !previewTimingSegments[index]) return;
  if (index === 0) {
    previewTimingSegments[1].start_frame = 0;
  } else {
    previewTimingSegments[index - 1].end_frame = previewTimingSegments[index].end_frame;
  }
  previewTimingSegments.splice(index, 1);
  updatePreviewTimingControls();
  const nextIndex = Math.max(0, index - 1);
  announcePreviewTimingChange(previewTimingSegments[nextIndex], nextIndex);
}

function changePreviewTimingBoundary(index, action, requestedValue) {
  const segment = previewTimingSegments[index];
  if (!segment || !Number.isFinite(requestedValue)) return;
  if (action === "start" && index > 0) {
    const previous = previewTimingSegments[index - 1];
    const end = segment.end_frame ?? previewTimingLastFrame();
    const min = previous.start_frame + PREVIEW_TIMING_MIN_SECTION_FRAMES;
    const max = end - PREVIEW_TIMING_MIN_SECTION_FRAMES + 1;
    segment.start_frame = THREE.MathUtils.clamp(Math.round(requestedValue), min, max);
    previous.end_frame = segment.start_frame - 1;
  } else if (action === "end" && index < previewTimingSegments.length - 1) {
    const next = previewTimingSegments[index + 1];
    const nextEnd = next.end_frame ?? previewTimingLastFrame();
    const min = segment.start_frame + PREVIEW_TIMING_MIN_SECTION_FRAMES - 1;
    const max = nextEnd - PREVIEW_TIMING_MIN_SECTION_FRAMES;
    segment.end_frame = THREE.MathUtils.clamp(Math.round(requestedValue), min, max);
    next.start_frame = segment.end_frame + 1;
  }
  updatePreviewTimingControls();
  announcePreviewTimingChange(previewTimingSegments[index], index);
}

previewTimingSegmentList.addEventListener("input", (event) => {
  const input = event.target.closest("input[data-timing-action]");
  if (!input || input.dataset.timingAction !== "offset") return;
  const index = Number(input.dataset.timingIndex);
  const segment = previewTimingSegments[index];
  if (!segment) return;
  segment.offset_sec = THREE.MathUtils.clamp(Number(input.value) || 0, -8, 8);
  const row = input.closest(".timing-segment-row");
  const numberInput = row?.querySelector('[data-timing-action="offset-number"]');
  const output = row?.querySelector("output");
  if (numberInput) numberInput.value = String(segment.offset_sec);
  if (output) output.value = formatTimingOffset(segment.offset_sec);
  announcePreviewTimingChange(segment, index);
});

previewTimingSegmentList.addEventListener("change", (event) => {
  const input = event.target.closest("input[data-timing-action]");
  if (!input) return;
  const index = Number(input.dataset.timingIndex);
  const action = input.dataset.timingAction;
  if (action === "start" || action === "end") {
    changePreviewTimingBoundary(index, action, Number(input.value));
  } else if (action === "offset-number") {
    const segment = previewTimingSegments[index];
    if (!segment) return;
    segment.offset_sec = THREE.MathUtils.clamp(Number(input.value) || 0, -8, 8);
    updatePreviewTimingControls();
    announcePreviewTimingChange(segment, index);
  } else if (action === "offset") {
    schedulePreviewTimingSave();
  }
});

previewTimingSegmentList.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-timing-action]");
  if (!button) return;
  const index = Number(button.dataset.timingIndex);
  if (button.dataset.timingAction === "add-before") splitPreviewTimingSegment(index, "before");
  else if (button.dataset.timingAction === "add-after") splitPreviewTimingSegment(index, "after");
  else if (button.dataset.timingAction === "remove") removePreviewTimingSegment(index);
});

togglePreviewTimingButton.addEventListener("click", () => {
  previewTimingCollapsed = !previewTimingCollapsed;
  previewTimingSegmentsPanel.classList.toggle("collapsed", previewTimingCollapsed);
  togglePreviewTimingButton.textContent = previewTimingCollapsed ? "Show" : "Hide";
  togglePreviewTimingButton.setAttribute("aria-expanded", String(!previewTimingCollapsed));
});
lockPreviewPathButton.addEventListener("click", () => savePreviewCalibration());
videoPanelSizeInput.addEventListener("input", () => {
  setVideoPanelWidth(videoPanelSizeInput.value);
  el("detail-text").textContent = "Video resized; the 3D map was fitted into the remaining space.";
});
sourceVideo.addEventListener("loadedmetadata", () => {
  updateVideoAspectRatio();
  if (!replayActive && !livePlaybackActive && localizedPoses.length) {
    const activeJob = ["queued", "running", "stopping"].includes(currentJobStatus) && !streamComplete;
    syncVideoToLocalizedFrame((activeJob ? localizedPoses[0] : localizedPoses.at(-1)).time_sec);
  }
  maybeStartLivePlayback();
});
sourceVideo.addEventListener("ended", () => {
  if (livePlaybackActive) stopLivePlayback(true);
  else stopReplay(true);
});

el("reset-view").addEventListener("click", () => setOrbit(false));
el("top-view").addEventListener("click", () => setOrbit(true));
el("toggle-walls").addEventListener("click", (event) => {
  wallsVisible = !wallsVisible;
  wallsGroup.visible = wallsVisible;
  fallbackDirty = true;
  event.currentTarget.textContent = wallsVisible ? "Walls" : "Walls off";
  event.currentTarget.setAttribute("aria-pressed", String(wallsVisible));
});
el("toggle-ceiling").addEventListener("click", (event) => {
  ceilingVisible = !ceilingVisible;
  updateCeilingVisibility();
  event.currentTarget.textContent = ceilingVisible ? "Ceiling" : "Ceiling off";
  event.currentTarget.setAttribute("aria-pressed", String(ceilingVisible));
});

setPlaybackRate(1);
initializeVideoPanelLayout();
cameraRig = makeCameraRig();
loadAnalogCameraModel().catch(() => { /* Procedural camera remains available. */ });
setOrbit(false);
installPointerControls();
window.addEventListener("resize", () => {
  setVideoPanelWidth(videoPanelWidth, { persist: false });
  resize();
});
if (typeof ResizeObserver === "function") {
  canvasResizeObserver = new ResizeObserver(() => resize());
  canvasResizeObserver.observe(container);
}
resize();
animate();

try {
  await loadReferenceMap();
  await loadDisplayMesh();
} catch (error) {
  meshBadge.textContent = "Reference unavailable";
  setStatus("error", error.message);
}
pollStatus();
