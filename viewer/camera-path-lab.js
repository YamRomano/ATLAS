import * as THREE from "./vendor/three.module.js";
import { GLTFLoader } from "./vendor/GLTFLoader.js";

const REFERENCE_MAP_ID = "map_copy_20260730_114851_cfefdc";
const MESH_GLB_URL = "./public/camera_path_lab/good_copy_mesh.glb";
const MESH_FALLBACK_URL = "./public/camera_path_lab/good_copy_mesh.json";
const CAMERA_MODEL_URL = "./public/camera_path_lab/analog_camera.glb";

const el = (id) => document.getElementById(id);
const container = el("lab-canvas");
const videoInput = el("video-input");
const sourceVideo = el("source-video");
const statusDot = el("status-dot");
const statusText = el("status-text");
const cameraLabel = el("camera-label");
const coordinates = el("camera-coordinates");
const meshBadge = el("mesh-badge");
const startButton = el("start-button");
const stopButton = el("stop-button");

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
try {
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x071b2c, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
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
const camera = new THREE.PerspectiveCamera(44, 1, 0.02, 180);
const orbit = { target: new THREE.Vector3(-1.7, 0.4, 0.2), yaw: -0.84, pitch: 0.52, distance: 17.5 };
const wallsGroup = new THREE.Group();
const pathGroup = new THREE.Group();
scene.add(wallsGroup, pathGroup);

scene.add(new THREE.HemisphereLight(0xffffff, 0xd9eef7, 1.45));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.05);
keyLight.position.set(5, 11, 4);
scene.add(keyLight);
const floorGrid = new THREE.GridHelper(24, 24, 0x69b7d8, 0x225a78);
floorGrid.position.y = -1.075;
floorGrid.material.opacity = 0.32;
floorGrid.material.transparent = true;
scene.add(floorGrid);

let mapEntry = null;
let roomMatrix = null;
let selectedFile = null;
let videoObjectUrl = null;
let latestPoseUrl = null;
let latestPoseSignature = "";
let pathLine = null;
let cameraRig = null;
let cameraPosePosition = null;
let wallsVisible = true;

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
    const triangles = fallbackTriangles.map((triangle) => {
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
  const points = fallbackVoxels.map((voxel) => {
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
  context.beginPath();
  context.moveTo(points[0].x, points[0].y);
  points.slice(1).forEach((point) => context.lineTo(point.x, point.y));
  context.strokeStyle = "#3d9bc6";
  context.lineWidth = 2.1;
  context.shadowColor = "rgba(86, 173, 213, 0.42)";
  context.shadowBlur = 8;
  context.stroke();
  context.restore();
}

function drawFallbackCamera(context) {
  if (!cameraPosePosition || !fallbackHeading) return;
  const position = projectFallback(cameraPosePosition);
  const front = projectFallback(cameraPosePosition.clone().add(fallbackHeading.clone().multiplyScalar(0.75)));
  if (!position || !front) return;
  const angle = Math.atan2(front.y - position.y, front.x - position.x);
  context.save();
  context.translate(position.x, position.y);
  context.rotate(angle);
  context.scale(1.28, 1.28);
  context.shadowColor = "rgba(86, 173, 213, 0.38)";
  context.shadowBlur = 12;
  context.strokeStyle = "rgba(79, 158, 193, 0.46)";
  context.lineWidth = 1;
  context.beginPath();
  context.arc(0, 0, 15, 0, Math.PI * 2);
  context.stroke();
  context.shadowBlur = 0;
  context.fillStyle = "rgba(255, 255, 255, 0.96)";
  context.strokeStyle = "#467f99";
  context.lineWidth = 1.35;
  context.fillRect(-14, -8, 23, 16);
  context.strokeRect(-14, -8, 23, 16);
  context.fillStyle = "#d8edf6";
  context.fillRect(-8, -11, 10, 4);
  context.strokeRect(-8, -11, 10, 4);
  context.beginPath();
  context.arc(-10, -4, 2, 0, Math.PI * 2);
  context.stroke();
  context.beginPath();
  context.moveTo(9, -5.5);
  context.lineTo(16, -6.5);
  context.lineTo(16, 6.5);
  context.lineTo(9, 5.5);
  context.closePath();
  context.fillStyle = "#c1e2ef";
  context.fill();
  context.stroke();
  context.strokeStyle = "rgba(70, 146, 181, 0.6)";
  context.beginPath();
  context.moveTo(16, -6.5);
  context.lineTo(34, -16);
  context.moveTo(16, 6.5);
  context.lineTo(34, 16);
  context.stroke();
  context.strokeStyle = "#4b9fc5";
  context.lineWidth = 1.2;
  context.beginPath();
  context.ellipse(14, 0, 3, 6, 0, 0, Math.PI * 2);
  context.ellipse(17, 0, 2, 5, 0, 0, Math.PI * 2);
  context.stroke();
  context.strokeStyle = "#347fa2";
  context.lineWidth = 1.8;
  context.beginPath();
  context.moveTo(15, 0);
  context.lineTo(40, 0);
  context.lineTo(34, -4);
  context.moveTo(40, 0);
  context.lineTo(34, 4);
  context.stroke();
  context.fillStyle = "#347fa2";
  context.beginPath();
  context.arc(15.5, 0, 2.4, 0, Math.PI * 2);
  context.fill();
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
  if (renderer) renderer.render(scene, camera);
  else if (fallbackDirty) drawFallbackScene();
}

function updateCameraLabel() {
  if (!cameraPosePosition || !cameraRig?.visible) {
    cameraLabel.hidden = true;
    return;
  }
  const projected = cameraPosePosition.clone().project(camera);
  if (projected.z < -1 || projected.z > 1) {
    cameraLabel.hidden = true;
    return;
  }
  cameraLabel.hidden = false;
  cameraLabel.style.left = `${(projected.x * 0.5 + 0.5) * container.clientWidth}px`;
  cameraLabel.style.top = `${(-projected.y * 0.5 + 0.5) * container.clientHeight}px`;
}

function installPointerControls() {
  const surface = renderer?.domElement || fallbackCanvas;
  if (!surface) return;
  let pointer = null;
  let lastX = 0;
  let lastY = 0;
  surface.addEventListener("pointerdown", (event) => {
    pointer = event.pointerId;
    lastX = event.clientX;
    lastY = event.clientY;
    surface.setPointerCapture(pointer);
  });
  surface.addEventListener("pointermove", (event) => {
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
  statusDot.className = `status-dot ${status || "idle"}`;
  statusText.textContent = message || "Ready";
  const active = ["queued", "running", "stopping"].includes(status);
  startButton.disabled = active || !selectedFile;
  stopButton.disabled = !active;
}

function formatTime(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  return `${String(minutes).padStart(2, "0")}:${(value % 60).toFixed(1).padStart(4, "0")}`;
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

function posePosition(pose) {
  return pose.rcenter ? new THREE.Vector3(...pose.rcenter.map(Number)) : roomPoint(pose.center);
}

function poseHeading(pose) {
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
  rig.add(new THREE.LineSegments(frustumGeometry, new THREE.LineBasicMaterial({ color: 0x4a9fc5, transparent: true, opacity: 0.68 })));
  rig.scale.setScalar(1.3);
  rig.visible = false;
  scene.add(rig);
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
        cameraRig.add(model);
        for (const part of cameraRig.userData.proceduralBody || []) part.visible = false;
        resolve(model);
      },
      undefined,
      reject,
    );
  });
}

function updatePath(payload) {
  const all = Array.isArray(payload?.poses) ? payload.poses : [];
  const poses = all.filter(acceptedPose);
  const signature = `${poses.length}:${payload?.processed_count || all.length}:${payload?.complete || false}`;
  if (signature === latestPoseSignature) return;
  latestPoseSignature = signature;
  const positions = poses.map(posePosition).filter(Boolean);
  fallbackPath = positions;
  if (pathLine) {
    pathGroup.remove(pathLine);
    pathLine.geometry.dispose();
    pathLine.material.dispose();
  }
  if (positions.length >= 2) {
    const geometry = new THREE.BufferGeometry().setFromPoints(positions);
    pathLine = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: 0x3d91b8, transparent: true, opacity: 0.92 }));
    pathGroup.add(pathLine);
  }
  const latest = poses.at(-1);
  if (latest) {
    const position = posePosition(latest);
    cameraPosePosition = position;
    cameraRig.position.copy(position);
    const heading = poseHeading(latest);
    if (heading) {
      targetHeading = heading.clone();
      if (!displayedHeading) displayedHeading = heading.clone();
      applyDisplayedHeading(displayedHeading);
    }
    cameraRig.visible = true;
    const yaw = targetHeading
      ? THREE.MathUtils.radToDeg(Math.atan2(targetHeading.x, targetHeading.z))
      : null;
    coordinates.textContent = `X ${position.x.toFixed(3)} · Y ${position.y.toFixed(3)} · Z ${position.z.toFixed(3)}`
      + (yaw === null ? "" : ` · YAW ${yaw.toFixed(1)}°`);
    const time = Number(latest.time_sec) || 0;
    el("video-time").textContent = formatTime(time);
    if (sourceVideo.src && Math.abs(sourceVideo.currentTime - time) > 0.26) sourceVideo.currentTime = time;
  }
  fallbackDirty = true;
  const accepted = Number(payload?.accepted_count ?? poses.length);
  const processed = Number(payload?.processed_count ?? all.length);
  const expected = Number(payload?.expected_count ?? 0);
  el("accepted-count").textContent = String(accepted);
  el("processed-count").textContent = String(processed);
  if (expected) el("target-count").textContent = String(expected);
  el("frame-chip").textContent = processed ? `FRAME ${processed}` : "FRAME —";
}

function addWalls(entry) {
  wallsGroup.clear();
  const walls = entry?.safety_barriers || entry?.barriers || [];
  fallbackWalls = walls
    .map((wall) => (wall.corners || []).map((point) => new THREE.Vector3(...point.map(Number))))
    .filter((corners) => corners.length >= 4);
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
      MESH_GLB_URL,
      (gltf) => {
        gltf.scene.updateMatrixWorld(true);
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
          node.material = new THREE.MeshBasicMaterial({
            color: node.geometry.getAttribute("color") ? 0xffffff : 0x82adba,
            vertexColors: Boolean(node.geometry.getAttribute("color")),
            transparent: true,
            opacity: 0.5,
            depthWrite: true,
            side: THREE.DoubleSide,
          });
        });
        scene.add(gltf.scene);
        fallbackDirty = true;
        resolve(gltf.scene);
      },
      undefined,
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
    meshBadge.textContent = renderer ? "COLMAP Delaunay mesh · GPU display" : "COLMAP Delaunay mesh";
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

async function pollStatus() {
  try {
    const response = await fetch("/api/camera-path-lab/status", { cache: "no-store" });
    const payload = await response.json();
    const stream = payload.stream || {};
    setStatus(payload.status || "idle", payload.message);
    const active = ["queued", "running"].includes(payload.status);
    if (active && selectedFile && sourceVideo.paused) {
      sourceVideo.play().catch(() => {});
    } else if (!active && !sourceVideo.paused) {
      sourceVideo.pause();
    }
    el("detail-text").textContent = stream.error || payload.message || "Ready.";
    const processed = Number(stream.pose_count || 0);
    const accepted = Number(stream.accepted_pose_count || 0);
    const expected = Number(stream.expected_count || 0);
    el("accepted-count").textContent = String(accepted);
    el("processed-count").textContent = String(processed);
    el("target-count").textContent = String(expected);
    el("progress-fill").style.width = `${expected ? Math.min(100, (processed / expected) * 100) : 0}%`;
    const poseUrl = stream.final_pose_url || stream.partial_pose_url;
    if (poseUrl) {
      latestPoseUrl = poseUrl;
      await fetchPoseStream(poseUrl);
    }
  } catch (_) {
    setStatus("error", "Local processing service is unavailable");
    el("detail-text").textContent = "Start the local processing service, then reopen this page.";
  } finally {
    window.setTimeout(pollStatus, 850);
  }
}

videoInput.addEventListener("change", () => {
  selectedFile = videoInput.files?.[0] || null;
  if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
  videoObjectUrl = selectedFile ? URL.createObjectURL(selectedFile) : null;
  sourceVideo.src = videoObjectUrl || "";
  el("video-empty").hidden = Boolean(selectedFile);
  el("file-name").textContent = selectedFile?.name || "Choose a lab video";
  videoInput.closest(".file-button").classList.toggle("has-file", Boolean(selectedFile));
  startButton.disabled = !selectedFile;
});

startButton.addEventListener("click", async () => {
  if (!selectedFile) return;
  const form = new FormData();
  form.append("video", selectedFile, selectedFile.name);
  form.append("map_id", REFERENCE_MAP_ID);
  setStatus("queued", "Uploading video…");
  try {
    const response = await fetch("/api/camera-path-lab/upload", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.error || "Upload failed");
    sourceVideo.currentTime = 0;
    sourceVideo.play().catch(() => {});
  } catch (error) {
    setStatus("error", error.message);
  }
});

stopButton.addEventListener("click", async () => {
  setStatus("stopping", "Stopping after the active localization step…");
  sourceVideo.pause();
  try { await fetch("/api/drone/stop", { method: "POST" }); } catch (_) { /* status poll will report it */ }
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

cameraRig = makeCameraRig();
loadAnalogCameraModel().catch(() => { /* Procedural camera remains available. */ });
setOrbit(false);
installPointerControls();
window.addEventListener("resize", resize);
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
