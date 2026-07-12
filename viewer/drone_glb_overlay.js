import * as THREE from "./vendor/three.module.js";
import { GLTFLoader } from "./vendor/GLTFLoader.js";

window.directDroneOverlayInstalled = true;
window.directDroneModelReady = false;
document.body.dataset.droneOverlay = "loading-real-glb";

function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function mul(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s];
}

function norm(a) {
  return Math.sqrt(Math.max(a[0] * a[0] + a[1] * a[1] + a[2] * a[2], 1e-12));
}

function normalize(a) {
  const n = norm(a);
  return [a[0] / n, a[1] / n, a[2] / n];
}

function inverseViewRotate(v, view) {
  const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw);
  const cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);

  // Inverse of app.js rotate(): first undo pitch, then undo yaw.
  const x1 = v[0];
  const y1 = cp * v[1] + sp * v[2];
  const z1 = -sp * v[1] + cp * v[2];
  return [
    cy * x1 - sy * z1,
    y1,
    sy * x1 + cy * z1,
  ];
}

function setCameraFromAtlasView(view) {
  // Match the ATLAS room camera so the GLB is viewed as a world object, not
  // as a screen-facing billboard. The camera may orbit with the user's mouse;
  // the drone attitude itself remains controlled only by TSolve/path yaw.
  const viewDistance = 5.4;
  const eye = inverseViewRotate([0, 0, viewDistance], view);
  const up = inverseViewRotate([0, 1, 0], view);
  camera.position.set(eye[0], eye[1], eye[2]);
  camera.up.set(up[0], up[1], up[2]);
  camera.lookAt(0, 0, 0);
}

const canvas = document.getElementById("drone3d");
let renderer = null;
try {
  renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
} catch (error) {
  window.directDroneOverlayInstalled = false;
  window.directDroneModelReady = false;
  document.body.dataset.droneOverlay = "webgl-unavailable";
  document.body.dataset.droneOverlayMessage = error?.message || String(error);
  console.warn("DJI Mini 3 Pro WebGL overlay unavailable; using canvas GLB renderer.", error);
  throw error;
}
renderer.setClearColor(0x000000, 0);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
if ("outputColorSpace" in renderer) renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1.9, 1.9, 1.9, -1.9, 0.01, 100);
camera.position.set(0.0, 5.4, 0.02);
camera.up.set(0, 0, -1);
camera.lookAt(0, 0, 0);

const root = new THREE.Group();
root.rotation.order = "YXZ";
scene.add(root);

const body = new THREE.Group();
body.rotation.order = "XYZ";
root.add(body);

scene.add(new THREE.AmbientLight(0xffffff, 1.55));
const key = new THREE.DirectionalLight(0xffffff, 2.1);
key.position.set(3.2, 4.5, 5.0);
scene.add(key);
const rim = new THREE.DirectionalLight(0x8fffe4, 1.2);
rim.position.set(-3.0, 2.5, -2.0);
scene.add(rim);

let model = null;
let naturalScale = 1;
let smoothedRoomYaw = null;
let lastYawUpdateMs = performance.now();
let lastPoseTimeSec = null;
let currentOverlaySize = 0;
// Fixed calibration between the DJI GLB's local nose direction and ATLAS
// room yaw. Positive Y is the room vertical. The DJI Mini 3 Pro GLB is flat
// as loaded; after the path-heading stabilization in app.js, roomYaw already
// points the nose along the selected TSolve/path heading.
const MODEL_YAW_CORRECTION = 0;

function unwrapAngleNear(target, reference) {
  let out = target;
  while (out - reference > Math.PI) out -= Math.PI * 2;
  while (out - reference < -Math.PI) out += Math.PI * 2;
  return out;
}

function normalizeLoadedModel(object) {
  const box = new THREE.Box3().setFromObject(object);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);

  object.position.sub(center);
  naturalScale = 2.35 / Math.max(size.x, size.y, size.z, 1e-6);
  object.scale.setScalar(naturalScale);

  object.traverse(child => {
    if (!child.isMesh) return;
    child.frustumCulled = false;
    if (child.material) {
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      for (const mat of materials) {
        mat.side = THREE.DoubleSide;
        mat.needsUpdate = true;
      }
    }
  });
}

function setOverlaySize(size) {
  if (currentOverlaySize === size) return;
  currentOverlaySize = size;
  canvas.style.width = `${size}px`;
  canvas.style.height = `${size}px`;
  renderer.setSize(size, size, false);
  camera.updateProjectionMatrix();
}

function hideOverlay() {
  canvas.style.display = "none";
  canvas.style.left = "-9999px";
  canvas.style.top = "-9999px";
  canvas.style.transform = "none";
}

new GLTFLoader().load(
  "public/models/dji-mini-3-pro.glb",
  gltf => {
    model = gltf.scene;
    normalizeLoadedModel(model);
    body.add(model);
    window.directDroneModelReady = true;
    document.body.classList.add("real-drone-glb-ready");
    document.body.dataset.droneOverlay = "real-glb-ready";
    document.body.dataset.droneOverlayMessage = "DJI Mini 3 Pro rendered directly from GLB";
  },
  undefined,
  error => {
    window.directDroneOverlayInstalled = false;
    window.directDroneModelReady = false;
    document.body.classList.remove("real-drone-glb-ready");
    document.body.dataset.droneOverlay = "real-glb-error";
    document.body.dataset.droneOverlayMessage = error?.message || String(error);
    console.warn("Failed to load DJI Mini 3 Pro GLB.", error);
  }
);

function animate() {
  requestAnimationFrame(animate);

  const api = window.TSOLVE_VIEWER;
  const pose = api?.getCurrentPose?.();
  if (!model || !pose?.rcenter) {
    hideOverlay();
    return;
  }

  const view = api.getView();
  const size = view?.mode === "top" ? 214 : 196;
  setOverlaySize(size);

  const p = api.projectRoomPoint(pose.rcenter);
  canvas.style.display = "block";
  canvas.style.left = `${p[0] - size * 0.5}px`;
  canvas.style.top = `${p[1] - size * 0.5}px`;
  canvas.style.transform = "none";

  const heading = normalize(api.getHeadingForPose(pose));
  const targetRoomYaw = Math.atan2(heading[0], heading[2]);
  const poseTimeSec = Number(pose.time_sec);
  if (Number.isFinite(poseTimeSec) && lastPoseTimeSec != null && poseTimeSec < lastPoseTimeSec - 0.25) {
    smoothedRoomYaw = null;
  }
  if (Number.isFinite(poseTimeSec)) lastPoseTimeSec = poseTimeSec;
  const now = performance.now();
  const dt = Math.max(0.001, Math.min(0.08, (now - lastYawUpdateMs) / 1000));
  lastYawUpdateMs = now;
  if (smoothedRoomYaw == null || !Number.isFinite(smoothedRoomYaw)) {
    smoothedRoomYaw = targetRoomYaw;
  } else {
    const unwrapped = unwrapAngleNear(targetRoomYaw, smoothedRoomYaw);
    const alpha = 1 - Math.exp(-dt * 5.5);
    smoothedRoomYaw += (unwrapped - smoothedRoomYaw) * alpha;
  }
  const roomYaw = smoothedRoomYaw;

  setCameraFromAtlasView(view);

  root.rotation.set(0, roomYaw + MODEL_YAW_CORRECTION, 0);
  // This DJI GLB is already Y-up: its smallest native span is the vertical
  // axis. Keep the body unpitched so front and back stay at the same room Y,
  // i.e. the drone remains parallel to the room XZ plane. Heading trims live
  // on root.rotation, where they rotate only around room +Y.
  body.rotation.set(0, 0, 0);
  root.position.set(0, 0, 0);
  renderer.render(scene, camera);
}

animate();
