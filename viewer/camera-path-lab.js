import * as THREE from "./vendor/three.module.js";
import { GLTFLoader } from "./vendor/GLTFLoader.js";

const REFERENCE_MAP_ID = "map_copy_20260730_114851_cfefdc";
const MESH_GLB_URL = "./public/camera_path_lab/good_copy_mesh.glb";
const MESH_FALLBACK_URL = "./public/camera_path_lab/good_copy_mesh.json";

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
try {
  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x03101a, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);
} catch (_) {
  container.classList.add("no-webgl");
  const notice = document.createElement("div");
  notice.className = "webgl-notice";
  notice.innerHTML = "<strong>3D preview unavailable here</strong><span>Open this page in Safari or Chrome to use the Mac GPU renderer.</span>";
  container.appendChild(notice);
}

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x03101a, 0.018);
const camera = new THREE.PerspectiveCamera(44, 1, 0.02, 180);
const orbit = { target: new THREE.Vector3(-1.7, 0.4, 0.2), yaw: -0.84, pitch: 0.52, distance: 17.5 };
const wallsGroup = new THREE.Group();
const pathGroup = new THREE.Group();
scene.add(wallsGroup, pathGroup);

scene.add(new THREE.HemisphereLight(0xc8efff, 0x071018, 1.25));
const keyLight = new THREE.DirectionalLight(0xb8e9ff, 1.1);
keyLight.position.set(5, 11, 4);
scene.add(keyLight);
const floorGrid = new THREE.GridHelper(24, 24, 0x24637a, 0x103548);
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
}

function resize() {
  if (!renderer) return;
  const width = Math.max(1, container.clientWidth);
  const height = Math.max(1, container.clientHeight);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function animate() {
  requestAnimationFrame(animate);
  updateOrbitCamera();
  updateCameraLabel();
  if (renderer) renderer.render(scene, camera);
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
  if (!renderer) return;
  let pointer = null;
  let lastX = 0;
  let lastY = 0;
  renderer.domElement.addEventListener("pointerdown", (event) => {
    pointer = event.pointerId;
    lastX = event.clientX;
    lastY = event.clientY;
    renderer.domElement.setPointerCapture(pointer);
  });
  renderer.domElement.addEventListener("pointermove", (event) => {
    if (pointer !== event.pointerId) return;
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
    orbit.yaw -= dx * 0.006;
    orbit.pitch = THREE.MathUtils.clamp(orbit.pitch + dy * 0.005, -1.25, 1.52);
  });
  const release = (event) => {
    if (pointer === event.pointerId) pointer = null;
  };
  renderer.domElement.addEventListener("pointerup", release);
  renderer.domElement.addEventListener("pointercancel", release);
  renderer.domElement.addEventListener("wheel", (event) => {
    event.preventDefault();
    orbit.distance = THREE.MathUtils.clamp(orbit.distance * Math.exp(event.deltaY * 0.001), 2.2, 46);
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
    color: 0x06131b,
    transparent: true,
    opacity: 0.9,
  }));
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(shell),
    new THREE.LineBasicMaterial({ color: 0xe9fbff, transparent: true, opacity: 0.96 }),
  );
  const lens = new THREE.Mesh(
    new THREE.CylinderGeometry(0.065, 0.085, 0.13, 18),
    new THREE.MeshStandardMaterial({
      color: 0x163849,
      emissive: 0x087d9e,
      emissiveIntensity: 0.9,
      roughness: 0.35,
    }),
  );
  lens.rotation.x = Math.PI / 2;
  lens.position.z = -0.15;
  const frontDot = new THREE.Mesh(
    new THREE.SphereGeometry(0.027, 12, 8),
    new THREE.MeshBasicMaterial({ color: 0xff5478 }),
  );
  frontDot.position.z = -0.225;
  const halo = new THREE.Mesh(
    new THREE.TorusGeometry(0.28, 0.006, 6, 40),
    new THREE.MeshBasicMaterial({ color: 0x62dcfb, transparent: true, opacity: 0.52 }),
  );
  halo.position.z = 0.015;
  rig.add(body, edges, lens, frontDot, halo);
  const points = [
    [0, 0, -0.18], [-0.32, -0.20, -0.75], [0, 0, -0.18], [0.32, -0.20, -0.75],
    [0, 0, -0.18], [-0.32, 0.20, -0.75], [0, 0, -0.18], [0.32, 0.20, -0.75],
    [-0.32, -0.20, -0.75], [0.32, -0.20, -0.75], [0.32, -0.20, -0.75], [0.32, 0.20, -0.75],
    [0.32, 0.20, -0.75], [-0.32, 0.20, -0.75], [-0.32, 0.20, -0.75], [-0.32, -0.20, -0.75],
  ].flat();
  const frustumGeometry = new THREE.BufferGeometry();
  frustumGeometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
  rig.add(new THREE.LineSegments(frustumGeometry, new THREE.LineBasicMaterial({ color: 0x5ce1ff, transparent: true, opacity: 0.76 })));
  rig.scale.setScalar(0.72);
  rig.visible = false;
  scene.add(rig);
  return rig;
}

function updatePath(payload) {
  const all = Array.isArray(payload?.poses) ? payload.poses : [];
  const poses = all.filter(acceptedPose);
  const signature = `${poses.length}:${payload?.processed_count || all.length}:${payload?.complete || false}`;
  if (signature === latestPoseSignature) return;
  latestPoseSignature = signature;
  const positions = poses.map(posePosition).filter(Boolean);
  if (pathLine) {
    pathGroup.remove(pathLine);
    pathLine.geometry.dispose();
    pathLine.material.dispose();
  }
  if (positions.length >= 2) {
    const geometry = new THREE.BufferGeometry().setFromPoints(positions);
    pathLine = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: 0x5ce1ff, transparent: true, opacity: 0.98 }));
    pathGroup.add(pathLine);
  }
  const latest = poses.at(-1);
  if (latest) {
    const position = posePosition(latest);
    cameraPosePosition = position;
    cameraRig.position.copy(position);
    const heading = poseHeading(latest);
    if (heading) {
      const target = position.clone().add(heading);
      cameraRig.up.set(0, 1, 0);
      cameraRig.lookAt(target);
      cameraRig.rotateY(Math.PI);
    }
    cameraRig.visible = true;
    coordinates.textContent = `X ${position.x.toFixed(3)} · Y ${position.y.toFixed(3)} · Z ${position.z.toFixed(3)}`;
    const time = Number(latest.time_sec) || 0;
    el("video-time").textContent = formatTime(time);
    if (sourceVideo.src && Math.abs(sourceVideo.currentTime - time) > 0.26) sourceVideo.currentTime = time;
  }
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
      color: wall.color || 0x94e3fe,
      transparent: true,
      opacity: Math.max(0.07, Number(wall.opacity) || 0.1),
      depthWrite: false,
      side: THREE.DoubleSide,
    }));
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), new THREE.LineBasicMaterial({ color: 0x7edcf5, transparent: true, opacity: 0.44 }));
    wallsGroup.add(mesh, edges);
  }
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
        gltf.scene.traverse((node) => {
          if (!node.isMesh) return;
          node.material = new THREE.MeshStandardMaterial({
            color: 0xb1cbd3,
            vertexColors: Boolean(node.geometry.getAttribute("color")),
            roughness: 0.92,
            metalness: 0,
            transparent: true,
            opacity: 0.82,
            side: THREE.DoubleSide,
          });
        });
        scene.add(gltf.scene);
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
  const geometry = new THREE.BoxGeometry(size * 0.95, size * 0.95, size * 0.95);
  const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.94, metalness: 0 });
  const instanced = new THREE.InstancedMesh(geometry, material, voxels.length);
  const matrix = new THREE.Matrix4();
  const color = new THREE.Color();
  voxels.forEach((voxel, index) => {
    matrix.makeTranslation(Number(voxel[0]), Number(voxel[1]), Number(voxel[2]));
    instanced.setMatrixAt(index, matrix);
    color.setRGB(Number(voxel[3]) / 255, Number(voxel[4]) / 255, Number(voxel[5]) / 255);
    instanced.setColorAt(index, color);
  });
  instanced.instanceMatrix.needsUpdate = true;
  if (instanced.instanceColor) instanced.instanceColor.needsUpdate = true;
  scene.add(instanced);
  return { count: voxels.length };
}

async function loadDisplayMesh() {
  try {
    await loadGlbMesh();
    meshBadge.textContent = "COLMAP Delaunay mesh · GPU display";
  } catch (_) {
    try {
      const fallback = await loadVoxelFallback();
      meshBadge.textContent = `${fallback.count.toLocaleString()} surface voxels · GPU display`;
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
    setStatus("error", "ATLAS server is unavailable");
    el("detail-text").textContent = "Start the local ATLAS server, then reopen this page.";
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
  event.currentTarget.textContent = wallsVisible ? "Walls" : "Walls off";
  event.currentTarget.setAttribute("aria-pressed", String(wallsVisible));
});

cameraRig = makeCameraRig();
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
