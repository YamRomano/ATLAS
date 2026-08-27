import * as THREE from "./vendor/three.module.js";
import { GLTFLoader } from "./vendor/GLTFLoader.js";

// This module is intentionally display-only. It uses a selected map's
// footprint, or a visual-only calibrated fallback, to place a separate GLB
// layer and never writes map metadata,
// localization poses, patrol coordinates, walls, or COLMAP assets.
const REGISTRY_URL = "./public/map_visual_layers.json";
const MAX_PIXEL_RATIO = 2;

const canvas = document.getElementById("map-mesh");
const toggleButton = document.getElementById("toggle-mesh");
const alignButton = document.getElementById("align-map-mesh");
const lockButton = document.getElementById("lock-map-mesh");
const ceilingButton = document.getElementById("toggle-mesh-ceiling");
const restoreWallButton = document.getElementById("restore-mesh-wall");
const mainCanvas = document.getElementById("map");

let renderer = null;
let registry = { layers: [] };
let currentLayer = null;
let currentMapId = null;
let currentMapSignature = "";
let visible = false;
let loading = false;
let loadError = "";
let meshObject = null;
let loadedLayerKey = "";
let meshSourceBounds = null;
let meshMaterials = [];
let ceilingHidden = false;
let ceilingCutY = 0;
let wallCutawayActive = false;
let automaticWallCutawayActive = false;
const ceilingPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), ceilingCutY);
const wallCutawayPlaneRoom = new THREE.Plane();
const wallCutawayPlane = new THREE.Plane();
const automaticWallCutawayPlaneRoom = new THREE.Plane();
const automaticWallCutawayPlane = new THREE.Plane();
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const PLACEMENT_STORAGE_PREFIX = "atlas.map-mesh-placement.v1";
let placementMode = false;
let placementDrag = null;
let draftVisualOffsetXZ = null;
let centeredLockedPlacementKey = "";

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 400);
camera.position.set(0, 0, 100);
camera.up.set(0, 1, 0);
camera.lookAt(0, 0, 0);

const viewRoot = new THREE.Group();
viewRoot.matrixAutoUpdate = false;
const alignmentRoot = new THREE.Group();
alignmentRoot.matrixAutoUpdate = false;
viewRoot.add(alignmentRoot);
scene.add(viewRoot);

scene.add(new THREE.AmbientLight(0xffffff, 1.35));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.6);
keyLight.position.set(4, 7, 6);
scene.add(keyLight);

function viewerApi() {
  return window.TSOLVE_VIEWER || null;
}

function mapsById() {
  const maps = viewerApi()?.getMapLibrary?.()?.maps || [];
  return new Map(maps.map((entry) => [entry.id, entry]));
}

function mapInheritsFrom(entry, rootMapId, inheritToCopies) {
  if (!entry || !rootMapId) return false;
  if (entry.id === rootMapId) return true;
  if (!inheritToCopies) return false;
  const byId = mapsById();
  const visited = new Set();
  let cursor = entry;
  while (cursor?.source_map_id && !visited.has(cursor.source_map_id)) {
    if (cursor.source_map_id === rootMapId) return true;
    visited.add(cursor.source_map_id);
    cursor = byId.get(cursor.source_map_id);
  }
  return false;
}

function layerForMap(entry) {
  return (registry.layers || []).find((layer) =>
    layer.kind === "textured_mesh" &&
    mapInheritsFrom(entry, layer.root_map_id, layer.inherit_to_copies !== false)
  ) || null;
}

function wallCorners(entry) {
  return (entry?.safety_barriers || entry?.barriers || [])
    .map((wall) => (wall.corners || []).map((point) => new THREE.Vector3(...point.map(Number))))
    .filter((corners) => corners.length >= 4);
}

function footprintForMap(entry) {
  const walls = wallCorners(entry);
  if (!walls.length) return null;
  const heights = walls.flatMap((corners) => corners.map((point) => point.y));
  const floorY = Math.min(...heights);
  const floorCorners = walls
    .flatMap((corners) => corners)
    .filter((point) => Math.abs(point.y - floorY) < 0.01);
  if (!floorCorners.length) return null;
  const center = floorCorners
    .reduce((sum, point) => sum.add(point), new THREE.Vector3())
    .multiplyScalar(1 / floorCorners.length);
  const wallsByLength = [...walls]
    .sort((left, right) => right[0].distanceToSquared(right[1]) - left[0].distanceToSquared(left[1]));
  const longestWall = wallsByLength[0];
  const shortestWall = wallsByLength[wallsByLength.length - 1];
  const direction = longestWall[1].clone().sub(longestWall[0]);
  let axisDeg = THREE.MathUtils.radToDeg(Math.atan2(direction.z, direction.x));
  if (axisDeg < 90) axisDeg += 180;
  return {
    floorY,
    center,
    axisDeg,
    longLength: direction.length(),
    shortLength: shortestWall[0].distanceTo(shortestWall[1]),
  };
}

function calibratedFootprint(layer) {
  const target = layer?.alignment?.target_footprint;
  const centerXZ = target?.center_xz;
  if (!target || !Array.isArray(centerXZ) || centerXZ.length < 2) return null;
  const values = [
    target.floor_y,
    centerXZ[0],
    centerXZ[1],
    target.axis_deg,
    target.long_m,
    target.short_m,
  ].map(Number);
  if (!values.every(Number.isFinite) || values[4] <= 0 || values[5] <= 0) return null;
  return {
    floorY: values[0],
    center: new THREE.Vector3(values[1], values[0], values[2]),
    axisDeg: values[3],
    longLength: values[4],
    shortLength: values[5],
  };
}

function configuredVisualOffset(layer) {
  const offset = layer?.alignment?.visual_anchor_offset_xz;
  const x = Number(offset?.[0]);
  const z = Number(offset?.[1]);
  return [Number.isFinite(x) ? x : 0, Number.isFinite(z) ? z : 0];
}

function placementStorageKey(entry, layer) {
  if (!entry?.id || !layer?.id) return "";
  return `${PLACEMENT_STORAGE_PREFIX}:${entry.id}:${layer.id}`;
}

function lockedPlacement(entry, layer) {
  const key = placementStorageKey(entry, layer);
  if (!key) return null;
  try {
    const saved = JSON.parse(localStorage.getItem(key) || "null");
    const offset = saved?.visual_anchor_offset_xz;
    if (
      saved?.locked !== true ||
      saved?.registration_version !== (layer?.version || "") ||
      !Array.isArray(offset) ||
      offset.length < 2 ||
      !offset.slice(0, 2).every((value) => Number.isFinite(Number(value)))
    ) return null;
    return { ...saved, visual_anchor_offset_xz: offset.slice(0, 2).map(Number) };
  } catch {
    return null;
  }
}

function effectiveVisualOffset(entry, layer) {
  if (placementMode && draftVisualOffsetXZ && entry?.id === currentMapId) {
    return [...draftVisualOffsetXZ];
  }
  return lockedPlacement(entry, layer)?.visual_anchor_offset_xz || configuredVisualOffset(layer);
}

function alignmentMatrix(entry, layer) {
  const alignment = layer?.alignment || {};
  // Fixed visual layers use their audited physical room footprint so manual
  // wall edits cannot move localization content. Older room_footprint layers
  // retain their editable-wall behavior.
  const footprint = alignment.mode === "fixed_room_footprint"
    ? calibratedFootprint(layer)
    : footprintForMap(entry) || calibratedFootprint(layer);
  if (!footprint || !["room_footprint", "fixed_room_footprint"].includes(alignment.mode)) return null;
  const sourceCenter = alignment.source_center_xz || [0, 0];
  const sourceAxisDeg = Number(alignment.source_axis_deg || 0);
  const sourceShort = Number(alignment.source_short_m || 0);
  const visualAnchorOffsetXZ = effectiveVisualOffset(entry, layer);
  const offsetX = Number.isFinite(visualAnchorOffsetXZ[0]) ? visualAnchorOffsetXZ[0] : 0;
  const offsetZ = Number.isFinite(visualAnchorOffsetXZ[1]) ? visualAnchorOffsetXZ[1] : 0;
  const longScale = THREE.MathUtils.clamp(
    footprint.longLength / Math.max(Number(alignment.source_long_m || 1), 1e-6),
    0.6,
    1.1,
  );
  const shortScale = THREE.MathUtils.clamp(
    footprint.shortLength / Math.max(sourceShort, 1e-6),
    0.6,
    1.1,
  );
  // Fit both room axes, not a single annotated screen point. In source-axis
  // coordinates this maps both corners of the lower short wall onto the
  // corresponding COLMAP room-footprint corners. No map content is changed.
  const correction = new THREE.Matrix4()
    .makeTranslation(
      footprint.center.x + offsetX,
      footprint.floorY - Number(alignment.source_floor_y || 0),
      footprint.center.z + offsetZ,
    )
    .multiply(new THREE.Matrix4().makeRotationY(THREE.MathUtils.degToRad(-footprint.axisDeg)))
    .multiply(new THREE.Matrix4().makeScale(longScale, 1, shortScale))
    .multiply(new THREE.Matrix4().makeRotationY(THREE.MathUtils.degToRad(sourceAxisDeg)))
    .multiply(new THREE.Matrix4().makeTranslation(-Number(sourceCenter[0] || 0), 0, -Number(sourceCenter[1] || 0)));
  return correction;
}

function prepareMaterials(object, opacity) {
  const prepared = [];
  object.traverse((child) => {
    if (!child.isMesh) return;
    child.frustumCulled = false;
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    const cloned = materials.map((source) => {
      const material = source.clone();
      material.transparent = true;
      material.opacity = THREE.MathUtils.clamp(Number(opacity) || 0.72, 0.12, 1);
      material.depthWrite = true;
      material.side = THREE.DoubleSide;
      material.toneMapped = false;
      if (material.map && "colorSpace" in material.map) material.map.colorSpace = THREE.SRGBColorSpace;
      material.needsUpdate = true;
      prepared.push(material);
      return material;
    });
    child.material = Array.isArray(child.material) ? cloned : cloned[0];
  });
  return prepared;
}

function disposeObject(object) {
  object?.traverse((child) => {
    child.geometry?.dispose?.();
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    for (const material of materials) {
      material?.map?.dispose?.();
      material?.dispose?.();
    }
  });
}

function setStatus(state, message = "") {
  document.body.dataset.mapMeshOverlay = state;
  document.body.dataset.mapMeshOverlayMessage = message;
  document.body.dataset.mapMeshRegistration = currentLayer?.version || "";
}

function updateButton() {
  const entry = viewerApi()?.getCurrentMapEntry?.();
  const available = Boolean(currentLayer && alignmentMatrix(entry, currentLayer));
  const locked = Boolean(lockedPlacement(entry, currentLayer));
  toggleButton.hidden = !currentLayer;
  toggleButton.disabled = !available || loading;
  toggleButton.classList.toggle("active", available && visible);
  toggleButton.classList.toggle("mesh-loading", loading);
  toggleButton.textContent = loading ? "Loading Mesh…" : (visible ? "Hide Mesh" : "Show Mesh");
  toggleButton.title = loadError
    ? `Mesh unavailable: ${loadError}`
    : available
      ? `Show or hide the read-only room mesh. Double-click a visible wall for a temporary cutaway. Registration: ${currentLayer.version || "unversioned"}.`
      : "This map has no compatible visual alignment";
  alignButton.hidden = !currentLayer || locked || placementMode;
  alignButton.disabled = !available || loading || !visible || !meshObject || locked;
  alignButton.classList.toggle("active", placementMode);
  alignButton.title = "Enter the one-time top-view mesh placement mode, then drag the mesh into position";
  lockButton.hidden = !placementMode || locked;
  lockButton.disabled = !placementMode || loading || !visible || !meshObject;
  lockButton.classList.toggle("active", placementMode);
  lockButton.title = "Lock this visual alignment for this map and hide the placement controls";
  canvas.style.display = available && visible ? "block" : "none";
  canvas.classList.toggle("mesh-placement-active", placementMode && available && visible);
  canvas.classList.toggle("mesh-placement-dragging", Boolean(placementDrag));
  document.body.dataset.mapMeshPlacement = locked ? "locked" : (placementMode ? "editing" : "available");
  ceilingButton.hidden = !currentLayer;
  ceilingButton.disabled = !available || loading || !visible || !meshObject;
  ceilingButton.classList.toggle("active", available && visible && ceilingHidden);
  ceilingButton.textContent = ceilingHidden ? "Show Ceiling" : "Hide Ceiling";
  ceilingButton.setAttribute("aria-pressed", String(ceilingHidden));
  ceilingButton.title = visible
    ? "Show or hide only the upper ceiling of the visual mesh"
    : "Show the visual mesh before hiding its ceiling";
  restoreWallButton.hidden = !wallCutawayActive;
  restoreWallButton.disabled = !wallCutawayActive || loading || !meshObject;
  restoreWallButton.classList.toggle("active", wallCutawayActive);
  restoreWallButton.setAttribute("aria-pressed", String(wallCutawayActive));
  restoreWallButton.title = "Restore the wall hidden by the temporary double-click cutaway";
}

function ensureRenderer() {
  if (renderer) return renderer;
  renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: "high-performance" });
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO));
  renderer.localClippingEnabled = true;
  if ("outputColorSpace" in renderer) renderer.outputColorSpace = THREE.SRGBColorSpace;
  return renderer;
}

async function loadLayer(layer) {
  const layerKey = `${layer.id}:${layer.version || layer.url}`;
  if (meshObject && loadedLayerKey === layerKey) return;
  loading = true;
  loadError = "";
  updateButton();
  setStatus("loading", layer.id);
  try {
    ensureRenderer();
    const gltf = await new GLTFLoader().loadAsync(`./${layer.url}?v=${encodeURIComponent(layer.version || "1")}`);
    meshSourceBounds = new THREE.Box3().setFromObject(gltf.scene);
    meshMaterials = prepareMaterials(gltf.scene, layer.opacity);
    if (meshObject) {
      alignmentRoot.remove(meshObject);
      disposeObject(meshObject);
    }
    meshObject = gltf.scene;
    loadedLayerKey = layerKey;
    alignmentRoot.add(meshObject);
    updateCeilingCut();
    updateClippingVisibility();
    setStatus("ready", layer.id);
  } catch (error) {
    loadError = error?.message || String(error);
    setStatus("error", loadError);
    visible = false;
    throw error;
  } finally {
    loading = false;
    updateButton();
  }
}

function updateCeilingCut() {
  if (!meshSourceBounds || !meshObject) return;
  // The source scan and aligned ATLAS room share the vertical axis. Applying
  // the visual-only registration matrix here gives the same 64% room cut used
  // by Camera Path, without changing the GLB or any map coordinates.
  const alignedBounds = meshSourceBounds.clone().applyMatrix4(alignmentRoot.matrix);
  const size = alignedBounds.getSize(new THREE.Vector3());
  ceilingCutY = alignedBounds.min.y + size.y * 0.64;
}

function updateClippingVisibility() {
  const planes = [];
  if (ceilingHidden) planes.push(ceilingPlane);
  if (automaticWallCutawayActive) planes.push(automaticWallCutawayPlane);
  if (wallCutawayActive) planes.push(wallCutawayPlane);
  for (const material of meshMaterials) {
    material.clippingPlanes = planes;
    material.needsUpdate = true;
  }
  updateButton();
}

function syncClippingPlanesToView() {
  // Materials clip in rendered world space. The cutoff is defined in room
  // space, so transform the plane with the same live ATLAS view matrix as the
  // mesh. This keeps ceiling and wall cutaways attached while orbiting.
  if (ceilingHidden) {
    ceilingPlane.set(new THREE.Vector3(0, -1, 0), ceilingCutY);
    ceilingPlane.applyMatrix4(viewRoot.matrix);
  }
  if (wallCutawayActive) {
    wallCutawayPlane.copy(wallCutawayPlaneRoom).applyMatrix4(viewRoot.matrix);
  }
  if (automaticWallCutawayActive) {
    automaticWallCutawayPlane.copy(automaticWallCutawayPlaneRoom).applyMatrix4(viewRoot.matrix);
  }
}

function footprintCorners(footprint) {
  if (!footprint) return [];
  const angle = THREE.MathUtils.degToRad(footprint.axisDeg);
  const longAxis = new THREE.Vector3(Math.cos(angle), 0, Math.sin(angle));
  const shortAxis = new THREE.Vector3(-Math.sin(angle), 0, Math.cos(angle));
  const longHalf = footprint.longLength * 0.5;
  const shortHalf = footprint.shortLength * 0.5;
  return [
    footprint.center.clone().addScaledVector(longAxis, longHalf).addScaledVector(shortAxis, shortHalf),
    footprint.center.clone().addScaledVector(longAxis, longHalf).addScaledVector(shortAxis, -shortHalf),
    footprint.center.clone().addScaledVector(longAxis, -longHalf).addScaledVector(shortAxis, shortHalf),
    footprint.center.clone().addScaledVector(longAxis, -longHalf).addScaledVector(shortAxis, -shortHalf),
  ];
}

function syncAutomaticWallCutaway(entry) {
  const footprint = currentLayer?.alignment?.mode === "fixed_room_footprint"
    ? calibratedFootprint(currentLayer)
    : footprintForMap(entry) || calibratedFootprint(currentLayer);
  const inverseView = new THREE.Matrix4().copy(viewRoot.matrix).invert();
  const outward = new THREE.Vector3(0, 0, 1).transformDirection(inverseView);
  const horizontal = Math.hypot(outward.x, outward.z);
  const nextActive = Boolean(visible && meshObject && footprint && horizontal > 0.48);
  document.body.dataset.mapMeshAutomaticWall = nextActive ? "cutaway" : "full";

  if (nextActive) {
    outward.set(outward.x / horizontal, 0, outward.z / horizontal);
    const inward = outward.clone().negate();
    const corners = footprintCorners(footprint);
    const frontProjection = Math.max(...corners.map((corner) => corner.dot(outward)));
    const centerProjection = footprint.center.dot(outward);
    const pointInsideFrontWall = footprint.center.clone()
      .addScaledVector(outward, frontProjection - centerProjection)
      .addScaledVector(inward, 0.16);
    automaticWallCutawayPlaneRoom.setFromNormalAndCoplanarPoint(inward, pointInsideFrontWall);
  }

  if (automaticWallCutawayActive !== nextActive) {
    automaticWallCutawayActive = nextActive;
    updateClippingVisibility();
  }
}

function resetWallCutaway() {
  if (!wallCutawayActive) return;
  wallCutawayActive = false;
  updateClippingVisibility();
}

function mapSignature(entry, layer = currentLayer) {
  const footprint = layer?.alignment?.mode === "fixed_room_footprint"
    ? calibratedFootprint(layer)
    : footprintForMap(entry) || calibratedFootprint(layer);
  if (!footprint) return `${entry?.id || "none"}:no-footprint`;
  const alignment = layer?.alignment || {};
  return [
    entry.id,
    layer?.version || "unversioned",
    footprint.floorY,
    footprint.center.x,
    footprint.center.z,
    footprint.axisDeg,
    footprint.longLength,
    footprint.shortLength,
    alignment.source_axis_deg || 0,
    alignment.source_long_m || 0,
    alignment.source_short_m || 0,
    alignment.visual_anchor_offset_xz?.[0] || 0,
    alignment.visual_anchor_offset_xz?.[1] || 0,
  ]
    .map((value) => typeof value === "number" ? value.toFixed(6) : value)
    .join(":");
}

function syncSelectedMap() {
  const entry = viewerApi()?.getCurrentMapEntry?.();
  const nextLayer = layerForMap(entry);
  const signature = mapSignature(entry, nextLayer);
  if (entry?.id === currentMapId && signature === currentMapSignature) return;
  placementMode = false;
  placementDrag = null;
  draftVisualOffsetXZ = null;
  currentMapId = entry?.id || null;
  currentMapSignature = signature;
  currentLayer = nextLayer;
  visible = false;
  wallCutawayActive = false;
  loadError = "";
  const matrix = currentLayer ? alignmentMatrix(entry, currentLayer) : null;
  alignmentRoot.matrix.copy(matrix || new THREE.Matrix4());
  alignmentRoot.matrixWorldNeedsUpdate = true;
  updateCeilingCut();
  updateClippingVisibility();
  updateButton();
  setStatus(currentLayer ? (matrix ? "available" : "unaligned") : "not-configured", currentMapId || "");
  const savedPlacementKey = lockedPlacement(entry, currentLayer)
    ? placementStorageKey(entry, currentLayer)
    : "";
  if (savedPlacementKey && savedPlacementKey !== centeredLockedPlacementKey) {
    centeredLockedPlacementKey = savedPlacementKey;
    requestAnimationFrame(() => document.getElementById("reset")?.click());
  }
}

function roomFloorPointFromClient(clientX, clientY) {
  const api = viewerApi();
  const entry = api?.getCurrentMapEntry?.();
  const footprint = currentLayer?.alignment?.mode === "fixed_room_footprint"
    ? calibratedFootprint(currentLayer)
    : footprintForMap(entry) || calibratedFootprint(currentLayer);
  const room = api?.getRoom?.();
  if (!footprint || !room?.bounds || !api?.projectRoomPointToViewport) return null;
  const base = [footprint.center.x, footprint.floorY, footprint.center.z];
  const step = Math.max(0.01, Number(room.bounds.radius || 1) * 0.04);
  const p0 = api.projectRoomPointToViewport(base);
  const px = api.projectRoomPointToViewport([base[0] + step, base[1], base[2]]);
  const pz = api.projectRoomPointToViewport([base[0], base[1], base[2] + step]);
  if (![p0, px, pz].every((point) => Array.isArray(point) && point.length >= 2)) return null;
  const ax = px[0] - p0[0];
  const ay = px[1] - p0[1];
  const bx = pz[0] - p0[0];
  const by = pz[1] - p0[1];
  const det = ax * by - ay * bx;
  if (Math.abs(det) < 1e-8) return null;
  const dx = clientX - p0[0];
  const dy = clientY - p0[1];
  const cx = (dx * by - dy * bx) / det;
  const cz = (ax * dy - ay * dx) / det;
  return [base[0] + cx * step, base[2] + cz * step];
}

function applyDraftPlacement() {
  const entry = viewerApi()?.getCurrentMapEntry?.();
  const matrix = alignmentMatrix(entry, currentLayer);
  if (!matrix) return;
  alignmentRoot.matrix.copy(matrix);
  alignmentRoot.matrixWorldNeedsUpdate = true;
  updateCeilingCut();
  syncClippingPlanesToView();
  document.body.dataset.mapMeshDraftOffset = draftVisualOffsetXZ?.map((value) => value.toFixed(4)).join(",") || "";
}

function beginPlacement() {
  const entry = viewerApi()?.getCurrentMapEntry?.();
  if (!visible || !meshObject || loading || !currentLayer || lockedPlacement(entry, currentLayer)) return;
  placementMode = true;
  placementDrag = null;
  draftVisualOffsetXZ = effectiveVisualOffset(entry, currentLayer);
  document.getElementById("view-top")?.click();
  updateButton();
}

function cancelPlacement() {
  if (!placementMode) return;
  placementMode = false;
  placementDrag = null;
  draftVisualOffsetXZ = null;
  const matrix = alignmentMatrix(viewerApi()?.getCurrentMapEntry?.(), currentLayer);
  if (matrix) alignmentRoot.matrix.copy(matrix);
  alignmentRoot.matrixWorldNeedsUpdate = true;
  updateCeilingCut();
  updateButton();
}

function lockPlacement() {
  const entry = viewerApi()?.getCurrentMapEntry?.();
  const key = placementStorageKey(entry, currentLayer);
  if (!placementMode || !draftVisualOffsetXZ || !key) return;
  const offset = draftVisualOffsetXZ.map((value) => Number(value.toFixed(4)));
  try {
    localStorage.setItem(key, JSON.stringify({
      locked: true,
      registration_version: currentLayer?.version || "",
      visual_anchor_offset_xz: offset,
      locked_at: new Date().toISOString(),
    }));
  } catch (error) {
    console.warn("The visual mesh placement could not be saved.", error);
    return;
  }
  placementMode = false;
  placementDrag = null;
  draftVisualOffsetXZ = null;
  const matrix = alignmentMatrix(entry, currentLayer);
  if (matrix) alignmentRoot.matrix.copy(matrix);
  alignmentRoot.matrixWorldNeedsUpdate = true;
  updateCeilingCut();
  updateButton();
  centeredLockedPlacementKey = key;
  requestAnimationFrame(() => document.getElementById("reset")?.click());
}

function syncAtlasView() {
  const api = viewerApi();
  const room = api?.getRoom?.();
  const view = api?.getView?.();
  if (!room?.bounds || !view || !renderer) return false;
  const rect = mainCanvas.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return false;

  // The pinned localization drawer changes .map-pane padding. The 2D map is
  // a normal-flow canvas inside that padded content box, while this WebGL
  // layer is absolutely positioned. Mirror the real map canvas rectangle on
  // every frame so opening, minimizing, pinning, or resizing the drawer cannot
  // stretch the mesh independently of the COLMAP points.
  const hostRect = canvas.offsetParent?.getBoundingClientRect?.()
    || mainCanvas.parentElement?.getBoundingClientRect?.()
    || rect;
  canvas.style.inset = "auto";
  canvas.style.left = `${rect.left - hostRect.left}px`;
  canvas.style.top = `${rect.top - hostRect.top}px`;
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;

  const dpr = Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO);
  renderer.setPixelRatio(dpr);
  renderer.setSize(rect.width, rect.height, false);

  const axis = view.axisScale || { x: 1, y: 1, z: 1 };
  const center = room.bounds.center;
  viewRoot.matrix
    .makeRotationX(Number(view.pitch) || 0)
    .multiply(new THREE.Matrix4().makeRotationY(Number(view.yaw) || 0))
    .multiply(new THREE.Matrix4().makeScale(Number(axis.x) || 1, Number(axis.y) || 1, Number(axis.z) || 1))
    .multiply(new THREE.Matrix4().makeTranslation(-center[0], -center[1], -center[2]));
  viewRoot.matrixWorldNeedsUpdate = true;
  syncAutomaticWallCutaway(viewerApi()?.getCurrentMapEntry?.());
  syncClippingPlanesToView();

  const scale = 0.46 * Math.min(rect.width, rect.height) * Number(view.zoom || 1) / Math.max(Number(room.bounds.radius), 1e-6);
  const centerX = rect.width * 0.5 + Number(view.panX || 0);
  const centerY = rect.height * 0.52 + Number(view.panY || 0);
  camera.left = -centerX / scale;
  camera.right = (rect.width - centerX) / scale;
  camera.top = centerY / scale;
  camera.bottom = -(rect.height - centerY) / scale;
  camera.updateProjectionMatrix();
  return true;
}

toggleButton?.addEventListener("click", async () => {
  if (!currentLayer || loading) return;
  if (visible) {
    visible = false;
    resetWallCutaway();
    setStatus("hidden", currentLayer.id);
    updateButton();
    return;
  }
  try {
    await loadLayer(currentLayer);
    const entry = viewerApi()?.getCurrentMapEntry?.();
    const matrix = alignmentMatrix(entry, currentLayer);
    if (!matrix) throw new Error("The selected map has no compatible visual alignment.");
    alignmentRoot.matrix.copy(matrix);
    alignmentRoot.matrixWorldNeedsUpdate = true;
    visible = true;
    setStatus("visible", currentLayer.id);
  } catch (error) {
    console.warn("Textured map visualization could not be loaded.", error);
  }
  updateButton();
});

alignButton?.addEventListener("click", beginPlacement);
lockButton?.addEventListener("click", lockPlacement);

canvas?.addEventListener("pointerdown", (event) => {
  if (!placementMode || event.button !== 0) return;
  const point = roomFloorPointFromClient(event.clientX, event.clientY);
  if (!point) return;
  placementDrag = {
    pointerId: event.pointerId,
    startPoint: point,
    startOffset: [...draftVisualOffsetXZ],
  };
  canvas.setPointerCapture?.(event.pointerId);
  updateButton();
  event.preventDefault();
  event.stopPropagation();
});

canvas?.addEventListener("pointermove", (event) => {
  if (!placementMode || !placementDrag || placementDrag.pointerId !== event.pointerId) return;
  const point = roomFloorPointFromClient(event.clientX, event.clientY);
  if (!point) return;
  draftVisualOffsetXZ = [
    placementDrag.startOffset[0] + point[0] - placementDrag.startPoint[0],
    placementDrag.startOffset[1] + point[1] - placementDrag.startPoint[1],
  ];
  applyDraftPlacement();
  event.preventDefault();
  event.stopPropagation();
});

function finishPlacementDrag(event) {
  if (!placementDrag || placementDrag.pointerId !== event.pointerId) return;
  canvas.releasePointerCapture?.(event.pointerId);
  placementDrag = null;
  updateButton();
  event.preventDefault();
  event.stopPropagation();
}

canvas?.addEventListener("pointerup", finishPlacementDrag);
canvas?.addEventListener("pointercancel", finishPlacementDrag);
window.addEventListener("keydown", (event) => {
  if (placementMode && event.key === "Escape") cancelPlacement();
});

ceilingButton?.addEventListener("click", () => {
  if (!visible || !meshObject || loading) return;
  ceilingHidden = !ceilingHidden;
  updateClippingVisibility();
});

restoreWallButton?.addEventListener("click", resetWallCutaway);

mainCanvas?.addEventListener("dblclick", (event) => {
  if (!visible || !meshObject || loading || viewerApi()?.isMapInteractionBusy?.()) return;
  const rect = mainCanvas.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return;
  pointer.set(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  );

  scene.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObject(meshObject, true).find((candidate) => candidate.face);
  if (!hit?.face) return;

  const normalMatrix = new THREE.Matrix3().getNormalMatrix(hit.object.matrixWorld);
  const worldNormal = hit.face.normal.clone().applyMatrix3(normalMatrix).normalize();
  const inverseView = new THREE.Matrix4().copy(viewRoot.matrixWorld).invert();
  const roomPlane = new THREE.Plane()
    .setFromNormalAndCoplanarPoint(worldNormal, hit.point)
    .applyMatrix4(inverseView);

  // Floor/ceiling double-clicks remain available to normal map navigation;
  // only a mostly vertical surface is treated as a wall cutaway.
  if (Math.abs(roomPlane.normal.y) > 0.65) return;
  const cameraRoom = camera.getWorldPosition(new THREE.Vector3()).applyMatrix4(inverseView);
  if (roomPlane.distanceToPoint(cameraRoom) > 0) roomPlane.negate();
  const pointInside = roomPlane.coplanarPoint(new THREE.Vector3())
    .addScaledVector(roomPlane.normal, 0.035);
  wallCutawayPlaneRoom.setFromNormalAndCoplanarPoint(roomPlane.normal, pointInside);
  wallCutawayActive = true;
  syncClippingPlanesToView();
  updateClippingVisibility();
  event.preventDefault();
  event.stopPropagation();
});

async function loadRegistry() {
  const response = await fetch(REGISTRY_URL, { cache: "no-store" });
  if (!response.ok) throw new Error(`Mesh registry request failed (${response.status}).`);
  registry = await response.json();
  // The map may finish loading before this small sidecar registry. Force one
  // eligibility pass so a ready map is not left in the pre-fetch state.
  currentMapSignature = "";
  syncSelectedMap();
}

function animate() {
  syncSelectedMap();
  if (visible && currentLayer && meshObject && syncAtlasView()) {
    renderer.render(scene, camera);
  }
  requestAnimationFrame(animate);
}

window.ATLAS_MAP_MESH = {
  isAvailable: () => Boolean(currentLayer && alignmentMatrix(viewerApi()?.getCurrentMapEntry?.(), currentLayer)),
  isVisible: () => visible,
  isCeilingHidden: () => ceilingHidden,
  isWallCutawayActive: () => wallCutawayActive,
  isAutomaticWallCutawayActive: () => automaticWallCutawayActive,
  getStatus: () => document.body.dataset.mapMeshOverlay || "initializing",
  getRegistrationVersion: () => currentLayer?.version || "",
  getVisualAnchorOffsetXZ: () => [...(currentLayer?.alignment?.visual_anchor_offset_xz || [0, 0])],
  getEffectiveVisualAnchorOffsetXZ: () => effectiveVisualOffset(viewerApi()?.getCurrentMapEntry?.(), currentLayer),
  isPlacementLocked: () => Boolean(lockedPlacement(viewerApi()?.getCurrentMapEntry?.(), currentLayer)),
  getCurrentMapId: () => currentMapId,
};

setStatus("initializing");
loadRegistry().catch((error) => {
  loadError = error?.message || String(error);
  setStatus("error", loadError);
  updateButton();
  console.warn("Map visualization registry unavailable.", error);
});
animate();
