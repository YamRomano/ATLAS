const canvas = document.getElementById("map");
let ctx = canvas.getContext("2d");
const staticCanvas = document.createElement("canvas");
const staticCtx = staticCanvas.getContext("2d");
const video = document.getElementById("video");
const liveFrameView = document.getElementById("live-frame-view");
const liveFrameStatus = document.getElementById("live-frame-status");
const stats = document.getElementById("stats");
const poseTime = document.getElementById("pose-time");
const poseTotal = document.getElementById("pose-total");
const poseAction = document.getElementById("pose-action");
const poseRoot = document.getElementById("pose-root");
const poseCenter = document.getElementById("pose-center");
const poseT = document.getElementById("pose-t");
const poseR = document.getElementById("pose-r");
const demoApp = document.getElementById("demo-app");
const startPage = document.getElementById("start-page");
const startPreview = document.getElementById("start-preview");
const navBack = document.getElementById("nav-back");
const atlasHome = document.getElementById("atlas-home");
const atlasScreenLabel = document.getElementById("atlas-screen-label");
const mapCardList = document.getElementById("map-card-list");
const uploadStatus = document.getElementById("upload-status");
const mapStatus = document.getElementById("map-status");
const droneStatus = document.getElementById("drone-status");
const jobLog = document.getElementById("job-log");
const djiLiveFeed = document.getElementById("dji-live-feed");
const djiLiveFeedSide = document.getElementById("dji-live-feed-side");
const djiLiveState = document.getElementById("dji-live-state");
const djiLiveStateSide = document.getElementById("dji-live-state-side");
const djiLiveMeta = document.getElementById("dji-live-meta");
const djiLiveMetaSide = document.getElementById("dji-live-meta-side");
const pipelineStatus = document.querySelector(".pipeline-status");
const libraryPanel = document.querySelector(".library-panel");
const collapseLibraryButton = document.getElementById("collapse-library");
const collapseConsoleButton = document.getElementById("collapse-console");
const mapModal = document.getElementById("map-modal");
const videoLibraryModal = document.getElementById("video-library-modal");
const videoLibraryTitle = document.getElementById("video-library-title");
const videoLibrarySubtitle = document.getElementById("video-library-subtitle");
const videoLibraryList = document.getElementById("video-library-list");
const mapUpload = document.getElementById("map-upload");
const mapVideoUpload = document.getElementById("map-video-upload");
const demoDroneUpload = document.getElementById("demo-drone-upload");
const liveAtlasPhoneIp = document.getElementById("live-atlas-phone-ip");
const liveAtlasFps = document.getElementById("live-atlas-fps");
const liveLocalizationControl = document.getElementById("live-localization-control");
const liveControlSummary = document.getElementById("live-control-summary");
const startLiveAtlasButton = document.getElementById("start-live-atlas");
const stopLiveAtlasButton = document.getElementById("stop-live-atlas");
const takeoffHeightInput = document.getElementById("takeoff-height-m");
const djiTakeoffButton = document.getElementById("dji-takeoff");
const djiLandButton = document.getElementById("dji-land");
const djiCommandStatus = document.getElementById("dji-command-status");
const confirmLocalizationButton = document.getElementById("confirm-localization");
const localizationGateStatus = document.getElementById("localization-gate-status");
const droneControlPanel = document.getElementById("drone-control-panel");
const missionSpeedSelect = document.getElementById("mission-speed");
const planMissionButton = document.getElementById("plan-mission");
const replayTabs = document.getElementById("replay-tabs");
const replayTabList = document.getElementById("replay-tab-list");
const sidePanel = document.querySelector(".side");
const liveMappingPanel = document.getElementById("live-mapping-panel");
const liveCameraFeed = document.getElementById("live-camera-feed");
const liveBuildPreview = document.getElementById("live-build-preview");
const liveMapCaption = document.getElementById("live-map-caption");
const stopMapping = document.getElementById("stop-mapping");
const viewIsoButton = document.getElementById("view-iso");
const viewDroneButton = document.getElementById("view-drone");
const togglePointsButton = document.getElementById("toggle-points");
const toggleCamerasButton = document.getElementById("toggle-cameras");
const selectTargetButton = document.getElementById("select-target");
const clearTargetButton = document.getElementById("clear-target");
const startMissionButton = document.getElementById("start-mission");
const targetStatus = document.getElementById("target-status");
const safetyBarrierPanel = document.getElementById("safety-barrier-panel");
const addBarrierButton = document.getElementById("add-barrier");
const adjustWallsButton = document.getElementById("adjust-walls");
const saveWallAdjustmentsButton = document.getElementById("save-wall-adjustments");
const cancelBarrierButton = document.getElementById("cancel-barrier");
const clearBarriersButton = document.getElementById("clear-barriers");
const barrierStatus = document.getElementById("barrier-status");
const barrierList = document.getElementById("barrier-list");
const barrierClearanceInput = document.getElementById("barrier-clearance-m");

let scene = null;
let poses = [];
let poseStreamMeta = null;
let scan = null;
let room = null;
let droneModel = null;
let droneModelPromise = null;
let mapLibraryData = { selected_map_id: "default_demo", maps: [] };
let currentMapEntry = null;
let renderStarted = false;
let lastMapStatus = null;
let lastDroneStatus = null;
let currentScreen = "start";
let screenHistory = [];
let pendingLiveReplayOpen = false;
let pendingLiveReplayMapId = null;
let liveReplayInFlight = false;
let liveReplayMessage = "";
let liveReplayStageDetail = "";
let liveReplayWaitingViewPrepared = false;
let liveVideoObjectUrl = null;
let liveReplayStartedAt = 0;
let livePoseStreamKey = "";
let livePoseStreamCount = 0;
let liveStatusPollBusy = false;
let liveVideoWaitingForFirstPose = false;
let liveVideoSyncedToFirstPose = false;
let liveCurrentPoseOverride = null;
let liveFrameMode = false;
let liveAtlasPreviewActive = false;
let lastReplayFrameUrl = "";
let replayFramePlaybackEnabled = false;
let pathPlaybackActive = false;
let pathPlaybackStartWallMs = 0;
let pathPlaybackStartTimeSec = 0;
let pathPlaybackEndTimeSec = 0;
const previewSceneCache = new Map();
const previewZoomByMap = new Map();
let view = {
  mode: "iso",
  yaw: -0.72,
  pitch: 0.68,
  zoom: 1.22,
  panX: 0,
  panY: 0,
  axisScale: { x: 1, y: 1, z: 1 },
  showPoints: true,
  showCameras: true,
};
let missionTarget = null;
let missionSelecting = false;
let missionDraggingTarget = false;
let missionDragMoved = false;
let firstLocalizationConfirmed = false;
let plannedMission = null;
let barrierDraft = null;
let barrierEditing = false;
let barrierAdjusting = false;
let barrierUnsaved = false;
let stagedSafetyBarrierMapId = null;
let stagedSafetyBarriers = null;
let barrierSaving = false;
let barrierCornerDrag = null;
let barrierTransformDrag = null;
let barrierCornerHover = null;
let barrierTransformHover = null;
let barrierDragMoved = false;
let barrierClickSuppress = false;
const isoViewPresets = [
  { yaw: -0.72, pitch: 0.68, zoom: 1.22 },
  { yaw: 0.78, pitch: 0.66, zoom: 1.22 },
  { yaw: 2.35, pitch: 0.70, zoom: 1.22 },
  { yaw: -2.28, pitch: 0.64, zoom: 1.22 },
];
let isoViewIndex = 0;
const sideViewPresets = [
  { yaw: 0.0, pitch: 0.0, zoom: 1.25 },
  { yaw: Math.PI / 4, pitch: 0.0, zoom: 1.25 },
  { yaw: Math.PI / 2, pitch: 0.0, zoom: 1.25 },
  { yaw: 3 * Math.PI / 4, pitch: 0.0, zoom: 1.25 },
  { yaw: Math.PI, pitch: 0.0, zoom: 1.25 },
  { yaw: -3 * Math.PI / 4, pitch: 0.0, zoom: 1.25 },
  { yaw: -Math.PI / 2, pitch: 0.0, zoom: 1.25 },
  { yaw: -Math.PI / 4, pitch: 0.0, zoom: 1.25 },
];
let sideViewIndex = 0;
let dragging = false;
let last = { x: 0, y: 0 };
let staticLayerKey = "";
let staticLayerPan = { x: 0, y: 0 };
let interactionFastUntil = 0;

function invalidateStaticLayer() {
  staticLayerKey = "";
}

function markFastInteraction(durationMs = 180) {
  interactionFastUntil = performance.now() + durationMs;
}

function isFastInteraction() {
  return dragging || performance.now() < interactionFastUntil;
}

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const nextWidth = Math.max(1, Math.floor(rect.width * dpr));
  const nextHeight = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
    canvas.width = nextWidth;
    canvas.height = nextHeight;
    invalidateStaticLayer();
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { rect, dpr };
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function mul(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s];
}

function norm(a) {
  return Math.sqrt(Math.max(dot(a, a), 1e-18));
}

function normalize(a) {
  const n = norm(a);
  return [a[0] / n, a[1] / n, a[2] / n];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function matVec(M, v) {
  return [
    M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
    M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
    M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2],
  ];
}

function mat4Identity() {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

function mat4Mul(a, b) {
  const out = new Array(16).fill(0);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      for (let k = 0; k < 4; k++) out[r * 4 + c] += a[r * 4 + k] * b[k * 4 + c];
    }
  }
  return out;
}

function quatToMat4(q) {
  const [x, y, z, w] = q;
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;
  return [
    1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0,
    2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0,
    2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0,
    0, 0, 0, 1,
  ];
}

function nodeMatrix(node) {
  if (node.matrix) return node.matrix.slice();
  const t = node.translation || [0, 0, 0];
  const r = quatToMat4(node.rotation || [0, 0, 0, 1]);
  const s = node.scale || [1, 1, 1];
  const S = [s[0], 0, 0, 0, 0, s[1], 0, 0, 0, 0, s[2], 0, 0, 0, 0, 1];
  const T = [1, 0, 0, t[0], 0, 1, 0, t[1], 0, 0, 1, t[2], 0, 0, 0, 1];
  return mat4Mul(T, mat4Mul(r, S));
}

function transformPoint4(M, p) {
  return [
    M[0] * p[0] + M[1] * p[1] + M[2] * p[2] + M[3],
    M[4] * p[0] + M[5] * p[1] + M[6] * p[2] + M[7],
    M[8] * p[0] + M[9] * p[1] + M[10] * p[2] + M[11],
  ];
}

const GLB_COMPONENTS = {
  5120: { getter: "getInt8", bytes: 1 },
  5121: { getter: "getUint8", bytes: 1 },
  5122: { getter: "getInt16", bytes: 2 },
  5123: { getter: "getUint16", bytes: 2 },
  5125: { getter: "getUint32", bytes: 4 },
  5126: { getter: "getFloat32", bytes: 4 },
};
const GLB_TYPE_COUNTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 };

function parseGLB(buffer) {
  const dv = new DataView(buffer);
  const magic = dv.getUint32(0, true);
  if (magic !== 0x46546c67) throw new Error("Not a GLB file");
  const version = dv.getUint32(4, true);
  if (version !== 2) throw new Error(`Unsupported GLB version ${version}`);
  const length = dv.getUint32(8, true);
  let off = 12;
  let gltf = null;
  let bin = null;
  while (off < length) {
    const chunkLength = dv.getUint32(off, true);
    const chunkType = dv.getUint32(off + 4, true);
    off += 8;
    const chunk = buffer.slice(off, off + chunkLength);
    off += chunkLength;
    if (chunkType === 0x4e4f534a) {
      gltf = JSON.parse(new TextDecoder().decode(chunk).replace(/\0+$/g, "").trim());
    } else if (chunkType === 0x004e4942) {
      bin = chunk;
    }
  }
  if (!gltf || !bin) throw new Error("GLB missing JSON or BIN chunk");
  return { gltf, bin };
}

function readAccessor(gltf, bin, accessorIndex) {
  const accessor = gltf.accessors[accessorIndex];
  const view = gltf.bufferViews[accessor.bufferView];
  const comp = GLB_COMPONENTS[accessor.componentType];
  const elemCount = GLB_TYPE_COUNTS[accessor.type];
  const count = accessor.count;
  const baseOffset = (view.byteOffset || 0) + (accessor.byteOffset || 0);
  const stride = view.byteStride || (comp.bytes * elemCount);
  const dv = new DataView(bin);
  const out = [];
  for (let i = 0; i < count; i++) {
    const base = baseOffset + i * stride;
    const vals = [];
    for (let j = 0; j < elemCount; j++) {
      const byteOffset = base + j * comp.bytes;
      vals.push(comp.bytes === 1 ? dv[comp.getter](byteOffset) : dv[comp.getter](byteOffset, true));
    }
    out.push(elemCount === 1 ? vals[0] : vals);
  }
  return out;
}

function normalizeModel(vertices) {
  const mins = [Infinity, Infinity, Infinity];
  const maxs = [-Infinity, -Infinity, -Infinity];
  for (const v of vertices) {
    for (let i = 0; i < 3; i++) {
      mins[i] = Math.min(mins[i], v[i]);
      maxs[i] = Math.max(maxs[i], v[i]);
    }
  }
  const center = mins.map((v, i) => 0.5 * (v + maxs[i]));
  const span = Math.max(...maxs.map((v, i) => v - mins[i]), 1e-9);
  return {
    vertices: vertices.map(v => [(v[0] - center[0]) / span, (v[1] - center[1]) / span, (v[2] - center[2]) / span]),
    bounds: { min: mins, max: maxs, center, span },
  };
}

async function loadDroneGLB() {
  const response = await fetch("public/models/dji-mini-3-pro.glb");
  if (!response.ok) throw new Error(`GLB load failed: ${response.status}`);
  const { gltf, bin } = parseGLB(await response.arrayBuffer());
  const vertices = [];
  const triangles = [];

  function appendMesh(meshIndex, worldMatrix) {
    const mesh = gltf.meshes?.[meshIndex];
    if (!mesh) return;
    for (const prim of mesh.primitives || []) {
      if ((prim.mode ?? 4) !== 4 || prim.attributes?.POSITION == null) continue;
      const localPositions = readAccessor(gltf, bin, prim.attributes.POSITION);
      const base = vertices.length;
      for (const p of localPositions) vertices.push(transformPoint4(worldMatrix, p));
      let indices;
      if (prim.indices != null) indices = readAccessor(gltf, bin, prim.indices);
      else indices = localPositions.map((_, i) => i);
      for (let i = 0; i + 2 < indices.length; i += 3) {
        triangles.push([base + Number(indices[i]), base + Number(indices[i + 1]), base + Number(indices[i + 2])]);
      }
    }
  }

  function walkNode(nodeIndex, parentMatrix) {
    const node = gltf.nodes[nodeIndex];
    const worldMatrix = mat4Mul(parentMatrix, nodeMatrix(node));
    if (node.mesh != null) appendMesh(node.mesh, worldMatrix);
    for (const child of node.children || []) walkNode(child, worldMatrix);
  }

  const sceneIndex = gltf.scene ?? 0;
  const sceneNodes = gltf.scenes?.[sceneIndex]?.nodes || [];
  if (sceneNodes.length) {
    for (const n of sceneNodes) walkNode(n, mat4Identity());
  } else {
    for (let i = 0; i < (gltf.meshes || []).length; i++) appendMesh(i, mat4Identity());
  }

  const normalized = normalizeModel(vertices);
  const triLimit = 5200;
  const step = Math.max(1, Math.ceil(triangles.length / triLimit));
  const sampledTriangles = [];
  for (let i = 0; i < triangles.length; i += step) sampledTriangles.push(triangles[i]);
  const edgeSet = new Set();
  const addEdge = (a, b) => {
    if (a === b) return;
    edgeSet.add(a < b ? `${a}:${b}` : `${b}:${a}`);
  };
  for (const [a, b, c] of sampledTriangles) {
    addEdge(a, b);
    addEdge(b, c);
    addEdge(c, a);
  }
  const allEdges = [...edgeSet].map(key => key.split(":").map(Number));
  const edgeLimit = 1800;
  const edgeStep = Math.max(1, Math.ceil(allEdges.length / edgeLimit));
  const sampledEdges = [];
  for (let i = 0; i < allEdges.length; i += edgeStep) sampledEdges.push(allEdges[i]);
  return {
    kind: "glb",
    name: "DJI Mini 3 Pro",
    vertices: normalized.vertices,
    triangles: sampledTriangles,
    edges: sampledEdges,
    sourceTriangleCount: triangles.length,
    bounds: normalized.bounds,
  };
}

function loadDroneModelOnce() {
  if (!droneModelPromise) {
    droneModelPromise = loadDroneGLB().catch(error => {
      console.warn("Canvas DJI Mini 3 Pro GLB load failed.", error);
      return null;
    });
  }
  return droneModelPromise;
}

function quantile(values, q) {
  if (!values.length) return 0;
  const a = [...values].sort((x, y) => x - y);
  const idx = Math.min(a.length - 1, Math.max(0, Math.floor(q * (a.length - 1))));
  return a[idx];
}

function median(values) {
  return quantile(values, 0.5);
}

function pointCloudBounds(points, low = 0.01, high = 0.99) {
  if (!points.length) return null;
  const xs = points.map(p => p[0]);
  const ys = points.map(p => p[1]);
  const zs = points.map(p => p[2]);
  const min = [quantile(xs, low), quantile(ys, low), quantile(zs, low)];
  const max = [quantile(xs, high), quantile(ys, high), quantile(zs, high)];
  const span = max.map((v, i) => Math.max(v - min[i], 1e-6));
  const margin = [span[0] * 0.035, span[1] * 0.055, span[2] * 0.035];
  const b = {
    min: min.map((v, i) => v - margin[i]),
    max: max.map((v, i) => v + margin[i]),
  };
  b.center = b.min.map((v, i) => 0.5 * (v + b.max[i]));
  b.radius = Math.max(...b.max.map((v, i) => Math.abs(v - b.center[i])), 1e-6);
  return b;
}

function poseReferenceError(pose) {
  const center = pose?.center;
  const ref = pose?.colmap_reference?.center;
  if (!Array.isArray(center) || !Array.isArray(ref) || center.length < 3 || ref.length < 3) return null;
  return norm(sub(center, ref));
}

function poseTrackMaxStep(a, b) {
  const dt = Math.abs(Number(b?.time_sec) - Number(a?.time_sec));
  return Number.isFinite(dt) && dt > 0
    ? Math.min(3.2, Math.max(1.15, 1.65 * dt + 0.35))
    : 1.35;
}

function posesAreLocallyStable(a, b) {
  if (!a?.rcenter || !b?.rcenter) return false;
  return norm(sub(b.rcenter, a.rcenter)) <= poseTrackMaxStep(a, b);
}

function filterReplayPoseTrack(roomPoses) {
  const out = roomPoses.map(p => ({
    ...p,
    rawRcenter: p.rcenter ? p.rcenter.slice() : null,
  }));

  const valid = out
    .map((pose, index) => ({ pose, index }))
    .filter(item => item.pose.success && item.pose.rcenter);

  const dp = new Array(valid.length).fill(1);
  const prev = new Array(valid.length).fill(-1);
  for (let i = 0; i < valid.length; i++) {
    for (let j = 0; j < i; j++) {
      if (!posesAreLocallyStable(valid[j].pose, valid[i].pose)) continue;
      if (dp[j] + 1 > dp[i]) {
        dp[i] = dp[j] + 1;
        prev[i] = j;
      }
    }
  }

  const acceptedIndices = new Set();
  if (valid.length) {
    let best = 0;
    for (let i = 1; i < valid.length; i++) {
      if (dp[i] > dp[best]) best = i;
    }
    for (let at = best; at >= 0; at = prev[at]) {
      acceptedIndices.add(valid[at].index);
      if (prev[at] < 0) break;
    }
  }

  let accepted = 0;
  for (let i = 0; i < out.length; i++) {
    const pose = out[i];
    const keep = acceptedIndices.has(i);
    pose.filtered = !keep && Boolean(pose.success && pose.rawRcenter);
    pose.filter_reason = null;
    pose.trackSegment = 0;
    const refErr = poseReferenceError(pose);
    if (Number.isFinite(refErr)) pose.colmap_reference_error_m = refErr;
    if (keep) {
      accepted += 1;
    } else if (pose.success && pose.rcenter) {
      pose.rcenter = null;
    }
  }

  out.poseQuality = {
    total: out.length,
    accepted,
    rejected: out.filter(p => p.filtered).length,
  };
  return out;
}

function covariance(points, center) {
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const p of points) {
    const d = sub(p, center);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) C[r][c] += d[r] * d[c];
    }
  }
  const inv = 1 / Math.max(points.length, 1);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) C[r][c] *= inv;
  return C;
}

function powerEigen(C, seed) {
  let v = normalize(seed);
  for (let i = 0; i < 64; i++) v = normalize(matVec(C, v));
  const lambda = dot(v, matVec(C, v));
  return { v, lambda };
}

function deflate(C, eig) {
  const out = C.map(row => row.slice());
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) out[r][c] -= eig.lambda * eig.v[r] * eig.v[c];
  }
  return out;
}

function robustBounds3(points, low = 0.01, high = 0.99) {
  const xs = points.map(p => p[0]);
  const ys = points.map(p => p[1]);
  const zs = points.map(p => p[2]);
  const min = [quantile(xs, low), quantile(ys, low), quantile(zs, low)];
  const max = [quantile(xs, high), quantile(ys, high), quantile(zs, high)];
  return {
    min,
    max,
    center: min.map((v, i) => 0.5 * (v + max[i])),
    span: max.map((v, i) => Math.max(v - min[i], 1e-6)),
  };
}

function buildScanVisualPoints(targetBounds, floorY) {
  const raw = scan?.points || [];
  if (!raw.length) return [];

  const rawPoints = raw.map(p => [p[0], p[1], p[2]]);
  const stride = Math.max(1, Math.ceil(rawPoints.length / 9000));
  const sample = [];
  for (let i = 0; i < rawPoints.length; i += stride) sample.push(rawPoints[i]);

  const center = [0, 0, 0];
  for (const p of sample) for (let i = 0; i < 3; i++) center[i] += p[i];
  for (let i = 0; i < 3; i++) center[i] /= Math.max(sample.length, 1);

  const C = covariance(sample, center);
  const e0 = powerEigen(C, [1, 0.15, 0.05]);
  const e1 = powerEigen(deflate(C, e0), [0.1, 1, 0.2]);
  const axisX = normalize(e0.v);
  const axisZ = normalize(sub(e1.v, mul(axisX, dot(e1.v, axisX))));
  let axisY = normalize(cross(axisZ, axisX));

  const scanToPca = xyz => {
    const d = sub(xyz, center);
    return [dot(d, axisX), dot(d, axisY), dot(d, axisZ)];
  };
  const pcaPoints = rawPoints.map(scanToPca);
  let scanBounds = robustBounds3(pcaPoints, 0.01, 0.99);

  // Keep the scan's vertical direction consistent with the room frame.
  const rawLowY = quantile(pcaPoints.map(p => p[1]), 0.05);
  const rawHighY = quantile(pcaPoints.map(p => p[1]), 0.95);
  if (Math.abs(rawLowY) > Math.abs(rawHighY)) {
    axisY = mul(axisY, -1);
    for (const p of pcaPoints) p[1] *= -1;
    scanBounds = robustBounds3(pcaPoints, 0.01, 0.99);
  }

  const targetSpan = targetBounds.max.map((v, i) => Math.max(v - targetBounds.min[i], 1e-6));
  const horizontalScale = Math.min(
    targetSpan[0] / scanBounds.span[0],
    targetSpan[2] / scanBounds.span[2]
  );
  const scale = Math.max(0.01, horizontalScale * 0.96);
  const offset = [
    targetBounds.center[0] - scanBounds.center[0] * scale,
    floorY - scanBounds.min[1] * scale,
    targetBounds.center[2] - scanBounds.center[2] * scale,
  ];

  return raw.map((p, i) => {
    const q = pcaPoints[i];
    return {
      rxyz: [
        q[0] * scale + offset[0],
        q[1] * scale + offset[1],
        q[2] * scale + offset[2],
      ],
      rgb: [p[3], p[4], p[5]],
    };
  });
}

function buildRoomFrame() {
  const sparsePointRows = scene.points3D || [];
  const densePointRows = Array.isArray(scene.dense_points3D) && scene.dense_points3D.length
    ? scene.dense_points3D
    : null;
  const visualPointRows = densePointRows || sparsePointRows;
  const cloud = visualPointRows.map(p => p.xyz).filter(Boolean);
  const cameras = (scene.map_cameras || []).map(c => c.center).filter(Boolean);
  const sampleStride = Math.max(1, Math.ceil(cloud.length / 7000));
  const sample = [];
  for (let i = 0; i < cloud.length; i += sampleStride) sample.push(cloud[i]);
  // The room coordinate frame must be fixed for a map. During live TSolve
  // replay, poses arrive one by one; including the growing path in the PCA
  // frame or projection bounds makes the entire map appear to jump.
  sample.push(...cameras);

  const center = [0, 0, 0];
  for (const p of sample) for (let i = 0; i < 3; i++) center[i] += p[i];
  for (let i = 0; i < 3; i++) center[i] /= Math.max(sample.length, 1);

  const C = covariance(sample, center);
  const e0 = powerEigen(C, [1, 0.2, 0.1]);
  const e1 = powerEigen(deflate(C, e0), [0.1, 1, 0.2]);
  let axisX = normalize(e0.v);
  let axisZ = normalize(sub(e1.v, mul(axisX, dot(e1.v, axisX))));
  const zSign = Number(currentMapEntry?.display_z_sign ?? -1) < 0 ? -1 : 1;
  axisZ = mul(axisZ, zSign);
  let axisY = normalize(cross(axisZ, axisX));

  const rawTransform = xyz => {
    const d = sub(xyz, center);
    return [dot(d, axisX), dot(d, axisY), dot(d, axisZ)];
  };
  const pointY = cloud.slice(0, 5000).map(p => rawTransform(p)[1]);
  const camY = cameras.map(p => rawTransform(p)[1]);
  if (camY.length && median(camY) < median(pointY)) axisY = mul(axisY, -1);

  const transform = xyz => {
    const d = sub(xyz, center);
    return [dot(d, axisX), dot(d, axisY), dot(d, axisZ)];
  };
  const transformDirection = dir => [dot(dir, axisX), dot(dir, axisY), dot(dir, axisZ)];

  const transformedPoints = visualPointRows.map(p => ({ ...p, dense: Boolean(densePointRows), rxyz: transform(p.xyz) }));
  const xs = transformedPoints.map(p => p.rxyz[0]);
  const ys = transformedPoints.map(p => p.rxyz[1]);
  const zs = transformedPoints.map(p => p.rxyz[2]);
  const robust = {
    min: [quantile(xs, 0.01), quantile(ys, 0.02), quantile(zs, 0.01)],
    max: [quantile(xs, 0.99), quantile(ys, 0.98), quantile(zs, 0.99)],
  };
  const span = robust.max.map((v, i) => Math.max(v - robust.min[i], 1e-6));
  const margin = [span[0] * 0.16, span[1] * 0.22, span[2] * 0.16];
  const displayBounds = {
    min: robust.min.map((v, i) => v - margin[i]),
    max: robust.max.map((v, i) => v + margin[i]),
  };
  const displayPoints = transformedPoints.filter(p =>
    p.rxyz[0] >= displayBounds.min[0] && p.rxyz[0] <= displayBounds.max[0] &&
    p.rxyz[1] >= displayBounds.min[1] && p.rxyz[1] <= displayBounds.max[1] &&
    p.rxyz[2] >= displayBounds.min[2] && p.rxyz[2] <= displayBounds.max[2]
  );
  const structureBounds = pointCloudBounds(displayPoints.map(p => p.rxyz), 0.015, 0.985);
  const allRoom = [
    ...displayPoints.map(p => p.rxyz),
    ...cameras.map(transform),
  ];
  const b = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  for (const p of allRoom) {
    for (let i = 0; i < 3; i++) {
      b.min[i] = Math.min(b.min[i], p[i]);
      b.max[i] = Math.max(b.max[i], p[i]);
    }
  }
  b.center = b.min.map((v, i) => 0.5 * (v + b.max[i]));
  b.radius = Math.max(...b.max.map((v, i) => Math.abs(v - b.center[i])), 1e-6);
  const floorY = quantile(displayPoints.map(p => p.rxyz[1]), 0.05);
  const scanPoints = buildScanVisualPoints(b, floorY);
  const rawRoomPoses = poses.map(p => ({ ...p, rcenter: p.center ? transform(p.center) : null }));
  const roomPoses = buildReplayDisplayPoses(rawRoomPoses, floorY, { applyLanding: false });
  const poseQuality = roomPoses.poseQuality || {
    total: roomPoses.length,
    accepted: roomPoses.filter(p => p.success && p.rcenter).length,
    rejected: roomPoses.filter(p => p.filtered).length,
  };
  const routeYs = roomPoses.filter(p => p.success && p.rcenter).map(p => p.rcenter[1]);
  const routeHeightBounds = routeYs.length
    ? {
      min: Math.min(floorY, quantile(routeYs, 0.03)),
      max: Math.max(floorY + 0.18, quantile(routeYs, 0.97)),
    }
    : { min: floorY, max: Math.max(floorY + 0.18, (structureBounds || b).max[1]) };

  const rawRotationYaw = pose => {
    const R = pose.R;
    if (!Array.isArray(R) || R.length < 3 || !Array.isArray(R[2])) return null;
    // OpenCV-style pose matrices store camera axes in R. We use the optical
    // axis as the drone/camera forward direction, then calibrate the first
    // frame to the visible route so the initial shot stays intuitive.
    const f = transformDirection([R[2][0], R[2][1], R[2][2]]);
    if (Math.abs(f[0]) + Math.abs(f[2]) < 1e-9) return null;
    return Math.atan2(f[0], f[2]);
  };

  assignStablePathHeadings(roomPoses);

  const firstWithHeading = roomPoses.find(p => p.success && p.rcenter && p.pathHeading);
  let rotationYawOffset = 0;
  if (firstWithHeading) {
    const route = firstWithHeading.pathHeading;
    const routeYaw = Math.atan2(route[0], route[2]);
    const firstRawYaw = rawRotationYaw(firstWithHeading);
    if (Number.isFinite(firstRawYaw)) rotationYawOffset = routeYaw - firstRawYaw;
  }
  for (const pose of roomPoses) {
    const rawYaw = rawRotationYaw(pose);
    if (Number.isFinite(rawYaw)) {
      pose.rotationYaw = rawYaw + rotationYawOffset;
      pose.rotationHeading = headingFromYaw(pose.rotationYaw);
    }
  }

  return {
    origin: center,
    axes: { x: axisX, y: axisY, z: axisZ },
    transform,
    bounds: b,
    structureBounds,
    displayPoints,
    scanPoints,
    floorY,
    routeHeightBounds,
    mapCameras: (scene.map_cameras || []).map(c => ({ ...c, rcenter: transform(c.center) })),
    poses: roomPoses,
    poseQuality,
    visualPointSource: densePointRows ? "dense" : "sparse",
  };
}

function displayPointSummaryLine() {
  if (!room) return "";
  if (room.scanPoints?.length) return `${room.scanPoints.length} LiDAR scan samples<br>`;
  if (room.visualPointSource === "dense") return `${room.displayPoints.length} dense COLMAP display points<br>`;
  return `${room.displayPoints.length} COLMAP display points<br>`;
}

function mapSourceLine() {
  if (!room) return "COLMAP map used directly";
  if (room.scanPoints?.length) return "LiDAR scan aligned to TSolve frame";
  if (room.visualPointSource === "dense") return "Dense COLMAP visualization, sparse map used for localization";
  return "COLMAP map used directly";
}

function updateViewButtons() {
  for (const id of ["view-iso", "view-top", "view-side", "view-drone"]) document.getElementById(id)?.classList.remove("active");
  document.getElementById(`view-${view.mode}`)?.classList.add("active");
  if (viewIsoButton) viewIsoButton.textContent = `3D ${isoViewIndex + 1}/4`;
  const sideButton = document.getElementById("view-side");
  if (sideButton) sideButton.textContent = view.mode === "side" ? `Side ${sideViewIndex + 1}/8` : "Side";
  if (togglePointsButton) {
    togglePointsButton.classList.toggle("active", Boolean(view.showPoints));
    togglePointsButton.textContent = view.showPoints ? "Hide Points" : "Show Points";
  }
  if (toggleCamerasButton) {
    toggleCamerasButton.classList.toggle("active", Boolean(view.showCameras));
    toggleCamerasButton.textContent = view.showCameras ? "Hide Cameras" : "Show Cameras";
  }
}

function setView(mode, options = {}) {
  view.mode = mode;
  if (mode === "top") Object.assign(view, { yaw: 0.0, pitch: -Math.PI / 2, zoom: 1.12, panX: 0, panY: 0 });
  if (mode === "side") {
    if (options.advance) sideViewIndex = (sideViewIndex + 1) % sideViewPresets.length;
    const preset = sideViewPresets[sideViewIndex];
    Object.assign(view, { ...preset, panX: 0, panY: 0 });
  }
  if (mode === "iso") {
    if (options.advance) isoViewIndex = (isoViewIndex + 1) % isoViewPresets.length;
    const preset = isoViewPresets[isoViewIndex];
    Object.assign(view, { ...preset, panX: 0, panY: 0 });
  }
  updateViewButtons();
}

function centerViewOn(rxyz, targetX = 0.50, targetY = 0.55, smooth = false) {
  if (!room || !rxyz) return;
  const rect = canvas.getBoundingClientRect();
  const oldPanX = view.panX;
  const oldPanY = view.panY;
  view.panX = 0;
  view.panY = 0;
  const p = project(rxyz);
  const targetPanX = rect.width * targetX - p[0];
  const targetPanY = rect.height * targetY - p[1];
  const alpha = smooth ? 0.16 : 1;
  view.panX = oldPanX + (targetPanX - oldPanX) * alpha;
  view.panY = oldPanY + (targetPanY - oldPanY) * alpha;
  if (Math.abs(view.panX - oldPanX) > 0.15 || Math.abs(view.panY - oldPanY) > 0.15) {
    invalidateStaticLayer();
  }
}

function setDroneView() {
  view.mode = "drone";
  const preset = isoViewPresets[isoViewIndex] || isoViewPresets[0];
  Object.assign(view, {
    yaw: preset.yaw,
    pitch: preset.pitch,
    zoom: 6.2,
    panX: 0,
    panY: 0,
  });
  const cur = closestPose();
  if (cur?.rcenter) centerViewOn(cur.rcenter, 0.50, 0.56, false);
  updateViewButtons();
}

function rotate(p) {
  const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw);
  const cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
  const x1 = cy * p[0] + sy * p[2];
  const z1 = -sy * p[0] + cy * p[2];
  const y2 = cp * p[1] - sp * z1;
  const z2 = sp * p[1] + cp * z1;
  return [x1, y2, z2];
}

function project(rxyz) {
  const rect = canvas.getBoundingClientRect();
  const axisScale = view.axisScale || { x: 1, y: 1, z: 1 };
  const p = [
    (rxyz[0] - room.bounds.center[0]) * axisScale.x,
    (rxyz[1] - room.bounds.center[1]) * axisScale.y,
    (rxyz[2] - room.bounds.center[2]) * axisScale.z,
  ];
  const r = rotate(p);
  const scale = 0.46 * Math.min(rect.width, rect.height) * view.zoom / room.bounds.radius;
  return [
    rect.width * 0.5 + view.panX + r[0] * scale,
    rect.height * 0.52 + view.panY - r[1] * scale,
    r[2],
  ];
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpVec(a, b, t) {
  return a.map((v, i) => lerp(v, b[i], t));
}

function angleNear(target, reference) {
  let out = target;
  while (out - reference > Math.PI) out -= Math.PI * 2;
  while (out - reference < -Math.PI) out += Math.PI * 2;
  return out;
}

function lerpAngle(a, b, t) {
  return lerp(a, angleNear(b, a), t);
}

function headingFromYaw(yaw) {
  return [Math.sin(yaw), 0, Math.cos(yaw)];
}

function formatVector(values) {
  if (!Array.isArray(values)) return "-";
  return values.map(v => Number(v).toFixed(3)).join(", ");
}

function formatMatrix(rows) {
  if (!Array.isArray(rows) || !rows.length) return "-";
  return rows
    .map(row => Array.isArray(row) ? row.map(v => Number(v).toFixed(3)).join("  ") : "-")
    .join("\n");
}

function selectedMap() {
  const selectedId = mapLibraryData?.selected_map_id || "default_demo";
  return (mapLibraryData?.maps || []).find(m => m.id === selectedId) || mapLibraryData?.maps?.[0] || null;
}

function assetUrl(entry, file) {
  const base = entry?.asset_base || "public";
  return `${base.replace(/\/$/, "")}/${file}`;
}

function replayList(entry = currentMapEntry) {
  const replays = Array.isArray(entry?.replays) ? entry.replays : [];
  if (replays.length) return replays;
  const counts = entry?.counts || {};
  if (entry?.has_drone_demo || Number(counts.poses || 0) > 0) {
    return [{
      id: "base",
      title: "Built-in Drone Path",
      asset_base: entry?.asset_base || "public",
      built_in: true,
      counts: { poses: Number(counts.poses || 0) },
    }];
  }
  return [];
}

function activeReplay(entry = currentMapEntry) {
  const replays = replayList(entry);
  if (!replays.length) return null;
  return replays.find(replay => replay.id === entry?.active_replay_id) || replays[0];
}

function replayAssetUrl(replay, file) {
  const base = replay?.asset_base || currentMapEntry?.asset_base || "public";
  return `${base.replace(/\/$/, "")}/${file}`;
}

function replayQueryFrameBaseUrl(replay) {
  if (!replay) return "";
  if (replay.query_frame_base_url) return String(replay.query_frame_base_url).replace(/\/$/, "");
  const poseFrameSource = poseStreamMeta?.query_frame_base_url || poseStreamMeta?.frame_source;
  if (poseFrameSource && String(poseFrameSource).includes("/query_frames")) {
    const raw = String(poseFrameSource);
    const publicIdx = raw.indexOf("public/");
    if (publicIdx >= 0) return raw.slice(publicIdx).replace(/\/$/, "");
    if (!raw.startsWith("/") && !raw.includes(":")) return raw.replace(/\/$/, "");
  }
  const id = String(replay.id || "");
  if (id.startsWith("dji_live_")) return `public/live_dji_sessions/atlas_${id}/query_frames`;
  return "";
}

function poseFrameFileName(pose) {
  const raw = String(pose?.image_name || pose?.instance_id || "").trim();
  if (!raw) return "";
  const name = raw.split(/[\\/]/).pop();
  if (!name) return "";
  if (/\.(jpe?g|png|webp)$/i.test(name)) return name;
  if (/^query_\d+$/i.test(name)) return `${name}.jpg`;
  return "";
}

function replayFrameUrlForPose(replay, pose) {
  const base = replayQueryFrameBaseUrl(replay);
  const name = poseFrameFileName(pose);
  if (!base || !name) return "";
  return `${base}/${encodeURIComponent(name)}`;
}

function cacheBust(url) {
  const sep = url.includes("?") ? "&" : "?";
  const stamp = currentMapEntry?.updated_at || currentMapEntry?.created_at || "atlas";
  return `${url}${sep}v=${encodeURIComponent(stamp)}`;
}

function clearUploadedVideoPreview() {
  if (liveVideoObjectUrl) {
    URL.revokeObjectURL(liveVideoObjectUrl);
    liveVideoObjectUrl = null;
  }
}

function hasLiveVideoSource() {
  return Boolean(liveFrameMode || liveVideoObjectUrl || video.getAttribute("src"));
}

function setLiveFrameMode(enabled) {
  liveFrameMode = Boolean(enabled);
  liveFrameView?.classList.toggle("hidden", !liveFrameMode);
  liveFrameStatus?.classList.toggle("hidden", !liveFrameMode);
  video?.classList.toggle("hidden", liveFrameMode);
  if (!liveFrameMode) {
    if (liveFrameView) liveFrameView.removeAttribute("src");
    if (liveFrameStatus) liveFrameStatus.textContent = "";
    lastReplayFrameUrl = "";
    replayFramePlaybackEnabled = false;
  }
}

function setLiveFrameStatus(text, visible = true) {
  if (!liveFrameStatus) return;
  liveFrameStatus.textContent = text || "";
  liveFrameStatus.classList.toggle("hidden", !visible || !liveFrameMode);
}

function sortedTimedPoses(sourcePoses = room?.poses || []) {
  return sourcePoses
    .filter(p => p?.success && p.rcenter && Number.isFinite(Number(p.time_sec)))
    .sort((a, b) => Number(a.time_sec) - Number(b.time_sec));
}

function currentReplayClockTime(good) {
  if (pathPlaybackActive) {
    const timed = sortedTimedPoses(good);
    if (timed.length >= 2) {
      const first = Number(timed[0].time_sec);
      const last = Number(timed[timed.length - 1].time_sec);
      const elapsed = (performance.now() - pathPlaybackStartWallMs) / 1000;
      const t = Math.min(pathPlaybackStartTimeSec + elapsed, last);
      if (t >= last - 1e-3) {
        pathPlaybackActive = false;
        replayFramePlaybackEnabled = false;
      }
      return t;
    }
    pathPlaybackActive = false;
    replayFramePlaybackEnabled = false;
  }
  const t = Number(video.currentTime);
  return Number.isFinite(t) ? t : 0;
}

function startPoseClockPlayback() {
  const timed = sortedTimedPoses();
  if (timed.length < 2) {
    uploadStatus.textContent = "No timestamped TSolve path is available to play.";
    return false;
  }
  const first = Number(timed[0].time_sec);
  const last = Number(timed[timed.length - 1].time_sec);
  if (!(last > first)) return false;
  pathPlaybackActive = true;
  replayFramePlaybackEnabled = true;
  pathPlaybackStartWallMs = performance.now();
  pathPlaybackStartTimeSec = first;
  pathPlaybackEndTimeSec = last;
  lastReplayFrameUrl = "";
  video.pause();
  uploadStatus.textContent = "Playing saved TSolve path and its captured DJI frames.";
  return true;
}

function playCurrentReplay() {
  if (sidePanel) sidePanel.scrollTo({ top: 0, behavior: "smooth" });
  const src = video.getAttribute("src");
  const canUseVideoNow = src && !liveFrameMode && Number.isFinite(Number(video.duration)) && Number(video.duration) > 0.05;
  if (canUseVideoNow) {
    pathPlaybackActive = false;
    replayFramePlaybackEnabled = false;
    video.play().catch(() => startPoseClockPlayback());
    return;
  }
  if (src && !liveFrameMode && video.readyState < 1) {
    let resolved = false;
    const cleanup = () => {
      video.removeEventListener("loadedmetadata", onLoaded);
      video.removeEventListener("error", onError);
    };
    const onLoaded = () => {
      if (resolved) return;
      resolved = true;
      cleanup();
      const duration = Number(video.duration);
      if (Number.isFinite(duration) && duration > 0.05) {
        pathPlaybackActive = false;
        replayFramePlaybackEnabled = false;
        video.play().catch(() => startPoseClockPlayback());
      } else {
        startPoseClockPlayback();
      }
    };
    const onError = () => {
      if (resolved) return;
      resolved = true;
      cleanup();
      startPoseClockPlayback();
    };
    video.addEventListener("loadedmetadata", onLoaded, { once: true });
    video.addEventListener("error", onError, { once: true });
    video.load();
    setTimeout(() => {
      if (resolved) return;
      resolved = true;
      cleanup();
      startPoseClockPlayback();
    }, 650);
    return;
  }
  startPoseClockPlayback();
}

function setVideoFrameSteppingMode(enabled) {
  video.controls = !enabled;
  video.classList.toggle("frame-stepping", Boolean(enabled));
}

function latestPoseFrame(poses) {
  if (!Array.isArray(poses) || !poses.length) return null;
  for (let i = poses.length - 1; i >= 0; i -= 1) {
    if (poses[i]?.image_name) return poses[i];
  }
  return null;
}

function liveFrameUrlForPayload(payload, stream = null, options = {}) {
  const liveStream = stream || payload?.stream || poseStreamMeta?.stream || {};
  const sourcePayload = payload || poseStreamMeta || {};
  const processedPose = latestPoseFrame(sourcePayload?.poses);
  const frame = processedPose || sourcePayload?.current_frame || {};
  const base = liveStream.query_frame_base_url;
  const name = String(frame.image_name || "").split("/").pop();
  if (base && name) return `${base.replace(/\/$/, "")}/${encodeURIComponent(name)}?t=${Date.now()}`;
  if (options.allowRawPreview && liveStream.live_preview_url) return `${liveStream.live_preview_url}?t=${Date.now()}`;
  return "";
}

function updateLiveFrameView(payload = null, stream = null, options = {}) {
  const url = liveFrameUrlForPayload(payload, stream, options);
  if (!url || !liveFrameView) return false;
  setLiveFrameMode(true);
  liveFrameView.src = url;
  setLiveFrameStatus("", false);
  return true;
}

function updateReplayFrameViewForPose(pose, options = {}) {
  if (!replayFramePlaybackEnabled && !options.force) return false;
  if (liveReplayInFlight || pendingLiveReplayOpen) return false;
  const replay = activeReplay(currentMapEntry);
  const url = replayFrameUrlForPose(replay, pose);
  if (!url || !liveFrameView) return false;
  if (lastReplayFrameUrl === url) return true;
  setLiveFrameMode(true);
  setVideoFrameSteppingMode(true);
  video.pause();
  video.removeAttribute("src");
  liveFrameView.src = url;
  lastReplayFrameUrl = url;
  setLiveFrameStatus("", false);
  return true;
}

function ensureLiveStreamVideoSource(stream) {
  if (stream?.live_preview_url || stream?.query_frame_base_url) {
    setLiveFrameMode(true);
    if (!updateLiveFrameView(poseStreamMeta, stream)) {
      setLiveFrameStatus("Waiting for first TSolve-processed DJI frame...", true);
    }
    video.pause();
    video.removeAttribute("src");
    video.load();
    return;
  }
  if (liveVideoObjectUrl) return;
  const mediaUrl = stream?.media_url;
  if (!mediaUrl) return;
  setLiveFrameMode(false);
  video.muted = true;
  video.playsInline = true;
  if (video.getAttribute("src") !== mediaUrl) {
    video.src = mediaUrl;
    video.load();
  }
}

function showUploadedVideoPreview(file) {
  clearUploadedVideoPreview();
  setLiveFrameMode(false);
  liveVideoObjectUrl = URL.createObjectURL(file);
  setVideoFrameSteppingMode(true);
  video.muted = true;
  video.playsInline = true;
  video.src = liveVideoObjectUrl;
  video.load();
  video.currentTime = 0;
  video.pause();
  liveVideoWaitingForFirstPose = true;
  liveVideoSyncedToFirstPose = false;
  uploadStatus.textContent = "Drone video loaded. Waiting for the first TSolve R,t before showing the stream.";
}

function poseTimestampSeconds(pose) {
  const t = Number(pose?.time_sec);
  return Number.isFinite(t) ? Math.max(0, t) : 0;
}

function syncUploadedVideoToFirstPose(partialPoses) {
  if (!liveVideoWaitingForFirstPose || liveVideoSyncedToFirstPose) return;
  const first = (partialPoses || []).find(p => p && p.success !== false && p.center);
  if (!first) return;
  const startTime = poseTimestampSeconds(first);
  liveVideoSyncedToFirstPose = true;
  liveVideoWaitingForFirstPose = false;
  const seekAndHold = () => {
    try {
      const safeTime = Number.isFinite(video.duration)
        ? Math.min(startTime, Math.max(video.duration - 0.05, 0))
        : startTime;
      video.currentTime = safeTime;
    } catch {
      video.currentTime = 0;
    }
    video.pause();
  };
  const seek = () => seekAndHold();
  if (video.readyState >= 1) seek();
  else video.addEventListener("loadedmetadata", seek, { once: true });
  uploadStatus.textContent = `First TSolve R,t is ready. Showing processed frame ${startTime.toFixed(2)} s.`;
}

function latestSuccessfulPose(partialPoses) {
  const good = (partialPoses || []).filter(p => p && p.success !== false && p.rcenter);
  return good.length ? good[good.length - 1] : null;
}

function latestLivePoseForDisplay(partialPoses) {
  const good = (partialPoses || [])
    .filter(p => p && p.success !== false && (p.rcenter || p.rawRcenter));
  if (!good.length) return null;
  const latest = good[good.length - 1];
  if (latest.rcenter) return latest;
  return { ...latest, rcenter: latest.rawRcenter };
}

function syncUploadedVideoToLatestPose(partialPoses) {
  if (!liveReplayInFlight || !hasLiveVideoSource()) return;
  if (liveFrameMode) {
    updateLiveFrameView(poseStreamMeta);
    return;
  }
  setVideoFrameSteppingMode(true);
  const latest = latestSuccessfulPose(partialPoses);
  if (!latest) return;
  const latestTime = poseTimestampSeconds(latest);
  const drift = Math.abs(Number(video.currentTime || 0) - latestTime);
  if (!liveVideoSyncedToFirstPose || drift > 0.08) {
    try {
      const safeTime = Number.isFinite(video.duration)
        ? Math.min(latestTime, Math.max(video.duration - 0.05, 0))
        : latestTime;
      video.currentTime = safeTime;
    } catch {
      video.currentTime = latestTime;
    }
  }
  video.pause();
  uploadStatus.textContent = `Showing current localized frame ${latestTime.toFixed(2)} s.`;
}

function syncUploadedVideoToProcessingFrame(payload) {
  if (!liveReplayInFlight || !hasLiveVideoSource()) return;
  if (liveFrameMode) {
    updateLiveFrameView(payload);
    const displayedPose = latestPoseFrame(payload?.poses);
    const frame = displayedPose || payload?.current_frame || {};
    const frameIndex = Number(frame.frame_index);
    const frameTime = Number(frame.time_sec);
    const label = Number.isFinite(frameIndex)
      ? `frame ${frameIndex + 1}`
      : (Number.isFinite(frameTime) ? `frame at ${frameTime.toFixed(2)} s` : "DJI frame");
    const verb = displayedPose ? "localized" : "processing";
    uploadStatus.textContent = liveVideoSyncedToFirstPose
      ? `Showing TSolve-${verb} ${label}.`
      : `Waiting for first TSolve R,t. Showing ${verb} ${label}.`;
    return;
  }
  setVideoFrameSteppingMode(true);
  const frame = payload?.current_frame || {};
  const frameTime = Number(payload?.current_frame_time_sec ?? frame.time_sec);
  if (!Number.isFinite(frameTime)) return;
  const seekTime = Math.max(0, frameTime);
  const drift = Math.abs(Number(video.currentTime || 0) - seekTime);
  if (drift > 0.08) {
    try {
      const safeTime = Number.isFinite(video.duration)
        ? Math.min(seekTime, Math.max(video.duration - 0.05, 0))
        : seekTime;
      video.currentTime = safeTime;
    } catch {
      video.currentTime = seekTime;
    }
  }
  video.pause();
  const frameIndex = Number(frame.frame_index);
  const label = Number.isFinite(frameIndex)
    ? `frame ${frameIndex + 1}`
    : "current frame";
  uploadStatus.textContent = liveVideoSyncedToFirstPose
    ? `Processing ${label} at ${seekTime.toFixed(2)} s.`
    : `Waiting for first TSolve R,t. Processing ${label} at ${seekTime.toFixed(2)} s.`;
}

async function startDroneReplayUpload(file, mapId) {
  if (!file || !mapId) return;
  await selectMap(mapId, false);
  pendingLiveReplayOpen = true;
  pendingLiveReplayMapId = mapId;
  liveReplayInFlight = true;
  liveReplayMessage = "Creating a new live TSolve path from the uploaded drone stream";
  liveReplayStartedAt = performance.now();
  livePoseStreamKey = "";
  livePoseStreamCount = 0;
  liveCurrentPoseOverride = null;
  liveVideoWaitingForFirstPose = false;
  liveVideoSyncedToFirstPose = false;
  uploadStatus.textContent = `Uploading drone path for ${currentMapEntry?.title || mapId}: ${file.name}`;
  await loadViewerData(false, currentMapEntry);
  liveReplayWaitingViewPrepared = true;
  showDemo({ resetVideo: false });
  showUploadedVideoPreview(file);
  await uploadVideo("/api/drone/upload", file, { map_id: mapId });
  await pollStatus();
}

function selectedLiveAtlasFps() {
  const raw = Number(liveAtlasFps?.value || 2);
  if (!Number.isFinite(raw)) return 2;
  return Math.min(5, Math.max(0.5, raw));
}

function updateLiveControlSummary() {
  const fps = selectedLiveAtlasFps();
  if (liveControlSummary) liveControlSummary.textContent = `${fps} FPS`;
}

function takeoffHeightM() {
  const raw = Number(takeoffHeightInput?.value || 1);
  if (!Number.isFinite(raw)) return 1;
  return Math.min(2, Math.max(0.1, raw));
}

function setDjiCommandStatus(text, tone = "") {
  if (!djiCommandStatus) return;
  djiCommandStatus.textContent = text || "Drone control idle.";
  djiCommandStatus.dataset.tone = tone;
}

function liveLocalizationStarted() {
  return Boolean(liveReplayInFlight || liveAtlasPreviewActive || poseStreamMeta?.stream?.live_atlas);
}

function firstConfirmedPoseReady() {
  return Boolean(
    liveCurrentPoseOverride ||
    latestLivePoseForDisplay(room?.poses || poses || []) ||
    latestLivePoseForDisplay(poseStreamMeta?.poses || [])
  );
}

function updateFlightControlState() {
  const liveStarted = liveLocalizationStarted();
  const poseReady = firstConfirmedPoseReady();
  if (djiTakeoffButton) djiTakeoffButton.disabled = !liveStarted;
  if (djiLandButton) djiLandButton.disabled = !liveStarted;
  if (confirmLocalizationButton) {
    confirmLocalizationButton.disabled = !poseReady;
    confirmLocalizationButton.textContent = firstLocalizationConfirmed
      ? "Localization Confirmed"
      : "Confirm First Localization";
  }
  if (localizationGateStatus) {
    if (firstLocalizationConfirmed) {
      localizationGateStatus.textContent = "Confirmed. Mission controls are unlocked.";
    } else if (poseReady) {
      localizationGateStatus.textContent = "First R,t is visible. Confirm it matches the map before mission planning.";
    } else if (!liveStarted) {
      localizationGateStatus.textContent = "Start live localization before takeoff or mission planning.";
    } else {
      localizationGateStatus.textContent = "Waiting for first TSolve R,t.";
    }
  }
  if (!liveStarted && djiCommandStatus) {
    setDjiCommandStatus("Start live localization to unlock takeoff and land.", "");
  }
  droneControlPanel?.classList.toggle("is-locked", false);
  if (selectTargetButton) selectTargetButton.disabled = !room;
  if (clearTargetButton) clearTargetButton.disabled = !missionTarget?.rxyz;
  if (planMissionButton) planMissionButton.disabled = !missionTarget?.rxyz;
  if (startMissionButton) startMissionButton.disabled = !firstLocalizationConfirmed || !plannedMission;
}

function resetLocalizationGate(options = {}) {
  const preserveMission = Boolean(options.preserveMission);
  firstLocalizationConfirmed = false;
  if (!preserveMission) {
    plannedMission = null;
    missionTarget = null;
    missionSelecting = false;
    selectTargetButton?.classList.remove("active");
  }
  updateFlightControlState();
  updateMissionStatus();
}

async function sendDjiFlightCommand(command, fields = {}) {
  const phoneIp = (liveAtlasPhoneIp?.value || "").trim();
  const data = await postJson("/api/drone/flight-command", {
    command,
    phone_ip: phoneIp,
    ...fields,
  });
  const resultText = data.queued
    ? `${command} queued through live bridge.`
    : `${command} sent.`;
  setDjiCommandStatus(resultText, "ok");
  if (data.result?.note) setDjiCommandStatus(data.result.note, "ok");
  return data;
}

function asVec3(value) {
  if (!Array.isArray(value) || value.length < 3) return null;
  const out = value.slice(0, 3).map(Number);
  return out.every(Number.isFinite) ? out : null;
}

function legacyBarrierCorners(barrier) {
  const a = asVec3(barrier.a || barrier.start);
  const b = asVec3(barrier.b || barrier.end);
  if (!a || !b) return null;
  const floorY = room?.floorY ?? Math.min(a[1], b[1]);
  const height = Math.max(0.25, Math.min(8, Number(barrier.height_m || 1.8)));
  return [
    [a[0], floorY, a[2]],
    [b[0], floorY, b[2]],
    [b[0], floorY + height, b[2]],
    [a[0], floorY + height, a[2]],
  ];
}

function canonicalVerticalWallCorners(corners) {
  const raw = (corners || []).map(asVec3).filter(Boolean);
  if (raw.length < 2) return null;
  const ys = raw.map(p => p[1]).filter(Number.isFinite);
  const floorY = room?.floorY ?? Math.min(...ys, raw[0][1], raw[1][1]);
  const topY = Math.max(floorY + 0.25, Math.max(...ys));
  const a0 = raw[0];
  const a1 = raw[3] || raw[0];
  const b0 = raw[1];
  const b1 = raw[2] || raw[1];
  const a = [(a0[0] + a1[0]) * 0.5, floorY, (a0[2] + a1[2]) * 0.5];
  const b = [(b0[0] + b1[0]) * 0.5, floorY, (b0[2] + b1[2]) * 0.5];
  return [
    [a[0], floorY, a[2]],
    [b[0], floorY, b[2]],
    [b[0], topY, b[2]],
    [a[0], topY, a[2]],
  ];
}

function normalizedBarrierCorners(barrier) {
  const raw = Array.isArray(barrier?.corners) ? barrier.corners.map(asVec3).filter(Boolean) : [];
  if (raw.length >= 4) return canonicalVerticalWallCorners(raw.slice(0, 4));
  return legacyBarrierCorners(barrier);
}

function mapSafetyBarriers() {
  const barriers = (
    barrierUnsaved &&
    stagedSafetyBarrierMapId === currentMapEntry?.id &&
    Array.isArray(stagedSafetyBarriers)
  )
    ? stagedSafetyBarriers
    : (Array.isArray(currentMapEntry?.safety_barriers) ? currentMapEntry.safety_barriers : []);
  return barriers
    .map((barrier, index) => {
      const corners = normalizedBarrierCorners(barrier);
      if (!corners) return null;
      const ys = corners.map(p => p[1]);
      return {
        id: String(barrier.id || `barrier_${index}`),
        label: String(barrier.label || `Wall ${index + 1}`),
        a: corners[0],
        b: corners[1],
        corners,
        height_m: Math.max(0.25, Math.min(8, Math.max(...ys) - Math.min(...ys) || Number(barrier.height_m || 1.8))),
        clearance_m: Math.max(0.05, Math.min(5, Number(barrier.clearance_m || 0.45))),
      };
    })
    .filter(Boolean);
}

function selectedBarrierClearance() {
  const value = Number(barrierClearanceInput?.value || 0.45);
  return Math.max(0.15, Math.min(2, Number.isFinite(value) ? value : 0.45));
}

function pointSegmentDistance2D(p, a, b) {
  const vx = b[0] - a[0];
  const vz = b[2] - a[2];
  const wx = p[0] - a[0];
  const wz = p[2] - a[2];
  const len2 = vx * vx + vz * vz;
  if (len2 <= 1e-12) return Math.hypot(p[0] - a[0], p[2] - a[2]);
  const t = Math.max(0, Math.min(1, (wx * vx + wz * vz) / len2));
  return Math.hypot(p[0] - (a[0] + t * vx), p[2] - (a[2] + t * vz));
}

function orient2D(a, b, c) {
  return (b[0] - a[0]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[0] - a[0]);
}

function segmentsIntersect2D(a, b, c, d) {
  const eps = 1e-9;
  const o1 = orient2D(a, b, c);
  const o2 = orient2D(a, b, d);
  const o3 = orient2D(c, d, a);
  const o4 = orient2D(c, d, b);
  if (Math.abs(o1) <= eps && pointSegmentDistance2D(c, a, b) <= eps) return true;
  if (Math.abs(o2) <= eps && pointSegmentDistance2D(d, a, b) <= eps) return true;
  if (Math.abs(o3) <= eps && pointSegmentDistance2D(a, c, d) <= eps) return true;
  if (Math.abs(o4) <= eps && pointSegmentDistance2D(b, c, d) <= eps) return true;
  return (o1 > 0) !== (o2 > 0) && (o3 > 0) !== (o4 > 0);
}

function segmentDistance2D(a, b, c, d) {
  if (segmentsIntersect2D(a, b, c, d)) return 0;
  return Math.min(
    pointSegmentDistance2D(a, c, d),
    pointSegmentDistance2D(b, c, d),
    pointSegmentDistance2D(c, a, b),
    pointSegmentDistance2D(d, a, b),
  );
}

function projectedPolygonArea2D(points) {
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    area += a[0] * b[2] - b[0] * a[2];
  }
  return area * 0.5;
}

function pointInProjectedPolygon2D(point, polygon) {
  if (!polygon?.length || Math.abs(projectedPolygonArea2D(polygon)) < 1e-8) return false;
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const pi = polygon[i];
    const pj = polygon[j];
    const crosses = ((pi[2] > point[2]) !== (pj[2] > point[2])) &&
      (point[0] < (pj[0] - pi[0]) * (point[2] - pi[2]) / ((pj[2] - pi[2]) || 1e-12) + pi[0]);
    if (crosses) inside = !inside;
  }
  return inside;
}

function barrierRouteDistance2D(routeA, routeB, barrier) {
  const footprint = (barrier.corners || []).map(p => [p[0], 0, p[2]]);
  if (footprint.length < 2) return Infinity;
  if (pointInProjectedPolygon2D(routeA, footprint) || pointInProjectedPolygon2D(routeB, footprint)) return 0;
  let best = Infinity;
  for (let i = 0; i < footprint.length; i++) {
    const a = footprint[i];
    const b = footprint[(i + 1) % footprint.length];
    if (Math.hypot(b[0] - a[0], b[2] - a[2]) < 1e-8) continue;
    best = Math.min(best, segmentDistance2D(routeA, routeB, a, b));
  }
  return best;
}

function missionRouteSafetyCheck(segments) {
  if (!segments.length) return { blocked: false, reason: "No route segments yet.", nearest: null };
  let nearest = null;
  for (const barrier of mapSafetyBarriers()) {
    const dist = Math.min(...segments.map(([a, b]) => barrierRouteDistance2D(a, b, barrier)));
    const clearance = barrier.clearance_m;
    if (!nearest || dist < nearest.distance) nearest = { barrier, distance: dist, clearance };
    if (dist <= clearance) {
      return {
        blocked: true,
        nearest: { barrier, distance: dist, clearance },
        reason: `${barrier.label} is ${dist.toFixed(2)} map units from the route; required clearance is ${clearance.toFixed(2)}.`,
      };
    }
  }
  return { blocked: false, nearest };
}

function missionBarrierCheck(target = missionTarget?.rxyz) {
  const cur = closestPose();
  if (!cur?.rcenter || !target) return { blocked: false, reason: "No current pose yet.", nearest: null };
  return missionRouteSafetyCheck(missionRouteSegments(target, cur));
}

function updateBarrierStatus(message = null, tone = "") {
  if (!barrierStatus) return;
  barrierStatus.dataset.tone = tone;
  if (message) {
    barrierStatus.textContent = message;
    return;
  }
  if (barrierEditing) {
    barrierStatus.textContent = barrierDraft?.a
      ? "Pick the second endpoint for this wall."
      : "Pick the first endpoint on the visible COLMAP point cloud.";
    return;
  }
  if (barrierUnsaved) {
    barrierStatus.textContent = "Wall edits are staged. Press Save Walls to commit them to this map.";
    return;
  }
  const count = mapSafetyBarriers().length;
  barrierStatus.textContent = count
    ? `${count} manual safety wall${count === 1 ? "" : "s"} saved. Press Adjust Walls to reshape, move, or rotate them.`
    : "Add manual walls so mission targets stay away from obstacles.";
}

function updateBarrierAdjustControls() {
  if (adjustWallsButton) {
    adjustWallsButton.classList.toggle("active", barrierAdjusting);
    adjustWallsButton.textContent = barrierAdjusting ? "Adjusting Walls" : "Adjust Walls";
  }
  if (saveWallAdjustmentsButton) saveWallAdjustmentsButton.disabled = !barrierUnsaved || barrierSaving;
  if (cancelBarrierButton && !barrierEditing) cancelBarrierButton.disabled = !barrierUnsaved;
}

function markBarrierAdjustUnsaved(message = "Wall adjusted. Press Save Walls to commit.") {
  barrierUnsaved = true;
  stagedSafetyBarrierMapId = currentMapEntry?.id || null;
  updateBarrierAdjustControls();
  updateBarrierStatus(message, "busy");
}

function renderBarrierList() {
  if (!barrierList) return;
  updateBarrierAdjustControls();
  const barriers = mapSafetyBarriers();
  barrierList.innerHTML = "";
  if (!barriers.length) {
    updateBarrierStatus();
    return;
  }
  for (const barrier of barriers) {
    const item = document.createElement("div");
    item.className = "barrier-item";
    const text = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = barrier.label;
    const detail = document.createElement("span");
    detail.textContent = `4 adjustable corners, clearance ${barrier.clearance_m.toFixed(2)} map units`;
    text.append(name, detail);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Remove";
    remove.addEventListener("click", event => {
      event.stopPropagation();
      saveSafetyBarriers(barriers.filter(candidate => candidate.id !== barrier.id));
    });
    item.append(text, remove);
    barrierList.appendChild(item);
  }
  updateBarrierStatus();
}

async function saveSafetyBarriers(nextBarriers) {
  if (!currentMapEntry?.id || barrierSaving) return;
  const barriersForSave = normalizeSafetyBarrierBank(nextBarriers);
  barrierSaving = true;
  updateBarrierStatus("Saving safety barriers...", "busy");
  try {
    const data = await postJson("/api/map/barriers", {
      map_id: currentMapEntry.id,
      barriers: barriersForSave,
    });
    if (data.state?.library) mapLibraryData = data.state.library;
    currentMapEntry = selectedMap() || data.map || currentMapEntry;
    barrierDraft = null;
    barrierEditing = false;
    barrierAdjusting = false;
    barrierUnsaved = false;
    stagedSafetyBarrierMapId = null;
    stagedSafetyBarriers = null;
    addBarrierButton?.classList.remove("active");
    if (cancelBarrierButton) cancelBarrierButton.disabled = true;
    updateBarrierAdjustControls();
    renderBarrierList();
    invalidateStaticLayer();
    updateMissionStatus();
  } catch (err) {
    updateBarrierStatus(`Could not save safety barriers: ${err.message || err}`, "error");
  } finally {
    barrierSaving = false;
    updateBarrierAdjustControls();
  }
}

function addBarrierFromPickedPoint(picked) {
  if (!picked?.rxyz) return false;
  if (!barrierDraft?.a) {
    barrierDraft = { a: picked.rxyz.slice(0, 3) };
    updateBarrierStatus();
    return true;
  }
  const a = barrierDraft.a.slice(0, 3);
  const b = picked.rxyz.slice(0, 3);
  if (horizontalPathDistance(a, b) < 0.08) {
    updateBarrierStatus("Wall endpoints are too close. Pick a second point farther away.", "error");
    return false;
  }
  const barriers = mapSafetyBarriers();
  const height = Math.max(1.2, Math.min(3.5, (room?.bounds?.max?.[1] ?? 2) - (room?.floorY ?? 0)));
  const floorY = room?.floorY ?? Math.min(a[1], b[1]);
  const corners = [
    [a[0], floorY, a[2]],
    [b[0], floorY, b[2]],
    [b[0], floorY + height, b[2]],
    [a[0], floorY + height, a[2]],
  ];
  const next = barriers.concat({
    id: `barrier_${Date.now().toString(36)}`,
    label: `Wall ${barriers.length + 1}`,
    a: corners[0],
    b: corners[1],
    corners,
    height_m: height,
    clearance_m: selectedBarrierClearance(),
  });
  saveSafetyBarriers(next);
  return true;
}

function planMissionPreview() {
  if (!missionTarget?.rxyz) {
    updateMissionStatus("Pick an existing COLMAP point before planning.");
    return;
  }
  const speed = Number(missionSpeedSelect?.value || 0.4);
  const profile = missionLandingProfile(missionTarget.rxyz);
  const currentPoseReady = Boolean(closestPose()?.rcenter);
  let routePlan = null;
  let distance = missionDistanceFromCurrent();
  let safety = { blocked: false, nearest: null, reason: "Safety check pending until first live R,t." };
  if (currentPoseReady) {
    routePlan = planWallAwareRoute(missionTarget.rxyz, closestPose());
    safety = routePlan.safety || missionBarrierCheck(missionTarget.rxyz);
    distance = routePlan.distance;
    if (routePlan.blocked) {
      plannedMission = null;
      updateMissionStatus(`Mission blocked by safety wall. ${routePlan.reason || safety.reason}`);
      updateFlightControlState();
      return;
    }
  }
  plannedMission = {
    target: missionTarget.rxyz,
    approach: routePlan?.profile?.approach || profile?.approach || null,
    profile: routePlan?.profile?.mode || profile?.mode || "horizontal-approach-then-land",
    route: routePlan?.waypoints || null,
    route_segments: routePlan?.segments || null,
    detoured: Boolean(routePlan?.detoured),
    speed,
    distance,
    safety,
    pending_current_pose: !currentPoseReady,
    created_at: Date.now(),
  };
  const distText = distance == null ? "distance pending until first live R,t" : `${distance.toFixed(2)} map units`;
  const clearanceText = safety.nearest
    ? ` Nearest wall clearance: ${safety.nearest.distance.toFixed(2)} map units.`
    : "";
  const actionText = profile?.targetLooksGround ? "horizontal approach above the point, then land" : "horizontal approach, then descend";
  const detourText = routePlan?.detoured ? " with a safety-wall detour" : "";
  const gateText = firstLocalizationConfirmed ? "Confirm before any autonomous command." : "Start live localization and confirm first R,t before execution.";
  updateMissionStatus(`Preflight path saved${detourText}: ${actionText} at ${speed.toFixed(1)} m/s (${distText}).${clearanceText} ${gateText}`);
  updateFlightControlState();
}

async function startLiveAtlas() {
  const mapId = currentMapEntry?.id || mapLibraryData?.selected_map_id || "default_demo";
  const phoneIp = (liveAtlasPhoneIp?.value || "").trim();
  const fps = selectedLiveAtlasFps();
  if (!phoneIp) {
    uploadStatus.textContent = "Enter the Android phone IP before starting Live ATLAS.";
    return;
  }
  resetLocalizationGate({ preserveMission: true });
  await selectMap(mapId, false);
  pendingLiveReplayOpen = true;
  pendingLiveReplayMapId = mapId;
  liveReplayInFlight = true;
  liveReplayMessage = "Starting DJI live ATLAS self-localization";
  liveReplayStageDetail = "Connecting to Android MSDK stream";
  liveReplayStartedAt = performance.now();
  livePoseStreamKey = "";
  livePoseStreamCount = 0;
  liveCurrentPoseOverride = null;
  liveVideoWaitingForFirstPose = false;
  liveVideoSyncedToFirstPose = false;
  liveAtlasPreviewActive = true;
  clearUploadedVideoPreview();
  setLiveFrameMode(true);
  if (liveFrameView) liveFrameView.removeAttribute("src");
  setLiveFrameStatus("Connecting to Android MSDK stream. Waiting for first live DJI frame...", true);
  setDjiCommandStatus("Live localization started. Takeoff is now unlocked; confirm before sending any flight command.", "ok");
  updateFlightControlState();
  uploadStatus.textContent = `Starting Live ATLAS on ${currentMapEntry?.title || mapId}`;
  await loadViewerData(false, currentMapEntry);
  liveReplayWaitingViewPrepared = true;
  showDemo({ resetVideo: false });
  renderReplayTabs();
  try {
    await postJson("/api/drone/live-atlas", {
      map_id: mapId,
      phone_ip: phoneIp,
      fps,
      max_size: 1200,
    });
  } catch (error) {
    liveReplayInFlight = false;
    liveAtlasPreviewActive = false;
    pendingLiveReplayOpen = false;
    pendingLiveReplayMapId = null;
    setDjiCommandStatus("Live localization failed to start. Takeoff remains locked.", "error");
    updateFlightControlState();
    throw error;
  }
  await pollStatus();
}

async function stopLiveAtlas() {
  liveReplayStageDetail = "Stopping DJI live localization and saving current path...";
  uploadStatus.textContent = "Stopping Live ATLAS and saving the current path";
  renderReplayTabs();
  await postJson("/api/drone/stop", {});
  liveAtlasPreviewActive = false;
  firstLocalizationConfirmed = false;
  plannedMission = null;
  updateFlightControlState();
  await pollStatus();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatDateTime(value) {
  if (!value) return "not recorded";
  const text = String(value).replace("T", " ").replace(/\.\d+Z?$/, "");
  return text.length > 16 ? text.slice(0, 16) : text;
}

function cardDescription(entry) {
  const hasReplay = replayList(entry).length > 0;
  const dense = Number(entry?.counts?.dense_points || 0) > 0;
  if (hasReplay) return "Localization-ready COLMAP map with saved TSolve drone paths.";
  if (dense) return "COLMAP map with a dense viewer cloud for inspection.";
  return "COLMAP point-cloud map ready for drone localization.";
}

function collectMapVideoSources(entry) {
  const names = new Set();
  const addName = name => {
    const cleaned = String(name || "").trim();
    if (!cleaned) return;
    names.add(cleaned);
  };

  for (const key of ["source_video", "source_videos", "map_videos", "videos"]) {
    const value = entry?.[key];
    if (Array.isArray(value)) value.forEach(addName);
    else if (value) addName(value);
  }

  const desc = String(entry?.description || "");
  const matches = desc.match(/[\w.-]+\.(?:mov|mp4|m4v|avi|mkv)/gi) || [];
  matches.forEach(addName);

  if (!names.size) {
    if (entry?.id === "default_demo" || String(entry?.frames_path || "").includes("/data/map_frames")) {
      addName("Indoor Patrol Map frame bank");
    } else {
      addName("COLMAP map frame bank");
    }
  }
  return [...names];
}

function showVideoLibrary(mapId) {
  const entry = (mapLibraryData.maps || []).find(m => m.id === mapId);
  if (!entry || !videoLibraryModal || !videoLibraryList) return;
  const names = collectMapVideoSources(entry);
  if (videoLibraryTitle) videoLibraryTitle.textContent = "Video Map Library";
  if (videoLibrarySubtitle) {
    videoLibrarySubtitle.textContent = `${entry.title || "3D map"} · ${names.length} source item${names.length === 1 ? "" : "s"}`;
  }
  videoLibraryList.innerHTML = names.map(name => `
    <article class="video-source-tile">
      <div class="video-source-thumb" aria-hidden="true"></div>
      <div class="video-source-name">${escapeHtml(name)}</div>
    </article>
  `).join("");
  videoLibraryModal.classList.remove("hidden");
}

function hideVideoLibrary() {
  videoLibraryModal?.classList.add("hidden");
}

function renderMapLibrary() {
  if (!mapCardList) return;
  const maps = mapLibraryData?.maps || [];
  currentMapEntry = selectedMap();
  mapCardList.innerHTML = "";
  if (!maps.length) {
    mapCardList.innerHTML = `<article class="map-card"><div class="map-card-body"><h3>No maps yet</h3><p>Create a new 3D map from video or live camera.</p></div></article>`;
    return;
  }

  for (const entry of maps) {
    const counts = entry.counts || {};
    const replays = replayList(entry);
    const active = activeReplay(entry);
    const isSelected = entry.id === (currentMapEntry?.id || mapLibraryData.selected_map_id);
    const hasReplay = replays.length > 0;
    const replayLabel = hasReplay
      ? `${replays.length} path${replays.length === 1 ? "" : "s"}, ${active?.counts?.poses ?? counts.poses ?? 0} poses active`
      : "upload drone video";
    const sourceCount = collectMapVideoSources(entry).length;
    const card = document.createElement("article");
    card.className = `map-card${isSelected ? " selected" : ""}`;
    card.dataset.mapId = entry.id;
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-label", `Open ${entry.title || "3D map"}`);
    card.innerHTML = `
      <button type="button" class="card-delete-bubble" data-map-id="${entry.id}" title="Delete this 3D map" aria-label="Delete ${escapeHtml(entry.title || "3D map")}">×</button>
      <div class="map-card-body">
        <div class="map-title-row">
          <h3 class="map-title" data-map-id="${entry.id}" title="Double-click to rename">${escapeHtml(entry.title || "Untitled Map")}</h3>
        </div>
        <div class="map-preview">
          <canvas class="map-preview-canvas" data-preview-map-id="${entry.id}" aria-hidden="true"></canvas>
        </div>
        <p class="map-description">${escapeHtml(cardDescription(entry))}</p>
        <dl>
          <dt>Map points</dt><dd>${counts.points ?? 0}</dd>
          <dt>Cameras</dt><dd>${counts.cameras ?? 0}</dd>
          <dt>Replay poses</dt><dd>${replayLabel}</dd>
          <dt>Created</dt><dd>${escapeHtml(formatDateTime(entry.created_at))}</dd>
          <dt>Updated</dt><dd>${escapeHtml(formatDateTime(entry.updated_at))}</dd>
          <dt>Video lib</dt><dd>${sourceCount} item${sourceCount === 1 ? "" : "s"}</dd>
        </dl>
        <div class="map-card-tools" aria-label="map tools">
          <span class="icon-action map-status-action" title="${hasReplay ? "Live TSolve paths available" : "3D map only"}" aria-label="${hasReplay ? "Live TSolve paths available" : "3D map only"}">${hasReplay ? "⌁" : "□"}</span>
          <button type="button" class="icon-action video-lib-button" data-map-id="${entry.id}" title="Show source video map library" aria-label="Show source video map library">▦</button>
          <button type="button" class="icon-action duplicate-map" data-map-id="${entry.id}" title="Duplicate this 3D map without drone paths" aria-label="Duplicate map">⧉</button>
          <label class="icon-action enhance-map-action" title="Enhance this map with more mapping videos" aria-label="Enhance map">
            ✦
            <input class="enhance-map-upload" data-map-id="${entry.id}" type="file" accept="video/*" multiple />
          </label>
        </div>
      </div>
    `;
    card.addEventListener("click", event => {
      if (event.target?.closest?.("button,label,input,.map-title")) return;
      selectMap(entry.id, true);
    });
    card.addEventListener("keydown", event => {
      if (event.key !== "Enter" && event.key !== " ") return;
      if (event.target?.closest?.("button,label,input,.map-title")) return;
      event.preventDefault();
      runUi(() => selectMap(entry.id, true));
    });
    mapCardList.appendChild(card);
  }

  for (const title of mapCardList.querySelectorAll(".map-title")) {
    title.addEventListener("click", event => event.stopPropagation());
    title.addEventListener("dblclick", event => {
      event.stopPropagation();
      runUi(() => renameMap(title.dataset.mapId));
    });
  }
  for (const btn of mapCardList.querySelectorAll(".video-lib-button")) {
    btn.addEventListener("click", event => {
      event.stopPropagation();
      showVideoLibrary(btn.dataset.mapId);
    });
  }
  for (const btn of mapCardList.querySelectorAll(".duplicate-map")) {
    btn.addEventListener("click", event => {
      event.stopPropagation();
      runUi(() => duplicateMap(btn.dataset.mapId));
    });
  }
  for (const btn of mapCardList.querySelectorAll(".card-delete-bubble")) {
    btn.addEventListener("click", event => {
      event.stopPropagation();
      runUi(() => deleteMap(btn.dataset.mapId));
    });
  }
  for (const input of mapCardList.querySelectorAll(".enhance-map-upload")) {
    input.addEventListener("change", event => runUi(async () => {
      const files = [...(event.target.files || [])];
      const mapId = input.dataset.mapId;
      if (!files.length || !mapId) return;
      await selectMap(mapId, false);
      const names = files.map(file => file.name).join(", ");
      uploadStatus.textContent = `Enhancing ${currentMapEntry?.title || mapId} with ${files.length} mapping video${files.length === 1 ? "" : "s"}: ${names}`;
      await uploadVideos("/api/map/enhance", files, { map_id: mapId });
      input.value = "";
      await pollStatus();
    }));
  }
  for (const preview of mapCardList.querySelectorAll(".map-preview-canvas")) {
    preview.addEventListener("wheel", event => {
      event.preventDefault();
      event.stopPropagation();
      const mapId = preview.dataset.previewMapId;
      if (!mapId) return;
      const current = previewZoomByMap.get(mapId) || 1;
      const factor = Math.exp(-event.deltaY * 0.0014);
      previewZoomByMap.set(mapId, Math.max(0.55, Math.min(3.2, current * factor)));
      drawMapCardPreview(preview, (mapLibraryData.maps || []).find(m => m.id === mapId) || currentMapEntry);
    }, { passive: false });
  }
  renderStartPreview();
}

function renderReplayTabs() {
  if (!replayTabList) return;
  const replays = replayList(currentMapEntry);
  const active = activeReplay(currentMapEntry);
  replayTabs?.classList.toggle("has-replays", replays.length > 0);
  replayTabList.innerHTML = "";
  if (!replays.length) {
    if (liveReplayInFlight) {
      replayTabList.appendChild(createPendingReplayTab());
      return;
    }
    replayTabList.innerHTML = `<span class="replay-empty">No drone paths yet. Add a drone video to localize online.</span>`;
    return;
  }
  for (const replay of replays) {
    const item = document.createElement("div");
    item.className = `replay-item${replay.id === active?.id ? " active" : ""}`;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `replay-tab${replay.id === active?.id ? " active" : ""}`;
    btn.dataset.replayId = replay.id;
    const poseCount = replay.counts?.poses ?? 0;
    btn.innerHTML = `
      <span>${replay.title || "Drone Path"}</span>
      <small>${poseCount} pose${poseCount === 1 ? "" : "s"}</small>
    `;
    btn.addEventListener("click", () => runUi(() => selectReplay(replay.id)));
    const rename = document.createElement("button");
    rename.type = "button";
    rename.className = "replay-rename";
    rename.title = "Rename this drone video path";
    rename.textContent = "Rename";
    rename.addEventListener("click", event => {
      event.stopPropagation();
      runUi(() => renameReplay(replay.id));
    });
    const del = document.createElement("button");
    del.type = "button";
    del.className = "replay-delete";
    del.title = "Delete this drone video path";
    del.textContent = "Delete";
    del.addEventListener("click", event => {
      event.stopPropagation();
      runUi(() => deleteReplay(replay.id));
    });
    item.appendChild(btn);
    item.appendChild(rename);
    item.appendChild(del);
    replayTabList.appendChild(item);
  }
  if (liveReplayInFlight) {
    replayTabList.appendChild(createPendingReplayTab());
  }
}

function livePathCreationStage() {
  const msg = liveReplayStageDetail || liveReplayMessage || "TSolve online localization running";
  const processed = Number(poseStreamMeta?.processed_count ?? poseStreamMeta?.stream?.pose_count ?? livePoseStreamCount ?? 0);
  const expected = Number(poseStreamMeta?.expected_count ?? poseStreamMeta?.stream?.expected_count ?? 0);
  const mapId = pendingLiveReplayMapId || poseStreamMeta?.stream?.map_id;
  const map = (mapLibraryData?.maps || []).find(m => m.id === mapId);
  const prefix = map && currentMapEntry?.id !== map.id ? `${map.title || "selected map"}: ` : "";
  if (expected > 0) {
    return `${prefix}${msg} (${processed}/${expected} frames)`;
  }
  return `${prefix}${msg}`;
}

function createPendingReplayTab() {
  const item = document.createElement("div");
  item.className = "replay-item pending-item";

  const pending = document.createElement("button");
  pending.type = "button";
  pending.className = "replay-tab active pending";

  const title = document.createElement("span");
  title.textContent = "Building new live path";

  const stage = document.createElement("small");
  stage.className = "replay-stage";
  stage.textContent = livePathCreationStage();

  pending.appendChild(title);
  pending.appendChild(stage);

  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "replay-delete cancel-live-path";
  const isLiveAtlas = Boolean(poseStreamMeta?.stream?.live_atlas);
  cancel.textContent = isLiveAtlas ? "Stop" : "Cancel";
  cancel.title = isLiveAtlas ? "Stop live localization and save this path" : "Cancel current live path creation";
  cancel.addEventListener("click", event => {
    event.stopPropagation();
    runUi(isLiveAtlas ? stopLiveAtlas : cancelLivePathCreation);
  });

  item.appendChild(pending);
  item.appendChild(cancel);
  return item;
}

async function cancelLivePathCreation() {
  liveReplayStageDetail = "Cancelling live path creation...";
  uploadStatus.textContent = "Cancelling live TSolve path creation";
  renderReplayTabs();
  await postJson("/api/drone/stop", {});
  await pollStatus();
}

async function selectReplay(replayId) {
  if (!currentMapEntry?.id || !replayId) return;
  const data = await postJson("/api/replay/select", { map_id: currentMapEntry.id, replay_id: replayId });
  if (data.state?.library) mapLibraryData = data.state.library;
  else await refreshMapLibrary();
  currentMapEntry = selectedMap();
  renderMapLibrary();
  renderReplayTabs();
  await loadViewerData(false, currentMapEntry);
  video.currentTime = 0;
  video.pause();
}

async function deleteReplay(replayId) {
  if (!currentMapEntry?.id || !replayId) return;
  const replay = replayList(currentMapEntry).find(r => r.id === replayId);
  const title = replay?.title || "this drone path";
  if (!window.confirm(`Delete "${title}" from this 3D map?`)) return;
  const data = await postJson("/api/replay/delete", { map_id: currentMapEntry.id, replay_id: replayId });
  if (data.state?.library) mapLibraryData = data.state.library;
  else await refreshMapLibrary();
  currentMapEntry = selectedMap();
  renderMapLibrary();
  renderReplayTabs();
  await loadViewerData(false, currentMapEntry);
  uploadStatus.textContent = `Deleted drone path: ${title}`;
}

async function renameReplay(replayId) {
  if (!currentMapEntry?.id || !replayId) return;
  const replay = replayList(currentMapEntry).find(r => r.id === replayId);
  const current = replay?.title || "Drone Path";
  const title = window.prompt("Rename drone path", current);
  if (title == null) return;
  const cleaned = title.trim();
  if (!cleaned || cleaned === current) return;
  const data = await postJson("/api/replay/rename", {
    map_id: currentMapEntry.id,
    replay_id: replayId,
    title: cleaned,
  });
  if (data.state?.library) mapLibraryData = data.state.library;
  else await refreshMapLibrary();
  currentMapEntry = selectedMap();
  renderMapLibrary();
  renderReplayTabs();
  await loadViewerData(false, currentMapEntry);
  uploadStatus.textContent = `Renamed drone path: ${cleaned}`;
}

function drawMapCardPlaceholder(canvas, entry) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const pctx = canvas.getContext("2d");
  pctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  pctx.clearRect(0, 0, rect.width, rect.height);
  pctx.fillStyle = "rgba(4, 12, 11, 0.84)";
  pctx.fillRect(0, 0, rect.width, rect.height);
  pctx.strokeStyle = "rgba(105,218,255,0.14)";
  pctx.lineWidth = 1;
  for (let i = 0; i <= 7; i++) {
    const y = 24 + i * (rect.height - 48) / 7;
    pctx.beginPath();
    pctx.moveTo(26, y);
    pctx.lineTo(rect.width - 26, y);
    pctx.stroke();
  }
  for (let i = 0; i <= 9; i++) {
    const x = 26 + i * (rect.width - 52) / 9;
    pctx.beginPath();
    pctx.moveTo(x, 24);
    pctx.lineTo(x, rect.height - 24);
    pctx.stroke();
  }
  const count = Math.min(900, Math.max(90, Math.floor((entry.counts?.points || 1000) / 40)));
  let seed = [...String(entry.id || "map")].reduce((a, c) => a + c.charCodeAt(0), 0) || 17;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  for (let i = 0; i < count; i++) {
    const cluster = i % 3;
    const cx = [0.35, 0.58, 0.72][cluster] * rect.width;
    const cy = [0.58, 0.38, 0.66][cluster] * rect.height;
    const x = cx + (rand() - 0.5) * rect.width * [0.30, 0.22, 0.18][cluster];
    const y = cy + (rand() - 0.5) * rect.height * [0.22, 0.28, 0.18][cluster];
    pctx.fillStyle = i % 5 === 0 ? "rgba(155,217,196,0.72)" : "rgba(221,237,231,0.54)";
    pctx.fillRect(x, y, 1.3, 1.3);
  }
  pctx.fillStyle = "rgba(2, 12, 11, 0.72)";
  pctx.fillRect(10, 10, 112, 24);
  pctx.fillStyle = "#a9eaff";
  pctx.font = "bold 12px Inter, system-ui, sans-serif";
  pctx.fillText(entry.has_drone_demo ? "3D replay" : "3D map", 22, 27);
}

function sceneBounds(points) {
  const mins = [Infinity, Infinity, Infinity];
  const maxs = [-Infinity, -Infinity, -Infinity];
  for (const p of points) {
    const xyz = p.xyz || p;
    for (let i = 0; i < 3; i++) {
      mins[i] = Math.min(mins[i], xyz[i]);
      maxs[i] = Math.max(maxs[i], xyz[i]);
    }
  }
  const center = mins.map((v, i) => 0.5 * (v + maxs[i]));
  let radius = 1;
  for (const p of points) radius = Math.max(radius, norm(sub(p.xyz || p, center)));
  return { mins, maxs, center, radius };
}

function robustPreviewBounds(points) {
  const sample = [];
  const stride = Math.max(1, Math.ceil(points.length / 6000));
  for (let i = 0; i < points.length; i += stride) sample.push(points[i].xyz || points[i]);
  const quantile = (axis, q) => {
    const values = sample.map(p => p[axis]).filter(Number.isFinite).sort((a, b) => a - b);
    if (!values.length) return 0;
    return values[Math.max(0, Math.min(values.length - 1, Math.floor(q * (values.length - 1))))];
  };
  const mins = [0, 1, 2].map(i => quantile(i, 0.025));
  const maxs = [0, 1, 2].map(i => quantile(i, 0.975));
  for (let i = 0; i < 3; i++) {
    const pad = Math.max(1e-6, (maxs[i] - mins[i]) * 0.10);
    mins[i] -= pad;
    maxs[i] += pad;
  }
  const center = mins.map((v, i) => 0.5 * (v + maxs[i]));
  const extent = maxs.map((v, i) => Math.max(1e-6, v - mins[i]));
  const radius = Math.max(1, Math.max(...extent) * 0.56);
  return { mins, maxs, center, radius, extent };
}

function drawSceneMiniPreview(canvas, sceneData, entry) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const pctx = canvas.getContext("2d");
  pctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  pctx.clearRect(0, 0, rect.width, rect.height);

  const points = sceneData?.points3D || [];
  if (!points.length) {
    drawMapCardPlaceholder(canvas, entry);
    return;
  }

  const bounds = robustPreviewBounds(points);
  const yaw = -0.72;
  const pitch = 0.52;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const previewZoom = previewZoomByMap.get(entry.id) || 1;
  const scale = 0.58 * Math.min(rect.width, rect.height) * previewZoom / bounds.radius;
  const cameraDistance = bounds.radius * 4.8;
  const toPreview = xyz => {
    const x0 = xyz[0] - bounds.center[0];
    const y0 = xyz[1] - bounds.center[1];
    const z0 = xyz[2] - bounds.center[2];
    const x1 = cy * x0 + sy * z0;
    const z1 = -sy * x0 + cy * z0;
    const y2 = cp * y0 - sp * z1;
    const z2 = sp * y0 + cp * z1;
    const perspective = cameraDistance / Math.max(bounds.radius * 1.45, cameraDistance + z2);
    return [
      rect.width * 0.52 + x1 * scale * perspective,
      rect.height * 0.64 - y2 * scale * perspective,
      z2,
      perspective,
    ];
  };

  const bg = pctx.createLinearGradient(0, 0, rect.width, rect.height);
  bg.addColorStop(0, "rgba(9, 38, 69, 0.98)");
  bg.addColorStop(0.48, "rgba(3, 15, 31, 0.96)");
  bg.addColorStop(1, "rgba(1, 5, 14, 0.98)");
  pctx.fillStyle = bg;
  pctx.fillRect(0, 0, rect.width, rect.height);

  pctx.save();
  pctx.globalCompositeOperation = "screen";
  pctx.fillStyle = "rgba(50, 196, 255, 0.055)";
  pctx.beginPath();
  pctx.ellipse(rect.width * 0.52, rect.height * 0.60, rect.width * 0.36, rect.height * 0.24, -0.15, 0, Math.PI * 2);
  pctx.fill();
  pctx.restore();

  const floorY = bounds.mins[1];
  const gridRadius = bounds.radius * 1.02;
  const floorIso = (x, z) => {
    const x0 = (x - bounds.center[0]) / gridRadius;
    const z0 = (z - bounds.center[2]) / gridRadius;
    return [
      rect.width * 0.52 + (x0 - z0) * rect.width * 0.20,
      rect.height * 0.91 + (x0 + z0) * rect.height * 0.045,
    ];
  };
  pctx.strokeStyle = "rgba(89, 212, 255, 0.13)";
  pctx.lineWidth = 1;
  for (let i = -4; i <= 4; i++) {
    const off = i * gridRadius / 4;
    const a = floorIso(bounds.center[0] + off, bounds.center[2] - gridRadius);
    const b = floorIso(bounds.center[0] + off, bounds.center[2] + gridRadius);
    pctx.beginPath();
    pctx.moveTo(a[0], a[1]);
    pctx.lineTo(b[0], b[1]);
    pctx.stroke();
    const c = floorIso(bounds.center[0] - gridRadius, bounds.center[2] + off);
    const d = floorIso(bounds.center[0] + gridRadius, bounds.center[2] + off);
    pctx.beginPath();
    pctx.moveTo(c[0], c[1]);
    pctx.lineTo(d[0], d[1]);
    pctx.stroke();
  }

  const floorCorners = [
    floorIso(bounds.center[0] - gridRadius, bounds.center[2] - gridRadius),
    floorIso(bounds.center[0] + gridRadius, bounds.center[2] - gridRadius),
    floorIso(bounds.center[0] + gridRadius, bounds.center[2] + gridRadius),
    floorIso(bounds.center[0] - gridRadius, bounds.center[2] + gridRadius),
  ];
  pctx.strokeStyle = "rgba(119, 226, 255, 0.28)";
  pctx.lineWidth = 1.1;
  pctx.beginPath();
  floorCorners.forEach((p, idx) => {
    if (idx === 0) pctx.moveTo(p[0], p[1]);
    else pctx.lineTo(p[0], p[1]);
  });
  pctx.closePath();
  pctx.stroke();

  const projected = [];
  const stride = Math.max(1, Math.ceil(points.length / 2800));
  for (let i = 0; i < points.length; i += stride) {
    const pt = points[i];
    const [x, y, depth, perspective] = toPreview(pt.xyz || pt);
    if (x < -4 || y < -4 || x > rect.width + 4 || y > rect.height + 4) continue;
    projected.push({ x, y, depth, perspective, rgb: pt.rgb || [210, 235, 226] });
  }
  projected.sort((a, b) => b.depth - a.depth);
  for (const pt of projected) {
    const rgb = pt.rgb || [210, 235, 226];
    const heightTint = Math.max(0, Math.min(1, (pt.y - rect.height * 0.15) / (rect.height * 0.75)));
    const alpha = 0.30 + Math.max(0, Math.min(0.42, pt.perspective * 0.18 + (1 - heightTint) * 0.16));
    pctx.fillStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
    const size = Math.max(1, Math.min(2.2, 1.05 * pt.perspective + 0.65));
    pctx.fillRect(pt.x, pt.y, size, size);
  }

  pctx.fillStyle = "rgba(2, 12, 11, 0.72)";
  pctx.fillRect(10, 10, 104, 24);
  pctx.fillStyle = "#a9eaff";
  pctx.font = "bold 12px Inter, system-ui, sans-serif";
  pctx.fillText(entry.has_drone_demo ? "3D replay" : "3D map", 22, 27);
}

function drawMapCardPreview(canvas, entry) {
  const cached = previewSceneCache.get(entry.id);
  if (cached?.scene) {
    drawSceneMiniPreview(canvas, cached.scene, entry);
    return;
  }
  drawMapCardPlaceholder(canvas, entry);
  if (cached?.loading) return;
  const loading = fetch(assetUrl(entry, "scene.json"))
    .then(resp => resp.ok ? resp.json() : null)
    .then(sceneData => {
      previewSceneCache.set(entry.id, { scene: sceneData });
      for (const target of document.querySelectorAll(`.map-preview-canvas[data-preview-map-id="${entry.id}"]`)) {
        if (sceneData) drawSceneMiniPreview(target, sceneData, entry);
      }
    })
    .catch(() => {
      previewSceneCache.set(entry.id, { scene: null });
    });
  previewSceneCache.set(entry.id, { loading });
}

async function refreshMapLibrary() {
  try {
    const resp = await fetch("/api/maps", { cache: "no-store" });
    if (!resp.ok) throw new Error(`maps ${resp.status}`);
    mapLibraryData = await resp.json();
  } catch {
    if (!mapLibraryData.maps?.length) {
      mapLibraryData = {
        selected_map_id: "default_demo",
        maps: [{
          id: "default_demo",
          title: "Indoor Patrol Map",
          description: "Original TSolve drone replay map.",
          asset_base: "public",
          deletable: false,
          has_drone_demo: true,
          counts: { points: 0, cameras: 0, poses: 0 },
        }],
      };
    }
  }
  renderMapLibrary();
  return mapLibraryData;
}

function buildReplayDisplayPoses(roomPoses, floorY, options = {}) {
  const out = filterReplayPoseTrack(roomPoses);
  if (options.applyLanding !== true) return out;

  const goodIdx = out
    .map((p, i) => (p.success && p.rcenter ? i : -1))
    .filter(i => i >= 0);
  if (goodIdx.length < 3) return out;

  const rawY = goodIdx.map(i => out[i].rcenter[1]);
  const medianY = quantile(rawY, 0.5);
  const cruiseCeil = medianY + 0.36;
  const maxStep = 0.18;

  let y = rawY.map((value, i) => {
    const window = rawY.slice(Math.max(0, i - 1), Math.min(rawY.length, i + 2));
    return Math.min(median(window), cruiseCeil);
  });

  for (let i = 1; i < y.length; i++) y[i] = Math.min(y[i], y[i - 1] + maxStep);
  for (let i = y.length - 2; i >= 0; i--) y[i] = Math.min(y[i], y[i + 1] + maxStep);

  if (options.applyLanding === true) {
    const landingCount = Math.min(7, Math.max(3, Math.floor(y.length * 0.22)));
    const landingStart = Math.max(0, y.length - landingCount);
    const startY = y[Math.max(0, landingStart - 1)];
    const landY = floorY + 0.16;
    for (let i = landingStart; i < y.length; i++) {
      const u = (i - landingStart + 1) / landingCount;
      const eased = u * u * (3 - 2 * u);
      y[i] = lerp(startY, landY, eased);
    }
  }

  for (let k = 0; k < goodIdx.length; k++) {
    const i = goodIdx[k];
    out[i].rcenter = [out[i].rcenter[0], y[k], out[i].rcenter[2]];
    out[i].displayCorrected = true;
  }
  return out;
}

function horizontalPathDistance(a, b) {
  if (!a || !b) return 0;
  const dx = b[0] - a[0];
  const dz = b[2] - a[2];
  return Math.sqrt(dx * dx + dz * dz);
}

function stablePathHeadingAt(good, i) {
  const origin = good[i]?.rcenter;
  if (!origin) return null;

  const maxLook = 14;
  const minMove = 0.22;
  let best = null;
  let bestDist = 0;

  for (let j = i + 1; j < good.length && j <= i + maxLook; j++) {
    if (!canConnectPath(good[j - 1], good[j])) break;
    const d = horizontalPathDistance(origin, good[j].rcenter);
    if (d > bestDist) {
      bestDist = d;
      best = sub(good[j].rcenter, origin);
    }
    if (d >= minMove) return best;
  }

  for (let j = i - 1; j >= 0 && j >= i - maxLook; j--) {
    if (!canConnectPath(good[j], good[j + 1])) break;
    const d = horizontalPathDistance(good[j].rcenter, origin);
    if (d > bestDist) {
      bestDist = d;
      best = sub(origin, good[j].rcenter);
    }
    if (d >= minMove) return best;
  }

  return bestDist > 1e-5 ? best : null;
}

function assignStablePathHeadings(roomPoses) {
  const good = roomPoses.filter(p => p.success && p.rcenter);
  for (let i = 0; i < good.length; i++) {
    const heading = stablePathHeadingAt(good, i);
    if (heading && norm(heading) > 1e-8) good[i].pathHeading = heading;
  }
}

function closestPose() {
  if ((liveReplayInFlight || pendingLiveReplayOpen) && liveCurrentPoseOverride?.rcenter) {
    return liveCurrentPoseOverride;
  }
  const good = room?.poses?.filter(p => p.success && p.rcenter) || [];
  if (!good.length) return null;
  const t = currentReplayClockTime(good);
  const timed = sortedTimedPoses(good);
  if (timed.length >= 2) {
    if (t <= Number(timed[0].time_sec)) {
      return {
        ...timed[0],
        rheading: timed[0].pathHeading || timed[0].rotationHeading || sub(timed[1].rcenter, timed[0].rcenter),
      };
    }
    for (let i = 0; i + 1 < timed.length; i++) {
      const a = timed[i];
      const b = timed[i + 1];
      const ta = Number(a.time_sec);
      const tb = Number(b.time_sec);
      if (t >= ta && t <= tb) {
        const u = clamp01((t - ta) / Math.max(tb - ta, 1e-9));
        const nearest = u < 0.5 ? a : b;
        const rotationYaw =
          Number.isFinite(a.rotationYaw) && Number.isFinite(b.rotationYaw)
            ? lerpAngle(a.rotationYaw, b.rotationYaw, u)
            : null;
        return {
          ...nearest,
          instance_id: `${a.instance_id}->${b.instance_id}`,
          time_sec: t,
          center: lerpVec(a.center, b.center, u),
          rcenter: lerpVec(a.rcenter, b.rcenter, u),
          rotationYaw,
          rheading: Number.isFinite(rotationYaw)
            ? headingFromYaw(rotationYaw)
            : (nearest.rotationHeading || nearest.pathHeading || a.pathHeading || b.pathHeading || sub(b.rcenter, a.rcenter)),
        };
      }
    }
    const last = timed[timed.length - 1];
    const prev = timed[timed.length - 2];
    return { ...last, rheading: last.rotationHeading || last.pathHeading || sub(last.rcenter, prev.rcenter) };
  }
  let best = good[0], bestD = Infinity;
  for (const p of good) {
    const pt = Number(p.time_sec);
    const d = Number.isFinite(pt) ? Math.abs(pt - t) : Math.abs(good.indexOf(p) - t);
    if (d < bestD) {
      best = p; bestD = d;
    }
  }
  return best;
}

function canConnectPath(a, b) {
  if (!a?.rcenter || !b?.rcenter) return false;
  if (Number.isInteger(a.trackSegment) && Number.isInteger(b.trackSegment) && a.trackSegment !== b.trackSegment) {
    return false;
  }
  const step = norm(sub(b.rcenter, a.rcenter));
  const ta = Number(a.time_sec);
  const tb = Number(b.time_sec);
  if (Number.isFinite(ta) && Number.isFinite(tb)) {
    const dt = Math.abs(tb - ta);
    const maxStep = Math.min(5.0, Math.max(1.15, 1.65 * dt + 0.55));
    return step <= maxStep;
  }
  return step <= 1.8;
}

function drawPoint(rxyz, color, size = 2) {
  const [x, y] = project(rxyz);
  ctx.fillStyle = color;
  ctx.fillRect(x - size * 0.5, y - size * 0.5, size, size);
}

function drawCircle(rxyz, color, radius = 4) {
  const [x, y] = project(rxyz);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
}

function drawLine(a, b, color, width = 1, dash = []) {
  const pa = project(a), pb = project(b);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(pa[0], pa[1]);
  ctx.lineTo(pb[0], pb[1]);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawPolyline(points, color, width = 1, dash = []) {
  for (let i = 1; i < points.length; i++) {
    drawLine(points[i - 1], points[i], color, width, dash);
  }
}

function drawLabel(rxyz, text, color = "#dff7ff") {
  const [x, y] = project(rxyz);
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.fillStyle = "rgba(2, 8, 18, 0.78)";
  const w = ctx.measureText(text).width + 12;
  ctx.fillRect(x + 8, y - 18, w, 22);
  ctx.fillStyle = color;
  ctx.fillText(text, x + 14, y - 3);
}

function missionDistanceFromCurrent() {
  const segments = plannedMission?.route_segments?.length
    ? plannedMission.route_segments
    : missionRouteSegments(missionTarget?.rxyz);
  if (!segments.length) return null;
  return segments.reduce((sum, [a, b]) => sum + norm(sub(b, a)), 0);
}

function missionApproachPoint(target, cur = closestPose()) {
  if (!target) return null;
  const fallbackY = (room?.floorY ?? target[1]) + takeoffHeightM();
  const cruiseY = cur?.rcenter ? cur.rcenter[1] : fallbackY;
  return [target[0], cruiseY, target[2]];
}

function missionRouteSegments(target = missionTarget?.rxyz, cur = closestPose()) {
  if (!target || !cur?.rcenter) return [];
  const approach = missionApproachPoint(target, cur);
  if (!approach) return [];
  return routeSegmentsFromWaypoints([cur.rcenter, approach, target]);
}

function routeSegmentsFromWaypoints(points) {
  const segments = [];
  const clean = (points || []).filter(Boolean);
  for (let i = 1; i < clean.length; i++) {
    if (norm(sub(clean[i], clean[i - 1])) > 1e-6) segments.push([clean[i - 1], clean[i]]);
  }
  return segments;
}

function routeLengthFromSegments(segments) {
  return (segments || []).reduce((sum, [a, b]) => sum + norm(sub(b, a)), 0);
}

function sameRoutePoint(a, b, eps = 1e-4) {
  return Boolean(a && b && norm(sub(a, b)) <= eps);
}

function uniqueBarrierFootprintPoints(barrier) {
  const out = [];
  for (const p of barrier?.corners || []) {
    if (!out.some(q => Math.hypot(q[0] - p[0], q[2] - p[2]) < 1e-5)) {
      out.push([p[0], 0, p[2]]);
    }
  }
  if (out.length >= 2) return out;
  if (barrier?.a && barrier?.b) return [[barrier.a[0], 0, barrier.a[2]], [barrier.b[0], 0, barrier.b[2]]];
  return out;
}

function detourCandidatesForBarrier(barrier, routeY) {
  const points = uniqueBarrierFootprintPoints(barrier);
  if (points.length < 2) return [];
  const center = [
    points.reduce((sum, p) => sum + p[0], 0) / points.length,
    0,
    points.reduce((sum, p) => sum + p[2], 0) / points.length,
  ];
  const a = points[0];
  const b = points[1];
  const dx = b[0] - a[0];
  const dz = b[2] - a[2];
  const len = Math.hypot(dx, dz) || 1;
  const along = [dx / len, dz / len];
  const perp = [-along[1], along[0]];
  const margin = Math.max(0.5, Number(barrier.clearance_m || 0.45) + 0.65);
  const directions = [
    perp,
    [-perp[0], -perp[1]],
    along,
    [-along[0], -along[1]],
    [perp[0] + along[0], perp[1] + along[1]],
    [perp[0] - along[0], perp[1] - along[1]],
    [-perp[0] + along[0], -perp[1] + along[1]],
    [-perp[0] - along[0], -perp[1] - along[1]],
  ];
  const candidates = [];
  for (const point of points) {
    for (const dir of directions) {
      const dlen = Math.hypot(dir[0], dir[1]) || 1;
      const candidate = [
        point[0] + (dir[0] / dlen) * margin,
        routeY,
        point[2] + (dir[1] / dlen) * margin,
      ];
      if (Math.hypot(candidate[0] - center[0], candidate[2] - center[2]) > 0.1) candidates.push(candidate);
    }
  }
  return candidates;
}

function candidateRouteGraphNodes(routeY) {
  const nodes = [];
  for (const barrier of mapSafetyBarriers()) {
    for (const candidate of detourCandidatesForBarrier(barrier, routeY)) {
      if (!nodes.some(p => Math.hypot(p[0] - candidate[0], p[2] - candidate[2]) < 0.08)) nodes.push(candidate);
    }
  }
  return nodes;
}

function shortestSafeHorizontalRoute(start, goal, routeY) {
  const nodes = [
    [start[0], routeY, start[2]],
    [goal[0], routeY, goal[2]],
    ...candidateRouteGraphNodes(routeY),
  ];
  const n = nodes.length;
  const dist = new Array(n).fill(Infinity);
  const prev = new Array(n).fill(-1);
  const used = new Array(n).fill(false);
  dist[0] = 0;
  for (let iter = 0; iter < n; iter++) {
    let best = -1;
    for (let i = 0; i < n; i++) {
      if (!used[i] && (best < 0 || dist[i] < dist[best])) best = i;
    }
    if (best < 0 || !Number.isFinite(dist[best])) break;
    if (best === 1) break;
    used[best] = true;
    for (let j = 0; j < n; j++) {
      if (j === best || used[j]) continue;
      const safety = missionRouteSafetyCheck([[nodes[best], nodes[j]]]);
      if (safety.blocked) continue;
      const weight = norm(sub(nodes[best], nodes[j]));
      if (dist[best] + weight < dist[j]) {
        dist[j] = dist[best] + weight;
        prev[j] = best;
      }
    }
  }
  if (!Number.isFinite(dist[1])) return null;
  const path = [];
  for (let at = 1; at >= 0; at = prev[at]) {
    path.push(nodes[at]);
    if (at === 0) break;
  }
  return path.reverse();
}

function planWallAwareRoute(target = missionTarget?.rxyz, cur = closestPose()) {
  if (!target || !cur?.rcenter) {
    return { blocked: false, waypoints: [], segments: [], distance: null, safety: { blocked: false, nearest: null } };
  }
  const profile = missionLandingProfile(target, cur);
  const approach = profile?.approach;
  if (!approach) return { blocked: false, waypoints: [], segments: [], distance: null, safety: { blocked: false, nearest: null } };
  const baseWaypoints = [cur.rcenter, approach, target];
  const baseSegments = routeSegmentsFromWaypoints(baseWaypoints);
  const baseSafety = missionRouteSafetyCheck(baseSegments);
  if (!baseSafety.blocked || !mapSafetyBarriers().length) {
    return {
      blocked: baseSafety.blocked,
      detoured: false,
      waypoints: baseWaypoints,
      segments: baseSegments,
      distance: routeLengthFromSegments(baseSegments),
      safety: baseSafety,
      profile,
    };
  }

  const horizontal = shortestSafeHorizontalRoute(cur.rcenter, approach, approach[1]);
  if (horizontal?.length) {
    const waypoints = sameRoutePoint(horizontal[horizontal.length - 1], target)
      ? horizontal
      : horizontal.concat([target]);
    const segments = routeSegmentsFromWaypoints(waypoints);
    const safety = missionRouteSafetyCheck(segments);
    if (!safety.blocked) {
      return {
        blocked: false,
        detoured: true,
        waypoints,
        segments,
        distance: routeLengthFromSegments(segments),
        safety,
        profile,
      };
    }
  }

  return {
    blocked: true,
    detoured: false,
    waypoints: baseWaypoints,
    segments: baseSegments,
    distance: routeLengthFromSegments(baseSegments),
    safety: baseSafety,
    profile,
    reason: baseSafety.reason || "No clear detour was found around the saved safety walls.",
  };
}

function missionLandingProfile(target = missionTarget?.rxyz, cur = closestPose()) {
  if (!target) return null;
  const approach = missionApproachPoint(target, cur);
  const floorY = room?.floorY;
  const targetLooksGround = Number.isFinite(floorY)
    ? Math.abs(target[1] - floorY) <= Math.max(0.25, takeoffHeightM() * 0.35)
    : true;
  return {
    target,
    approach,
    targetLooksGround,
    mode: targetLooksGround ? "horizontal-approach-then-land" : "horizontal-approach-then-descend",
  };
}

function updateMissionStatus(message = null) {
  if (!targetStatus) return;
  if (message) {
    targetStatus.textContent = message;
    updateFlightControlState();
    return;
  }
  if (missionSelecting) {
    targetStatus.textContent = "Click a visible point in the 3D map to set the destination.";
    updateFlightControlState();
    return;
  }
  if (!missionTarget?.rxyz) {
    targetStatus.textContent = firstLocalizationConfirmed
      ? "No destination selected."
      : "No destination selected. You can pre-plan one now, before starting live localization.";
    updateFlightControlState();
    return;
  }
  const d = missionDistanceFromCurrent();
  const profile = missionLandingProfile();
  const suffix = d == null
    ? "Preflight target saved; path will anchor after first live R,t."
    : `Planned path: horizontal approach above target, then ${profile?.targetLooksGround ? "land" : "descend"} (${d.toFixed(2)} map units).`;
  const detoured = plannedMission?.detoured ? " Safety-wall detour active." : "";
  const planned = plannedMission ? `${detoured} Preview ready; confirm localization before autonomous execution.` : "";
  targetStatus.textContent = `Destination selected. ${suffix}${planned}`;
  updateFlightControlState();
}

function nearestVisibleMapPoint(clientX, clientY) {
  if (!room?.displayPoints?.length) return null;
  const rect = canvas.getBoundingClientRect();
  const sx = clientX - rect.left;
  const sy = clientY - rect.top;
  const points = room.displayPoints;
  const stride = Math.max(1, Math.ceil(points.length / 35000));
  let best = null;
  let bestD2 = Infinity;
  for (let i = 0; i < points.length; i += stride) {
    const p = points[i];
    const [x, y] = project(p.rxyz);
    const dx = x - sx;
    const dy = y - sy;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestD2) {
      bestD2 = d2;
      best = p;
    }
  }
  if (!best || bestD2 > 48 * 48) return null;
  return best;
}

function missionTargetHit(clientX, clientY) {
  if (!missionTarget?.rxyz) return false;
  const rect = canvas.getBoundingClientRect();
  const sx = clientX - rect.left;
  const sy = clientY - rect.top;
  const [x, y] = project(missionTarget.rxyz);
  const dx = x - sx;
  const dy = y - sy;
  return dx * dx + dy * dy <= 28 * 28;
}

function updateMissionTargetFromPointer(clientX, clientY) {
  if (!missionTarget?.rxyz) return false;
  const next = screenToViewEditPlane(clientX, clientY, missionTarget.rxyz);
  if (!next) return false;
  missionTarget = { rxyz: next, rgb: missionTarget.rgb || null };
  plannedMission = null;
  updateMissionStatus();
  return true;
}

function drawMissionTarget(cur = null) {
  if (!missionTarget?.rxyz) return;
  const target = missionTarget.rxyz;
  if (cur?.rcenter) {
    const usePlanned = plannedMission?.route?.length >= 2 && sameRoutePoint(plannedMission.target, target);
    const routePlan = usePlanned
      ? plannedMission
      : { waypoints: [cur.rcenter, missionApproachPoint(target, cur), target].filter(Boolean), safety: missionBarrierCheck(target) };
    const safety = routePlan?.safety || missionBarrierCheck(target);
    const routeColor = safety.blocked ? "rgba(255, 72, 110, 0.98)" : "rgba(255, 220, 95, 0.95)";
    const route = routePlan?.route || routePlan?.waypoints;
    if (route?.length >= 2) {
      drawPolyline(route, routeColor, safety.blocked ? 3.2 : 2.4, [8, 8]);
      for (let i = 1; i < route.length - 1; i++) drawRouteMarker(route[i], routePlan?.detoured ? 7 : 8);
      const approach = route[route.length - 2];
      drawRouteMarker(approach, 8);
      drawLabel(approach, routePlan?.detoured ? "detour" : "approach", "#ffe58c");
    }
  }
  const [x, y] = project(target);
  ctx.save();
  ctx.shadowColor = "rgba(255, 220, 95, 0.95)";
  ctx.shadowBlur = 14;
  ctx.strokeStyle = "rgba(255, 240, 160, 0.98)";
  ctx.fillStyle = "rgba(255, 196, 56, 0.35)";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.arc(x, y, 12, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x - 18, y);
  ctx.lineTo(x + 18, y);
  ctx.moveTo(x, y - 18);
  ctx.lineTo(x, y + 18);
  ctx.stroke();
  ctx.restore();
  drawLabel(target, "destination", "#ffe58c");
}

function barrierCenter(corners) {
  return [
    corners.reduce((sum, p) => sum + p[0], 0) / corners.length,
    corners.reduce((sum, p) => sum + p[1], 0) / corners.length,
    corners.reduce((sum, p) => sum + p[2], 0) / corners.length,
  ];
}

function barrierEditPlane(anchor = [0, 0, 0]) {
  if (view.mode === "side") {
    return { axisA: 0, axisB: 1, lockedAxis: 2, lockedValue: anchor[2], label: "X/Y" };
  }
  return { axisA: 0, axisB: 2, lockedAxis: 1, lockedValue: anchor[1], label: "X/Z" };
}

function axisPoint(base, axis, delta) {
  const next = base.slice(0, 3);
  next[axis] += delta;
  return next;
}

function screenToBarrierPlane(clientX, clientY, plane, anchor) {
  if (!room) return null;
  const rect = canvas.getBoundingClientRect();
  const sx = clientX - rect.left;
  const sy = clientY - rect.top;
  const base = anchor.slice(0, 3);
  base[plane.lockedAxis] = plane.lockedValue;
  const step = Math.max(0.01, (room.bounds?.radius || 1) * 0.04);
  const p0 = project(base);
  const pa = project(axisPoint(base, plane.axisA, step));
  const pb = project(axisPoint(base, plane.axisB, step));
  const ax = pa[0] - p0[0];
  const ay = pa[1] - p0[1];
  const bx = pb[0] - p0[0];
  const by = pb[1] - p0[1];
  const det = ax * by - ay * bx;
  if (Math.abs(det) < 1e-8) return null;
  const dx = sx - p0[0];
  const dy = sy - p0[1];
  const ca = (dx * by - dy * bx) / det;
  const cb = (ax * dy - ay * dx) / det;
  const next = base.slice(0, 3);
  next[plane.axisA] += ca * step;
  next[plane.axisB] += cb * step;
  next[plane.lockedAxis] = plane.lockedValue;
  return next;
}

function screenToBasisPlane(clientX, clientY, anchor, basisA, basisB) {
  if (!room) return null;
  const rect = canvas.getBoundingClientRect();
  const sx = clientX - rect.left;
  const sy = clientY - rect.top;
  const base = anchor.slice(0, 3);
  const step = Math.max(0.01, (room.bounds?.radius || 1) * 0.04);
  const p0 = project(base);
  const pa = project([
    base[0] + basisA[0] * step,
    base[1] + basisA[1] * step,
    base[2] + basisA[2] * step,
  ]);
  const pb = project([
    base[0] + basisB[0] * step,
    base[1] + basisB[1] * step,
    base[2] + basisB[2] * step,
  ]);
  const ax = pa[0] - p0[0];
  const ay = pa[1] - p0[1];
  const bx = pb[0] - p0[0];
  const by = pb[1] - p0[1];
  const det = ax * by - ay * bx;
  if (Math.abs(det) < 1e-8) return null;
  const dx = sx - p0[0];
  const dy = sy - p0[1];
  const ca = (dx * by - dy * bx) / det;
  const cb = (ax * dy - ay * dx) / det;
  return [
    base[0] + (basisA[0] * ca + basisB[0] * cb) * step,
    base[1] + (basisA[1] * ca + basisB[1] * cb) * step,
    base[2] + (basisA[2] * ca + basisB[2] * cb) * step,
  ];
}

function screenToViewEditPlane(clientX, clientY, anchor) {
  const base = anchor?.slice?.(0, 3);
  if (!base) return null;
  if (view.mode === "side") {
    const cy = Math.cos(view.yaw);
    const sy = Math.sin(view.yaw);
    return screenToBasisPlane(clientX, clientY, base, [cy, 0, sy], [0, 1, 0]);
  }
  return screenToBasisPlane(clientX, clientY, base, [1, 0, 0], [0, 0, 1]);
}

function barrierRotateHandlePoint(corners, center = barrierCenter(corners)) {
  const plane = barrierEditPlane(center);
  const edges = [[0, 1], [1, 2], [2, 3], [3, 0]];
  let da = 1;
  let db = 0;
  let len = 1;
  for (const [aIdx, bIdx] of edges) {
    const a = corners[aIdx] || center;
    const b = corners[bIdx] || center;
    const ea = b[plane.axisA] - a[plane.axisA];
    const eb = b[plane.axisB] - a[plane.axisB];
    const edgeLen = Math.hypot(ea, eb);
    if (edgeLen > len) {
      da = ea;
      db = eb;
      len = edgeLen;
    }
  }
  const offset = Math.max(0.18, Math.min(1.2, (room?.bounds?.radius || 2) * 0.10));
  const handle = center.slice(0, 3);
  handle[plane.axisA] += (-db / len) * offset;
  handle[plane.axisB] += (da / len) * offset;
  handle[plane.lockedAxis] = plane.lockedValue;
  return handle;
}

function drawBarrierTransformHandles(barrier, corners, center) {
  if (!barrierAdjusting) return;
  const moveActive = barrierTransformDrag?.barrierId === barrier.id && barrierTransformDrag?.type === "move";
  const rotateActive = barrierTransformDrag?.barrierId === barrier.id && barrierTransformDrag?.type === "rotate";
  const moveHover = barrierTransformHover?.barrierId === barrier.id && barrierTransformHover?.type === "move";
  const rotateHover = barrierTransformHover?.barrierId === barrier.id && barrierTransformHover?.type === "rotate";
  const rotateHandle = barrierRotateHandlePoint(corners, center);
  const [cx, cy] = project(center);
  const [rx, ry] = project(rotateHandle);

  ctx.save();
  if (view.mode !== "side") {
    ctx.strokeStyle = rotateActive || rotateHover ? "rgba(235, 242, 250, 0.95)" : "rgba(165, 178, 188, 0.62)";
    ctx.lineWidth = rotateActive || rotateHover ? 2.2 : 1.4;
    ctx.setLineDash([4, 5]);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(rx, ry);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.shadowColor = moveActive || moveHover ? "rgba(235, 242, 250, 0.95)" : "rgba(160, 170, 180, 0.65)";
  ctx.shadowBlur = moveActive || moveHover ? 18 : 10;
  ctx.fillStyle = moveActive || moveHover ? "rgba(235, 242, 250, 0.95)" : "rgba(122, 134, 146, 0.88)";
  ctx.strokeStyle = "rgba(245, 248, 252, 0.96)";
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.arc(cx, cy, 7.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  if (view.mode !== "side") {
    ctx.shadowColor = rotateActive || rotateHover ? "rgba(235, 242, 250, 0.95)" : "rgba(160, 170, 180, 0.65)";
    ctx.shadowBlur = rotateActive || rotateHover ? 18 : 10;
    ctx.fillStyle = rotateActive || rotateHover ? "rgba(235, 242, 250, 0.95)" : "rgba(104, 118, 132, 0.88)";
    ctx.beginPath();
    ctx.moveTo(rx, ry - 8);
    ctx.lineTo(rx + 8, ry);
    ctx.lineTo(rx, ry + 8);
    ctx.lineTo(rx - 8, ry);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function drawBarrierPanelFill(corners) {
  const projected = corners.map(p => project(p));
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(projected[0][0], projected[0][1]);
  for (let i = 1; i < projected.length; i++) ctx.lineTo(projected[i][0], projected[i][1]);
  ctx.closePath();
  ctx.shadowColor = "rgba(220, 230, 240, 0.07)";
  ctx.shadowBlur = 8;
  ctx.fillStyle = "rgba(128, 134, 142, 0.12)";
  ctx.strokeStyle = "rgba(224, 230, 238, 0.40)";
  ctx.lineWidth = 1.7;
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawBarrierNameOnWall(barrier, center) {
  const [x, y] = project(center);
  ctx.save();
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "800 24px Inter, system-ui, sans-serif";
  ctx.fillStyle = "rgba(238, 242, 246, 0.32)";
  ctx.strokeStyle = "rgba(4, 8, 12, 0.34)";
  ctx.lineWidth = 4;
  ctx.strokeText(barrier.label || "WALL", x, y);
  ctx.fillText(barrier.label || "WALL", x, y);
  ctx.restore();
}

function drawSafetyBarriers() {
  const barriers = mapSafetyBarriers();
  if (!barriers.length && !barrierDraft?.a) return;

  for (const barrier of barriers) {
    const corners = barrier.corners || [];
    if (corners.length < 4) continue;
    const edges = [[0, 1], [1, 2], [2, 3], [3, 0]];
    drawBarrierPanelFill(corners);
    drawLine(corners[0], corners[1], "rgba(235, 240, 245, 0.55)", 2.5);
    for (const [aIdx, bIdx] of edges.slice(1)) {
      drawLine(corners[aIdx], corners[bIdx], "rgba(188, 198, 208, 0.44)", 1.6);
    }
    drawLine(corners[0], corners[2], "rgba(230, 236, 242, 0.12)", 0.8, [5, 7]);
    drawLine(corners[1], corners[3], "rgba(230, 236, 242, 0.12)", 0.8, [5, 7]);

    const mid = barrierCenter(corners);
    drawBarrierNameOnWall(barrier, mid);
    drawBarrierTransformHandles(barrier, corners, mid);

    for (let i = 0; i < corners.length; i++) {
      const [x, y] = project(corners[i]);
      const active = barrierCornerDrag?.barrierId === barrier.id && barrierCornerDrag?.cornerIndex === i;
      const hover = barrierCornerHover?.barrierId === barrier.id && barrierCornerHover?.cornerIndex === i;
      if (!barrierAdjusting || (!active && !hover)) continue;
      ctx.save();
      ctx.shadowColor = active ? "rgba(255, 230, 120, 0.95)" : "rgba(235, 242, 250, 0.82)";
      ctx.shadowBlur = active ? 18 : 10;
      ctx.fillStyle = active ? "rgba(255, 230, 120, 0.92)" : "rgba(196, 206, 216, 0.9)";
      ctx.strokeStyle = "rgba(255, 244, 250, 0.96)";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.rect(x - 5.5, y - 5.5, 11, 11);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }
  }

  if (barrierDraft?.a) {
    const floorY = room?.floorY ?? barrierDraft.a[1];
    const a = [barrierDraft.a[0], floorY + 0.05, barrierDraft.a[2]];
    const [x, y] = project(a);
    ctx.save();
    ctx.shadowColor = "rgba(255, 100, 130, 0.85)";
    ctx.shadowBlur = 12;
    ctx.strokeStyle = "rgba(255, 158, 180, 0.95)";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
    drawLabel(a, "wall start", "#ffc3d1");
  }
}

function barrierCornerHit(clientX, clientY) {
  if (!barrierAdjusting) return null;
  const rect = canvas.getBoundingClientRect();
  const sx = clientX - rect.left;
  const sy = clientY - rect.top;
  const barriers = mapSafetyBarriers();
  for (let bi = barriers.length - 1; bi >= 0; bi--) {
    const barrier = barriers[bi];
    const corners = barrier.corners || [];
    for (let ci = corners.length - 1; ci >= 0; ci--) {
      const [x, y] = project(corners[ci]);
      const dx = x - sx;
      const dy = y - sy;
      if (dx * dx + dy * dy <= 17 * 17) {
        return { barrierId: barrier.id, cornerIndex: ci };
      }
    }
  }
  return null;
}

function barrierTransformHit(clientX, clientY) {
  if (!barrierAdjusting) return null;
  const rect = canvas.getBoundingClientRect();
  const sx = clientX - rect.left;
  const sy = clientY - rect.top;
  const barriers = mapSafetyBarriers();
  for (let bi = barriers.length - 1; bi >= 0; bi--) {
    const barrier = barriers[bi];
    const corners = barrier.corners || [];
    if (corners.length < 4) continue;
    const center = barrierCenter(corners);
    const rotateHandle = barrierRotateHandlePoint(corners, center);
    if (view.mode !== "side") {
      const [rx, ry] = project(rotateHandle);
      const rdx = rx - sx;
      const rdy = ry - sy;
      if (rdx * rdx + rdy * rdy <= 20 * 20) {
        return { type: "rotate", barrierId: barrier.id };
      }
    }
    const [cx, cy] = project(center);
    const cdx = cx - sx;
    const cdy = cy - sy;
    if (cdx * cdx + cdy * cdy <= 22 * 22) {
      return { type: "move", barrierId: barrier.id };
    }
  }
  return null;
}

function hitKey(hit) {
  if (!hit) return "";
  return `${hit.type || "corner"}:${hit.barrierId}:${hit.cornerIndex ?? ""}`;
}

function updateBarrierHover(clientX, clientY) {
  if (!barrierAdjusting) {
    clearBarrierHover();
    return;
  }
  if (barrierCornerDrag || barrierTransformDrag || barrierEditing) return;
  const corner = barrierCornerHit(clientX, clientY);
  const transform = corner ? null : barrierTransformHit(clientX, clientY);
  const changed = hitKey(corner) !== hitKey(barrierCornerHover) || hitKey(transform) !== hitKey(barrierTransformHover);
  barrierCornerHover = corner;
  barrierTransformHover = transform;
  canvas.style.cursor = corner || transform ? "grab" : (missionSelecting ? "crosshair" : "");
  if (changed) markFastInteraction(120);
}

function clearBarrierHover() {
  barrierCornerHover = null;
  barrierTransformHover = null;
  canvas.style.cursor = missionSelecting ? "crosshair" : "";
}

function barrierPayloadForSave(barrier) {
  const corners = canonicalVerticalWallCorners(barrier.corners || []) || (barrier.corners || []).map(p => p.slice(0, 3));
  if (corners.length < 4) return { ...barrier, corners: [] };
  const ys = corners.map(p => p[1]);
  return {
    id: barrier.id,
    label: barrier.label,
    a: corners[0],
    b: corners[1],
    corners,
    height_m: Math.max(0.25, Math.min(8, Math.max(...ys) - Math.min(...ys) || barrier.height_m || 1.8)),
    clearance_m: barrier.clearance_m,
  };
}

function replaceBarrierInCurrentMap(barrierId, updater) {
  const barriers = mapSafetyBarriers();
  const next = barriers.map(barrier => (
    barrier.id === barrierId ? barrierPayloadForSave(updater(barrier)) : barrierPayloadForSave(barrier)
  ));
  stagedSafetyBarrierMapId = currentMapEntry?.id || null;
  stagedSafetyBarriers = next;
  if (currentMapEntry) currentMapEntry.safety_barriers = next;
  const libEntry = (mapLibraryData.maps || []).find(m => m.id === currentMapEntry?.id);
  if (libEntry) libEntry.safety_barriers = next;
  plannedMission = null;
  invalidateStaticLayer();
  return next;
}

function barrierDragViewHint() {
  if (view.mode === "top") return "Top view edits X/Z only; wall height is locked.";
  if (view.mode === "side") return "Side view edits visible horizontal direction and height; hidden depth is locked.";
  return "3D view uses the current horizontal edit plane.";
}

function buildVerticalWallCorners(a, b, floorY, topY) {
  return [
    [a[0], floorY, a[2]],
    [b[0], floorY, b[2]],
    [b[0], topY, b[2]],
    [a[0], topY, a[2]],
  ];
}

function applyWallCornerEdit(corners, cornerIndex, pointer) {
  const current = canonicalVerticalWallCorners(corners);
  if (!current || !pointer) return corners;
  const floorY = current[0][1];
  let topY = current[2][1];
  const a = current[0].slice(0, 3);
  const b = current[1].slice(0, 3);
  const endpoint = cornerIndex === 0 || cornerIndex === 3 ? a : b;
  endpoint[0] = pointer[0];
  endpoint[2] = pointer[2];
  if (cornerIndex === 2 || cornerIndex === 3) topY = Math.max(floorY + 0.25, pointer[1]);
  return buildVerticalWallCorners(a, b, floorY, topY);
}

function snapWallCornersToNeighbors(barrierId, corners) {
  const current = canonicalVerticalWallCorners(corners);
  if (!current) return corners;
  const snapDistance = Math.max(0.10, Math.min(0.45, (room?.bounds?.radius || 2) * 0.045));
  const endpoints = [current[0].slice(0, 3), current[1].slice(0, 3)];
  let topY = current[2][1];
  const otherEndpoints = [];
  for (const barrier of mapSafetyBarriers()) {
    if (barrier.id === barrierId) continue;
    const other = canonicalVerticalWallCorners(barrier.corners || []);
    if (!other) continue;
    const otherTop = other[2][1];
    otherEndpoints.push({ point: other[0], topY: otherTop });
    otherEndpoints.push({ point: other[1], topY: otherTop });
  }
  for (const endpoint of endpoints) {
    let best = null;
    for (const other of otherEndpoints) {
      const d = Math.hypot(endpoint[0] - other.point[0], endpoint[2] - other.point[2]);
      if (d <= snapDistance && (!best || d < best.d)) best = { ...other, d };
    }
    if (best) {
      endpoint[0] = best.point[0];
      endpoint[2] = best.point[2];
      topY = Math.max(topY, best.topY);
    }
  }
  return buildVerticalWallCorners(endpoints[0], endpoints[1], current[0][1], topY);
}

function setWallEndpointAndHeight(corners, endpointIndex, endpoint, topY) {
  const current = canonicalVerticalWallCorners(corners);
  if (!current) return corners;
  const a = current[0].slice(0, 3);
  const b = current[1].slice(0, 3);
  const target = endpointIndex === 0 ? a : b;
  target[0] = endpoint[0];
  target[2] = endpoint[2];
  return buildVerticalWallCorners(a, b, current[0][1], Math.max(current[0][1] + 0.25, topY));
}

function normalizeSafetyBarrierBank(barriers) {
  const next = (barriers || [])
    .map(barrierPayloadForSave)
    .filter(barrier => Array.isArray(barrier.corners) && barrier.corners.length >= 4);
  const snapDistance = Math.max(0.10, Math.min(0.45, (room?.bounds?.radius || 2) * 0.045));
  for (let pass = 0; pass < 3; pass++) {
    let changed = false;
    for (let i = 0; i < next.length; i++) {
      for (let j = i + 1; j < next.length; j++) {
        for (const ei of [0, 1]) {
          for (const ej of [0, 1]) {
            const ci = canonicalVerticalWallCorners(next[i].corners);
            const cj = canonicalVerticalWallCorners(next[j].corners);
            if (!ci || !cj) continue;
            const pi = ci[ei];
            const pj = cj[ej];
            const d = Math.hypot(pi[0] - pj[0], pi[2] - pj[2]);
            if (d > snapDistance) continue;
            const shared = [(pi[0] + pj[0]) * 0.5, 0, (pi[2] + pj[2]) * 0.5];
            const topY = Math.max(ci[2][1], cj[2][1]);
            next[i].corners = setWallEndpointAndHeight(ci, ei, shared, topY);
            next[j].corners = setWallEndpointAndHeight(cj, ej, shared, topY);
            next[i] = barrierPayloadForSave({ ...next[i], a: next[i].corners[0], b: next[i].corners[1] });
            next[j] = barrierPayloadForSave({ ...next[j], a: next[j].corners[0], b: next[j].corners[1] });
            changed = true;
          }
        }
      }
    }
    if (!changed) break;
  }
  return next.map(barrierPayloadForSave);
}

function startBarrierTransformDrag(hit, clientX, clientY) {
  const barrier = mapSafetyBarriers().find(candidate => candidate.id === hit.barrierId);
  const corners = (barrier?.corners || []).map(p => p.slice(0, 3));
  if (!barrier || corners.length < 4) return false;
  const center = barrierCenter(corners);
  if (hit.type === "move") {
    const pointer = screenToViewEditPlane(clientX, clientY, center);
    if (!pointer) return false;
    barrierTransformDrag = {
      type: hit.type,
      barrierId: hit.barrierId,
      startCorners: corners,
      center,
      startPointer: pointer,
      plane: { label: view.mode === "side" ? "side plane" : "X/Z" },
    };
    barrierDragMoved = false;
    return true;
  }
  const plane = barrierEditPlane(center);
  const pointer = screenToBarrierPlane(clientX, clientY, plane, center) || barrierRotateHandlePoint(corners, center);
  const startAngle = Math.atan2(pointer[plane.axisB] - center[plane.axisB], pointer[plane.axisA] - center[plane.axisA]);
  barrierTransformDrag = {
    type: hit.type,
    barrierId: hit.barrierId,
    startCorners: corners,
    center,
    plane,
    startPointer: pointer,
    startAngle,
  };
  barrierDragMoved = false;
  return true;
}

function updateBarrierCornerFromPointer(clientX, clientY) {
  if (!barrierCornerDrag) return false;
  replaceBarrierInCurrentMap(barrierCornerDrag.barrierId, barrier => {
    const corners = (barrier.corners || []).map(p => p.slice(0, 3));
    const currentCorner = corners[barrierCornerDrag.cornerIndex];
    const pointer = screenToViewEditPlane(clientX, clientY, currentCorner);
    if (!pointer) return barrier;
    const nextCorners = snapWallCornersToNeighbors(
      barrier.id,
      applyWallCornerEdit(corners, barrierCornerDrag.cornerIndex, pointer),
    );
    return { ...barrier, corners: nextCorners, a: nextCorners[0], b: nextCorners[1] };
  });
  barrierDragMoved = true;
  updateBarrierStatus(`Corner adjusted. ${barrierDragViewHint()} Release to save.`, "busy");
  updateMissionStatus();
  return true;
}

function updateBarrierTransformFromPointer(clientX, clientY) {
  if (!barrierTransformDrag) return false;
  const drag = barrierTransformDrag;
  let nextCorners = drag.startCorners.map(p => p.slice(0, 3));
  if (drag.type === "move") {
    const pointer = screenToViewEditPlane(clientX, clientY, drag.center);
    if (!pointer) return false;
    const delta = [
      pointer[0] - drag.startPointer[0],
      0,
      pointer[2] - drag.startPointer[2],
    ];
    nextCorners = nextCorners.map(corner => [corner[0] + delta[0], corner[1], corner[2] + delta[2]]);
  } else if (drag.type === "rotate") {
    const pointer = screenToBarrierPlane(clientX, clientY, drag.plane, drag.center);
    if (!pointer) return false;
    const plane = drag.plane;
    const angle = Math.atan2(pointer[plane.axisB] - drag.center[plane.axisB], pointer[plane.axisA] - drag.center[plane.axisA]);
    const delta = angle - drag.startAngle;
    const cos = Math.cos(delta);
    const sin = Math.sin(delta);
    nextCorners = nextCorners.map(corner => {
      const next = corner.slice(0, 3);
      const da = corner[plane.axisA] - drag.center[plane.axisA];
      const db = corner[plane.axisB] - drag.center[plane.axisB];
      next[plane.axisA] = drag.center[plane.axisA] + cos * da - sin * db;
      next[plane.axisB] = drag.center[plane.axisB] + sin * da + cos * db;
      return next;
    });
  }
  const finalCorners = snapWallCornersToNeighbors(drag.barrierId, nextCorners);
  replaceBarrierInCurrentMap(drag.barrierId, barrier => ({ ...barrier, corners: finalCorners, a: finalCorners[0], b: finalCorners[1] }));
  barrierDragMoved = true;
  updateBarrierStatus(`${drag.type === "move" ? "Wall moved" : "Wall rotated"} on ${drag.plane.label}. Release to save.`, "busy");
  updateMissionStatus();
  return true;
}

function saveDraggedBarrierCorner() {
  if (!barrierCornerDrag) return;
  barrierCornerDrag = null;
  if (barrierDragMoved) {
    barrierDragMoved = false;
    barrierClickSuppress = true;
    markBarrierAdjustUnsaved("Corner staged. Press Save Walls before using this barrier for missions.");
  } else {
    barrierClickSuppress = true;
    updateBarrierStatus();
  }
}

function saveDraggedBarrierTransform() {
  if (!barrierTransformDrag) return;
  barrierTransformDrag = null;
  if (barrierDragMoved) {
    barrierDragMoved = false;
    barrierClickSuppress = true;
    markBarrierAdjustUnsaved("Wall transform staged. Press Save Walls before using this barrier for missions.");
  } else {
    barrierClickSuppress = true;
    updateBarrierStatus();
  }
}

function drawGrid() {
  const b = room.bounds;
  const floorY = room.floorY;
  const sx = b.max[0] - b.min[0];
  const sz = b.max[2] - b.min[2];
  const step = Math.max(0.25, Math.pow(10, Math.floor(Math.log10(Math.max(sx, sz) / 8))));
  const startX = Math.floor(b.min[0] / step) * step;
  const endX = Math.ceil(b.max[0] / step) * step;
  const startZ = Math.floor(b.min[2] / step) * step;
  const endZ = Math.ceil(b.max[2] / step) * step;

  for (let x = startX; x <= endX; x += step) {
    const major = Math.abs(Math.round(x / step) % 5) === 0;
    drawLine([x, floorY, startZ], [x, floorY, endZ], major ? "rgba(75,205,255,0.16)" : "rgba(75,205,255,0.07)", major ? 1.2 : 0.8);
  }
  for (let z = startZ; z <= endZ; z += step) {
    const major = Math.abs(Math.round(z / step) % 5) === 0;
    drawLine([startX, floorY, z], [endX, floorY, z], major ? "rgba(75,205,255,0.16)" : "rgba(75,205,255,0.07)", major ? 1.2 : 0.8);
  }

  drawLine([0, floorY, 0], [Math.min(endX, sx * 0.25), floorY, 0], "rgba(255,105,140,0.8)", 2);
  drawLine([0, floorY, 0], [0, floorY, Math.min(endZ, sz * 0.25)], "rgba(91,169,255,0.8)", 2);
  drawLine([0, floorY, 0], [0, floorY + Math.max(0.2, (b.max[1] - b.min[1]) * 0.2), 0], "rgba(105,218,255,0.82)", 2);
  drawLabel([Math.min(endX, sx * 0.25), floorY, 0], "room X", "#ff9db5");
  drawLabel([0, floorY, Math.min(endZ, sz * 0.25)], "room Z", "#95c7ff");
}

function drawRoomStructure() {
  const b = room?.structureBounds || room?.bounds;
  if (!b) return;
  const floorY = Math.max(room.floorY, b.min[1]);
  const ceilingY = Math.max(floorY + 0.24, b.max[1]);

  const x0 = b.min[0], x1 = b.max[0];
  const z0 = b.min[2], z1 = b.max[2];
  const bottom = [
    [x0, floorY, z0],
    [x1, floorY, z0],
    [x1, floorY, z1],
    [x0, floorY, z1],
  ];
  const top = bottom.map(p => [p[0], ceilingY, p[2]]);
  const floorLine = "rgba(65, 190, 255, 0.34)";
  const topLine = "rgba(198, 245, 255, 0.58)";
  const cornerLine = "rgba(120, 220, 255, 0.42)";

  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4;
    drawLine(bottom[i], bottom[j], floorLine, view.mode === "top" ? 1.6 : 1.15);
    drawLine(top[i], top[j], topLine, view.mode === "top" ? 1.9 : 1.35);
    if (view.mode !== "top") drawLine(bottom[i], top[i], cornerLine, 1.0);
  }
}

function ensureMapLineCells() {
  if (room?.lineCells) return room.lineCells;
  const source = room.scanPoints?.length ? room.scanPoints : room.displayPoints;
  const b = room.bounds;
  const sx = Math.max(b.max[0] - b.min[0], 1e-6);
  const sz = Math.max(b.max[2] - b.min[2], 1e-6);
  const cell = Math.max(Math.max(sx, sz) / 88, 0.045);
  const stride = Math.max(1, Math.ceil(source.length / 70000));
  const cells = new Map();

  for (let i = 0; i < source.length; i += stride) {
    const p = source[i];
    const ix = Math.floor((p.rxyz[0] - b.min[0]) / cell);
    const iz = Math.floor((p.rxyz[2] - b.min[2]) / cell);
    const key = `${ix},${iz}`;
    const rgb = p.rgb || [190, 225, 235];
    let c = cells.get(key);
    if (!c) {
      c = { ix, iz, count: 0, minY: Infinity, maxY: -Infinity, r: 0, g: 0, blue: 0 };
      cells.set(key, c);
    }
    c.count += 1;
    c.minY = Math.min(c.minY, p.rxyz[1]);
    c.maxY = Math.max(c.maxY, p.rxyz[1]);
    c.r += rgb[0];
    c.g += rgb[1];
    c.blue += rgb[2];
  }

  const minCount = room.scanPoints?.length ? 2 : 1;
  const out = [...cells.values()]
    .filter(c => c.count >= minCount)
    .sort((a, bCell) => a.ix - bCell.ix || a.iz - bCell.iz);
  room.lineCells = { cell, cells: out };
  return room.lineCells;
}

function drawMapLineModel() {
  const b = room?.bounds;
  if (!b) return;
  const { cell, cells } = ensureMapLineCells();
  const maxCells = view.mode === "drone" ? 2600 : 5600;
  const stride = Math.max(1, Math.ceil(cells.length / maxCells));
  const floorY = room.floorY;
  const topCap = b.max[1];

  for (let i = 0; i < cells.length; i += stride) {
    const c = cells[i];
    const x0 = b.min[0] + c.ix * cell;
    const z0 = b.min[2] + c.iz * cell;
    const x1 = x0 + cell;
    const z1 = z0 + cell;
    const cx = x0 + cell * 0.5;
    const cz = z0 + cell * 0.5;
    const topY = Math.min(topCap, Math.max(floorY + 0.035, c.maxY));
    const density = Math.min(1, Math.log2(c.count + 1) / 5);
    const alpha = view.mode === "drone" ? 0.30 + density * 0.40 : 0.24 + density * 0.34;
    const line = `rgba(92, 219, 255, ${alpha.toFixed(3)})`;
    const bright = `rgba(218, 249, 255, ${(alpha + 0.10).toFixed(3)})`;

    if (topY - floorY > 0.16) {
      drawLine([cx, floorY, cz], [cx, topY, cz], line, 0.9);
      if (c.count > 3 || topY - floorY > 0.42) {
        drawLine([x0, topY, cz], [x1, topY, cz], bright, 0.85);
        drawLine([cx, topY, z0], [cx, topY, z1], bright, 0.85);
      }
    } else if (c.count > 2) {
      drawLine([x0, floorY, cz], [x1, floorY, cz], line, 0.75);
      drawLine([cx, floorY, z0], [cx, floorY, z1], line, 0.75);
    }
  }
}

function drawFootprint() {
  const stride = Math.max(1, Math.ceil(room.displayPoints.length / 8500));
  for (let i = 0; i < room.displayPoints.length; i += stride) {
    const p = room.displayPoints[i];
    const fp = [p.rxyz[0], room.floorY, p.rxyz[2]];
    drawPoint(fp, "rgba(47, 119, 164, 0.18)", view.mode === "top" ? 3.0 : 2.0);
  }
}

function drawMapCameras() {
  for (const cam of room.mapCameras) {
    drawCircle(cam.rcenter, "rgba(74,163,255,0.9)", view.mode === "top" ? 3.2 : 2.7);
  }
}

function drawRouteMarker(rxyz, radius = 6) {
  const [x, y] = project(rxyz);
  const color = routeColorForHeight(rxyz, 0.78);
  ctx.save();
  ctx.fillStyle = routeColorForHeight(rxyz, 0.22);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function liveAcquisitionPoint() {
  const good = room?.poses?.filter(p => p.success && p.rcenter) || [];
  if (good.length) return good[good.length - 1].rcenter;
  if (room?.mapCameras?.length) {
    const sorted = [...room.mapCameras].sort((a, b) => (a.name || "").localeCompare(b.name || ""));
    return sorted[Math.floor(sorted.length * 0.50)]?.rcenter || sorted[0]?.rcenter;
  }
  return room?.bounds?.center || null;
}

function drawLiveAcquisitionMarker() {
  const p = liveAcquisitionPoint();
  if (!p) return;
  const [x, y] = project(p);
  const t = performance.now() * 0.004;
  const pulse = 9 + Math.sin(t) * 3;
  ctx.save();
  ctx.shadowColor = "rgba(105,218,255,0.75)";
  ctx.shadowBlur = 18;
  ctx.strokeStyle = "rgba(105,218,255,0.90)";
  ctx.fillStyle = "rgba(105,218,255,0.18)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, pulse, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.setLineDash([4, 5]);
  ctx.beginPath();
  ctx.arc(x, y, pulse + 11, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(2,8,18,0.72)";
  ctx.fillRect(x + 13, y - 19, 136, 25);
  ctx.fillStyle = "#dff8ff";
  ctx.font = "700 12px Inter, system-ui, sans-serif";
  ctx.fillText("acquiring first R,t", x + 21, y - 3);
  ctx.restore();
}

function drawStepMarker(rxyz, index, total) {
  const [x, y] = project(rxyz);
  const latest = index === total - 1;
  const radius = latest ? (view.mode === "top" ? 5.2 : 4.5) : (view.mode === "top" ? 3.9 : 3.2);
  const fade = total <= 1 ? 1 : 0.48 + 0.42 * (index / Math.max(total - 1, 1));
  const color = routeColorForHeight(rxyz, fade);
  ctx.save();
  ctx.shadowColor = latest ? routeColorForHeight(rxyz, 0.82) : routeColorForHeight(rxyz, 0.38);
  ctx.shadowBlur = latest ? 12 : 5;
  ctx.fillStyle = latest
    ? "rgba(255,245,247,0.98)"
    : color;
  ctx.strokeStyle = latest
    ? routeColorForHeight(rxyz, 0.98)
    : "rgba(255,168,183,0.75)";
  ctx.lineWidth = latest ? 1.7 : 0.9;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function routeHeight01(rxyz) {
  const low = room?.routeHeightBounds?.min ?? room?.floorY ?? room?.bounds?.min?.[1] ?? 0;
  const high = room?.routeHeightBounds?.max ?? room?.structureBounds?.max?.[1] ?? room?.bounds?.max?.[1] ?? low + 1;
  return clamp01((rxyz[1] - low) / Math.max(high - low, 1e-6));
}

function routeColorForHeight(rxyz, alpha = 1) {
  const h = Math.pow(routeHeight01(rxyz), 0.58);
  const r = Math.round(lerp(28, 255, h));
  const g = Math.round(lerp(0, 228, h));
  const b = Math.round(lerp(8, 54, h));
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function drawPath() {
  const good = room.poses.filter(p => p.success && p.rcenter);
  for (let i = 1; i < good.length; i++) {
    if (canConnectPath(good[i - 1], good[i])) {
      const mid = lerpVec(good[i - 1].rcenter, good[i].rcenter, 0.5);
      drawLine(good[i - 1].rcenter, good[i].rcenter, routeColorForHeight(mid, 0.16), view.mode === "top" ? 16 : 12);
    }
  }
  for (let i = 1; i < good.length; i++) {
    if (canConnectPath(good[i - 1], good[i])) {
      const mid = lerpVec(good[i - 1].rcenter, good[i].rcenter, 0.5);
      drawLine(good[i - 1].rcenter, good[i].rcenter, routeColorForHeight(mid, 0.94), 3.0, [10, 8]);
    }
  }
  for (let i = 0; i < good.length; i++) {
    drawStepMarker(good[i].rcenter, i, good.length);
  }
  if (good.length) {
    const start = good[0].rcenter;
    const land = good[good.length - 1].rcenter;
    const replayStillStreaming = Boolean(liveReplayInFlight || pendingLiveReplayOpen || poseStreamMeta?.complete === false);
    drawLabel(start, "start", "#d8fff2");
    if (replayStillStreaming) {
      drawRouteMarker(land, view.mode === "top" ? 10 : 8);
      drawLabel(land, "live", "#d8fff2");
    } else {
      const landGround = [land[0], room.floorY, land[2]];
      if (view.mode !== "top") drawLine(land, landGround, "rgba(255,64,92,0.56)", 2.0, [4, 5]);
      drawRouteMarker(landGround, view.mode === "top" ? 10 : 8);
      drawLabel(landGround, "land", "#d8fff2");
    }
  }
}

function pointOnPolyline(points, u) {
  if (!points.length) return [0, 0, 0];
  if (points.length === 1) return points[0];
  const lengths = [];
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    const len = norm(sub(points[i], points[i - 1]));
    lengths.push(len);
    total += len;
  }
  let target = clamp01(u) * Math.max(total, 1e-9);
  for (let i = 1; i < points.length; i++) {
    const len = lengths[i - 1];
    if (target <= len || i === points.length - 1) {
      const t = len <= 1e-9 ? 0 : target / len;
      return [
        lerp(points[i - 1][0], points[i][0], t),
        lerp(points[i - 1][1], points[i][1], t),
        lerp(points[i - 1][2], points[i][2], t),
      ];
    }
    target -= len;
  }
  return points[points.length - 1];
}

function partialPolyline(points, u) {
  if (points.length < 2) return points.slice();
  const out = [points[0]];
  const lengths = [];
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    const len = norm(sub(points[i], points[i - 1]));
    lengths.push(len);
    total += len;
  }
  let target = clamp01(u) * Math.max(total, 1e-9);
  for (let i = 1; i < points.length; i++) {
    const len = lengths[i - 1];
    if (target >= len) {
      out.push(points[i]);
      target -= len;
      continue;
    }
    const t = len <= 1e-9 ? 0 : target / len;
    out.push([
      lerp(points[i - 1][0], points[i][0], t),
      lerp(points[i - 1][1], points[i][1], t),
      lerp(points[i - 1][2], points[i][2], t),
    ]);
    break;
  }
  return out;
}

function headingForPose(cur) {
  if (cur?.rheading && norm(cur.rheading) > 1e-8) return cur.rheading;
  if (cur?.rotationHeading && norm(cur.rotationHeading) > 1e-8) return cur.rotationHeading;
  if (cur?.pathHeading && norm(cur.pathHeading) > 1e-8) return cur.pathHeading;
  const good = room.poses.filter(p => p.success && p.rcenter);
  const idx = good.findIndex(p => p.instance_id === cur.instance_id);
  if (idx >= 0) {
    const prev = good[Math.max(0, idx - 1)]?.rcenter;
    const next = good[Math.min(good.length - 1, idx + 1)]?.rcenter;
    if (prev && next && norm(sub(next, prev)) > 1e-8) return sub(next, prev);
    if (idx < good.length - 1 && norm(sub(good[idx + 1].rcenter, cur.rcenter)) > 1e-8) return sub(good[idx + 1].rcenter, cur.rcenter);
    if (idx > 0 && norm(sub(cur.rcenter, good[idx - 1].rcenter)) > 1e-8) return sub(cur.rcenter, good[idx - 1].rcenter);
  }
  return [1, 0, 0];
}

// Fixed calibration between the fallback drone glyph and ATLAS room yaw.
// The live heading still comes from TSolve/path rotation; this trim only
// turns the drawn vehicle 90 degrees left around the room vertical axis.
const DRONE_VISUAL_YAW_OFFSET = Math.PI / 2;

function rotateHorizontalHeading(heading, yaw) {
  const h = [heading[0], 0, heading[2]];
  if (norm(h) < 1e-6) return [1, 0, 0];
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  return [
    c * h[0] + s * h[2],
    0,
    -s * h[0] + c * h[2],
  ];
}

function droneVisualHeading(heading) {
  return normalize(rotateHorizontalHeading(heading, DRONE_VISUAL_YAW_OFFSET));
}

function drawDroneIcon(rxyz, heading = [1, 0, 0]) {
  const [x, y] = project(rxyz);
  const visualHeading = droneVisualHeading(heading);
  // Fallback icon attitude must not depend on the current map orbit.  Use the
  // TSolve/world heading directly instead of projecting a second point through
  // the user-controlled view camera.
  const angle = Math.atan2(-visualHeading[2], visualHeading[0]);
  const s = view.mode === "top" ? 1.05 : 0.92;
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.scale(s, s);
  ctx.shadowColor = "rgba(255,79,123,0.55)";
  ctx.shadowBlur = 12;

  ctx.strokeStyle = "rgba(245,255,251,0.95)";
  ctx.lineWidth = 2.2;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(-13, -10);
  ctx.lineTo(0, 0);
  ctx.lineTo(13, -10);
  ctx.moveTo(-13, 10);
  ctx.lineTo(0, 0);
  ctx.lineTo(13, 10);
  ctx.stroke();

  ctx.shadowBlur = 0;
  ctx.fillStyle = "rgba(4,16,14,0.9)";
  ctx.strokeStyle = "rgba(245,255,251,0.96)";
  for (const [rx, ry] of [[-15, -12], [15, -12], [-15, 12], [15, 12]]) {
    ctx.beginPath();
    ctx.arc(rx, ry, 5.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  ctx.fillStyle = "rgba(245,255,251,0.98)";
  ctx.strokeStyle = "rgba(255,79,123,0.96)";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(12, 0);
  ctx.lineTo(-3, -6);
  ctx.lineTo(-7, 0);
  ctx.lineTo(-3, 6);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "rgba(2,12,11,0.95)";
  ctx.font = "bold 5px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("DJI", -1, 0);
  ctx.restore();
}

function droneBasis(heading) {
  const up = [0, 1, 0];
  const forward = droneVisualHeading(heading);
  const right = normalize(cross(forward, up));
  return { right, forward, up };
}

function droneModelPoint(local, center, basis, scale) {
  // The uploaded DJI model is widest along local X, longer front/back along
  // local Y, and thin vertically along local Z.
  return add(
    center,
    add(
      add(mul(basis.right, local[0] * scale), mul(basis.forward, local[1] * scale)),
      mul(basis.up, local[2] * scale)
    )
  );
}

function drawDroneModel(rxyz, heading = [1, 0, 0]) {
  if (!droneModel?.vertices?.length) {
    drawDroneIcon(rxyz, heading);
    return;
  }
  const basis = droneBasis(heading);
  const scale = Math.max(room.bounds.radius * 0.20, 0.25);
  const projected = new Map();
  const roomPoint = idx => droneModelPoint(droneModel.vertices[idx], rxyz, basis, scale);
  const localPoint = local => droneModelPoint(local, rxyz, basis, scale);
  const localScreen = local => project(localPoint(local));
  const getProjected = idx => {
    if (!projected.has(idx)) {
      const p = roomPoint(idx);
      projected.set(idx, project(p));
    }
    return projected.get(idx);
  };

  ctx.save();
  const center2 = project(rxyz);
  ctx.strokeStyle = "rgba(128, 230, 255, 0.18)";
  ctx.lineWidth = 1.1;
  ctx.shadowColor = "rgba(88, 214, 255, 0.26)";
  ctx.shadowBlur = 10;
  ctx.beginPath();
  ctx.arc(center2[0], center2[1], view.mode === "top" ? 20 : 16, 0, Math.PI * 2);
  ctx.stroke();
  ctx.shadowBlur = 0;

  const edgeSource = droneModel.edges?.length
    ? droneModel.edges
    : (droneModel.triangles || []).flatMap(([a, b, c]) => [[a, b], [b, c], [c, a]]);
  if (edgeSource.length) {
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.shadowColor = "rgba(98, 220, 255, 0.46)";
    ctx.shadowBlur = 7;

    ctx.strokeStyle = "rgba(5, 18, 24, 0.68)";
    ctx.lineWidth = view.mode === "top" ? 3.0 : 2.4;
    ctx.beginPath();
    for (const [a, b] of edgeSource) {
      const pa = getProjected(a), pb = getProjected(b);
      ctx.moveTo(pa[0], pa[1]);
      ctx.lineTo(pb[0], pb[1]);
    }
    ctx.stroke();

    ctx.shadowBlur = 0;
    ctx.strokeStyle = "rgba(235, 255, 255, 0.94)";
    ctx.lineWidth = view.mode === "top" ? 1.15 : 0.9;
    ctx.beginPath();
    for (const [a, b] of edgeSource) {
      const pa = getProjected(a), pb = getProjected(b);
      ctx.moveTo(pa[0], pa[1]);
      ctx.lineTo(pb[0], pb[1]);
    }
    ctx.stroke();
  }

  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  const armTips = [
    [-0.47, 0.32, 0.03],
    [0.47, 0.32, 0.03],
    [-0.47, -0.32, 0.03],
    [0.47, -0.32, 0.03],
  ];
  const armRoots = [
    [-0.12, 0.10, 0.03],
    [0.12, 0.10, 0.03],
    [-0.12, -0.10, 0.03],
    [0.12, -0.10, 0.03],
  ];
  ctx.shadowColor = "rgba(86, 223, 255, 0.68)";
  ctx.shadowBlur = 12;
  ctx.strokeStyle = "rgba(3, 15, 22, 0.88)";
  ctx.lineWidth = view.mode === "top" ? 8.5 : 6.4;
  ctx.beginPath();
  for (let i = 0; i < armTips.length; i++) {
    const a = localScreen(armRoots[i]);
    const b = localScreen(armTips[i]);
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
  }
  ctx.stroke();

  ctx.strokeStyle = "rgba(225, 252, 255, 0.96)";
  ctx.lineWidth = view.mode === "top" ? 3.8 : 2.9;
  ctx.beginPath();
  for (let i = 0; i < armTips.length; i++) {
    const a = localScreen(armRoots[i]);
    const b = localScreen(armTips[i]);
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
  }
  ctx.stroke();

  const body = [
    [-0.15, 0.20, 0.05],
    [0.15, 0.20, 0.05],
    [0.18, -0.10, 0.05],
    [0.06, -0.23, 0.05],
    [-0.06, -0.23, 0.05],
    [-0.18, -0.10, 0.05],
  ].map(localScreen);
  ctx.fillStyle = "rgba(216, 245, 247, 0.88)";
  ctx.strokeStyle = "rgba(255, 255, 255, 0.98)";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  body.forEach(([x, y], i) => {
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.shadowBlur = 0;

  ctx.fillStyle = "rgba(255,79,123,0.96)";
  const nose = droneModelPoint([0, 0.54, 0.02], rxyz, basis, scale);
  const n = project(nose);
  ctx.beginPath();
  ctx.arc(n[0], n[1], 4.2, 0, Math.PI * 2);
  ctx.fill();

  const rotorLocs = [
    [-0.46, 0.30, 0.02],
    [0.46, 0.30, 0.02],
    [-0.46, -0.30, 0.02],
    [0.46, -0.30, 0.02],
  ];
  for (const loc of rotorLocs) {
    const rp = project(droneModelPoint(loc, rxyz, basis, scale));
    ctx.fillStyle = "rgba(2, 12, 11, 0.88)";
    ctx.strokeStyle = "rgba(255, 255, 255, 0.96)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(rp[0], rp[1], view.mode === "top" ? 7.2 : 5.6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  ctx.restore();
}

function drawPoints() {
  const isScan = Boolean(room.scanPoints?.length);
  const sources = room.scanPoints?.length
    ? [room.scanPoints, room.displayPoints]
    : [room.displayPoints];
  const totalPoints = sources.reduce((sum, source) => sum + source.length, 0);
  const fast = isFastInteraction();
  const pointBudget = fast
    ? (view.mode === "drone" ? 4200 : 6200)
    : (view.mode === "drone"
      ? (isScan ? 18000 : 14000)
      : (isScan ? 30000 : 22000));
  const stride = Math.max(1, Math.ceil(totalPoints / pointBudget));
  const sorted = [];
  for (const source of sources) {
    for (let i = 0; i < source.length; i += stride) sorted.push(source[i]);
  }
  sorted.sort((a, b) => project(a.rxyz)[2] - project(b.rxyz)[2]);
  const yMin = room.bounds.min[1];
  const ySpan = Math.max(room.bounds.max[1] - room.bounds.min[1], 1e-6);
  for (let i = 0; i < sorted.length; i += stride) {
    const p = sorted[i];
    const rgb = p.rgb || [220, 230, 225];
    const h = Math.max(0, Math.min(1, (p.rxyz[1] - yMin) / ySpan));
    const alpha = isScan ? (view.mode === "top" ? 0.90 : 0.94) : (view.mode === "top" ? 0.78 : 0.86);
    const boost = isScan ? (0.88 + 0.25 * h) : (0.76 + 0.42 * h);
    drawPoint(
      p.rxyz,
      `rgba(${Math.min(255, rgb[0] * boost)},${Math.min(255, rgb[1] * boost)},${Math.min(255, rgb[2] * boost)},${alpha})`,
      isScan ? (view.mode === "top" ? 2.35 : 1.95) : (view.mode === "top" ? 2.45 : 2.05)
    );
  }
}

function buildStaticLayerKey() {
  const axis = view.axisScale || { x: 1, y: 1, z: 1 };
  return [
    currentMapEntry?.id || "map",
    room?.displayPoints?.length || 0,
    room?.scanPoints?.length || 0,
    room?.mapCameras?.length || 0,
    room?.poses?.length || 0,
    canvas.width,
    canvas.height,
    view.mode,
    view.yaw.toFixed(4),
    view.pitch.toFixed(4),
    view.zoom.toFixed(4),
    axis.x.toFixed(3),
    axis.y.toFixed(3),
    axis.z.toFixed(3),
    view.showPoints ? "points" : "frame",
    view.showCameras ? "cameras" : "no-cameras",
    isFastInteraction() ? "fast" : "full",
  ].join("|");
}

function drawStaticLayer(rect, dpr) {
  const key = buildStaticLayerKey();
  const panDx = view.panX - staticLayerPan.x;
  const panDy = view.panY - staticLayerPan.y;
  const panMovedTooFar = Math.abs(panDx) > 72 || Math.abs(panDy) > 72;
  if (key !== staticLayerKey || staticCanvas.width !== canvas.width || staticCanvas.height !== canvas.height || panMovedTooFar) {
    staticCanvas.width = canvas.width;
    staticCanvas.height = canvas.height;
    staticCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    staticCtx.clearRect(0, 0, rect.width, rect.height);
    staticLayerPan = { x: view.panX, y: view.panY };

    const liveCtx = ctx;
    ctx = staticCtx;
    try {
      drawGrid();
      drawRoomStructure();
      if (view.showPoints) {
        drawFootprint();
        drawPoints();
      }
      if (view.showCameras) drawMapCameras();
      drawPath();
    } finally {
      ctx = liveCtx;
    }
    staticLayerKey = key;
  }

  ctx.drawImage(staticCanvas, view.panX - staticLayerPan.x, view.panY - staticLayerPan.y, rect.width, rect.height);
}

function renderStartPreview() {
  for (const canvas of document.querySelectorAll(".map-preview-canvas")) {
    if (canvas.dataset.previewMapId !== currentMapEntry?.id) {
      const entry = (mapLibraryData.maps || []).find(m => m.id === canvas.dataset.previewMapId);
      if (entry) drawMapCardPreview(canvas, entry);
    }
  }
  const previewCanvas = document.querySelector(`.map-preview-canvas[data-preview-map-id="${currentMapEntry?.id || ""}"]`) || startPreview;
  if (!previewCanvas || !room) return;
  const miniPreviewSource = room.scanPoints?.length ? room.scanPoints : room.displayPoints;
  drawSceneMiniPreview(
    previewCanvas,
    { points3D: miniPreviewSource.map(p => ({ xyz: p.rxyz, rgb: p.rgb })) },
    currentMapEntry || { id: "default_demo", has_drone_demo: true },
  );
  return;
  const rect = previewCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  previewCanvas.width = Math.max(1, Math.floor(rect.width * dpr));
  previewCanvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const pctx = previewCanvas.getContext("2d");
  pctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  pctx.clearRect(0, 0, rect.width, rect.height);

  const b = room.bounds;
  const preview = { yaw: -0.72, pitch: 0.68 };
  const cy = Math.cos(preview.yaw), sy = Math.sin(preview.yaw);
  const cp = Math.cos(preview.pitch), sp = Math.sin(preview.pitch);
  const currentPreviewId = currentMapEntry?.id || "default_demo";
  const previewZoom = previewZoomByMap.get(currentPreviewId) || 1;
  const scale = 0.72 * Math.min(rect.width, rect.height) * previewZoom / b.radius;
  const toPreview = p => {
    const x0 = p[0] - b.center[0];
    const y0 = p[1] - b.center[1];
    const z0 = p[2] - b.center[2];
    const x1 = cy * x0 + sy * z0;
    const z1 = -sy * x0 + cy * z0;
    const y2 = cp * y0 - sp * z1;
    const z2 = sp * y0 + cp * z1;
    return [
      rect.width * 0.52 + x1 * scale,
      rect.height * 0.60 - y2 * scale,
      z2,
    ];
  };
  const drawPreviewLine = (a, c, color, width = 1, dash = []) => {
    const pa = toPreview(a), pc = toPreview(c);
    pctx.strokeStyle = color;
    pctx.lineWidth = width;
    pctx.setLineDash(dash);
    pctx.beginPath();
    pctx.moveTo(pa[0], pa[1]);
    pctx.lineTo(pc[0], pc[1]);
    pctx.stroke();
    pctx.setLineDash([]);
  };

  pctx.fillStyle = "rgba(4, 12, 11, 0.7)";
  pctx.fillRect(0, 0, rect.width, rect.height);

  const floorY = room.floorY;
  const sx = b.max[0] - b.min[0];
  const sz = b.max[2] - b.min[2];
  const step = Math.max(0.25, Math.pow(10, Math.floor(Math.log10(Math.max(sx, sz) / 5))));
  const startX = Math.floor(b.min[0] / step) * step;
  const endX = Math.ceil(b.max[0] / step) * step;
  const startZ = Math.floor(b.min[2] / step) * step;
  const endZ = Math.ceil(b.max[2] / step) * step;
  for (let x = startX; x <= endX; x += step) {
    drawPreviewLine([x, floorY, startZ], [x, floorY, endZ], "rgba(105,218,255,0.13)", 0.9);
  }
  for (let z = startZ; z <= endZ; z += step) {
    drawPreviewLine([startX, floorY, z], [endX, floorY, z], "rgba(105,218,255,0.13)", 0.9);
  }

  const corners = [
    [b.min[0], b.min[1], b.min[2]],
    [b.max[0], b.min[1], b.min[2]],
    [b.max[0], b.min[1], b.max[2]],
    [b.min[0], b.min[1], b.max[2]],
    [b.min[0], b.max[1], b.min[2]],
    [b.max[0], b.max[1], b.min[2]],
    [b.max[0], b.max[1], b.max[2]],
    [b.min[0], b.max[1], b.max[2]],
  ];
  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  for (const [a, c] of edges) {
    drawPreviewLine(corners[a], corners[c], "rgba(119,226,255,0.50)", 1.2);
  }

  const previewSource = room.scanPoints?.length ? room.scanPoints : room.displayPoints;
  const stride = Math.max(1, Math.ceil(previewSource.length / 5200));
  const previewPoints = [];
  const yMin = b.min[1];
  const ySpan = Math.max(b.max[1] - b.min[1], 1e-6);
  for (let i = 0; i < previewSource.length; i += stride) previewPoints.push(previewSource[i]);
  previewPoints.sort((a, c) => toPreview(a.rxyz)[2] - toPreview(c.rxyz)[2]);
  for (const p of previewPoints) {
    const [x, y] = toPreview(p.rxyz);
    const rgb = p.rgb || [220, 230, 225];
    const h = Math.max(0, Math.min(1, (p.rxyz[1] - yMin) / ySpan));
    const boost = 0.64 + 0.42 * h;
    pctx.fillStyle = `rgba(${Math.min(255, rgb[0] * boost)},${Math.min(255, rgb[1] * boost)},${Math.min(255, rgb[2] * boost)},0.72)`;
    pctx.fillRect(x - 0.7, y - 0.7, 1.4, 1.4);
  }

  const good = room.poses.filter(p => p.success && p.rcenter);
  if (good.length > 1) {
    pctx.strokeStyle = "rgba(255,64,92,0.18)";
    pctx.lineWidth = 9;
    pctx.setLineDash([]);
    pctx.beginPath();
    for (let i = 0; i < good.length; i++) {
      const [x, y] = toPreview(good[i].rcenter);
      if (i === 0) pctx.moveTo(x, y);
      else pctx.lineTo(x, y);
    }
    pctx.stroke();

    pctx.strokeStyle = "rgba(255,64,92,0.96)";
    pctx.lineWidth = 2.4;
    pctx.setLineDash([9, 7]);
    pctx.beginPath();
    for (let i = 0; i < good.length; i++) {
      const [x, y] = toPreview(good[i].rcenter);
      if (i === 0) pctx.moveTo(x, y);
      else pctx.lineTo(x, y);
    }
    pctx.stroke();
    pctx.setLineDash([]);
  }

  pctx.fillStyle = "rgba(2, 12, 11, 0.72)";
  pctx.fillRect(10, 10, 88, 24);
  pctx.fillStyle = "#a9eaff";
  pctx.font = "bold 12px Inter, system-ui, sans-serif";
  pctx.fillText("3D preview", 22, 27);
}

function drawLiveBuildPreview(frameCount = 0, status = "idle") {
  if (!liveBuildPreview) return;
  const rect = liveBuildPreview.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  liveBuildPreview.width = Math.max(1, Math.floor(rect.width * dpr));
  liveBuildPreview.height = Math.max(1, Math.floor(rect.height * dpr));
  const lctx = liveBuildPreview.getContext("2d");
  lctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  lctx.clearRect(0, 0, rect.width, rect.height);

  const w = rect.width;
  const h = rect.height;
  lctx.fillStyle = "rgba(3, 12, 11, 0.96)";
  lctx.fillRect(0, 0, w, h);

  const cx = w * 0.52;
  const cy = h * 0.60;
  const yaw = -0.62;
  const pitch = 0.72;
  const projectMini = p => {
    const x0 = p[0], y0 = p[1], z0 = p[2];
    const x1 = Math.cos(yaw) * x0 + Math.sin(yaw) * z0;
    const z1 = -Math.sin(yaw) * x0 + Math.cos(yaw) * z0;
    const y2 = Math.cos(pitch) * y0 - Math.sin(pitch) * z1;
    const scale = Math.min(w, h) * 0.22;
    return [cx + x1 * scale, cy - y2 * scale];
  };

  lctx.strokeStyle = "rgba(105,218,255,0.14)";
  lctx.lineWidth = 1;
  for (let i = -4; i <= 4; i++) {
    const a = projectMini([i * 0.25, 0, -1.1]);
    const b = projectMini([i * 0.25, 0, 1.1]);
    lctx.beginPath();
    lctx.moveTo(a[0], a[1]);
    lctx.lineTo(b[0], b[1]);
    lctx.stroke();
    const c = projectMini([-1.1, 0, i * 0.25]);
    const d = projectMini([1.1, 0, i * 0.25]);
    lctx.beginPath();
    lctx.moveTo(c[0], c[1]);
    lctx.lineTo(d[0], d[1]);
    lctx.stroke();
  }

  const n = Math.max(0, Number(frameCount) || 0);
  const points = [];
  for (let i = 0; i < n; i++) {
    const t = i / Math.max(n - 1, 1);
    const angle = t * Math.PI * 1.45 - 0.5;
    points.push([
      Math.cos(angle) * (0.25 + 0.55 * t),
      0.18 + 0.28 * Math.sin(t * Math.PI),
      Math.sin(angle) * (0.25 + 0.55 * t),
    ]);
  }

  if (points.length > 1) {
    lctx.strokeStyle = "rgba(255,64,92,0.90)";
    lctx.lineWidth = 2.5;
    lctx.setLineDash([7, 6]);
    lctx.beginPath();
    points.forEach((p, i) => {
      const [x, y] = projectMini(p);
      if (i === 0) lctx.moveTo(x, y);
      else lctx.lineTo(x, y);
    });
    lctx.stroke();
    lctx.setLineDash([]);
  }

  for (const [i, p] of points.entries()) {
    const [x, y] = projectMini(p);
    lctx.fillStyle = i === points.length - 1 ? "#ff5d88" : "rgba(74,163,255,0.9)";
    lctx.beginPath();
    lctx.arc(x, y, i === points.length - 1 ? 5 : 3, 0, Math.PI * 2);
    lctx.fill();
    const ground = projectMini([p[0], 0, p[2]]);
    lctx.strokeStyle = "rgba(74,163,255,0.16)";
    lctx.lineWidth = 1;
    lctx.beginPath();
    lctx.moveTo(x, y);
    lctx.lineTo(ground[0], ground[1]);
    lctx.stroke();
  }

  lctx.fillStyle = "rgba(2,12,11,0.76)";
  lctx.fillRect(12, 12, 158, 48);
  lctx.fillStyle = "#d9fff5";
  lctx.font = "bold 13px Inter, system-ui, sans-serif";
  lctx.fillText(`${n} captured frames`, 24, 34);
  lctx.fillStyle = "#75dfff";
  lctx.font = "11px Inter, system-ui, sans-serif";
  lctx.fillText(status === "stopping" ? "reconstruction queued" : "mapping path preview", 24, 50);
}

function render() {
  const { rect, dpr } = resize();
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!scene || !room) {
    requestAnimationFrame(render);
    return;
  }

  const cur = closestPose();
  if (view.mode === "drone" && cur?.rcenter) {
    centerViewOn(cur.rcenter, 0.50, 0.56, true);
  }

  drawStaticLayer(rect, dpr);
  drawSafetyBarriers();
  drawMissionTarget(cur);

  if (cur && cur.rcenter) {
    updateReplayFrameViewForPose(cur);
    drawRouteMarker(cur.rcenter, 12);
    if (window.directDroneOverlayInstalled === false && !window.directDroneModelReady) {
      drawDroneIcon(cur.rcenter, headingForPose(cur));
    }
    drawLabel(cur.rcenter, "drone", "#ffd5df");
    poseTime.textContent = cur.time_sec == null ? cur.instance_id : `${Number(cur.time_sec).toFixed(2)} s`;
    poseTotal.textContent = cur.total_ms == null ? "-" : `${Number(cur.total_ms).toFixed(2)} ms`;
    poseAction.textContent = cur.stages_ms?.ysolve_static_action_double_ms == null ? "-" : `${Number(cur.stages_ms.ysolve_static_action_double_ms).toFixed(2)} ms`;
    poseRoot.textContent = cur.stages_ms?.ysolve_static_root_total_ms == null ? "-" : `${Number(cur.stages_ms.ysolve_static_root_total_ms).toFixed(2)} ms`;
    poseCenter.textContent = formatVector(cur.center);
    poseT.textContent = formatVector(cur.t);
    poseR.textContent = formatMatrix(cur.R);
    updateMissionStatus();
  } else if (liveReplayInFlight || pendingLiveReplayOpen) {
    drawLiveAcquisitionMarker();
    poseTime.textContent = "processing";
    poseTotal.textContent = "waiting for first R,t";
    poseAction.textContent = "-";
    poseRoot.textContent = "-";
    poseCenter.textContent = liveReplayMessage || "localizing incoming frames";
    poseT.textContent = "-";
    poseR.textContent = "-";
  }

  requestAnimationFrame(render);
}

async function loadViewerData(resetView = false, entry = null) {
  pathPlaybackActive = false;
  replayFramePlaybackEnabled = false;
  currentMapEntry = entry || selectedMap() || currentMapEntry || {
    id: "default_demo",
    asset_base: "public",
    title: "Indoor Patrol Map",
  };
  const base = currentMapEntry.asset_base || "public";
  const replay = activeReplay(currentMapEntry);
  const scanPath = currentMapEntry.scan_path || (currentMapEntry.id === "default_demo" ? "public/scan_mesh/scan_points.json" : null);
  const scanPromise = scanPath ? fetch(scanPath)
    .then(resp => resp.ok ? resp.json() : null)
    .catch(() => null) : Promise.resolve(null);
  const poseUrl = replay ? replayAssetUrl(replay, "poses.json") : assetUrl(currentMapEntry, "poses.json");
  const [sceneResp, poseResp, scanData] = await Promise.all([
    fetch(cacheBust(assetUrl(currentMapEntry, "scene.json")), { cache: "no-store" }),
    fetch(cacheBust(poseUrl), { cache: "no-store" }),
    scanPromise,
  ]);
  if (!sceneResp.ok) throw new Error(`missing scene for ${currentMapEntry.title || currentMapEntry.id}`);
  scene = await sceneResp.json();
  poseStreamMeta = poseResp.ok ? await poseResp.json() : null;
  poses = liveReplayInFlight ? [] : (poseStreamMeta?.poses || []);
  scan = scanData;
  droneModel = null;
  document.body.dataset.canvasDroneModel = "fallback-icon";
  room = buildRoomFrame();
  invalidateStaticLayer();
  const scanLine = displayPointSummaryLine();
  const sourceLine = mapSourceLine();
  const activeReplayLine = replay ? `Active drone path: ${replay.title || "Drone Path"}<br>` : "Active drone path: none<br>";
  const quality = room.poseQuality || {};
  const accepted = Number(quality.accepted ?? poses.filter(p => p.success).length ?? 0);
  const rejected = Number(quality.rejected ?? 0);
  const replayLine = liveReplayInFlight
    ? `Live TSolve initializing${accepted ? `: ${accepted} accepted` : ""}`
    : (poses.length ? `${accepted}/${poses.length} live TSolve R,t updates` : "No TSolve live replay yet");
  const streamLine = liveReplayInFlight
    ? "Live replay processing: waiting for exported R,t stream"
    : (poses.length ? "MP4 stream replay drives pose time" : "Upload drone video to localize online");
  stats.innerHTML = `Selected 3D map: ${currentMapEntry.title || "Selected Map"}<br>${scene.points3D.length} COLMAP map points<br>${scanLine}${scene.map_cameras.length} map cameras<br>${activeReplayLine}${replayLine}<br>${streamLine}<br>${sourceLine}`;
  const mediaUrl = replay ? replayAssetUrl(replay, "media/drone_query.mp4") : assetUrl(currentMapEntry, "media/drone_query.mp4");
  const replayFrameBase = replayQueryFrameBaseUrl(replay);
  if (!liveReplayInFlight && replayFrameBase && poses.length) {
    setVideoFrameSteppingMode(true);
    clearUploadedVideoPreview();
    video.pause();
    video.removeAttribute("src");
    video.load();
    lastReplayFrameUrl = "";
    const firstPose = sortedTimedPoses(poses)[0] || poses.find(p => p?.success && (p.rcenter || p.center)) || poses[0];
    if (!updateReplayFrameViewForPose(firstPose, { force: true })) {
      setLiveFrameMode(true);
      setLiveFrameStatus("Saved live path has poses, but its query-frame image could not be resolved.", true);
    }
  } else if (!liveReplayInFlight && (replay || currentMapEntry.has_drone_demo || poses.length)) {
    setLiveFrameMode(false);
    setVideoFrameSteppingMode(false);
    clearUploadedVideoPreview();
    if (video.getAttribute("src") !== mediaUrl) {
      video.src = mediaUrl;
      video.load();
    }
  } else if (!liveReplayInFlight) {
    setLiveFrameMode(false);
    setVideoFrameSteppingMode(false);
    clearUploadedVideoPreview();
    video.pause();
    video.removeAttribute("src");
    video.load();
  }
  if (resetView) setView("iso");
  renderReplayTabs();
  renderBarrierList();
  renderStartPreview();
}

async function init() {
  await refreshMapLibrary();
  await loadViewerData(true);
  renderStarted = true;
  updateNavState();
  setupLiveControlDrag();
  updateFlightControlState();
  renderBarrierList();
  render();
}

function screenTitle(screen) {
  if (screen === "modal") return "Create Map";
  return "";
}

function updateNavState() {
  if (atlasScreenLabel) atlasScreenLabel.textContent = screenTitle(currentScreen);
  if (navBack) navBack.disabled = screenHistory.length === 0 && currentScreen === "start";
}

function rememberScreen(target) {
  if (currentScreen !== target) screenHistory.push(currentScreen);
}

function showDemo(options = {}) {
  if (options.push !== false) rememberScreen("demo");
  document.body.classList.remove("show-start");
  document.body.classList.add("show-demo");
  mapModal?.classList.add("hidden");
  demoApp?.setAttribute("aria-hidden", "false");
  startPage?.setAttribute("aria-hidden", "true");
  currentScreen = "demo";
  updateNavState();
  setView("iso");
  if (sidePanel) sidePanel.scrollTop = 0;
  renderReplayTabs();
  updateFlightControlState();
  resize();
  if (options.resetVideo !== false && !liveReplayInFlight && !pendingLiveReplayOpen) {
    video.currentTime = 0;
    video.pause();
  }
}

function showLibrary(options = {}) {
  if (options.push !== false) rememberScreen("start");
  video.pause();
  document.body.classList.remove("show-demo");
  document.body.classList.add("show-start");
  mapModal?.classList.add("hidden");
  demoApp?.setAttribute("aria-hidden", "true");
  startPage?.setAttribute("aria-hidden", "false");
  currentScreen = "start";
  updateNavState();
  renderMapLibrary();
  renderStartPreview();
}

async function selectMap(mapId, openAfter = false) {
  const data = await postJson("/api/map/select", { map_id: mapId });
  if (data.state?.library) mapLibraryData = data.state.library;
  else await refreshMapLibrary();
  currentMapEntry = selectedMap();
  renderMapLibrary();
  renderReplayTabs();
  await loadViewerData(false, currentMapEntry);
  uploadStatus.textContent = `Selected map: ${currentMapEntry?.title || mapId}`;
  if (openAfter) showDemo();
}

async function deleteMap(mapId) {
  const entry = (mapLibraryData.maps || []).find(m => m.id === mapId);
  if (!entry) return;
  if (!window.confirm(`Delete 3D map "${entry.title}"?`)) return;
  uploadStatus.textContent = `Deleting map: ${entry.title || mapId}`;
  const data = await postJson("/api/map/delete", { map_id: mapId });
  if (data.state?.library) mapLibraryData = data.state.library;
  else await refreshMapLibrary();
  mapLibraryData.maps = (mapLibraryData.maps || []).filter(m => m.id !== mapId);
  if (mapLibraryData.selected_map_id === mapId) {
    mapLibraryData.selected_map_id = mapLibraryData.maps[0]?.id || "";
  }
  currentMapEntry = selectedMap();
  renderMapLibrary();
  renderReplayTabs();
  if (currentMapEntry) {
    await loadViewerData(false, currentMapEntry);
  } else {
    scene = null;
    poses = [];
    room = null;
    showLibrary({ push: false });
  }
  uploadStatus.textContent = `Deleted map: ${entry.title}`;
}

async function duplicateMap(mapId) {
  const entry = (mapLibraryData.maps || []).find(m => m.id === mapId);
  if (!entry) return;
  uploadStatus.textContent = `Duplicating 3D map without drone paths: ${entry.title || mapId}`;
  const data = await postJson("/api/map/duplicate", { map_id: mapId });
  if (data.state?.library) mapLibraryData = data.state.library;
  else await refreshMapLibrary();
  currentMapEntry = selectedMap();
  renderMapLibrary();
  renderReplayTabs();
  await loadViewerData(false, currentMapEntry);
  uploadStatus.textContent = `Duplicated map without paths: ${currentMapEntry?.title || "new 3D map"}`;
}

async function renameMap(mapId) {
  const entry = (mapLibraryData.maps || []).find(m => m.id === mapId);
  if (!entry) return;
  const current = entry.title || "Untitled Map";
  const title = window.prompt("Rename 3D map", current);
  if (title == null) return;
  const cleaned = title.trim();
  if (!cleaned || cleaned === current) return;
  const data = await postJson("/api/map/rename", { map_id: mapId, title: cleaned });
  if (data.state?.library) mapLibraryData = data.state.library;
  else await refreshMapLibrary();
  currentMapEntry = selectedMap();
  renderMapLibrary();
  renderReplayTabs();
  if (currentMapEntry?.id === mapId) await loadViewerData(false, currentMapEntry);
  uploadStatus.textContent = `Renamed map: ${cleaned}`;
}

canvas.addEventListener("mousedown", e => {
  const cornerHit = barrierCornerHit(e.clientX, e.clientY);
  if (cornerHit) {
    barrierCornerDrag = cornerHit;
    barrierCornerHover = cornerHit;
    barrierTransformHover = null;
    barrierDragMoved = false;
    dragging = false;
    missionDraggingTarget = false;
    canvas.style.cursor = "grabbing";
    markFastInteraction(120);
    updateBarrierStatus(`Drag this corner to reshape the safety wall. ${barrierDragViewHint()}`, "busy");
    return;
  }
  const transformHit = barrierTransformHit(e.clientX, e.clientY);
  if (transformHit && startBarrierTransformDrag(transformHit, e.clientX, e.clientY)) {
    barrierCornerHover = null;
    barrierTransformHover = transformHit;
    dragging = false;
    missionDraggingTarget = false;
    canvas.style.cursor = "grabbing";
    markFastInteraction(120);
    updateBarrierStatus(
      transformHit.type === "move"
        ? "Drag the center handle to move this safety wall."
        : "Drag the diamond handle to rotate this safety wall.",
      "busy",
    );
    return;
  }
  if (barrierEditing) {
    dragging = false;
    markFastInteraction(120);
    return;
  }
  if (missionTargetHit(e.clientX, e.clientY)) {
    missionDraggingTarget = true;
    missionDragMoved = false;
    dragging = false;
    markFastInteraction(120);
    updateMissionStatus("Drag the destination marker to another visible 3D point.");
    return;
  }
  dragging = true;
  markFastInteraction(320);
  last = { x: e.clientX, y: e.clientY };
});
window.addEventListener("mouseup", () => {
  if (barrierCornerDrag) {
    saveDraggedBarrierCorner();
  }
  if (barrierTransformDrag) {
    saveDraggedBarrierTransform();
  }
  if (missionDraggingTarget) {
    missionDraggingTarget = false;
    updateMissionStatus();
  }
  dragging = false;
  canvas.style.cursor = barrierCornerHover || barrierTransformHover ? "grab" : (missionSelecting ? "crosshair" : "");
  markFastInteraction(160);
});
window.addEventListener("mousemove", e => {
  if (barrierCornerDrag) {
    markFastInteraction(120);
    updateBarrierCornerFromPointer(e.clientX, e.clientY);
    return;
  }
  if (barrierTransformDrag) {
    markFastInteraction(120);
    updateBarrierTransformFromPointer(e.clientX, e.clientY);
    return;
  }
  if (missionDraggingTarget) {
    missionDragMoved = true;
    markFastInteraction(120);
    updateMissionTargetFromPointer(e.clientX, e.clientY);
    return;
  }
  updateBarrierHover(e.clientX, e.clientY);
  if (!dragging) return;
  markFastInteraction(220);
  view.yaw += (e.clientX - last.x) * 0.006;
  view.pitch += (e.clientY - last.y) * 0.006;
  view.pitch = Math.max(-1.55, Math.min(1.35, view.pitch));
  last = { x: e.clientX, y: e.clientY };
});
canvas.addEventListener("mouseleave", () => {
  if (!barrierCornerDrag && !barrierTransformDrag) clearBarrierHover();
});
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  markFastInteraction(220);
  view.zoom *= Math.exp(-e.deltaY * 0.001);
  view.zoom = Math.max(0.12, Math.min(20, view.zoom));
}, { passive: false });
canvas.addEventListener("click", e => {
  if (barrierClickSuppress) {
    barrierClickSuppress = false;
    return;
  }
  if (barrierDragMoved) {
    barrierDragMoved = false;
    return;
  }
  if (missionDragMoved) {
    missionDragMoved = false;
    return;
  }
  if (barrierEditing) {
    const picked = nearestVisibleMapPoint(e.clientX, e.clientY);
    if (!picked?.rxyz) {
      updateBarrierStatus("No visible map point under the cursor. Pick a point on the obstacle/wall edge.", "error");
      return;
    }
    addBarrierFromPickedPoint(picked);
    return;
  }
  if (!missionSelecting) return;
  const picked = nearestVisibleMapPoint(e.clientX, e.clientY);
  if (!picked?.rxyz) {
    updateMissionStatus("No visible map point under the cursor. Try a denser point area.");
    return;
  }
  missionTarget = { rxyz: picked.rxyz, rgb: picked.rgb || null };
  plannedMission = null;
  missionSelecting = false;
  selectTargetButton?.classList.remove("active");
  updateMissionStatus();
});

document.getElementById("open-demo")?.addEventListener("click", showDemo);
document.querySelector(".map-card.selected")?.addEventListener("dblclick", showDemo);
document.getElementById("back-library").addEventListener("click", showLibrary);
navBack?.addEventListener("click", goBack);
atlasHome?.addEventListener("click", goHome);
document.getElementById("start").addEventListener("click", event => {
  event.preventDefault();
  playCurrentReplay();
});
document.getElementById("reset").addEventListener("click", () => {
  if (view.mode === "drone") setDroneView();
  else setView(view.mode || "top");
});
viewIsoButton?.addEventListener("click", () => setView("iso", { advance: true }));
document.getElementById("view-top").addEventListener("click", () => setView("top"));
document.getElementById("view-side").addEventListener("click", () => setView("side", { advance: view.mode === "side" }));
viewDroneButton?.addEventListener("click", setDroneView);
document.getElementById("flip-z")?.addEventListener("click", () => runUi(async () => {
  if (!currentMapEntry?.id) return;
  const current = Number(currentMapEntry.display_z_sign ?? -1) < 0 ? -1 : 1;
  const data = await postJson("/api/map/display-z", {
    map_id: currentMapEntry.id,
    display_z_sign: current < 0 ? 1 : -1,
  });
  if (data.state?.library) mapLibraryData = data.state.library;
  else await refreshMapLibrary();
  currentMapEntry = selectedMap();
  await loadViewerData(false, currentMapEntry);
  renderMapLibrary();
  renderReplayTabs();
  renderStartPreview();
  uploadStatus.textContent = `Flipped Z display for ${currentMapEntry?.title || "selected map"}`;
}));
selectTargetButton?.addEventListener("click", () => {
  barrierEditing = false;
  barrierDraft = null;
  addBarrierButton?.classList.remove("active");
  if (cancelBarrierButton) cancelBarrierButton.disabled = true;
  updateBarrierStatus();
  missionSelecting = !missionSelecting;
  selectTargetButton.classList.toggle("active", missionSelecting);
  updateMissionStatus();
});
clearTargetButton?.addEventListener("click", () => {
  missionTarget = null;
  plannedMission = null;
  missionSelecting = false;
  selectTargetButton?.classList.remove("active");
  updateMissionStatus();
});
planMissionButton?.addEventListener("click", planMissionPreview);
startMissionButton?.addEventListener("click", () => {
  if (!firstLocalizationConfirmed) {
    updateMissionStatus("Confirm the first TSolve localization before confirming a mission.");
    return;
  }
  if (!plannedMission) {
    updateMissionStatus("Plan the path before confirming the mission.");
    return;
  }
  if (!missionTarget?.rxyz) {
    updateMissionStatus("Select a destination before starting guided patrol.");
    return;
  }
  planMissionPreview();
  const safety = plannedMission?.route_segments?.length
    ? missionRouteSafetyCheck(plannedMission.route_segments)
    : missionBarrierCheck(missionTarget.rxyz);
  if (safety.blocked) {
    plannedMission = null;
    updateMissionStatus(`Mission blocked by safety wall. ${safety.reason}`);
    updateFlightControlState();
    return;
  }
  const profile = missionLandingProfile(missionTarget.rxyz);
  const landingText = profile?.targetLooksGround ? "land at the ground target" : "descend to the selected point";
  updateMissionStatus(`Mission confirmed as preview only: yaw toward segment, cruise to approach point, hover, then ${landingText}. Autonomous TSolve-to-RC path following remains locked until verified; takeoff/land are live.`);
});
addBarrierButton?.addEventListener("click", () => {
  if (barrierUnsaved) {
    updateBarrierStatus("Save or discard the staged wall edits before adding a new wall.", "error");
    return;
  }
  barrierEditing = true;
  barrierAdjusting = false;
  barrierDraft = null;
  missionSelecting = false;
  plannedMission = null;
  selectTargetButton?.classList.remove("active");
  addBarrierButton.classList.add("active");
  if (cancelBarrierButton) cancelBarrierButton.disabled = false;
  updateBarrierAdjustControls();
  updateBarrierStatus();
  updateMissionStatus();
});
adjustWallsButton?.addEventListener("click", () => {
  if (!mapSafetyBarriers().length) {
    updateBarrierStatus("Add a wall first, then press Adjust Walls.", "error");
    return;
  }
  if (barrierUnsaved && barrierAdjusting) {
    updateBarrierStatus("Press Save Walls before leaving wall adjustment mode.", "error");
    return;
  }
  barrierEditing = false;
  barrierDraft = null;
  barrierAdjusting = !barrierAdjusting;
  addBarrierButton?.classList.remove("active");
  if (cancelBarrierButton) cancelBarrierButton.disabled = true;
  clearBarrierHover();
  updateBarrierAdjustControls();
  updateBarrierStatus(
    barrierAdjusting
      ? "Wall adjustment mode is active. Drag corners, center, or diamond handles, then press Save Walls."
      : null,
    barrierAdjusting ? "busy" : "",
  );
});
saveWallAdjustmentsButton?.addEventListener("click", () => {
  if (!barrierUnsaved) return;
  saveSafetyBarriers(mapSafetyBarriers().map(barrierPayloadForSave));
});
cancelBarrierButton?.addEventListener("click", () => {
  if (barrierUnsaved) {
    barrierUnsaved = false;
    stagedSafetyBarrierMapId = null;
    stagedSafetyBarriers = null;
    currentMapEntry = selectedMap() || currentMapEntry;
    barrierAdjusting = false;
    clearBarrierHover();
    updateBarrierAdjustControls();
    renderBarrierList();
    invalidateStaticLayer();
    updateMissionStatus();
    return;
  }
  barrierEditing = false;
  barrierDraft = null;
  addBarrierButton?.classList.remove("active");
  if (cancelBarrierButton) cancelBarrierButton.disabled = true;
  updateBarrierStatus();
});
clearBarriersButton?.addEventListener("click", () => {
  if (!mapSafetyBarriers().length) return;
  if (!window.confirm("Clear all manual safety walls for this 3D map?")) return;
  plannedMission = null;
  saveSafetyBarriers([]);
});
djiTakeoffButton?.addEventListener("click", async () => {
  if (!liveLocalizationStarted()) {
    setDjiCommandStatus("Press Start Localization first. Takeoff is locked until the live stream is active.", "error");
    return;
  }
  const height = takeoffHeightM();
  const ok = window.confirm(`Send TAKEOFF to the DJI drone?\n\nSafety checklist:\n- Propellers are clear.\n- The live ATLAS stream is running.\n- Requested guarded height: ${height.toFixed(1)} m.\n\nContinue?`);
  if (!ok) return;
  try {
    setDjiCommandStatus("Sending takeoff command...", "busy");
    await sendDjiFlightCommand("takeoff", { height_m: height });
    setDjiCommandStatus("Takeoff sent through the active live bridge. Wait for first TSolve pose, then confirm localization.", "ok");
  } catch (err) {
    setDjiCommandStatus(`Takeoff failed: ${err.message || err}`, "error");
  }
});
djiLandButton?.addEventListener("click", async () => {
  if (!liveLocalizationStarted()) {
    setDjiCommandStatus("Start live localization first so landing remains visible and logged.", "error");
    return;
  }
  const ok = window.confirm("Send LAND to the DJI drone?\n\nThe drone should have a clear landing area. Continue?");
  if (!ok) return;
  try {
    setDjiCommandStatus("Sending land command...", "busy");
    await sendDjiFlightCommand("land");
    setDjiCommandStatus("Land command sent. Keep localization running until touchdown is visible.", "ok");
  } catch (err) {
    setDjiCommandStatus(`Land failed: ${err.message || err}`, "error");
  }
});
confirmLocalizationButton?.addEventListener("click", () => {
  if (!firstConfirmedPoseReady()) {
    updateFlightControlState();
    return;
  }
  firstLocalizationConfirmed = true;
  setDjiCommandStatus("Localization confirmed. Mission controls unlocked.", "ok");
  if (missionTarget?.rxyz) {
    planMissionPreview();
  } else {
    updateMissionStatus("Localization confirmed. Pick a COLMAP point destination.");
  }
  updateFlightControlState();
});
togglePointsButton?.addEventListener("click", () => {
  view.showPoints = !view.showPoints;
  invalidateStaticLayer();
  updateViewButtons();
});
toggleCamerasButton?.addEventListener("click", () => {
  view.showCameras = !view.showCameras;
  invalidateStaticLayer();
  updateViewButtons();
});
document.getElementById("axis-reset").addEventListener("click", () => {
  view.axisScale = { x: 1, y: 1, z: 1 };
});
for (const btn of document.querySelectorAll(".axis-controls button[data-axis]")) {
  btn.addEventListener("click", () => {
    const axis = btn.dataset.axis;
    const dir = Number(btn.dataset.dir);
    const factor = dir > 0 ? 1.18 : 1 / 1.18;
    view.axisScale[axis] = Math.max(0.25, Math.min(4.0, view.axisScale[axis] * factor));
  });
}
function openMapModal(options = {}) {
  if (options.push !== false) rememberScreen("modal");
  mapModal?.classList.remove("hidden");
  currentScreen = "modal";
  updateNavState();
}

function closeMapModal(options = {}) {
  mapModal?.classList.add("hidden");
  const fallback = document.body.classList.contains("show-demo") ? "demo" : "start";
  const target = options.pop === false ? fallback : (screenHistory.pop() || fallback);
  if (target === "demo") showDemo({ push: false });
  else showLibrary({ push: false });
}

function goBack() {
  const target = screenHistory.pop();
  if (target === "demo") showDemo({ push: false });
  else if (target === "modal") openMapModal({ push: false });
  else showLibrary({ push: false });
}

function goHome() {
  screenHistory = [];
  showLibrary({ push: false });
}

function setupLiveControlDrag() {
  const panel = liveLocalizationControl;
  const handle = panel?.querySelector("summary");
  const parent = panel?.parentElement;
  if (!panel || !handle || !parent) return;
  let drag = null;
  let suppressClick = false;
  handle.addEventListener("pointerdown", event => {
    if (event.button !== 0) return;
    if (event.target.closest("button, input, select, label")) return;
    const panelRect = panel.getBoundingClientRect();
    const fixed = getComputedStyle(panel).position === "fixed";
    const parentRect = fixed
      ? { left: 0, top: 0, width: window.innerWidth, height: window.innerHeight }
      : parent.getBoundingClientRect();
    drag = {
      pointerId: event.pointerId,
      dx: event.clientX - panelRect.left,
      dy: event.clientY - panelRect.top,
      startX: event.clientX,
      startY: event.clientY,
      parentRect,
      moved: false,
    };
    handle.setPointerCapture?.(event.pointerId);
  });
  handle.addEventListener("pointermove", event => {
    if (!drag || drag.pointerId !== event.pointerId) return;
    const moveD = Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY);
    if (moveD > 3) drag.moved = true;
    const rect = drag.parentRect;
    const panelRect = panel.getBoundingClientRect();
    const maxLeft = Math.max(0, rect.width - panelRect.width - 10);
    const maxTop = Math.max(0, rect.height - panelRect.height - 10);
    const left = Math.max(10, Math.min(maxLeft, event.clientX - rect.left - drag.dx));
    const top = Math.max(10, Math.min(maxTop, event.clientY - rect.top - drag.dy));
    panel.style.left = `${left}px`;
    panel.style.top = `${top}px`;
    panel.style.right = "auto";
    panel.style.bottom = "auto";
    panel.classList.add("is-user-placed");
    suppressClick = drag.moved;
  });
  handle.addEventListener("pointerup", event => {
    if (drag?.pointerId === event.pointerId) {
      handle.releasePointerCapture?.(event.pointerId);
      suppressClick = drag.moved;
      drag = null;
    }
  });
  handle.addEventListener("click", event => {
    if (!suppressClick) return;
    event.preventDefault();
    event.stopPropagation();
    suppressClick = false;
  }, true);
}

async function postJson(url, payload) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  const data = await resp.json();
  if (!resp.ok || data.ok === false) throw new Error(data.error || `Request failed: ${url}`);
  return data;
}

async function uploadVideos(url, files, fields = {}) {
  const form = new FormData();
  for (const file of files) form.append("video", file);
  for (const [key, value] of Object.entries(fields || {})) {
    form.append(key, String(value));
  }
  const resp = await fetch(url, { method: "POST", body: form });
  const data = await resp.json();
  if (!resp.ok || data.ok === false) throw new Error(data.error || `Upload failed: ${url}`);
  return data;
}

async function uploadVideo(url, file, fields = {}) {
  return uploadVideos(url, [file], fields);
}

function runUi(action) {
  action().catch(error => {
    const message = error?.message || String(error);
    uploadStatus.textContent = message;
    jobLog.textContent = `ERROR: ${message}`;
    console.error(error);
  });
}

function formatJob(kind, job) {
  const label = kind === "map" ? "Map" : "Drone replay";
  return `${label}: ${job?.status || "idle"} - ${job?.message || ""}`;
}

function setDjiLiveText(state, meta) {
  if (djiLiveState) djiLiveState.textContent = state;
  if (djiLiveStateSide) djiLiveStateSide.textContent = state;
  if (djiLiveMeta) djiLiveMeta.textContent = meta;
  if (djiLiveMetaSide) djiLiveMetaSide.textContent = meta;
}

async function pollDjiLivePreview() {
  if (!djiLiveFeed && !djiLiveFeedSide) return;
  try {
    const resp = await fetch(`public/live_dji/status.json?t=${Date.now()}`, { cache: "no-store" });
    if (!resp.ok) {
      setDjiLiveText("offline", "No DJI bridge status yet.");
      return;
    }
    const status = await resp.json();
    const frames = Number(status.frames_saved || 0);
    const updated = Number(status.updated_at || 0);
    const age = updated ? Math.max(0, Date.now() / 1000 - updated) : null;
    const state = status.status || "unknown";
    const imageUrl = `public/live_dji/latest.jpg?t=${Date.now()}`;
    const shouldDriveMainLiveFrame = liveFrameMode && (liveAtlasPreviewActive || poseStreamMeta?.stream?.live_atlas);
    if (frames > 0) {
      if (djiLiveFeed) djiLiveFeed.src = imageUrl;
      if (djiLiveFeedSide) djiLiveFeedSide.src = imageUrl;
      if (shouldDriveMainLiveFrame) {
        const hasTsolveFrame = Boolean(liveFrameUrlForPayload(poseStreamMeta, poseStreamMeta?.stream || null));
        if (!hasTsolveFrame) {
          setLiveFrameStatus("Live DJI frames received. Waiting for TSolve to process the first frame...", true);
        }
      }
    } else if (shouldDriveMainLiveFrame) {
      if (liveFrameView) liveFrameView.removeAttribute("src");
      const stateText = String(state).replaceAll("_", " ");
      setLiveFrameStatus(`${stateText}: 0 live frames received from phone`, true);
    }
    const ageText = Number.isFinite(age) ? ` · ${age.toFixed(1)}s ago` : "";
    const frameText = frames > 0
      ? `${frames} frames · ${status.latest_frame || "latest"}${ageText}`
      : (status.message || "Waiting for first DJI frame.");
    setDjiLiveText(state, `${status.session || "DJI session"} · ${frameText}`);
    if (["stopped", "cancelled", "error"].includes(String(state).toLowerCase()) && liveFrameMode && !liveReplayInFlight) {
      setLiveFrameMode(false);
    }
  } catch (error) {
    setDjiLiveText("offline", "Live DJI status is not reachable.");
  }
}

function updateLivePoseStats(stream, payload) {
  if (!stats || !scene || !room) return;
  const title = currentMapEntry?.title || "Selected map";
  const scanLine = displayPointSummaryLine();
  const processed = Number(payload?.processed_count ?? stream?.pose_count ?? poses.length ?? 0);
  const expected = Number(payload?.expected_count ?? stream?.expected_count ?? 0);
  const quality = room.poseQuality || {};
  const accepted = Number(quality.accepted ?? 0);
  const rejected = Number(quality.rejected ?? 0);
  const countLine = expected > 0
    ? `Live TSolve R,t stream: ${accepted}/${processed}/${expected} accepted/processed/target`
    : `Live TSolve R,t stream: ${accepted}/${processed} accepted/processed`;
  const sourceLine = mapSourceLine();
  stats.innerHTML = `${title}<br>${scene.points3D.length} COLMAP map points<br>${scanLine}${scene.map_cameras.length} map cameras<br>${countLine}<br>${liveReplayMessage || "online self-localization active"}<br>${sourceLine}`;
}

async function loadLiveReplayPartial(stream = null) {
  if (!liveReplayInFlight && !pendingLiveReplayOpen) return false;
  if (!scene) return false;
  const resp = await fetch(`/api/live-replay?t=${Date.now()}`, { cache: "no-store" });
  if (!resp.ok && resp.status !== 404) return false;
  const payload = await resp.json().catch(() => null);
  if (!payload?.ok || !Array.isArray(payload.poses)) return false;

  const processed = Number(payload.processed_count ?? payload.poses.length ?? 0);
  const currentFrameKey = payload.current_frame
    ? `${payload.current_frame.frame_index ?? ""}:${payload.current_frame_time_sec ?? payload.current_frame.time_sec ?? ""}`
    : "";
  const key = `${payload.updated_at || ""}:${processed}:${currentFrameKey}`;
  if (key === livePoseStreamKey && processed === livePoseStreamCount) {
    updateLivePoseStats(stream || payload.stream, payload);
    return false;
  }

  livePoseStreamKey = key;
  livePoseStreamCount = processed;
  poseStreamMeta = payload;
  const expected = Number(payload.expected_count ?? payload.stream?.expected_count ?? 0);
  const firstLatency = Number(stream?.first_pose_latency_seconds ?? payload.stream?.first_pose_latency_seconds);
  const latencyLine = Number.isFinite(firstLatency)
    ? `; first R,t in ${firstLatency.toFixed(1)} s`
    : "";
  liveReplayStageDetail = expected > 0
    ? `TSolve R,t stream: ${processed}/${expected} frame updates${latencyLine}`
    : `TSolve R,t stream: ${processed} frame updates${latencyLine}`;
  poses = payload.poses;
  room = buildRoomFrame();
  ensureLiveStreamVideoSource(payload.stream || stream);
  const latestPose = latestSuccessfulPose(room.poses);
  liveCurrentPoseOverride = latestPose ? { ...latestPose } : null;
  if (liveReplayInFlight) {
    syncUploadedVideoToProcessingFrame(payload);
    if (latestPose && !liveVideoSyncedToFirstPose) {
      liveVideoSyncedToFirstPose = true;
      liveVideoWaitingForFirstPose = false;
    }
  } else if (latestPose) {
    syncUploadedVideoToLatestPose(room.poses);
  } else {
    syncUploadedVideoToProcessingFrame(payload);
  }
  invalidateStaticLayer();
  updateLivePoseStats(stream || payload.stream, payload);
  updateFlightControlState();
  renderReplayTabs();
  renderStartPreview();
  return true;
}

async function pollStatus() {
  if (liveStatusPollBusy) return;
  liveStatusPollBusy = true;
  try {
    const resp = await fetch("/api/status", { cache: "no-store" });
    if (!resp.ok) throw new Error(`status ${resp.status}`);
    const state = await resp.json();
    if (state.library) {
      const before = JSON.stringify(mapLibraryData);
      mapLibraryData = state.library;
      currentMapEntry = selectedMap();
      if (JSON.stringify(mapLibraryData) !== before) {
        renderMapLibrary();
        renderReplayTabs();
      }
    }
    mapStatus.textContent = formatJob("map", state.map);
    droneStatus.textContent = formatJob("drone", state.drone);
    const activeJobStates = new Set(["queued", "running", "stopping", "error"]);
    const liveJobStates = new Set(["queued", "running", "stopping"]);
    liveReplayInFlight = liveJobStates.has(state.drone?.status);
    if (liveReplayInFlight && !liveReplayStartedAt) liveReplayStartedAt = performance.now();
    liveReplayMessage = state.drone?.message || liveReplayMessage;
    if (state.drone?.status === "queued" || state.drone?.status === "running" || state.drone?.status === "stopping") {
      const stream = state.drone?.live_stream || {};
      if (stream.map_id) pendingLiveReplayMapId = stream.map_id;
      const poseCount = Number(stream.pose_count ?? livePoseStreamCount ?? 0);
      const expectedCount = Number(stream.expected_count ?? poseStreamMeta?.expected_count ?? 0);
      liveReplayStageDetail = expectedCount > 0
        ? `${liveReplayMessage || "TSolve online localization running"} · ${poseCount}/${expectedCount} R,t updates`
        : (liveReplayMessage || "TSolve online localization running");
    } else if (state.drone?.status === "done") {
      liveReplayStageDetail = state.drone?.message || "Live TSolve path ready";
    } else if (state.drone?.status === "error") {
      liveReplayStageDetail = state.drone?.message || "Live TSolve path failed";
    }
    if (currentScreen === "demo" || liveReplayInFlight) renderReplayTabs();
    const pipelineActive = activeJobStates.has(state.map?.status) || activeJobStates.has(state.drone?.status);
    pipelineStatus?.classList.toggle("is-active", pipelineActive);
    const mapLog = state.map?.log || [];
    const droneLog = state.drone?.log || [];
    jobLog.textContent = [...mapLog.slice(-8), ...droneLog.slice(-10)].join("\n") || "Create a map, then upload a drone video to simulate live TSolve localization.";
    const liveStatus = state.map?.status;
    const livePreview = state.map?.live_preview;
    const framesSaved = Number(state.map?.frames_saved || 0);
    const showLiveMapping = Boolean(livePreview && ["queued", "running", "stopping", "error"].includes(liveStatus));
    liveMappingPanel?.classList.toggle("hidden", !showLiveMapping);
    if (showLiveMapping) {
      if (liveCameraFeed && livePreview) {
        liveCameraFeed.src = `${livePreview}?t=${Date.now()}`;
      }
      if (liveMapCaption) {
        liveMapCaption.textContent = liveStatus === "stopping"
          ? `${framesSaved} frames captured. Closing camera and starting reconstruction.`
          : `${framesSaved} frames captured. Move the camera slowly, then press Stop Mapping.`;
      }
      drawLiveBuildPreview(framesSaved, liveStatus);
    }

    const previousMapStatus = lastMapStatus;
    const previousDroneStatus = lastDroneStatus;
    const mapDoneNow = state.map?.status === "done" && previousMapStatus && previousMapStatus !== "done";
    const droneDoneNow = state.drone?.status === "done" && (pendingLiveReplayOpen || (previousDroneStatus && previousDroneStatus !== "done"));
    const droneErroredNow = state.drone?.status === "error" && pendingLiveReplayOpen;
    lastMapStatus = state.map?.status || null;
    lastDroneStatus = state.drone?.status || null;
    if (liveReplayInFlight && renderStarted && !liveReplayWaitingViewPrepared) {
      await loadViewerData(false, currentMapEntry);
      liveReplayWaitingViewPrepared = true;
      renderStartPreview();
    }
    if (liveReplayInFlight && renderStarted) {
      await loadLiveReplayPartial(state.drone?.live_stream || null);
    }
    if (liveReplayInFlight && currentScreen !== "demo" && renderStarted) {
      showDemo();
    }
    if (liveReplayInFlight && scene && room && !poses.length) {
      const title = currentMapEntry?.title || "Selected map";
      const scanLine = displayPointSummaryLine();
      stats.innerHTML = `${title}<br>${scene.points3D.length} COLMAP map points<br>${scanLine}${scene.map_cameras.length} map cameras<br>Live TSolve initializing<br>${liveReplayMessage}`;
    }
    if ((mapDoneNow || droneDoneNow) && renderStarted) {
      await loadViewerData(Boolean(droneDoneNow), currentMapEntry);
      renderReplayTabs();
      renderStartPreview();
      if (droneDoneNow && pendingLiveReplayOpen && poses.length) {
        if (state.drone?.live_stream?.live_atlas) setLiveFrameMode(false);
        if (state.drone?.live_stream?.live_atlas) liveAtlasPreviewActive = false;
        liveReplayInFlight = false;
        liveCurrentPoseOverride = null;
        liveReplayWaitingViewPrepared = false;
        liveReplayStartedAt = 0;
        pendingLiveReplayOpen = false;
        pendingLiveReplayMapId = null;
        uploadStatus.textContent = `Live TSolve replay ready: ${currentMapEntry?.title || "selected map"}`;
        showDemo();
      } else if (droneDoneNow && pendingLiveReplayOpen) {
        if (state.drone?.live_stream?.live_atlas) setLiveFrameMode(false);
        if (state.drone?.live_stream?.live_atlas) liveAtlasPreviewActive = false;
        liveReplayInFlight = false;
        liveCurrentPoseOverride = null;
        liveReplayWaitingViewPrepared = false;
        liveReplayStartedAt = 0;
        pendingLiveReplayOpen = false;
        pendingLiveReplayMapId = null;
        uploadStatus.textContent = "Live replay finished, but no TSolve poses were produced for this video.";
      }
    }
    if (droneErroredNow) {
      if (state.drone?.live_stream?.live_atlas) setLiveFrameMode(false);
      if (state.drone?.live_stream?.live_atlas) liveAtlasPreviewActive = false;
      liveReplayInFlight = false;
      liveCurrentPoseOverride = null;
      liveReplayWaitingViewPrepared = false;
      liveReplayStartedAt = 0;
      pendingLiveReplayOpen = false;
      pendingLiveReplayMapId = null;
    }
  } catch {
    mapStatus.textContent = "Map: local backend not connected";
    droneStatus.textContent = "Drone replay: start scripts/atlas_app_server.py";
  } finally {
    liveStatusPollBusy = false;
  }
}

setInterval(pollDjiLivePreview, 1000);
pollDjiLivePreview();

document.getElementById("create-map").addEventListener("click", openMapModal);
document.getElementById("close-map-modal").addEventListener("click", closeMapModal);
mapModal?.addEventListener("click", event => {
  if (event.target === mapModal) closeMapModal();
});
document.getElementById("close-video-library-modal")?.addEventListener("click", hideVideoLibrary);
videoLibraryModal?.addEventListener("click", event => {
  if (event.target === videoLibraryModal) hideVideoLibrary();
});
collapseLibraryButton?.addEventListener("click", () => {
  const collapsed = !libraryPanel?.classList.contains("is-collapsed");
  libraryPanel?.classList.toggle("is-collapsed", collapsed);
  collapseLibraryButton.textContent = collapsed ? "Expand" : "Minimize";
  collapseLibraryButton.setAttribute("aria-expanded", String(!collapsed));
});
collapseConsoleButton?.addEventListener("click", () => {
  const collapsed = !pipelineStatus?.classList.contains("is-collapsed");
  pipelineStatus?.classList.toggle("is-collapsed", collapsed);
  collapseConsoleButton.textContent = collapsed ? "Expand" : "Minimize";
  collapseConsoleButton.setAttribute("aria-expanded", String(!collapsed));
});
document.getElementById("live-map").addEventListener("click", () => runUi(async () => {
  closeMapModal();
  uploadStatus.textContent = "Live map capture started";
  liveMappingPanel?.classList.remove("hidden");
  drawLiveBuildPreview(0, "queued");
  await postJson("/api/map/live", { duration: 75, fps: 1.5, camera_index: 0 });
  await pollStatus();
}));
stopMapping?.addEventListener("click", () => runUi(async () => {
  uploadStatus.textContent = "Stopping live map capture";
  await postJson("/api/map/stop", {});
  await pollStatus();
}));
mapUpload?.addEventListener("change", event => runUi(async () => {
  const files = [...(event.target.files || [])];
  if (!files.length) return;
  uploadStatus.textContent = `Uploading ${files.length} map video${files.length === 1 ? "" : "s"}`;
  await uploadVideos("/api/map/upload", files);
  await pollStatus();
}));
mapVideoUpload?.addEventListener("change", event => runUi(async () => {
  const files = [...(event.target.files || [])];
  if (!files.length) return;
  closeMapModal();
  uploadStatus.textContent = `Uploading ${files.length} map video${files.length === 1 ? "" : "s"} for one combined COLMAP map`;
  await uploadVideos("/api/map/upload", files);
  await pollStatus();
}));
demoDroneUpload?.addEventListener("change", event => runUi(async () => {
  const file = event.target.files?.[0];
  if (!file) return;
  const mapId = currentMapEntry?.id || mapLibraryData?.selected_map_id || "default_demo";
  await startDroneReplayUpload(file, mapId);
  event.target.value = "";
}));
startLiveAtlasButton?.addEventListener("click", () => runUi(startLiveAtlas));
stopLiveAtlasButton?.addEventListener("click", () => runUi(stopLiveAtlas));
liveAtlasFps?.addEventListener("change", updateLiveControlSummary);
window.addEventListener("resize", renderStartPreview);
updateLiveControlSummary();
setInterval(pollStatus, 2000);
setInterval(() => {
  if (liveReplayInFlight || pendingLiveReplayOpen) pollStatus();
}, 500);

window.TSOLVE_VIEWER = {
  getCurrentPose: () => closestPose(),
  getHeadingForPose: pose => headingForPose(pose),
  projectRoomPoint: rxyz => project(rxyz),
  getRoom: () => room,
  getView: () => view,
  getDroneModel: () => droneModel,
};

init().catch(err => {
  stats.textContent = `failed to load viewer data: ${err}`;
  console.error(err);
});
