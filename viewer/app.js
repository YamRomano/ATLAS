const canvas = document.getElementById("map");
let ctx = canvas.getContext("2d");
const launchParams = new URLSearchParams(window.location.search);
const fleetEmbedDroneId = String(launchParams.get("fleet-drone") || "").trim();
const fleetEmbedMode = launchParams.get("fleet-embed") === "1" && Boolean(fleetEmbedDroneId);
if (fleetEmbedMode) document.body.classList.add("fleet-map-embed");
let currentRenderedPose = null;
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
const diagState = document.getElementById("diag-state");
const diagFrameMethod = document.getElementById("diag-frame-method");
const diagReason = document.getElementById("diag-reason");
const diagExtracted = document.getElementById("diag-extracted");
const diagMatched = document.getElementById("diag-matched");
const diagFlowInput = document.getElementById("diag-flow-input");
const diagTracked = document.getElementById("diag-tracked");
const diagPnp = document.getElementById("diag-pnp");
const diagSelected = document.getElementById("diag-selected");
const diagPruned = document.getElementById("diag-pruned");
const diagTotalMs = document.getElementById("diag-total-ms");
const diagFrameLoadMs = document.getElementById("diag-frame-load-ms");
const diagHeadingFlowMs = document.getElementById("diag-heading-flow-ms");
const diagFeatureMs = document.getElementById("diag-feature-ms");
const diagMatchMs = document.getElementById("diag-match-ms");
const diagRegisterMs = document.getElementById("diag-register-ms");
const diagFlowMs = document.getElementById("diag-flow-ms");
const diagVisualRouteMs = document.getElementById("diag-visual-route-ms");
const diagVisualHeadingMs = document.getElementById("diag-visual-heading-ms");
const diagRouteLogicMs = document.getElementById("diag-route-logic-ms");
const diagLocalRecoveryMs = document.getElementById("diag-local-recovery-ms");
const diagCaseBuildMs = document.getElementById("diag-case-build-ms");
const diagCaseOutputMs = document.getElementById("diag-case-output-ms");
const diagTsolveMs = document.getElementById("diag-tsolve-ms");
const diagBackgroundApplyMs = document.getElementById("diag-background-apply-ms");
const diagPoseUpdateMs = document.getElementById("diag-pose-update-ms");
const diagStreamPublishMs = document.getElementById("diag-stream-publish-ms");
const diagPaceMs = document.getElementById("diag-pace-ms");
const diagBackgroundWorkerMs = document.getElementById("diag-background-worker-ms");
const diagOtherMs = document.getElementById("diag-other-ms");
const demoApp = document.getElementById("demo-app");
const startPage = document.getElementById("start-page");
const enemyPage = document.getElementById("enemy-page");
const fleetPage = document.getElementById("fleet-page");
const startPreview = document.getElementById("start-preview");
const navBack = document.getElementById("nav-back");
const atlasHome = document.getElementById("atlas-home");
const atlasScreenLabel = document.getElementById("atlas-screen-label");
const enemyLabButton = document.getElementById("enemy-lab-button");
const fleetMonitorButton = document.getElementById("fleet-monitor-button");
const fleetDroneSelect = document.getElementById("fleet-drone-select");
const fleetMapSelect = document.getElementById("fleet-map-select");
const fleetPatrolSelect = document.getElementById("fleet-patrol-select");
const fleetFpsSelect = document.getElementById("fleet-fps-select");
const fleetDispatchButton = document.getElementById("fleet-dispatch");
const fleetDispatchStatus = document.getElementById("fleet-dispatch-status");
const fleetStopAllButton = document.getElementById("fleet-stop-all");
const fleetDroneNameInput = document.getElementById("fleet-drone-name");
const fleetDroneIpInput = document.getElementById("fleet-drone-ip");
const fleetSaveDroneButton = document.getElementById("fleet-save-drone");
const fleetSwitcher = document.getElementById("fleet-switcher");
const fleetOverviewGrid = document.getElementById("fleet-overview-grid");
const fleetOverviewCount = document.getElementById("fleet-overview-count");
const fleetFocus = document.getElementById("fleet-focus");
const fleetSummaryRegistered = document.getElementById("fleet-summary-registered");
const fleetSummaryActive = document.getElementById("fleet-summary-active");
const fleetSummaryAirborne = document.getElementById("fleet-summary-airborne");
const fleetSummaryAttention = document.getElementById("fleet-summary-attention");
const fleetFocusOverline = document.getElementById("fleet-focus-overline");
const fleetFocusName = document.getElementById("fleet-focus-name");
const fleetFocusAssignment = document.getElementById("fleet-focus-assignment");
const fleetFocusBadge = document.getElementById("fleet-focus-badge");
const fleetLivePreview = document.getElementById("fleet-live-preview");
const fleetLiveIndicator = document.getElementById("fleet-live-indicator");
const fleetLiveFrameCount = document.getElementById("fleet-live-frame-count");
const fleetMetricLocalization = document.getElementById("fleet-metric-localization");
const fleetMetricPoses = document.getElementById("fleet-metric-poses");
const fleetMetricMap = document.getElementById("fleet-metric-map");
const fleetMetricPatrol = document.getElementById("fleet-metric-patrol");
const fleetMetricBridge = document.getElementById("fleet-metric-bridge");
const fleetMetricAction = document.getElementById("fleet-metric-action");
const fleetTakeoffButton = document.getElementById("fleet-takeoff");
const fleetStartPatrolButton = document.getElementById("fleet-start-patrol");
const fleetHoverButton = document.getElementById("fleet-hover");
const fleetLandButton = document.getElementById("fleet-land");
const fleetEndSessionButton = document.getElementById("fleet-end-session");
const fleetSmartLog = document.getElementById("fleet-smart-log");
const fleetLogCount = document.getElementById("fleet-log-count");
const enemyDroneList = document.getElementById("enemy-drone-list");
const enemyNameInput = document.getElementById("enemy-name");
const enemyVideoUpload = document.getElementById("enemy-video-upload");
const enemyUploadSubmit = document.getElementById("enemy-upload-submit");
const enemyUploadStatus = document.getElementById("enemy-upload-status");
const enemyModelStatus = document.getElementById("enemy-model-status");
const enemyModelNote = document.getElementById("enemy-model-note");
const enemyPrepareAllButton = document.getElementById("enemy-prepare-all");
const enemyTrainModelButton = document.getElementById("enemy-train-model");
const enemyTrainEpochs = document.getElementById("enemy-train-epochs");
const enemyTrainImgsz = document.getElementById("enemy-train-imgsz");
const enemyRefreshButton = document.getElementById("enemy-refresh");
const enemyAnnotationProfile = document.getElementById("enemy-annotation-profile");
const enemyExtractFramesButton = document.getElementById("enemy-extract-frames");
const enemyAnnotationStatus = document.getElementById("enemy-annotation-status");
const enemyFrameStrip = document.getElementById("enemy-frame-strip");
const enemyAnnotationCanvas = document.getElementById("enemy-annotation-canvas");
const enemySaveBoxButton = document.getElementById("enemy-save-box");
const enemyTrackBoxButton = document.getElementById("enemy-track-box");
const enemyCopyPrevBoxButton = document.getElementById("enemy-copy-prev-box");
const enemySkipFrameButton = document.getElementById("enemy-skip-frame");
const enemyNegativeFrameButton = document.getElementById("enemy-negative-frame");
const enemyClearBoxButton = document.getElementById("enemy-clear-box");
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
const patrolImportModal = document.getElementById("patrol-import-modal");
const patrolImportSubtitle = document.getElementById("patrol-import-subtitle");
const patrolImportList = document.getElementById("patrol-import-list");
const importPatrolButton = document.getElementById("import-patrol");
const mapUpload = document.getElementById("map-upload");
const mapVideoUpload = document.getElementById("map-video-upload");
const demoDroneUpload = document.getElementById("demo-drone-upload");
const liveAtlasPhoneIp = document.getElementById("live-atlas-phone-ip");
const phoneIpOptions = document.getElementById("phone-ip-options");
const phoneIpSelect = document.getElementById("phone-ip-select");
const savePhoneIpButton = document.getElementById("save-phone-ip");
const liveAtlasFps = document.getElementById("live-atlas-fps");
const liveAtlasPatrolSelect = document.getElementById("live-atlas-patrol");
const droneHeadingTrimSelect = document.getElementById("drone-heading-trim");
const droneHeadingTrimValue = document.getElementById("drone-heading-trim-value");
const useModelHeadingForFlightInput = document.getElementById("use-model-heading-for-flight");
const enemyLiveDetectorState = document.getElementById("enemy-live-detector-state");
const enemyLiveDetection = document.getElementById("enemy-live-detection");
const enemyResponseStatus = document.getElementById("enemy-response-status");
const enemyDetectionEnabledInput = document.getElementById("enemy-detection-enabled");
const enemyConfirmLockButton = document.getElementById("enemy-confirm-lock");
const enemyStartPursuitButton = document.getElementById("enemy-start-pursuit");
const enemyClearAlertButton = document.getElementById("enemy-clear-alert");
const enemyRangeStatus = document.getElementById("enemy-range-status");
const enemyMeasuredClearanceInput = document.getElementById("enemy-measured-clearance");
const enemyStopClearanceInput = document.getElementById("enemy-stop-clearance");
const enemyPursuitYawDirection = document.getElementById("enemy-pursuit-yaw-direction");
const enemySaveRangeSampleButton = document.getElementById("enemy-save-range-sample");
const enemyValidateRangeButton = document.getElementById("enemy-validate-range");
const enemyResetRangeButton = document.getElementById("enemy-reset-range");
const enemyRangeHelp = document.getElementById("enemy-range-help");
const liveLocalizationControl = document.getElementById("live-localization-control");
const liveControlSummary = document.getElementById("live-control-summary");
const pinLiveControlButton = document.getElementById("pin-live-control");
const startLiveAtlasButton = document.getElementById("start-live-atlas");
const stopLiveAtlasButton = document.getElementById("stop-live-atlas");
const takeoffHeightInput = document.getElementById("takeoff-height-m");
const djiTakeoffButton = document.getElementById("dji-takeoff");
const djiLandButton = document.getElementById("dji-land");
const djiEmergencyHoverButton = document.getElementById("dji-emergency-hover");
const guidedMotionEnable = document.getElementById("guided-motion-enable");
const djiCommandStatus = document.getElementById("dji-command-status");
const confirmLocalizationButton = document.getElementById("confirm-localization");
const correctInitialPositionButton = document.getElementById("correct-initial-position");
const resetInitialPositionButton = document.getElementById("reset-initial-position");
const initialPositionStatus = document.getElementById("initial-position-status");
const localizationGateStatus = document.getElementById("localization-gate-status");
const droneControlPanel = document.getElementById("drone-control-panel");
const missionSpeedSelect = document.getElementById("mission-speed");
const planMissionButton = document.getElementById("plan-mission");
const patrolControlPanel = document.getElementById("patrol-control-panel");
const editPatrolButton = document.getElementById("edit-patrol");
const patrolNameInput = document.getElementById("patrol-name");
const patrolSpeedSelect = document.getElementById("patrol-speed");
const patrolAltitudeInput = document.getElementById("patrol-altitude-m");
const patrolDwellSelect = document.getElementById("patrol-dwell-s");
const patrolScanModeSelect = document.getElementById("patrol-scan-mode");
const patrolModeSelect = document.getElementById("patrol-mode");
const patrolLoopInput = document.getElementById("patrol-loop");
const clearPatrolButton = document.getElementById("clear-patrol");
const validatePatrolButton = document.getElementById("validate-patrol");
const startPatrolButton = document.getElementById("start-patrol");
const stopPatrolButton = document.getElementById("stop-patrol");
const patrolStatus = document.getElementById("patrol-status");
const patrolCommandList = document.getElementById("patrol-command-list");
const savedPatrolList = document.getElementById("saved-patrol-list");
const newPatrolButton = document.getElementById("new-patrol");
const replayTabs = document.getElementById("replay-tabs");
const replayTabList = document.getElementById("replay-tab-list");
const simulateLivePathButton = document.getElementById("simulate-live-path");
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
const toggleCoverageRiskButton = document.getElementById("toggle-coverage-risk");
const selectTargetButton = document.getElementById("select-target");
const clearTargetButton = document.getElementById("clear-target");
const startMissionButton = document.getElementById("start-mission");
const targetStatus = document.getElementById("target-status");
const missionCommandList = document.getElementById("mission-command-list");
const safetyBarrierPanel = document.getElementById("safety-barrier-panel");
const addBarrierButton = document.getElementById("add-barrier");
const adjustWallsButton = document.getElementById("adjust-walls");
const undoWallEditButton = document.getElementById("undo-wall-edit");
const saveWallAdjustmentsButton = document.getElementById("save-wall-adjustments");
const cancelBarrierButton = document.getElementById("cancel-barrier");
const clearBarriersButton = document.getElementById("clear-barriers");
const barrierStatus = document.getElementById("barrier-status");
const barrierList = document.getElementById("barrier-list");
const barrierClearanceInput = document.getElementById("barrier-clearance-m");
const barrierColorInput = document.getElementById("barrier-color");
const barrierOpacityInput = document.getElementById("barrier-opacity");
const safetyTabWallsButton = document.getElementById("safety-tab-walls");
const safetyTabObstaclesButton = document.getElementById("safety-tab-obstacles");
const wallTools = document.getElementById("wall-tools");
const obstacleTools = document.getElementById("obstacle-tools");
const addObstacleButton = document.getElementById("add-obstacle");
const finishObstacleButton = document.getElementById("finish-obstacle");
const cancelObstacleButton = document.getElementById("cancel-obstacle");
const clearObstaclePointsButton = document.getElementById("clear-obstacle-points");
const undoObstacleEditButton = document.getElementById("undo-obstacle-edit");
const clearObstaclesButton = document.getElementById("clear-obstacles");
const obstacleStatus = document.getElementById("obstacle-status");
const obstacleList = document.getElementById("obstacle-list");
const obstacleClearanceInput = document.getElementById("obstacle-clearance-m");
const obstacleColorInput = document.getElementById("obstacle-color");
const obstacleOpacityInput = document.getElementById("obstacle-opacity");

let scene = null;
let poses = [];
let poseStreamMeta = null;
let scan = null;
let room = null;
let droneModel = null;
let droneModelPromise = null;
let mapLibraryData = { selected_map_id: "default_demo", maps: [] };
let enemyLibraryData = { model_status: "not_trained", enemies: [] };
let selectedEnemyId = "";
let selectedEnemyFrameId = "";
let enemyAnnotationImage = new Image();
let enemyAnnotationImageReady = false;
let enemyAnnotationImageFrameId = "";
let enemyBoxDraft = null;
let enemyBoxDrag = null;
let enemyCanvasRectCache = null;
let currentMapEntry = null;
let renderStarted = false;
let lastMapStatus = null;
let lastDroneStatus = null;
let currentScreen = "start";
let screenHistory = [];
let fleetData = { drones: [], maps: [], summary: {} };
let selectedFleetDroneId = "";
let fleetPollTimer = null;
let fleetEmbedSession = null;
let fleetEmbedAssignmentKey = "";
let fleetEmbedPollBusy = false;
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
let livePosePollBusy = false;
let liveVideoWaitingForFirstPose = false;
let liveVideoSyncedToFirstPose = false;
let liveCurrentPoseOverride = null;
let liveFrameLockedQueue = [];
let liveFrameLockedDisplayedPose = null;
let liveFrameLockedReplayId = "";
let liveFrameLockedQueuedCount = 0;
let liveFrameLockedDisplayedCount = 0;
let liveFrameLockedDroppedCount = 0;
let liveFrameLockedNextAtMs = 0;
let liveFrameLockedStream = null;
let liveFramePreloadCache = new Map();
let liveReplayCompletionPending = false;
let liveRotationPositionAnchor = null;
let liveFrameMode = false;
let liveAtlasPreviewActive = false;
let pendingLivePatrolId = "";
let lastDjiControlStatusId = "";
let lastDjiControlStatusKey = "";
let latestDjiLiveStatus = null;
let pendingDjiControlAck = null;
let lastReplayFrameUrl = "";
let lastLiveFrameUrl = "";
let replayFramePlaybackEnabled = false;
let replayFramePlaybackRaf = 0;

const PHONE_IP_STORAGE_KEY = "atlas.savedPhoneIps";
const DEFAULT_PHONE_IPS = ["192.168.50.235"];
const LIVE_CONTROL_PIN_STORAGE_KEY = "atlas.liveControlPinned";
const LIVE_CONTROL_SECTIONS_STORAGE_KEY = "atlas.liveControlCollapsedSections";
const DRONE_HEADING_TRIM_STORAGE_KEY = "atlas.droneHeadingTrimDeg";
const ENEMY_DETECTION_ENABLED_STORAGE_KEY = "atlas.enemyDetectionEnabled";
const DEFAULT_DRONE_HEADING_TRIM_DEG = -45;
// Live localization is polled in bursts.  Keeping six timed poses queued made
// the camera/model pair visibly trail the actual aircraft even though each
// displayed frame was internally aligned.  Retain only a short jitter buffer
// and drop a backlog to its newest aligned frame/pose pair.
const LIVE_FRAME_MAX_BUFFER = 12;
const LIVE_FRAME_TARGET_BUFFER = 1;
// A recorded-flight simulation must preserve the short physical RC pulses;
// dropping a burst to catch up can erase the only frames that prove motion.
// Real DJI display remains on the small low-latency buffer above.  The
// simulation catches up by rendering faster, while retaining up to 18 seconds
// of exact frame/pose pairs instead of deleting them.
const SIMULATED_LIVE_FRAME_MAX_BUFFER = 180;
const SIMULATED_LIVE_FRAME_TARGET_BUFFER = 24;
let pathPlaybackActive = false;
let pathPlaybackStartWallMs = 0;
let pathPlaybackStartTimeSec = 0;
let pathPlaybackEndTimeSec = 0;
let replayFrameHoldTimeSec = null;
const MISSION_AUTONOMY_SPEED_LIMIT_MPS = 0.16;
const PATROL_AUTONOMY_SPEED_LIMIT_MPS = 0.12;
const PATROL_SCAN_YAW_DEG = 90;
const MISSION_RELOCALIZE_HOVER_SECONDS = 1.5;
const FLIGHT_SAFETY_PULSE_BUFFER_M = 0.30;
const ENEMY_ALERT_MIN_CONFIDENCE = 0.35;
const ENEMY_HOVER_COOLDOWN_MS = 7000;
const ENEMY_LOCKON_MAX_YAW_DEG = 24;
const ENEMY_LOCKON_FOLLOW_TARGET_HEIGHT = 0.28;
const ENEMY_CONFIRM_HITS = 3;
const ENEMY_CONFIRM_WINDOW = 5;
let enemyDetectionHistory = [];
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
  showCoverageRisk: true,
};
let missionTarget = null;
let missionSelecting = false;
let missionDraggingTarget = false;
let missionDragMoved = false;
let patrolPoints = [];
let patrolSelecting = false;
let patrolDraggingIndex = -1;
let patrolDragMoved = false;
let patrolPointHover = null;
let patrolCoverageCache = { room: null, key: "", samples: [] };
let editingPatrolId = null;
let activePatrolId = null;
let firstLocalizationConfirmed = false;
let initialPositionSelecting = false;
let initialPoseOffsetRoom = [0, 0, 0];
let plannedMission = null;
let plannedPatrol = null;
let activeExecutionPatrolRoute = null;
let activePatrolExecutionContext = null;
let interruptedPatrolExecutionContext = null;
let enemyPursuitResumeContext = null;
const handledEnemyPursuitResumeIds = new Set();
let enemyPatrolResumeInFlight = false;
let enemyTargetSuppressedUntilClear = false;
let manualPatrolRecording = null;
let enemyAlertState = {
  active: false,
  hoverSent: false,
  hoverSentAt: 0,
  lockOnSentAt: 0,
  target: null,
  frame: "",
  updatedAt: 0,
};
let enemyPursuitInFlight = false;
let enemyPursuitCommandId = "";
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
let selectedBarrierId = null;
let safetyBarrierMode = "walls";
let obstacleEditing = false;
let obstacleDraft = null;
let selectedObstacleId = null;
let obstaclePointHover = null;
let obstaclePointDrag = null;
let obstacleTransformHover = null;
let obstacleTransformDrag = null;
let obstacleDragMoved = false;
let obstacleClickSuppress = false;
let wallUndoStack = [];
let obstacleUndoStack = [];
let patrolPointSafetyIssues = new Map();
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
  b.radius = Math.max(
    ...b.max.map((v, i) => Math.abs(v - b.center[i])),
    ...b.min.map((v, i) => Math.abs(v - b.center[i])),
    1e-6
  );
  return b;
}

function orientedStructureBox(points, fallbackBounds, floorY, forcedYaw = null) {
  const raw = (points || []).map(p => p.rxyz || p).filter(p =>
    Array.isArray(p) &&
    p.length >= 3 &&
    Number.isFinite(p[0]) &&
    Number.isFinite(p[1]) &&
    Number.isFinite(p[2])
  );
  if (raw.length < 24 || !fallbackBounds) return null;

  const yLow = quantile(raw.map(p => p[1]), 0.06);
  const yHigh = quantile(raw.map(p => p[1]), 0.88);
  const horizontal = raw.filter(p => p[1] >= yLow && p[1] <= yHigh);
  const sample = horizontal.length >= 24 ? horizontal : raw;

  const cx = median(sample.map(p => p[0]));
  const cz = median(sample.map(p => p[2]));
  let cxx = 0, cxz = 0, czz = 0;
  for (const p of sample) {
    const x = p[0] - cx;
    const z = p[2] - cz;
    cxx += x * x;
    cxz += x * z;
    czz += z * z;
  }
  cxx /= sample.length;
  cxz /= sample.length;
  czz /= sample.length;

  // Maps with an explicit room alignment are already rotated into their
  // intended X/Z frame. Running another PCA here rotates only the skeleton
  // away from those axes and creates misleading triangular gaps.
  const yaw = Number.isFinite(forcedYaw)
    ? forcedYaw
    : 0.5 * Math.atan2(2 * cxz, cxx - czz);
  const ux = Math.cos(yaw), uz = Math.sin(yaw);
  const vx = -uz, vz = ux;
  const us = sample.map(p => (p[0] - cx) * ux + (p[2] - cz) * uz);
  const vs = sample.map(p => (p[0] - cx) * vx + (p[2] - cz) * vz);
  let u0 = quantile(us, 0.015), u1 = quantile(us, 0.985);
  let v0 = quantile(vs, 0.015), v1 = quantile(vs, 0.985);
  const uPad = Math.max(0.08, (u1 - u0) * 0.055);
  const vPad = Math.max(0.08, (v1 - v0) * 0.055);
  u0 -= uPad; u1 += uPad;
  v0 -= vPad; v1 += vPad;

  const toRoom = (u, v, y) => [cx + ux * u + vx * v, y, cz + uz * u + vz * v];
  const y0 = Math.max(floorY, fallbackBounds.min[1]);
  const y1 = Math.max(y0 + 0.24, fallbackBounds.max[1]);
  const bottom = [
    toRoom(u0, v0, y0),
    toRoom(u1, v0, y0),
    toRoom(u1, v1, y0),
    toRoom(u0, v1, y0),
  ];
  const top = bottom.map(p => [p[0], y1, p[2]]);
  // cx/cz is the median of the sampled point density, not the geometric
  // center of the room.  A wall with many reconstructed features can pull
  // that median far away from the footprint center and make mouse rotation
  // orbit around the wrong anchor.  Use the midpoint of the oriented box we
  // actually draw so the wireframe, map points, and mesh share one stable
  // visual pivot.
  const center = toRoom(0.5 * (u0 + u1), 0.5 * (v0 + v1), 0.5 * (y0 + y1));
  return { bottom, top, yaw, center };
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
    ? Math.min(1.35, Math.max(0.42, 0.85 * dt + 0.28))
    : 0.75;
}

function posesAreLocallyStable(a, b) {
  if (!a?.rcenter || !b?.rcenter) return false;
  return norm(sub(b.rcenter, a.rcenter)) <= poseTrackMaxStep(a, b);
}

function isRealPose(pose) {
  return Boolean(pose?.success && !pose?.held_pose && pose?.rcenter);
}

function markLegacyHeldPoses(roomPoses) {
  let lastReal = null;
  for (const pose of roomPoses) {
    if (isRealPose(pose)) {
      lastReal = pose;
      continue;
    }
    const totalMs = Number(pose?.total_ms);
    const looksLikeHeld =
      pose?.success === false &&
      !pose?.held_pose &&
      pose?.rcenter &&
      lastReal?.rcenter &&
      Number.isFinite(totalMs) &&
      Math.abs(totalMs) < 1e-6 &&
      (!pose.rejected_reason || norm(sub(pose.rcenter, lastReal.rcenter)) < 1e-6);
    if (!looksLikeHeld) continue;
    pose.held_pose = true;
    pose.output_rejected = true;
    pose.hold_reason = pose.hold_reason || "legacy_held_pose";
  }
  return roomPoses;
}

function filterReplayPoseTrack(roomPoses, options = {}) {
  const out = chronologicalPoseObservations(roomPoses).map(p => ({
    ...p,
    rawRcenter: p.rcenter ? p.rcenter.slice() : null,
  }));

  if (options.filterTrack === false) {
    out.poseQuality = {
      total: out.length,
      accepted: out.filter(p => isRealPose(p)).length,
      rejected: out.filter(p => p.filtered).length,
    };
    return out;
  }

  const valid = out
    .map((pose, index) => ({ pose, index }))
    .filter(item => isRealPose(item.pose));

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
    pose.filtered = !keep && Boolean(pose.success && !pose.held_pose && pose.rawRcenter);
    pose.filter_reason = null;
    pose.trackSegment = 0;
    const refErr = poseReferenceError(pose);
    if (Number.isFinite(refErr)) pose.colmap_reference_error_m = refErr;
    if (keep) {
      accepted += 1;
    } else if (pose.rcenter && pose.success && !pose.held_pose) {
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

  const alignment = currentMapEntry?.room_alignment?.matrix;
  const hasExplicitAlignment = Array.isArray(alignment) && alignment.length === 3
    && alignment.every(row => Array.isArray(row) && row.length === 4 && row.every(Number.isFinite));
  let center = [0, 0, 0];
  let axisX = [1, 0, 0];
  let axisY = [0, 1, 0];
  let axisZ = [0, 0, 1];
  let transform;
  let transformDirection;
  if (hasExplicitAlignment) {
    transform = xyz => alignment.map(row => (
      row[0] * xyz[0] + row[1] * xyz[1] + row[2] * xyz[2] + row[3]
    ));
    transformDirection = dir => alignment.map(row => (
      row[0] * dir[0] + row[1] * dir[1] + row[2] * dir[2]
    ));
  } else {
    for (const p of sample) for (let i = 0; i < 3; i++) center[i] += p[i];
    for (let i = 0; i < 3; i++) center[i] /= Math.max(sample.length, 1);

    const C = covariance(sample, center);
    const e0 = powerEigen(C, [1, 0.2, 0.1]);
    const e1 = powerEigen(deflate(C, e0), [0.1, 1, 0.2]);
    axisX = normalize(e0.v);
    axisZ = normalize(sub(e1.v, mul(axisX, dot(e1.v, axisX))));
    const zSign = Number(currentMapEntry?.display_z_sign ?? -1) < 0 ? -1 : 1;
    axisZ = mul(axisZ, zSign);
    axisY = normalize(cross(axisZ, axisX));

    const rawTransform = xyz => {
      const d = sub(xyz, center);
      return [dot(d, axisX), dot(d, axisY), dot(d, axisZ)];
    };
    const pointY = cloud.slice(0, 5000).map(p => rawTransform(p)[1]);
    const camY = cameras.map(p => rawTransform(p)[1]);
    if (camY.length && median(camY) < median(pointY)) axisY = mul(axisY, -1);

    transform = xyz => {
      const d = sub(xyz, center);
      return [dot(d, axisX), dot(d, axisY), dot(d, axisZ)];
    };
    transformDirection = dir => [dot(dir, axisX), dot(dir, axisY), dot(dir, axisZ)];
  }

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
  const configuredStructureYawDeg = Number(currentMapEntry?.room_alignment?.structure_yaw_deg);
  const forcedStructureYaw = hasExplicitAlignment
    ? (Number.isFinite(configuredStructureYawDeg) ? configuredStructureYawDeg * Math.PI / 180 : 0)
    : null;
  const structureBox = orientedStructureBox(
    displayPoints,
    structureBounds,
    quantile(displayPoints.map(p => p.rxyz[1]), 0.05),
    forcedStructureYaw
  );
  // Camera centers can include badly registered frames far outside the room.
  // They must not pull the orbit pivot away from the visible structure once a
  // map has an explicit room alignment.
  const allRoom = [
    ...displayPoints.map(p => p.rxyz),
    ...(hasExplicitAlignment ? [] : cameras.map(transform)),
  ];
  const b = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  for (const p of allRoom) {
    for (let i = 0; i < 3; i++) {
      b.min[i] = Math.min(b.min[i], p[i]);
      b.max[i] = Math.max(b.max[i], p[i]);
    }
  }
  b.center = b.min.map((v, i) => 0.5 * (v + b.max[i]));
  if (hasExplicitAlignment && structureBox?.center) {
    b.center[0] = structureBox.center[0];
    b.center[2] = structureBox.center[2];
  }
  b.radius = Math.max(
    ...b.max.map((v, i) => Math.abs(v - b.center[i])),
    ...b.min.map((v, i) => Math.abs(v - b.center[i])),
    1e-6
  );
  const floorY = quantile(displayPoints.map(p => p.rxyz[1]), 0.05);
  const scanPoints = buildScanVisualPoints(b, floorY);
  const rawRoomPoses = poses
    .map((p, index) => ({
      ...p,
      _poseOrder: index,
      rcenter: Array.isArray(p.rcenter) ? p.rcenter : (p.center ? transform(p.center) : null),
    }))
    .sort((a, bPose) => {
      const ta = Number(a.time_sec);
      const tb = Number(bPose.time_sec);
      const aFinite = Number.isFinite(ta);
      const bFinite = Number.isFinite(tb);
      if (aFinite && bFinite && ta !== tb) return ta - tb;
      if (aFinite !== bFinite) return aFinite ? -1 : 1;
      return (a._poseOrder || 0) - (bPose._poseOrder || 0);
    });
  markLegacyHeldPoses(rawRoomPoses);
  const replay = activeReplay(currentMapEntry);
  const activeReplayIsLive = Boolean(
    replay?.id?.startsWith("dji_live_") ||
    replay?.id?.startsWith("video_live_") ||
    replay?.asset_base?.includes("/replays/dji_live_") ||
    replay?.asset_base?.includes("/replays/video_live_")
  );
  const livePoseStreamActive = Boolean(
    liveReplayInFlight ||
    pendingLiveReplayOpen ||
    poseStreamMeta?.complete === false ||
    activeReplayIsLive
  );
  const completedReplayDisplay = Boolean(
    !liveReplayInFlight &&
    !pendingLiveReplayOpen &&
    poseStreamMeta?.complete !== false
  );
  const recordedLiveFlight = replay?.kind === "simulated_live_tsolve_recorded_frames";
  const roomPoses = buildReplayDisplayPoses(rawRoomPoses, floorY, {
    // A finite recorded patrol ends on its last localized airborne frame. It
    // must not visually descend merely because the replay worker completed.
    applyLanding: completedReplayDisplay && !recordedLiveFlight,
    filterTrack: !livePoseStreamActive,
  });
  const poseQuality = roomPoses.poseQuality || {
    total: roomPoses.length,
    accepted: roomPoses.filter(p => isRealPose(p)).length,
    rejected: roomPoses.filter(p => p.filtered).length,
  };
  const routeYs = roomPoses.filter(p => isRealPose(p)).map(p => p.rcenter[1]);
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

  const firstWithHeading = roomPoses.find(p => isRealPose(p) && p.pathHeading);
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
    structureBox,
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

function updateLiveRoomPoseStream(sourcePoses) {
  if (!room || !Array.isArray(sourcePoses)) return false;
  const previousSourceCount = Number(room._liveSourcePoseCount || 0);
  const appendOnly = Boolean(
    previousSourceCount > 0 &&
    sourcePoses.length >= previousSourceCount &&
    Array.isArray(room.poses) &&
    room.poses.length === previousSourceCount
  );
  const sourceStart = appendOnly ? previousSourceCount : 0;
  const rawRoomPoses = sourcePoses
    .slice(sourceStart)
    .map((pose, index) => ({
      ...pose,
      _poseOrder: sourceStart + index,
      rcenter: Array.isArray(pose.rcenter)
        ? pose.rcenter.slice()
        : (pose.center ? room.transform(pose.center) : null),
    }));
  if (!appendOnly) rawRoomPoses.sort((a, b) => {
      const ta = Number(a.time_sec);
      const tb = Number(b.time_sec);
      const aFinite = Number.isFinite(ta);
      const bFinite = Number.isFinite(tb);
      if (aFinite && bFinite && ta !== tb) return ta - tb;
      if (aFinite !== bFinite) return aFinite ? -1 : 1;
      return (a._poseOrder || 0) - (b._poseOrder || 0);
    });
  // Legacy-held detection needs the preceding accepted pose when processing
  // only the new delta.  Do not clone/re-sort thousands of old poses on every
  // 100 ms live poll.
  const heldContext = appendOnly && room.poses.length
    ? [room.poses[room.poses.length - 1], ...rawRoomPoses]
    : rawRoomPoses;
  markLegacyHeldPoses(heldContext);
  const newRoomPoses = buildReplayDisplayPoses(rawRoomPoses, room.floorY, {
    applyLanding: false,
    filterTrack: false,
  });
  const oldAccepted = appendOnly
    ? room.poses.filter(pose => isRealPose(pose)).length
    : 0;
  room.poses = appendOnly ? room.poses.concat(newRoomPoses) : newRoomPoses;
  room._liveSourcePoseCount = sourcePoses.length;
  assignStablePathHeadings(room.poses, Math.max(0, oldAccepted - 14));
  room.poseQuality = {
    total: room.poses.length,
    accepted: room.poses.filter(pose => isRealPose(pose)).length,
    rejected: room.poses.filter(pose => pose.filtered).length,
  };
  // Height quantiles do not need to be re-sorted for every single-frame
  // append.  Refresh them on a reset and at coarse 100-pose checkpoints.
  if (!appendOnly || sourcePoses.length % 100 < rawRoomPoses.length) {
    const routeYs = room.poses.filter(pose => isRealPose(pose)).map(pose => pose.rcenter[1]);
    if (routeYs.length) {
    room.routeHeightBounds = {
      min: Math.min(room.floorY, quantile(routeYs, 0.03)),
      max: Math.max(room.floorY + 0.18, quantile(routeYs, 0.97)),
    };
    }
  }
  return true;
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
  if (toggleCoverageRiskButton) {
    toggleCoverageRiskButton.classList.toggle("active", Boolean(view.showCoverageRisk));
    toggleCoverageRiskButton.textContent = view.showCoverageRisk ? "Hide Coverage Risk" : "Coverage Risk";
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

function projectToViewport(rxyz) {
  const p = project(rxyz);
  const rect = canvas.getBoundingClientRect();
  return [rect.left + p[0], rect.top + p[1], p[2]];
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

function lerpOptionalVec(a, b, t, fallback = null) {
  if (Array.isArray(a) && Array.isArray(b) && a.length === b.length) {
    return lerpVec(a, b, t);
  }
  return Array.isArray(fallback) ? fallback.slice() : null;
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

function patrolList(entry = currentMapEntry) {
  return Array.isArray(entry?.patrols) ? entry.patrols : [];
}

function activeSavedPatrol(entry = currentMapEntry) {
  const patrols = patrolList(entry);
  return patrols.find(patrol => patrol.id === activePatrolId)
    || patrols.find(patrol => patrol.id === editingPatrolId)
    || patrols[0]
    || null;
}

function replayPoseCountText(replay, fallbackCounts = {}) {
  const counts = replay?.counts || fallbackCounts || {};
  const accepted = Number(counts.poses ?? counts.accepted ?? 0);
  const frames = Number(counts.frames ?? 0);
  const held = Number(counts.held ?? 0);
  if (["route_constrained_taught_baseline", "simulated_live_patrol_baseline"].includes(replay?.kind)) {
    return `${accepted} validated frame pose${accepted === 1 ? "" : "s"}`;
  }
  if (frames > accepted) {
    const heldText = held ? `, ${held} held` : "";
    return `${accepted}/${frames} real R,t${heldText}`;
  }
  return `${accepted} pose${accepted === 1 ? "" : "s"}`;
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

function replayUsesCapturedFrames(replay = activeReplay(currentMapEntry)) {
  return Boolean(replayQueryFrameBaseUrl(replay));
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
    stopPoseClockPlayback();
  }
}

function setLiveFrameStatus(text, visible = true) {
  if (!liveFrameStatus) return;
  liveFrameStatus.textContent = text || "";
  liveFrameStatus.classList.toggle("hidden", !visible || !liveFrameMode);
}

function sortedTimedPoses(sourcePoses = room?.poses || []) {
  return sourcePoses
    .filter(p => isRealPose(p) && Number.isFinite(Number(p.time_sec)))
    .sort((a, b) => Number(a.time_sec) - Number(b.time_sec));
}

function sortedTimedFramePoses(sourcePoses = room?.poses || []) {
  return sourcePoses
    .filter(p => p?.image_name && Number.isFinite(Number(p.time_sec)))
    .sort((a, b) => Number(a.time_sec) - Number(b.time_sec));
}

function replayFramePoseAt(timeSec) {
  const frames = sortedTimedFramePoses();
  if (!frames.length || !Number.isFinite(Number(timeSec))) return null;
  const t = Number(timeSec);
  let best = frames[0];
  let bestDt = Math.abs(Number(best.time_sec) - t);
  for (const frame of frames) {
    const dt = Math.abs(Number(frame.time_sec) - t);
    if (dt < bestDt) {
      best = frame;
      bestDt = dt;
    }
    if (Number(frame.time_sec) > t && dt > bestDt) break;
  }
  return best;
}

function firstPlayableReplayPose() {
  const timedFrames = sortedTimedFramePoses(room?.poses || []);
  if (timedFrames.length) return timedFrames[0];
  const timedPoses = sortedTimedPoses(room?.poses || []);
  if (timedPoses.length) return timedPoses[0];
  return (room?.poses || []).find(p => p?.success && !p?.held_pose && (p.rcenter || p.center))
    || (room?.poses || [])[0]
    || null;
}

function stopPoseClockPlayback() {
  pathPlaybackActive = false;
  replayFramePlaybackEnabled = false;
  if (replayFramePlaybackRaf) {
    cancelAnimationFrame(replayFramePlaybackRaf);
    replayFramePlaybackRaf = 0;
  }
}

function tickReplayFramePlayback() {
  if (!replayFramePlaybackEnabled) {
    replayFramePlaybackRaf = 0;
    return;
  }
  const framePose = replayFramePoseAt(currentReplayClockTime(room?.poses || []));
  if (!replayFramePlaybackEnabled) {
    replayFramePlaybackRaf = 0;
    return;
  }
  if (framePose) updateReplayFrameViewForPose(framePose);
  replayFramePlaybackRaf = requestAnimationFrame(tickReplayFramePlayback);
}

function startReplayFrameTicker() {
  if (replayFramePlaybackRaf) cancelAnimationFrame(replayFramePlaybackRaf);
  replayFramePlaybackRaf = requestAnimationFrame(tickReplayFramePlayback);
}

function currentReplayClockTime(good) {
  if (pathPlaybackActive) {
    const timed = sortedTimedFramePoses(room?.poses || []);
    const fallbackTimed = sortedTimedPoses(good);
    const clockSource = timed.length >= 2 ? timed : fallbackTimed;
    if (clockSource.length >= 2) {
      const first = Number(clockSource[0].time_sec);
      const last = Number(clockSource[clockSource.length - 1].time_sec);
      const elapsed = (performance.now() - pathPlaybackStartWallMs) / 1000;
      const t = Math.min(pathPlaybackStartTimeSec + elapsed, last);
      if (t >= last - 1e-3) {
        stopPoseClockPlayback();
      }
      return t;
    }
    stopPoseClockPlayback();
  }
  if (Number.isFinite(Number(replayFrameHoldTimeSec))) {
    return Number(replayFrameHoldTimeSec);
  }
  const t = Number(video.currentTime);
  return Number.isFinite(t) ? t : 0;
}

function startPoseClockPlayback() {
  const timedFrames = sortedTimedFramePoses();
  const timedPoses = sortedTimedPoses();
  const timed = timedFrames.length >= 2 ? timedFrames : timedPoses;
  if (timed.length < 2) {
    uploadStatus.textContent = "No timestamped TSolve frames are available to play.";
    return false;
  }
  const first = Number(timed[0].time_sec);
  const last = Number(timed[timed.length - 1].time_sec);
  if (!(last > first)) return false;
  replayFrameHoldTimeSec = null;
  pathPlaybackActive = true;
  replayFramePlaybackEnabled = true;
  pathPlaybackStartWallMs = performance.now();
  pathPlaybackStartTimeSec = first;
  pathPlaybackEndTimeSec = last;
  lastReplayFrameUrl = "";
  video.pause();
  const firstFrame = replayFramePoseAt(first);
  if (firstFrame) updateReplayFrameViewForPose(firstFrame, { force: true });
  startReplayFrameTicker();
  uploadStatus.textContent = timedFrames.length >= 2
    ? "Playing saved captured DJI frames with the TSolve pose track."
    : "Playing saved TSolve path.";
  return true;
}

function playCurrentReplay() {
  if (sidePanel) sidePanel.scrollTo({ top: 0, behavior: "smooth" });
  if (replayUsesCapturedFrames()) {
    startPoseClockPlayback();
    return;
  }
  const src = video.getAttribute("src");
  const canUseVideoNow = src && !liveFrameMode && Number.isFinite(Number(video.duration)) && Number(video.duration) > 0.05;
  if (canUseVideoNow) {
    stopPoseClockPlayback();
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
        stopPoseClockPlayback();
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

function poseObservationFrameIndex(pose) {
  const explicitRaw = pose?.frame_index;
  const explicit = Number(explicitRaw);
  if (explicitRaw !== null && explicitRaw !== undefined && explicitRaw !== "" && Number.isFinite(explicit)) {
    return explicit;
  }
  const raw = String(pose?.image_name || pose?.instance_id || "");
  const match = raw.match(/(\d+)(?:\.[^.]+)?$/);
  return match ? Number(match[1]) : -1;
}

function poseObservationOrderKey(pose) {
  const frameIndex = poseObservationFrameIndex(pose);
  const receivedUnix = Number(pose?.received_unix);
  const timeSec = Number(pose?.time_sec);
  return [
    frameIndex >= 0 ? 1 : 0,
    frameIndex,
    Number.isFinite(receivedUnix) ? 1 : 0,
    Number.isFinite(receivedUnix) ? receivedUnix : -Infinity,
    Number.isFinite(timeSec) ? 1 : 0,
    Number.isFinite(timeSec) ? timeSec : -Infinity,
  ];
}

function comparePoseObservations(a, b) {
  const ak = poseObservationOrderKey(a);
  const bk = poseObservationOrderKey(b);
  for (let i = 0; i < ak.length; i += 1) {
    if (ak[i] !== bk[i]) return ak[i] - bk[i];
  }
  return 0;
}

function chronologicalPoseObservations(poses) {
  if (!Array.isArray(poses)) return [];
  return poses
    .map((pose, index) => ({ pose, index }))
    .sort((a, b) => comparePoseObservations(a.pose, b.pose) || a.index - b.index)
    .map(item => item.pose);
}

function latestPoseFrame(poses) {
  const ordered = chronologicalPoseObservations(poses)
    .filter(pose => pose?.image_name);
  return ordered.length ? ordered[ordered.length - 1] : null;
}

function liveFrameUrlForPayload(payload, stream = null, options = {}) {
  const liveStream = stream || payload?.stream || poseStreamMeta?.stream || {};
  const sourcePayload = payload || poseStreamMeta || {};
  const processedPose = latestPoseFrame(sourcePayload?.poses);
  const preferCurrentFrame = Boolean(
    options.preferCurrentFrame ||
    liveReplayInFlight ||
    pendingLiveReplayOpen ||
    liveStream.live_atlas
  );
  const frame = preferCurrentFrame
    ? (sourcePayload?.current_frame || processedPose || {})
    : (processedPose || sourcePayload?.current_frame || {});
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

function exactLiveFrameUrl(pose, stream = null) {
  if (!pose?.image_name) return "";
  const liveStream = stream || poseStreamMeta?.stream || {};
  const base = String(liveStream.query_frame_base_url || poseStreamMeta?.query_frame_base_url || "").replace(/\/$/, "");
  const name = String(pose.image_name).split("/").pop();
  return base && name ? `${base}/${encodeURIComponent(name)}` : "";
}

function preloadLiveFrame(pose, stream = null) {
  const url = exactLiveFrameUrl(pose, stream);
  if (!url || liveFramePreloadCache.has(url)) return;
  const preloader = new Image();
  preloader.decoding = "async";
  preloader.src = url;
  liveFramePreloadCache.set(url, preloader);
  const preloadLimit = liveFrameLockedStream?.simulated_live ? 60 : 12;
  while (liveFramePreloadCache.size > preloadLimit) {
    liveFramePreloadCache.delete(liveFramePreloadCache.keys().next().value);
  }
}

function updateLiveFrameViewForPose(pose, stream = null) {
  if (!liveFrameView) return false;
  const url = exactLiveFrameUrl(pose, stream);
  if (!url) return false;
  if (lastLiveFrameUrl === url) return true;
  setLiveFrameMode(true);
  setVideoFrameSteppingMode(true);
  video.pause();
  video.removeAttribute("src");
  liveFrameView.src = url;
  lastLiveFrameUrl = url;
  setLiveFrameStatus("", false);
  return true;
}

function updateReplayFrameViewForPose(pose, options = {}) {
  if (!replayFramePlaybackEnabled && !options.force) return false;
  if (liveReplayInFlight || (pendingLiveReplayOpen && !options.force)) return false;
  const replay = activeReplay(currentMapEntry);
  const url = replayFrameUrlForPose(replay, pose);
  if (!url || !liveFrameView) return false;
  if (lastReplayFrameUrl === url) return true;
  setLiveFrameMode(true);
  setVideoFrameSteppingMode(true);
  video.pause();
  video.removeAttribute("src");
  liveFrameView.src = `${url}?t=${Date.now()}`;
  lastReplayFrameUrl = url;
  setLiveFrameStatus("", false);
  return true;
}

function ensureLiveStreamVideoSource(stream) {
  if (stream?.live_preview_url || stream?.query_frame_base_url) {
    setLiveFrameMode(true);
    if (liveFrameLockedDisplayedPose?.image_name) {
      updateLiveFrameViewForPose(liveFrameLockedDisplayedPose, stream);
    } else {
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
  const first = (partialPoses || []).find(p => p && p.success !== false && !p.held_pose && p.center);
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
  const good = chronologicalPoseObservations(partialPoses)
    .filter(p => p && p.success !== false && !p.held_pose && (p.rcenter || p.center));
  return good.length ? good[good.length - 1] : null;
}

function latestLivePoseForDisplay(partialPoses) {
  const good = chronologicalPoseObservations(partialPoses)
    .filter(p => p && (p.success !== false || p.held_pose) && (p.rcenter || p.rawRcenter || p.center));
  if (!good.length) return null;
  const latest = good[good.length - 1];
  if (latest.rcenter) return latest;
  if (latest.rawRcenter) return { ...latest, rcenter: latest.rawRcenter };
  if (latest.center && room?.transform) return { ...latest, rcenter: room.transform(latest.center) };
  return null;
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
  resetLiveFrameLockedPlayback();
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

async function startSimulatedBaselineLive() {
  const active = activeReplay(currentMapEntry);
  const patrolProfile = selectedLivePatrolProfile();
  const lockedBaselineId = patrolProfile?.baseline_replay_id;
  const replay = active?.kind === "route_constrained_taught_baseline"
    && (!lockedBaselineId || active.id === lockedBaselineId)
    ? active
    : replayList(currentMapEntry).find(item => (
      item?.kind === "route_constrained_taught_baseline"
      && (!lockedBaselineId || item.id === lockedBaselineId)
    ));
  if (!currentMapEntry?.id || !replay) {
    uploadStatus.textContent = "Select Full Patrol Baseline 15:47:14 before creating a new live path.";
    return;
  }
  pendingLiveReplayOpen = true;
  pendingLiveReplayMapId = currentMapEntry.id;
  liveReplayInFlight = true;
  liveReplayMessage = "Creating a new path by localizing recorded DJI frames through the live pipeline";
  liveReplayStartedAt = performance.now();
  livePoseStreamKey = "";
  livePoseStreamCount = 0;
  liveCurrentPoseOverride = null;
  resetLiveFrameLockedPlayback();
  liveVideoWaitingForFirstPose = false;
  liveVideoSyncedToFirstPose = false;
  uploadStatus.textContent = "Preparing the locked two-lap precision patrol simulation.";
  stopPoseClockPlayback();
  await loadViewerData(false, currentMapEntry);
  liveReplayWaitingViewPrepared = true;
  showDemo({ resetVideo: false });
  await postJson("/api/drone/simulate-patrol-baseline", {
    map_id: currentMapEntry.id,
    patrol_id: patrolProfile?.patrol_id || null,
    baseline_replay_id: replay.id,
    fps: selectedLiveAtlasFps(),
    recorded_timing: true,
    laps: 2,
  });
  await pollStatus();
}

function selectedLiveAtlasFps() {
  const raw = Number(liveAtlasFps?.value || 10);
  if (!Number.isFinite(raw)) return 10;
  return Math.min(10, Math.max(0.5, raw));
}

function configuredLivePatrolProfiles(mapId = currentMapEntry?.id) {
  const rawProfiles = [];
  if (mapLibraryData?.live_patrol_lock?.enabled === true) {
    rawProfiles.push(mapLibraryData.live_patrol_lock);
  }
  const extras = Array.isArray(mapLibraryData?.live_patrol_profiles)
    ? mapLibraryData.live_patrol_profiles
    : (Array.isArray(mapLibraryData?.live_patrol_profiles?.profiles)
      ? mapLibraryData.live_patrol_profiles.profiles
      : []);
  rawProfiles.push(...extras.filter(profile => profile?.enabled === true));
  const seen = new Set();
  return rawProfiles.filter(profile => {
    const key = `${profile?.map_id || ""}:${profile?.patrol_id || ""}`;
    if (!profile?.map_id || !profile?.patrol_id || seen.has(key)) return false;
    seen.add(key);
    return !mapId || profile.map_id === mapId;
  });
}

function livePatrolProfile(patrolId, mapId = currentMapEntry?.id) {
  const requested = String(patrolId || "").trim();
  return configuredLivePatrolProfiles(mapId).find(profile => profile.patrol_id === requested) || null;
}

function selectedLivePatrolProfile() {
  const profiles = configuredLivePatrolProfiles();
  if (!profiles.length) return null;
  const selectedId = String(liveAtlasPatrolSelect?.value || "").trim();
  return livePatrolProfile(selectedId)
    || livePatrolProfile(activePatrolId)
    || profiles[0];
}

function boundLivePatrolId() {
  return String(
    poseStreamMeta?.stream?.patrol_id
    || (liveLocalizationStarted() ? pendingLivePatrolId : "")
    || ""
  ).trim();
}

function renderLivePatrolSelector(preferredPatrolId = "") {
  if (!liveAtlasPatrolSelect) return;
  const profiles = configuredLivePatrolProfiles();
  const patrols = patrolList(currentMapEntry);
  // Once localization is active, the backend-bound patrol is authoritative.
  // Keeping a different value visible in the disabled selector makes it look
  // as though another patrol can be started with the current pose stream.
  const boundPatrol = liveLocalizationStarted() ? boundLivePatrolId() : "";
  const prior = String(boundPatrol || preferredPatrolId || liveAtlasPatrolSelect.value || "").trim();
  const fallback = profiles.some(profile => profile.patrol_id === prior)
    ? prior
    : (profiles.some(profile => profile.patrol_id === activePatrolId)
      ? activePatrolId
      : (profiles[0]?.patrol_id || ""));
  liveAtlasPatrolSelect.innerHTML = profiles.length
    ? profiles.map(profile => {
      const patrol = patrols.find(item => item.id === profile.patrol_id);
      const label = patrol ? patrolTitle(patrol) : profile.patrol_id;
      return `<option value="${escapeHtml(profile.patrol_id)}">${escapeHtml(label)} · isolated profile</option>`;
    }).join("")
    : '<option value="">Map localization only</option>';
  liveAtlasPatrolSelect.value = fallback;
  liveAtlasPatrolSelect.disabled = liveLocalizationStarted();
  const profile = selectedLivePatrolProfile();
  if (startLiveAtlasButton) {
    const label = profile
      ? patrolTitle(patrols.find(item => item.id === profile.patrol_id) || { title: profile.patrol_id })
      : "";
    startLiveAtlasButton.textContent = liveLocalizationStarted()
      ? `${label || "Live"} Localization Active`
      : (profile ? `Start ${label} Localization` : "Start Localization");
  }
}

function updateLiveControlSummary() {
  const fps = selectedLiveAtlasFps();
  const profile = selectedLivePatrolProfile();
  const patrol = patrolList(currentMapEntry).find(item => item.id === profile?.patrol_id);
  if (liveControlSummary) {
    liveControlSummary.textContent = `${fps} FPS${patrol ? ` · ${patrolTitle(patrol)}` : ""}`;
  }
}

function takeoffHeightM() {
  const raw = Number(takeoffHeightInput?.value || 1);
  if (!Number.isFinite(raw)) return 1;
  return Math.min(2, Math.max(0.1, raw));
}

function normalizePhoneIp(value) {
  return String(value || "").trim();
}

function savedPhoneIps() {
  let stored = [];
  try {
    stored = JSON.parse(localStorage.getItem(PHONE_IP_STORAGE_KEY) || "[]");
  } catch {
    stored = [];
  }
  const all = [...DEFAULT_PHONE_IPS, ...(Array.isArray(stored) ? stored : [])]
    .map(normalizePhoneIp)
    .filter(Boolean);
  return [...new Set(all)].slice(0, 12);
}

function renderPhoneIpOptions() {
  const ips = savedPhoneIps();
  if (phoneIpOptions) {
    phoneIpOptions.innerHTML = "";
    for (const ip of ips) {
      const option = document.createElement("option");
      option.value = ip;
      phoneIpOptions.appendChild(option);
    }
  }
  if (phoneIpSelect) {
    const current = normalizePhoneIp(liveAtlasPhoneIp?.value) || ips[0] || "";
    phoneIpSelect.innerHTML = "";
    for (const ip of ips) {
      const option = document.createElement("option");
      option.value = ip;
      option.textContent = ip;
      phoneIpSelect.appendChild(option);
    }
    if (current && !ips.includes(current)) {
      const option = document.createElement("option");
      option.value = current;
      option.textContent = `${current} (new)`;
      phoneIpSelect.insertBefore(option, phoneIpSelect.firstChild);
    }
    phoneIpSelect.value = current;
  }
}

function rememberPhoneIp(value) {
  const ip = normalizePhoneIp(value);
  if (!ip) return;
  const known = savedPhoneIps();
  const next = [ip, ...known.filter(v => v !== ip)].slice(0, 12);
  try {
    localStorage.setItem(PHONE_IP_STORAGE_KEY, JSON.stringify(next));
  } catch {
    // Local storage is a convenience only; keep live controls working if blocked.
  }
  renderPhoneIpOptions();
}

function storedDroneHeadingTrimDeg() {
  try {
    const stored = localStorage.getItem(DRONE_HEADING_TRIM_STORAGE_KEY);
    if (stored != null && stored !== "") {
      const value = Number(stored);
      if (Number.isFinite(value)) return value;
    }
  } catch {
    // Local storage is optional.
  }
  return DEFAULT_DRONE_HEADING_TRIM_DEG;
}

function selectedDroneHeadingTrimDeg() {
  const value = Number(droneHeadingTrimSelect?.value);
  return Number.isFinite(value) ? value : storedDroneHeadingTrimDeg();
}

function selectedDroneHeadingTrimRad() {
  return selectedDroneHeadingTrimDeg() * Math.PI / 180;
}

function renderDroneHeadingTrim() {
  if (!droneHeadingTrimSelect) return;
  const stored = Math.max(-180, Math.min(180, storedDroneHeadingTrimDeg()));
  droneHeadingTrimSelect.value = String(Math.round(stored / 5) * 5);
  droneHeadingTrimSelect.disabled = false;
  if (droneHeadingTrimValue) {
    const value = selectedDroneHeadingTrimDeg();
    droneHeadingTrimValue.textContent = `${value > 0 ? "+" : ""}${value} deg`;
  }
  const label = droneHeadingTrimSelect.closest("label");
  if (label) label.hidden = false;
}

function storedLiveControlPinned() {
  try {
    return localStorage.getItem(LIVE_CONTROL_PIN_STORAGE_KEY) === "true";
  } catch {
    return false;
  }
}

function setLiveControlPinned(pinned) {
  const enabled = Boolean(pinned);
  document.body.classList.toggle("live-control-pinned", enabled);
  if (enabled && liveLocalizationControl) liveLocalizationControl.open = true;
  if (pinLiveControlButton) {
    pinLiveControlButton.setAttribute("aria-pressed", String(enabled));
    pinLiveControlButton.textContent = enabled ? "Unpin" : "Pin";
    pinLiveControlButton.title = enabled
      ? "Release this control panel from the left dock"
      : "Pin this control panel to the left edge";
  }
  if (enabled && liveLocalizationControl) {
    liveLocalizationControl.classList.remove("is-user-placed");
    liveLocalizationControl.style.left = "";
    liveLocalizationControl.style.top = "";
    liveLocalizationControl.style.right = "";
    liveLocalizationControl.style.bottom = "";
  }
  try {
    localStorage.setItem(LIVE_CONTROL_PIN_STORAGE_KEY, enabled ? "true" : "false");
  } catch {
    // Pinning is visual convenience; keep the UI functional if storage is blocked.
  }
  syncLiveControlCollapsedState();
  if (renderStarted) render();
}

function liveControlIsPinned() {
  return document.body.classList.contains("live-control-pinned");
}

function syncLiveControlCollapsedState() {
  const collapsed = liveControlIsPinned() && liveLocalizationControl && !liveLocalizationControl.open;
  document.body.classList.toggle("live-control-collapsed", Boolean(collapsed));
}

function storedCollapsedLiveControlSections() {
  try {
    const stored = JSON.parse(localStorage.getItem(LIVE_CONTROL_SECTIONS_STORAGE_KEY) || "[]");
    return new Set(Array.isArray(stored) ? stored.filter(Boolean) : []);
  } catch {
    return new Set();
  }
}

function setLiveControlSectionCollapsed(section, collapsed, options = {}) {
  const key = section?.dataset.liveSection;
  const toggle = section?.querySelector(":scope > .live-control-section-toggle, :scope > .live-control-section-head > .live-control-section-toggle");
  if (!section || !key || !toggle) return;
  const isCollapsed = Boolean(collapsed);
  const title = toggle.querySelector(".live-control-section-title")?.textContent?.trim() || "section";
  section.classList.toggle("is-collapsed", isCollapsed);
  toggle.setAttribute("aria-expanded", String(!isCollapsed));
  toggle.setAttribute("aria-label", `${isCollapsed ? "Expand" : "Minimize"} ${title}`);
  toggle.title = `${isCollapsed ? "Expand" : "Minimize"} ${title}`;
  if (options.persist === false) return;
  const collapsedSections = storedCollapsedLiveControlSections();
  if (isCollapsed) collapsedSections.add(key);
  else collapsedSections.delete(key);
  try {
    localStorage.setItem(LIVE_CONTROL_SECTIONS_STORAGE_KEY, JSON.stringify([...collapsedSections]));
  } catch {
    // Section collapsing remains available for this session if storage is blocked.
  }
}

function setupLiveControlSections() {
  const collapsedSections = storedCollapsedLiveControlSections();
  for (const section of document.querySelectorAll("#live-localization-control .live-control-section")) {
    const toggle = section.querySelector(":scope > .live-control-section-toggle, :scope > .live-control-section-head > .live-control-section-toggle");
    if (!toggle) continue;
    setLiveControlSectionCollapsed(section, collapsedSections.has(section.dataset.liveSection), { persist: false });
    toggle.addEventListener("click", () => {
      setLiveControlSectionCollapsed(section, !section.classList.contains("is-collapsed"));
    });
  }
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

function correctedLivePose(pose) {
  if (!pose?.rcenter) return pose;
  const corrected = {
    ...pose,
    rcenter: pose.rcenter.map((value, index) => value + Number(initialPoseOffsetRoom[index] || 0)),
  };
  const opticalHeading = Array.isArray(pose.rotation_heading)
    ? pose.rotation_heading.slice(0, 3).map(Number)
    : null;
  const opticalHeadingTracks = Number(pose.rotation_heading_tracks || 0);
  const backendHeadingSource = String(
    pose.rheading_source || pose.rheadingSource || ""
  );
  const absoluteRouteHeading = new Set([
    "recorded_departure_image_alignment",
    "recorded_departure_image_tracking_consensus",
  ]).has(backendHeadingSource);
  const currentFrameOpticalHeading = Boolean(
    opticalHeading?.length === 3 &&
    opticalHeading.every(Number.isFinite) &&
    opticalHeadingTracks >= 16
  );
  if (
    currentFrameOpticalHeading &&
    pose.rotation_position_locked &&
    !absoluteRouteHeading
  ) {
    corrected.rheadingRaw = corrected.rheading;
    corrected.rheading = opticalHeading;
    corrected.rheadingSource = "optical_flow_yaw";
  }
  if (Array.isArray(liveRotationPositionAnchor) && liveRotationPositionAnchor.length >= 3) {
    corrected.rawRotationRcenter = corrected.rcenter;
    corrected.rcenter = liveRotationPositionAnchor.slice(0, 3).map(Number);
    corrected.rotationPositionLocked = true;
  }
  return corrected;
}

function resetLiveFrameLockedPlayback() {
  liveFrameLockedQueue = [];
  liveFrameLockedDisplayedPose = null;
  liveFrameLockedReplayId = "";
  liveFrameLockedQueuedCount = 0;
  liveFrameLockedDisplayedCount = 0;
  liveFrameLockedDroppedCount = 0;
  liveFrameLockedNextAtMs = 0;
  liveFrameLockedStream = null;
  liveFramePreloadCache = new Map();
  liveReplayCompletionPending = false;
  lastLiveFrameUrl = "";
}

function liveFrameIntervalMs(a, b) {
  const aTime = Number(a?.time_sec);
  const bTime = Number(b?.time_sec);
  const sourceDelta = (Number.isFinite(aTime) && Number.isFinite(bTime))
    ? (bTime - aTime) * 1000
    : NaN;
  if (Number.isFinite(sourceDelta) && sourceDelta > 1 && sourceDelta < 1000) {
    return sourceDelta;
  }
  const fps = Number(liveFrameLockedStream?.fps || 10);
  return 1000 / Math.max(0.5, Math.min(30, Number.isFinite(fps) ? fps : 10));
}

function liveFramePlaybackIntervalMs(a, b) {
  const interval = liveFrameIntervalMs(a, b);
  const backlog = liveFrameLockedQueue.length;
  if (liveFrameLockedStream?.simulated_live) {
    if (backlog > SIMULATED_LIVE_FRAME_TARGET_BUFFER) return Math.max(25, interval * 0.25);
    if (backlog > 12) return Math.max(33, interval * 0.40);
  }
  if (backlog > 6) return Math.max(40, interval * 0.60);
  if (backlog > LIVE_FRAME_TARGET_BUFFER) return Math.max(50, interval * 0.80);
  return interval;
}

function presentLiveFrameLockedPose(sourcePose) {
  if (!sourcePose?.rcenter) return;
  liveFrameLockedDisplayedPose = correctedLivePose(sourcePose);
  liveFrameLockedDisplayedCount += 1;
  const showedCapturedFrame = updateLiveFrameViewForPose(sourcePose, liveFrameLockedStream);
  if (!showedCapturedFrame && !liveFrameMode && video) {
    const frameTime = Number(sourcePose.time_sec);
    if (Number.isFinite(frameTime)) {
      try {
        video.currentTime = frameTime;
      } catch {
        // The exact R,t remains visible even if a browser cannot seek this video frame.
      }
      video.pause();
    }
  }
}

function advanceLiveFrameLockedPlayback(nowMs = performance.now()) {
  if (!liveFrameLockedDisplayedPose && liveFrameLockedQueue.length) {
    presentLiveFrameLockedPose(liveFrameLockedQueue.shift());
    liveFrameLockedNextAtMs = liveFrameLockedQueue.length
      ? nowMs + liveFramePlaybackIntervalMs(liveFrameLockedDisplayedPose, liveFrameLockedQueue[0])
      : 0;
  } else if (
    liveFrameLockedDisplayedPose &&
    liveFrameLockedQueue.length &&
    (!liveFrameLockedNextAtMs || nowMs >= liveFrameLockedNextAtMs)
  ) {
    presentLiveFrameLockedPose(liveFrameLockedQueue.shift());
    liveFrameLockedNextAtMs = liveFrameLockedQueue.length
      ? nowMs + liveFramePlaybackIntervalMs(liveFrameLockedDisplayedPose, liveFrameLockedQueue[0])
      : 0;
  }
  return liveFrameLockedDisplayedPose;
}

function enqueueLiveFrameLockedPoses(displayPoses, payload, stream = null) {
  if (!Array.isArray(displayPoses)) return;
  displayPoses = chronologicalPoseObservations(displayPoses);
  const replayId = String(payload?.replay_id || payload?.stream?.replay_id || stream?.replay_id || "live");
  if (liveFrameLockedReplayId && replayId !== liveFrameLockedReplayId) {
    resetLiveFrameLockedPlayback();
  }
  liveFrameLockedReplayId = replayId;
  liveFrameLockedStream = payload?.stream || stream || liveFrameLockedStream || {};
  const start = Math.min(liveFrameLockedQueuedCount, displayPoses.length);
  for (const pose of displayPoses.slice(start)) {
    if (pose?.rcenter && (pose.success !== false || pose.held_pose)) {
      liveFrameLockedQueue.push(pose);
    }
  }
  liveFrameLockedQueuedCount = displayPoses.length;
  const maxBuffer = liveFrameLockedStream?.simulated_live
    ? SIMULATED_LIVE_FRAME_MAX_BUFFER
    : LIVE_FRAME_MAX_BUFFER;
  const targetBuffer = liveFrameLockedStream?.simulated_live
    ? SIMULATED_LIVE_FRAME_TARGET_BUFFER
    : LIVE_FRAME_TARGET_BUFFER;
  if (liveFrameLockedQueue.length > maxBuffer) {
    const dropCount = liveFrameLockedQueue.length - targetBuffer;
    liveFrameLockedQueue.splice(0, dropCount);
    liveFrameLockedDroppedCount += dropCount;
    liveFrameLockedNextAtMs = performance.now();
  }
  for (const pose of liveFrameLockedQueue.slice(0, maxBuffer)) {
    preloadLiveFrame(pose, liveFrameLockedStream);
  }
  if (liveFrameLockedDisplayedPose && liveFrameLockedQueue.length && !liveFrameLockedNextAtMs) {
    liveFrameLockedNextAtMs = performance.now() + liveFramePlaybackIntervalMs(
      liveFrameLockedDisplayedPose,
      liveFrameLockedQueue[0],
    );
  }
  advanceLiveFrameLockedPlayback();
}

function currentLiveDisplayPose(fallback = liveCurrentPoseOverride) {
  return advanceLiveFrameLockedPlayback() || fallback;
}

function liveFrameLockedPlaybackDrained() {
  return liveFrameLockedQueuedCount > 0 &&
    liveFrameLockedQueue.length === 0 &&
    liveFrameLockedDisplayedCount + liveFrameLockedDroppedCount >= liveFrameLockedQueuedCount;
}

function initialPoseOffsetMagnitude() {
  return Math.hypot(Number(initialPoseOffsetRoom[0] || 0), Number(initialPoseOffsetRoom[2] || 0));
}

function updateInitialPositionControls() {
  const ready = firstConfirmedPoseReady();
  if (correctInitialPositionButton) {
    correctInitialPositionButton.disabled = !ready;
    correctInitialPositionButton.classList.toggle("active", initialPositionSelecting);
    correctInitialPositionButton.textContent = initialPositionSelecting
      ? "Click Actual Drone Position"
      : "Correct Initial Position";
  }
  if (resetInitialPositionButton) {
    resetInitialPositionButton.disabled = initialPoseOffsetMagnitude() < 1e-6;
  }
  if (initialPositionStatus) {
    const magnitude = initialPoseOffsetMagnitude();
    initialPositionStatus.textContent = magnitude < 1e-6
      ? "No manual position correction."
      : `Position corrected ${magnitude.toFixed(2)} map units. This same offset will be used by flight control.`;
  }
}

function djiBridgeAgeSeconds(status = latestDjiLiveStatus) {
  const updated = Number(status?.updated_at || 0);
  return updated > 0 ? Math.max(0, Date.now() / 1000 - updated) : Infinity;
}

function djiBridgeFresh(status = latestDjiLiveStatus, maxAgeSeconds = 5) {
  return djiBridgeAgeSeconds(status) <= maxAgeSeconds;
}

function djiBridgeState(status = latestDjiLiveStatus) {
  return String(status?.status || "").trim().toLowerCase();
}

function djiBridgeReadyForControl() {
  const state = djiBridgeState();
  return ["streaming", "waiting_for_video"].includes(state) && djiBridgeFresh(latestDjiLiveStatus, 5);
}

function djiBridgeReadyForMission() {
  const state = djiBridgeState();
  return state === "streaming" && djiBridgeFresh(latestDjiLiveStatus, 3.5) && Boolean(latestDjiLiveStatus?.control_enabled);
}

function liveMovementLockReason() {
  if (!liveLocalizationStarted()) return "Start Live ATLAS before executing drone movement.";
  if (!djiBridgeReadyForMission()) {
    const state = djiBridgeState() || "offline";
    const age = djiBridgeAgeSeconds();
    const ageText = Number.isFinite(age) ? `${age.toFixed(1)}s old` : "not available";
    return `Live DJI bridge is not streaming fresh command-ready frames (${state}, heartbeat ${ageText}).`;
  }
  if (!firstLocalizationConfirmed) return "Confirm the first TSolve localization before movement.";
  if (!firstConfirmedPoseReady()) return "Wait for a visible TSolve R,t pose before movement.";
  if (!guidedMotionEnable?.checked) return "Enable guided movement after localization confirmation.";
  return "";
}

function guidedMotionArmed() {
  return Boolean(
    guidedMotionEnable?.checked &&
    liveLocalizationStarted() &&
    djiBridgeReadyForMission() &&
    firstLocalizationConfirmed &&
    firstConfirmedPoseReady()
  );
}

function bestEnemyDetection(detections) {
  return [...(detections || [])]
    .filter(d => Number(d?.confidence) >= ENEMY_ALERT_MIN_CONFIDENCE && d?.box)
    .sort((a, b) => Number(b.confidence || 0) - Number(a.confidence || 0))[0] || null;
}

function enemyDetectionIsFresh(payload) {
  const updatedAt = Number(payload?.updated_at || 0) * 1000;
  return Number.isFinite(updatedAt) && updatedAt > 0 && Date.now() - updatedAt <= 2500;
}

function enemyDetectionEnabled() {
  return Boolean(enemyDetectionEnabledInput?.checked);
}

function storedEnemyDetectionEnabled() {
  try {
    return localStorage.getItem(ENEMY_DETECTION_ENABLED_STORAGE_KEY) === "true";
  } catch {
    return false;
  }
}

function renderEnemyDetectionDisabled() {
  if (enemyLiveDetectorState) enemyLiveDetectorState.textContent = "off";
  if (enemyLiveDetection) {
    enemyLiveDetection.textContent = "Enemy detection is off. Detector results cannot interrupt this patrol.";
    enemyLiveDetection.dataset.tone = "";
  }
  updateEnemyResponseStatus("Automatic enemy response is disabled for patrol validation.", "");
}

async function syncEnemyDetectionRuntime(enabled) {
  try {
    await postJson("/api/enemy-drone/live-detection", { enabled: Boolean(enabled) });
    return true;
  } catch (error) {
    if (!enabled) {
      updateEnemyResponseStatus(
        "Automatic flight response is off in this page. Restart the local server before the next flight to also suspend YOLO inference.",
        "error"
      );
    }
    return false;
  }
}

function setEnemyDetectionEnabled(enabled, options = {}) {
  const next = Boolean(enabled);
  if (enemyDetectionEnabledInput) enemyDetectionEnabledInput.checked = next;
  if (options.persist !== false) {
    try {
      localStorage.setItem(ENEMY_DETECTION_ENABLED_STORAGE_KEY, next ? "true" : "false");
    } catch {
      // The safety gate still applies for this page when local storage is unavailable.
    }
  }
  if (!next) {
    enemyDetectionHistory = [];
    enemyTargetSuppressedUntilClear = false;
    enemyAlertState = {
      active: false,
      hoverSent: false,
      hoverSentAt: 0,
      lockOnSentAt: 0,
      target: null,
      frame: "",
      updatedAt: Date.now(),
    };
    renderEnemyDetectionDisabled();
  } else {
    if (enemyLiveDetectorState) enemyLiveDetectorState.textContent = "armed";
    if (enemyLiveDetection) {
      enemyLiveDetection.textContent = "Waiting for fresh trained YOLO enemy-drone detections.";
      enemyLiveDetection.dataset.tone = "";
    }
    updateEnemyResponseStatus(
      liveLocalizationStarted()
        ? "Enemy detection is armed. A confirmed target will pause the patrol in hover."
        : "Enemy detection will arm when live localization starts.",
      ""
    );
  }
  updateEnemyResponseControls();
}

function enemyTargetEstimate(target) {
  const box = target?.box || {};
  const centerX = Number(box.x1 ?? 0) + Number(box.width ?? 0) * 0.5;
  const centerY = Number(box.y1 ?? 0) + Number(box.height ?? 0) * 0.5;
  const width = Math.max(0, Number(box.width || 0));
  const height = Math.max(0, Number(box.height || 0));
  const area = width * height;
  const dx = centerX - 0.5;
  const dy = centerY - 0.5;
  const yawDeg = Math.max(-ENEMY_LOCKON_MAX_YAW_DEG, Math.min(ENEMY_LOCKON_MAX_YAW_DEG, dx * 58));
  const followDuration = Math.max(0, Math.min(1.15, (ENEMY_LOCKON_FOLLOW_TARGET_HEIGHT - height) * 3.0));
  return { centerX, centerY, width, height, area, dx, dy, yawDeg, followDuration };
}

function updateEnemyResponseStatus(text = "", tone = "") {
  if (!enemyResponseStatus) return;
  enemyResponseStatus.textContent = text || "Patrol response is armed only after live localization starts.";
  enemyResponseStatus.dataset.tone = tone;
}

function currentEnemyRangeCalibration() {
  const calibration = currentEnemyProfile()?.range_calibration;
  return calibration && typeof calibration === "object"
    ? calibration
    : { status: "needs_samples", samples: [], sample_count: 0, model: null, validation: null };
}

function selectedEnemyStopClearanceM() {
  const value = Number(enemyStopClearanceInput?.value || 0.50);
  return Math.max(0.50, Math.min(2.0, Number.isFinite(value) ? value : 0.50));
}

function renderEnemyRangeCalibration() {
  const profile = currentEnemyProfile();
  const calibration = currentEnemyRangeCalibration();
  const count = Number(calibration.sample_count ?? calibration.samples?.length ?? 0);
  const validation = calibration.validation || {};
  const validated = calibration.status === "validated" && calibration.model;
  if (enemyRangeStatus) {
    enemyRangeStatus.textContent = validated
      ? `validated · ${count} samples · p90 ${Number(validation.p90_absolute_error_m || 0).toFixed(2)} m`
      : calibration.status === "rejected"
      ? `rejected · ${count} samples · add more varied measurements`
      : `${count} sample${count === 1 ? "" : "s"} · pursuit locked`;
  }
  if (enemyRangeHelp) {
    enemyRangeHelp.textContent = validated
      ? `Calibrated ${Number(validation.min_clearance_m || 0).toFixed(2)}–${Number(validation.max_clearance_m || 0).toFixed(2)} m. Pursuit uses a conservative ${Number(calibration.model?.conservative_margin_m || 0).toFixed(2)} m error margin.`
      : "Record at least 8 samples at 4 measured distances spanning 0.75 m. Forward pursuit stays locked until validation passes.";
  }
  const liveDetectionReady = Boolean(enemyAlertState.target && enemyDetectionIsFresh({updated_at: enemyAlertState.updatedAt / 1000}));
  if (enemySaveRangeSampleButton) enemySaveRangeSampleButton.disabled = !enemyDetectionEnabled() || !profile || !liveDetectionReady || enemyPursuitInFlight;
  if (enemyValidateRangeButton) enemyValidateRangeButton.disabled = !profile || count < 8 || enemyPursuitInFlight;
  if (enemyResetRangeButton) enemyResetRangeButton.disabled = !profile || count === 0 || enemyPursuitInFlight;
}

function updateEnemyResponseControls() {
  if (!enemyDetectionEnabled()) {
    if (enemyConfirmLockButton) enemyConfirmLockButton.disabled = true;
    if (enemyStartPursuitButton) enemyStartPursuitButton.disabled = true;
    if (enemyClearAlertButton) enemyClearAlertButton.disabled = true;
    renderEnemyRangeCalibration();
    renderEnemyDetectionDisabled();
    return;
  }
  const hasTarget = Boolean(enemyAlertState.active && enemyAlertState.target);
  const calibration = currentEnemyRangeCalibration();
  const pursuitReady = calibration.status === "validated" && Boolean(calibration.model);
  if (enemyConfirmLockButton) {
    enemyConfirmLockButton.disabled = !hasTarget || !guidedMotionArmed() || enemyPursuitInFlight;
  }
  if (enemyStartPursuitButton) {
    enemyStartPursuitButton.disabled = !hasTarget || !guidedMotionArmed() || !pursuitReady || enemyPursuitInFlight;
    enemyStartPursuitButton.title = pursuitReady
      ? "Start calibrated moving-target pursuit."
      : "Record and validate live range samples before forward pursuit.";
  }
  if (enemyClearAlertButton) {
    enemyClearAlertButton.disabled = !hasTarget || enemyPursuitInFlight;
  }
  if (!hasTarget) {
    updateEnemyResponseStatus(
      liveLocalizationStarted()
        ? "Waiting for trained YOLO enemy-drone detections."
        : "Patrol response is armed only after live localization starts.",
      ""
    );
  }
  renderEnemyRangeCalibration();
}

async function saveEnemyRangeSample() {
  const profile = currentEnemyProfile();
  if (!profile) throw new Error("Choose the NEO enemy profile first.");
  const measured = Number(enemyMeasuredClearanceInput?.value);
  if (!Number.isFinite(measured) || measured < 0.20 || measured > 10.0) {
    throw new Error("Enter the measured body-to-body clearance between 0.20 m and 10.0 m.");
  }
  updateEnemyResponseStatus(`Saving the live ${measured.toFixed(2)} m range sample...`, "busy");
  const data = await postJson("/api/enemy-drone/range-sample", {
    enemy_id: profile.id,
    measured_clearance_m: measured,
  });
  enemyLibraryData = data.library || enemyLibraryData;
  updateEnemyResponseStatus(`Saved live range sample at ${measured.toFixed(2)} m.`, "ok");
  renderEnemyLibrary();
  updateEnemyResponseControls();
}

async function validateEnemyRangeCalibration() {
  const profile = currentEnemyProfile();
  if (!profile) throw new Error("Choose the NEO enemy profile first.");
  updateEnemyResponseStatus("Validating range accuracy across measured distances...", "busy");
  const data = await postJson("/api/enemy-drone/range-validate", {
    enemy_id: profile.id,
    stop_clearance_m: selectedEnemyStopClearanceM(),
  });
  enemyLibraryData = data.library || enemyLibraryData;
  const validation = data.validation || {};
  const accepted = Boolean(validation.accepted);
  updateEnemyResponseStatus(
    accepted
      ? `Range validated: MAE ${Number(validation.mean_absolute_error_m || 0).toFixed(2)} m, p90 ${Number(validation.p90_absolute_error_m || 0).toFixed(2)} m. Guarded pursuit is available.`
      : `Range validation failed: MAE ${Number(validation.mean_absolute_error_m || 0).toFixed(2)} m, p90 ${Number(validation.p90_absolute_error_m || 0).toFixed(2)} m. Add varied samples before flight.`,
    accepted ? "ok" : "error"
  );
  renderEnemyLibrary();
  updateEnemyResponseControls();
}

async function resetEnemyRangeCalibration() {
  const profile = currentEnemyProfile();
  if (!profile) throw new Error("Choose the NEO enemy profile first.");
  const ok = window.confirm("Delete all saved NEO range-calibration samples and lock pursuit again?");
  if (!ok) return;
  const data = await postJson("/api/enemy-drone/range-reset", {enemy_id: profile.id});
  enemyLibraryData = data.library || enemyLibraryData;
  updateEnemyResponseStatus("Range samples reset. Forward pursuit is locked.", "busy");
  renderEnemyLibrary();
  updateEnemyResponseControls();
}

async function pauseForEnemyDetection(payload, target) {
  if (!enemyDetectionEnabled()) return;
  const now = Date.now();
  const sameFrame = enemyAlertState.frame && payload.frame && enemyAlertState.frame === payload.frame;
  enemyAlertState = {
    ...enemyAlertState,
    active: true,
    target,
    frame: String(payload.frame || ""),
    updatedAt: now,
  };
  if (enemyPursuitInFlight) {
    updateEnemyResponseControls();
    return;
  }
  const est = enemyTargetEstimate(target);
  updateEnemyResponseStatus(
    `Enemy drone candidate: ${target.class_name || "target"} ${(Number(target.confidence || 0) * 100).toFixed(0)}% · yaw estimate ${est.yawDeg.toFixed(1)} deg. Confirm lock-on for a slow guarded follow pulse.`,
    "alert"
  );
  updateEnemyResponseControls();
  if (!liveLocalizationStarted()) return;
  if (sameFrame || enemyAlertState.hoverSent && now - enemyAlertState.hoverSentAt < ENEMY_HOVER_COOLDOWN_MS) return;
  if (activePatrolExecutionContext) {
    interruptedPatrolExecutionContext = {
      ...activePatrolExecutionContext,
      interrupted_at: now,
    };
    activePatrolExecutionContext = null;
  }
  enemyAlertState.hoverSent = true;
  enemyAlertState.hoverSentAt = now;
  try {
    setDjiCommandStatus("Enemy drone detected. Sending guarded hover and waiting for lock-on confirmation.", "busy");
    await sendDjiFlightCommand("hover", { enemy_alert: true, emergency_stop: true });
  } catch (error) {
    setDjiCommandStatus(`Enemy alert hover failed: ${error.message || error}`, "error");
    updateEnemyResponseStatus(`Enemy detected, but hover command failed: ${error.message || error}`, "error");
  }
}

function clearEnemyAlert() {
  enemyAlertState = {
    active: false,
    hoverSent: false,
    hoverSentAt: 0,
    lockOnSentAt: 0,
    target: null,
    frame: "",
    updatedAt: Date.now(),
  };
  updateEnemyResponseControls();
}

function buildEnemyLockOnPlan(target) {
  const est = enemyTargetEstimate(target);
  const commands = [
    {
      type: "hover",
      title: "Enemy lock-on confirmation hold",
      detail: "Hold while ATLAS verifies fresh TSolve pose before lock-on pulse.",
      duration_s: 0.45,
      safety: "enemy-lock-on-gate",
    },
  ];
  if (Math.abs(est.yawDeg) >= 3) {
    commands.push({
      type: "yaw",
      title: "Align to enemy drone bearing",
      yaw_delta_deg: est.yawDeg,
      detail: "Slowly rotate toward the detected target in the live camera frame.",
      duration_s: 0.75,
      safety: "enemy-yaw-align",
    });
  }
  commands.push({
    type: "hover",
    title: "Post lock-on TSolve refresh",
    detail: "Stop and wait for the next live TSolve update before any further movement.",
    duration_s: 0.75,
    safety: "enemy-relocalize",
  });
  return { estimate: est, commands };
}

async function confirmEnemyLockOn() {
  if (!enemyAlertState.target) {
    updateEnemyResponseStatus("No enemy target is available for lock-on.", "error");
    return;
  }
  if (!guidedMotionArmed()) {
    updateEnemyResponseStatus("Confirm first localization and enable guided movement before lock-on.", "error");
    setDjiCommandStatus("Enemy lock-on blocked. Confirm localization and arm guided movement first.", "error");
    updateFlightControlState();
    return;
  }
  const { estimate, commands } = buildEnemyLockOnPlan(enemyAlertState.target);
  const ok = window.confirm(
    `Confirm guarded enemy-drone lock-on?\n\n` +
    `ATLAS will hover and may rotate in place to center the target. Forward movement is disabled until range estimation is validated.\n` +
    `Estimated yaw: ${estimate.yawDeg.toFixed(1)} deg.\n` +
    `Keep the physical controller ready and press Hover Now if anything looks wrong.\n\nContinue?`
  );
  if (!ok) {
    updateEnemyResponseStatus("Enemy lock-on cancelled. Drone remains in hover/normal patrol control.", "busy");
    return;
  }
  enemyAlertState.lockOnSentAt = Date.now();
  updateEnemyResponseStatus("Sending guarded enemy lock-on pulse to DJI bridge...", "busy");
  try {
    const result = await sendDjiFlightCommand("mission", {
      mission: {
        guided_enabled: true,
        enemy_lock_on: true,
        pose_max_age_seconds: 1.8,
        pose_recovery_seconds: 6.0,
        pulse_seconds: 0.16,
        max_forward_rc: 0.055,
        max_yaw_rc: 0.040,
        max_vertical_rc: 0.020,
        max_step_seconds: 1.2,
        map_id: currentMapEntry?.id || null,
        map_title: currentMapEntry?.title || null,
        target_detection: enemyAlertState.target,
        target_estimate: estimate,
        commands,
        safety_barriers: mapSafetyBarriers(),
        safety_obstacles: mapSafetyObstacles(),
        barrier_clearance_m: selectedBarrierClearance(),
        obstacle_clearance_m: selectedObstacleClearance(),
        heading_trim_deg: 0,
        operator_heading_calibrated: Boolean(useModelHeadingForFlightInput?.checked),
        initial_body_heading_offset_deg: -selectedDroneHeadingTrimDeg(),
        initial_pose_offset_room: initialPoseOffsetRoom.slice(0, 3),
        confirmed_at: new Date().toISOString(),
      },
    });
    const bridgeMessage = result.result?.message || result.message || "Enemy lock-on packet queued.";
    updateEnemyResponseStatus(`${bridgeMessage} Drone will hover after this pulse and wait for the next confirmed lock-on.`, "busy");
  } catch (error) {
    updateEnemyResponseStatus(`Enemy lock-on failed: ${error.message || error}`, "error");
    setDjiCommandStatus(`Enemy lock-on failed: ${error.message || error}`, "error");
  } finally {
    updateEnemyResponseControls();
  }
}

async function confirmEnemyPursuit() {
  const profile = currentEnemyProfile();
  const calibration = currentEnemyRangeCalibration();
  if (!profile || calibration.status !== "validated" || !calibration.model) {
    updateEnemyResponseStatus("Guarded pursuit is locked until live range calibration passes.", "error");
    return;
  }
  if (!enemyAlertState.target) {
    updateEnemyResponseStatus("No fresh NEO target is available for pursuit.", "error");
    return;
  }
  if (!guidedMotionArmed()) {
    updateEnemyResponseStatus("Confirm localization and enable guided movement before pursuit.", "error");
    return;
  }
  const stopClearance = selectedEnemyStopClearanceM();
  const resumeContext = interruptedPatrolExecutionContext;
  const resumeText = resumeContext
    ? `After interception, ATLAS will use a fresh TSolve pose, rejoin "${resumeContext.patrolName}" at its nearest safe patrol point, and continue in the saved order.\n`
    : "No active patrol was interrupted, so ATLAS will hold after interception.\n";
  const ok = window.confirm(
    `Start guarded moving-target pursuit?\n\n` +
    `ATLAS will continuously re-detect NEO, predict lateral motion, rotate to keep it centered, and advance with one short forward pulse per fresh detector frame.\n\n` +
    `Stop clearance: ${stopClearance.toFixed(2)} m (body-to-body).\n` +
    resumeText +
    `Walls and obstacles retain their saved clearance plus a ${FLIGHT_SAFETY_PULSE_BUFFER_M.toFixed(2)} m motion buffer.\n` +
    `Loss of NEO, stale range, stale TSolve pose, timeout, or Hover Now will stop translation and hold position.\n` +
    `Vertical tracking remains disabled for the first validation runs.\n\n` +
    `Keep the physical controller ready. Continue?`
  );
  if (!ok) return;
  enemyPursuitResumeContext = resumeContext;
  interruptedPatrolExecutionContext = null;
  updateEnemyResponseStatus("Queueing calibrated moving-target pursuit...", "busy");
  enemyPursuitInFlight = true;
  updateEnemyResponseControls();
  let data;
  try {
    data = await sendDjiFlightCommand("mission", {
      mission: {
        client_safety_version: 3,
        guided_enabled: true,
        enemy_pursuit: true,
        enemy_id: profile.id,
        operator_confirmed: true,
        stop_clearance_m: stopClearance,
        target_detection: enemyAlertState.target,
        pose_max_age_seconds: 1.2,
        pose_recovery_seconds: 4.0,
        pulse_seconds: 0.14,
        max_forward_rc: 0.025,
        max_yaw_rc: 0.028,
        max_vertical_rc: 0.010,
        vertical_tracking_enabled: false,
        detection_max_age_seconds: 1.0,
        lost_target_abort_seconds: 4.0,
        max_pursuit_seconds: 45.0,
        minimum_confidence: 0.40,
        pursuit_yaw_sign: Number(enemyPursuitYawDirection?.value || 1) < 0 ? -1 : 1,
        map_id: currentMapEntry?.id || null,
        map_title: currentMapEntry?.title || null,
        safety_barriers: mapSafetyBarriers(),
        safety_obstacles: mapSafetyObstacles(),
        safety_motion_buffer_m: FLIGHT_SAFETY_PULSE_BUFFER_M,
        initial_pose_offset_room: initialPoseOffsetRoom.slice(0, 3),
        confirmed_at: new Date().toISOString(),
      },
    });
  } catch (error) {
    enemyPursuitInFlight = false;
    enemyPursuitCommandId = "";
    enemyPursuitResumeContext = null;
    updateEnemyResponseControls();
    throw error;
  }
  enemyPursuitCommandId = String(data.command_id || "");
  updateEnemyResponseStatus(
    `Guarded pursuit started. ATLAS will hover until 3/5 live NEO detections confirm the moving track, then use short closed-loop pulses toward ${stopClearance.toFixed(2)} m.`,
    "busy"
  );
  updateEnemyResponseControls();
}

function updateFlightControlState() {
  const liveStarted = liveLocalizationStarted();
  const poseReady = firstConfirmedPoseReady();
  const bridgeControlReady = djiBridgeReadyForControl();
  const bridgeMissionReady = djiBridgeReadyForMission();
  if (djiTakeoffButton) djiTakeoffButton.disabled = !liveStarted || !bridgeControlReady;
  if (djiLandButton) djiLandButton.disabled = !liveStarted || !bridgeControlReady;
  if (djiEmergencyHoverButton) djiEmergencyHoverButton.disabled = !liveStarted || !bridgeControlReady;
  if (guidedMotionEnable) {
    guidedMotionEnable.disabled = !liveStarted || !firstLocalizationConfirmed || !poseReady || !bridgeMissionReady;
    if (guidedMotionEnable.disabled) guidedMotionEnable.checked = false;
  }
  if (confirmLocalizationButton) {
    confirmLocalizationButton.disabled = !poseReady;
    confirmLocalizationButton.textContent = firstLocalizationConfirmed
      ? "Localization Confirmed"
      : "Confirm First Localization";
  }
  updateInitialPositionControls();
  if (localizationGateStatus) {
    if (firstLocalizationConfirmed && bridgeMissionReady) {
      localizationGateStatus.textContent = "Confirmed. Mission controls are unlocked.";
    } else if (firstLocalizationConfirmed && !bridgeMissionReady) {
      const state = djiBridgeState() || "offline";
      const age = djiBridgeAgeSeconds();
      const ageText = Number.isFinite(age) ? `${age.toFixed(1)}s old` : "not available";
      localizationGateStatus.textContent = `Localization confirmed, but movement is locked until the live DJI bridge is streaming and fresh (${state}, heartbeat ${ageText}).`;
    } else if (poseReady) {
      localizationGateStatus.textContent = "First R,t is visible. Confirm it matches the map before mission planning.";
    } else if (!liveStarted) {
      localizationGateStatus.textContent = "Start live localization before takeoff or mission planning.";
    } else if (!bridgeMissionReady) {
      const state = djiBridgeState() || "offline";
      const age = djiBridgeAgeSeconds();
      const ageText = Number.isFinite(age) ? `${age.toFixed(1)}s old` : "not available";
      localizationGateStatus.textContent = `DJI bridge is not ready for movement (${state}, heartbeat ${ageText}). Keep live localization running until it is streaming.`;
    } else {
      localizationGateStatus.textContent = "Waiting for first TSolve R,t.";
    }
  }
  if (!liveStarted && djiCommandStatus) {
    setDjiCommandStatus("Start live localization to unlock takeoff and land.", "");
  } else if (liveStarted && !bridgeControlReady && djiCommandStatus) {
    const state = djiBridgeState() || "offline";
    setDjiCommandStatus(`Waiting for active DJI bridge before commands can move the drone (${state}).`, "busy");
  }
  droneControlPanel?.classList.toggle("is-locked", false);
  if (selectTargetButton) selectTargetButton.disabled = !room;
  if (clearTargetButton) clearTargetButton.disabled = !missionTarget?.rxyz;
  if (planMissionButton) planMissionButton.disabled = !missionTarget?.rxyz;
  if (startMissionButton) startMissionButton.disabled = !firstLocalizationConfirmed || !plannedMission || !guidedMotionArmed();
  if (editPatrolButton) editPatrolButton.disabled = !room;
  if (clearPatrolButton) clearPatrolButton.disabled = patrolPoints.length === 0;
  if (validatePatrolButton) validatePatrolButton.disabled = patrolPoints.length < 2;
  if (startPatrolButton) startPatrolButton.disabled = patrolPoints.length < 2;
  if (stopPatrolButton) stopPatrolButton.disabled = !liveStarted;
  if (liveAtlasPatrolSelect) liveAtlasPatrolSelect.disabled = liveStarted;
  if (startLiveAtlasButton) {
    startLiveAtlasButton.disabled = liveStarted;
    startLiveAtlasButton.title = liveStarted
      ? "A live localization profile is already active. Stop Live before selecting or starting another patrol profile."
      : "Start the selected patrol's isolated localization profile.";
  }
  const movementLock = liveMovementLockReason();
  const boundPatrol = boundLivePatrolId();
  for (const button of document.querySelectorAll('.saved-patrol-actions [data-action="play"]')) {
    const wrongProfile = Boolean(boundPatrol && button.dataset.patrolId !== boundPatrol);
    const reason = wrongProfile
      ? "This localization session is isolated to another patrol. Stop Live, select this patrol, then start localization again."
      : movementLock;
    button.disabled = Boolean(reason);
    button.title = reason || "Execute this saved patrol through the live DJI bridge.";
    const hint = button.closest(".saved-patrol-item")?.querySelector(".saved-patrol-profile-status");
    if (hint) {
      hint.textContent = reason || "Ready with the matching localization profile.";
      hint.dataset.tone = reason ? "error" : "ok";
    }
  }
  updateEnemyResponseControls();
}

function resetLocalizationGate(options = {}) {
  const preserveMission = Boolean(options.preserveMission);
  firstLocalizationConfirmed = false;
  initialPositionSelecting = false;
  initialPoseOffsetRoom = [0, 0, 0];
  if (!preserveMission) {
    plannedMission = null;
    missionTarget = null;
    missionSelecting = false;
    patrolSelecting = false;
    patrolDraggingIndex = -1;
    patrolPointHover = null;
    plannedPatrol = null;
    selectTargetButton?.classList.remove("active");
    editPatrolButton?.classList.remove("active");
    patrolControlPanel?.classList.remove("is-selecting");
    renderMissionCommands([]);
    renderPatrolCommands([]);
  }
  updateFlightControlState();
  updateMissionStatus();
  updatePatrolStatus();
}

async function sendDjiFlightCommand(command, fields = {}) {
  const normalizedCommand = String(command || "").toLowerCase();
  if (normalizedCommand === "mission" && !djiBridgeReadyForMission()) {
    const state = djiBridgeState() || "offline";
    const age = djiBridgeAgeSeconds();
    const ageText = Number.isFinite(age) ? `${age.toFixed(1)}s old` : "not available";
    throw new Error(`Live DJI bridge is not ready for movement (${state}, heartbeat ${ageText}). Start Live ATLAS and wait for streaming frames before pressing Play/Confirm.`);
  }
  if (["takeoff", "land", "hover"].includes(normalizedCommand) && !djiBridgeReadyForControl()) {
    const state = djiBridgeState() || "offline";
    throw new Error(`Live DJI bridge is not active for flight commands (${state}). Start Live ATLAS first.`);
  }
  const phoneIp = (liveAtlasPhoneIp?.value || "").trim();
  rememberPhoneIp(phoneIp);
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
  if (data.queued && data.command_id) watchDjiCommandAcknowledgement(command, data.command_id);
  return data;
}

function watchDjiCommandAcknowledgement(command, commandId) {
  pendingDjiControlAck = {
    id: commandId,
    command: String(command || "command"),
    sentAt: Date.now(),
  };
  window.setTimeout(async () => {
    if (!pendingDjiControlAck || pendingDjiControlAck.id !== commandId) return;
    try {
      const resp = await fetch(`public/live_dji/control_status.json?t=${Date.now()}`, { cache: "no-store" });
      const status = resp.ok ? await resp.json() : null;
      if (status?.id === commandId) {
        pendingDjiControlAck = null;
        if (status.status === "running") {
          setDjiCommandStatus(status.message || `${command} accepted by DJI bridge.`, "ok");
        }
        return;
      }
    } catch {
      // The bridge status poll below will surface the same problem; keep this check best-effort.
    }
    if (latestDjiLiveStatus?.last_control?.id === commandId) {
      pendingDjiControlAck = null;
      return;
    }
    const state = djiBridgeState() || "offline";
    const age = djiBridgeAgeSeconds();
    const ageText = Number.isFinite(age) ? `${age.toFixed(1)}s old` : "not available";
    const message = `${command} was queued, but the DJI bridge did not acknowledge it. Bridge state: ${state}, heartbeat ${ageText}. Keep Live ATLAS running and try again.`;
    setDjiCommandStatus(message, "error");
    if (String(command).toLowerCase() === "mission") {
      updateMissionStatus(message);
      updatePatrolStatus(message, "error");
    }
  }, 9000);
}

function asVec3(value) {
  if (!Array.isArray(value) || value.length < 3) return null;
  const out = value.slice(0, 3).map(Number);
  return out.every(Number.isFinite) ? out : null;
}

function sanitizeHexColor(value, fallback = "#cfd8df") {
  const text = String(value || "").trim();
  return /^#[0-9a-fA-F]{6}$/.test(text) ? text.toLowerCase() : fallback;
}

function selectedOpacity(input, fallback = 0.24) {
  const value = Number(input?.value ?? fallback);
  return Math.max(0.05, Math.min(0.95, Number.isFinite(value) ? value : fallback));
}

function hexToRgb(color) {
  const hex = sanitizeHexColor(color).slice(1);
  return [
    parseInt(hex.slice(0, 2), 16),
    parseInt(hex.slice(2, 4), 16),
    parseInt(hex.slice(4, 6), 16),
  ];
}

function rgbaFromHex(color, alpha = 1) {
  const [r, g, b] = hexToRgb(color);
  return `rgba(${r}, ${g}, ${b}, ${Math.max(0, Math.min(1, alpha))})`;
}

function selectedBarrierColor() {
  return sanitizeHexColor(barrierColorInput?.value, "#cfd8df");
}

function selectedObstacleColor() {
  return sanitizeHexColor(obstacleColorInput?.value, "#86dfff");
}

function selectedBarrierOpacity() {
  return selectedOpacity(barrierOpacityInput, 0.24);
}

function selectedObstacleOpacity() {
  return selectedOpacity(obstacleOpacityInput, 0.24);
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
        color: sanitizeHexColor(barrier.color, "#cfd8df"),
        opacity: Math.max(0.05, Math.min(0.95, Number(barrier.opacity ?? 0.24))),
      };
    })
    .filter(Boolean);
}

function selectedBarrierClearance() {
  const value = Number(barrierClearanceInput?.value || 0.45);
  return Math.max(0.15, Math.min(2, Number.isFinite(value) ? value : 0.45));
}

function selectedObstacleClearance() {
  const value = Number(obstacleClearanceInput?.value || 0.35);
  return Math.max(0.10, Math.min(2, Number.isFinite(value) ? value : 0.35));
}

function obstacleBoundsFromPoints(points, clearance = 0) {
  const clean = (points || []).map(asVec3).filter(Boolean);
  if (!clean.length) return null;
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const point of clean) {
    for (let axis = 0; axis < 3; axis++) {
      min[axis] = Math.min(min[axis], point[axis]);
      max[axis] = Math.max(max[axis], point[axis]);
    }
  }
  for (let axis = 0; axis < 3; axis++) {
    min[axis] -= clearance;
    max[axis] += clearance;
    if (max[axis] - min[axis] < 0.06) {
      min[axis] -= 0.03;
      max[axis] += 0.03;
    }
  }
  return { min, max };
}

function normalizedObstacleBounds(obstacle, clearance = 0) {
  const rawMin = asVec3(obstacle?.bounds?.min);
  const rawMax = asVec3(obstacle?.bounds?.max);
  const bounds = rawMin && rawMax
    ? {
        min: [Math.min(rawMin[0], rawMax[0]), Math.min(rawMin[1], rawMax[1]), Math.min(rawMin[2], rawMax[2])],
        max: [Math.max(rawMin[0], rawMax[0]), Math.max(rawMin[1], rawMax[1]), Math.max(rawMin[2], rawMax[2])],
      }
    : obstacleBoundsFromPoints(obstacle?.points || [], 0);
  if (!bounds) return null;
  const min = bounds.min.slice(0, 3);
  const max = bounds.max.slice(0, 3);
  for (let axis = 0; axis < 3; axis++) {
    min[axis] -= clearance;
    max[axis] += clearance;
    if (max[axis] - min[axis] < 0.06) {
      min[axis] -= 0.03;
      max[axis] += 0.03;
    }
  }
  return { min, max };
}

function mapSafetyObstacles() {
  const obstacles = Array.isArray(currentMapEntry?.safety_obstacles) ? currentMapEntry.safety_obstacles : [];
  return obstacles
    .map((obstacle, index) => {
      const points = (obstacle.points || []).map(asVec3).filter(Boolean);
      const clearance = Math.max(0.05, Math.min(3, Number(obstacle.clearance_m || 0.35)));
      const bounds = normalizedObstacleBounds(obstacle, 0) || obstacleBoundsFromPoints(points, 0);
      if (!bounds || points.length < 2) return null;
      return {
        id: String(obstacle.id || `obstacle_${index}`),
        label: String(obstacle.label || `Obstacle ${index + 1}`),
        points,
        bounds,
        clearance_m: clearance,
        color: sanitizeHexColor(obstacle.color, "#86dfff"),
        opacity: Math.max(0.05, Math.min(0.95, Number(obstacle.opacity ?? 0.24))),
      };
    })
    .filter(Boolean);
}

function safetyBlockerCount() {
  return mapSafetyBarriers().length + mapSafetyObstacles().length;
}

function obstaclePayloadForSave(obstacle) {
  const points = (obstacle?.points || []).map(asVec3).filter(Boolean);
  if (points.length < 2) return null;
  const clearance = Math.max(0.05, Math.min(3, Number(obstacle.clearance_m || selectedObstacleClearance())));
  const bounds = normalizedObstacleBounds(obstacle, 0) || obstacleBoundsFromPoints(points, 0);
  if (!bounds) return null;
  return {
    id: String(obstacle.id || `obstacle_${Date.now().toString(36)}`),
    label: String(obstacle.label || "Obstacle"),
    points,
    bounds,
    clearance_m: clearance,
    color: sanitizeHexColor(obstacle.color || selectedObstacleColor(), "#86dfff"),
    opacity: Math.max(0.05, Math.min(0.95, Number(obstacle.opacity ?? selectedObstacleOpacity()))),
    created_at: obstacle.created_at,
  };
}

function normalizeSafetyObstacleBank(obstacles) {
  return (obstacles || []).map(obstaclePayloadForSave).filter(Boolean);
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

function pointAabbDistance(point, bounds) {
  let sum = 0;
  for (let axis = 0; axis < 3; axis++) {
    const v = point[axis];
    const d = v < bounds.min[axis] ? bounds.min[axis] - v : (v > bounds.max[axis] ? v - bounds.max[axis] : 0);
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function segmentIntersectsAabb(a, b, bounds) {
  let tMin = 0;
  let tMax = 1;
  for (let axis = 0; axis < 3; axis++) {
    const delta = b[axis] - a[axis];
    if (Math.abs(delta) < 1e-12) {
      if (a[axis] < bounds.min[axis] || a[axis] > bounds.max[axis]) return false;
      continue;
    }
    const inv = 1 / delta;
    let t1 = (bounds.min[axis] - a[axis]) * inv;
    let t2 = (bounds.max[axis] - a[axis]) * inv;
    if (t1 > t2) [t1, t2] = [t2, t1];
    tMin = Math.max(tMin, t1);
    tMax = Math.min(tMax, t2);
    if (tMin > tMax) return false;
  }
  return true;
}

function obstacleRouteDistance3D(routeA, routeB, obstacle) {
  const bounds = normalizedObstacleBounds(obstacle, 0);
  if (!bounds) return Infinity;
  if (segmentIntersectsAabb(routeA, routeB, bounds)) return 0;
  let best = Math.min(pointAabbDistance(routeA, bounds), pointAabbDistance(routeB, bounds));
  for (let i = 1; i < 24; i++) {
    const t = i / 24;
    const p = [
      routeA[0] + (routeB[0] - routeA[0]) * t,
      routeA[1] + (routeB[1] - routeA[1]) * t,
      routeA[2] + (routeB[2] - routeA[2]) * t,
    ];
    best = Math.min(best, pointAabbDistance(p, bounds));
  }
  return best;
}

function missionRouteSafetyCheck(segments, extraClearance = FLIGHT_SAFETY_PULSE_BUFFER_M) {
  if (!segments.length) return { blocked: false, reason: "No route segments yet.", nearest: null };
  let nearest = null;
  for (const barrier of mapSafetyBarriers()) {
    const dist = Math.min(...segments.map(([a, b]) => barrierRouteDistance2D(a, b, barrier)));
    const clearance = barrier.clearance_m + extraClearance;
    if (!nearest || dist < nearest.distance) nearest = { type: "wall", barrier, label: barrier.label, distance: dist, clearance };
    if (dist <= clearance) {
      return {
        blocked: true,
        nearest: { type: "wall", barrier, label: barrier.label, distance: dist, clearance },
        reason: `${barrier.label} is ${dist.toFixed(2)} map units from the route; required clearance is ${clearance.toFixed(2)} (${barrier.clearance_m.toFixed(2)} saved + ${extraClearance.toFixed(2)} motion buffer).`,
      };
    }
  }
  for (const obstacle of mapSafetyObstacles()) {
    const dist = Math.min(...segments.map(([a, b]) => obstacleRouteDistance3D(a, b, obstacle)));
    const clearance = obstacle.clearance_m + extraClearance;
    if (!nearest || dist < nearest.distance) nearest = { type: "obstacle", obstacle, label: obstacle.label, distance: dist, clearance };
    if (dist <= clearance) {
      return {
        blocked: true,
        nearest: { type: "obstacle", obstacle, label: obstacle.label, distance: dist, clearance },
        reason: `${obstacle.label} is ${dist.toFixed(2)} map units from the route; required clearance is ${clearance.toFixed(2)} (${obstacle.clearance_m.toFixed(2)} saved + ${extraClearance.toFixed(2)} motion buffer). The route may pass over it only above its saved height.`,
      };
    }
  }
  return { blocked: false, nearest };
}

function pointSafetyIssue(point) {
  if (!point) return null;
  for (const barrier of mapSafetyBarriers()) {
    const dist = barrierRouteDistance2D(point, point, barrier);
    const clearance = barrier.clearance_m;
    if (dist <= clearance) {
      return {
        type: "wall",
        label: barrier.label,
        distance: dist,
        clearance,
        reason: `${barrier.label} is ${dist.toFixed(2)} map units from the point; required clearance is ${clearance.toFixed(2)}.`,
      };
    }
  }
  for (const obstacle of mapSafetyObstacles()) {
    const bounds = normalizedObstacleBounds(obstacle, 0);
    if (!bounds) continue;
    const dist = pointAabbDistance(point, bounds);
    const clearance = obstacle.clearance_m;
    if (dist <= clearance) {
      return {
        type: "obstacle",
        label: obstacle.label,
        distance: dist,
        clearance,
        reason: `${obstacle.label} is ${dist.toFixed(2)} map units from the point; required clearance is ${clearance.toFixed(2)}.`,
      };
    }
  }
  return null;
}

function updatePatrolPointSafetyIssues() {
  const next = new Map();
  for (let i = 0; i < patrolPoints.length; i++) {
    const target = patrolTargetPoint(patrolPoints[i]);
    const issue = pointSafetyIssue(target);
    if (issue) next.set(i, issue);
  }
  patrolPointSafetyIssues = next;
  return next;
}

function missionBarrierCheck(target = missionTarget?.rxyz) {
  const cur = closestPose();
  currentRenderedPose = cur || null;
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
  if (undoWallEditButton) undoWallEditButton.disabled = barrierSaving || wallUndoStack.length === 0;
  if (saveWallAdjustmentsButton) saveWallAdjustmentsButton.disabled = !barrierUnsaved || barrierSaving;
  if (cancelBarrierButton && !barrierEditing) cancelBarrierButton.disabled = !barrierUnsaved;
}

function markBarrierAdjustUnsaved(message = "Wall adjusted. Press Save Walls to commit.") {
  barrierUnsaved = true;
  stagedSafetyBarrierMapId = currentMapEntry?.id || null;
  updateBarrierAdjustControls();
  updateBarrierStatus(message, "busy");
}

function syncBarrierStyleInputs(barrier = null) {
  if (!barrier) {
    barrier = mapSafetyBarriers().find(candidate => candidate.id === selectedBarrierId) || null;
  }
  if (!barrier) return;
  if (barrierClearanceInput) barrierClearanceInput.value = Number(barrier.clearance_m || selectedBarrierClearance()).toFixed(2);
  if (barrierColorInput) barrierColorInput.value = sanitizeHexColor(barrier.color, "#cfd8df");
  if (barrierOpacityInput) barrierOpacityInput.value = String(Math.max(0.05, Math.min(0.95, Number(barrier.opacity ?? 0.24))));
}

function syncObstacleStyleInputs(obstacle = null) {
  if (!obstacle) {
    obstacle = mapSafetyObstacles().find(candidate => candidate.id === selectedObstacleId) || null;
  }
  if (!obstacle) return;
  if (obstacleClearanceInput) obstacleClearanceInput.value = Number(obstacle.clearance_m || selectedObstacleClearance()).toFixed(2);
  if (obstacleColorInput) obstacleColorInput.value = sanitizeHexColor(obstacle.color, "#86dfff");
  if (obstacleOpacityInput) obstacleOpacityInput.value = String(Math.max(0.05, Math.min(0.95, Number(obstacle.opacity ?? 0.24))));
}

function setSelectedBarrier(barrierId) {
  selectedBarrierId = barrierId || null;
  const barrier = mapSafetyBarriers().find(candidate => candidate.id === selectedBarrierId) || null;
  syncBarrierStyleInputs(barrier);
  renderBarrierList();
  invalidateStaticLayer();
}

function setSelectedObstacle(obstacleId, options = {}) {
  const { renderList = true } = options;
  selectedObstacleId = obstacleId || null;
  const obstacle = mapSafetyObstacles().find(candidate => candidate.id === selectedObstacleId) || null;
  syncObstacleStyleInputs(obstacle);
  if (renderList) renderObstacleList();
  else {
    obstacleList?.querySelectorAll(".obstacle-item").forEach(item => {
      item.classList.toggle("selected", item.dataset.obstacleId === selectedObstacleId);
    });
  }
  invalidateStaticLayer();
}

function updateSelectedBarrierPatch(patch, saveNow = true) {
  if (!selectedBarrierId) {
    updateBarrierStatus("Select a wall from the list or on the map before changing style.", "error");
    return;
  }
  const next = mapSafetyBarriers().map(barrier => (
    barrier.id === selectedBarrierId
      ? barrierPayloadForSave({ ...barrier, ...patch })
      : barrierPayloadForSave(barrier)
  ));
  if (currentMapEntry) currentMapEntry.safety_barriers = next;
  const libEntry = (mapLibraryData.maps || []).find(m => m.id === currentMapEntry?.id);
  if (libEntry) libEntry.safety_barriers = next;
  renderBarrierList();
  invalidateStaticLayer();
  plannedPatrol = null;
  renderPatrolCommands([]);
  if (patrolPoints.length >= 2) validatePatrolPreview(false);
  if (saveNow) saveSafetyBarriers(next);
}

function updateSelectedObstaclePatch(patch, saveNow = true) {
  if (!selectedObstacleId) {
    updateObstacleStatus("Select an object from the list or on the map before changing style.", "error");
    return;
  }
  const next = mapSafetyObstacles().map(obstacle => (
    obstacle.id === selectedObstacleId
      ? obstaclePayloadForSave({ ...obstacle, ...patch })
      : obstaclePayloadForSave(obstacle)
  )).filter(Boolean);
  if (currentMapEntry) currentMapEntry.safety_obstacles = next;
  const libEntry = (mapLibraryData.maps || []).find(m => m.id === currentMapEntry?.id);
  if (libEntry) libEntry.safety_obstacles = next;
  renderObstacleList();
  invalidateStaticLayer();
  plannedPatrol = null;
  renderPatrolCommands([]);
  if (patrolPoints.length >= 2) validatePatrolPreview(false);
  if (saveNow) saveSafetyObstacles(next);
}

function commitObstacleRename(obstacleId, rawLabel) {
  const label = String(rawLabel || "").trim();
  if (!label) {
    updateObstacleStatus("Object name cannot be empty.", "error");
    renderObstacleList();
    return false;
  }
  pushObstacleUndoSnapshot();
  const next = mapSafetyObstacles().map(obstacle => (
    obstacle.id === obstacleId
      ? obstaclePayloadForSave({ ...obstacle, label })
      : obstaclePayloadForSave(obstacle)
  )).filter(Boolean);
  selectedObstacleId = obstacleId;
  if (currentMapEntry) currentMapEntry.safety_obstacles = next;
  const libEntry = (mapLibraryData.maps || []).find(m => m.id === currentMapEntry?.id);
  if (libEntry) libEntry.safety_obstacles = next;
  renderObstacleList();
  invalidateStaticLayer();
  plannedPatrol = null;
  plannedMission = null;
  renderPatrolCommands([]);
  renderMissionCommands([]);
  updateObstacleStatus(`Renamed object to "${label}".`, "busy");
  saveSafetyObstacles(next);
  return true;
}

function beginObstacleRename(obstacle, nameElement) {
  if (!obstacle || !nameElement) return;
  selectedObstacleId = obstacle.id;
  syncObstacleStyleInputs(obstacle);
  nameElement.closest(".obstacle-item")?.classList.add("selected");
  invalidateStaticLayer();
  const input = document.createElement("input");
  input.type = "text";
  input.className = "obstacle-name-edit";
  input.value = obstacle.label || "";
  input.setAttribute("aria-label", "Rename safety object");
  let finished = false;
  const finish = save => {
    if (finished) return;
    finished = true;
    if (save) {
      const nextLabel = input.value.trim();
      if (nextLabel && nextLabel !== (obstacle.label || "")) {
        commitObstacleRename(obstacle.id, nextLabel);
        return;
      }
    }
    renderObstacleList();
  };
  input.addEventListener("click", event => event.stopPropagation());
  input.addEventListener("dblclick", event => event.stopPropagation());
  input.addEventListener("keydown", event => {
    if (event.key === "Enter") {
      event.preventDefault();
      event.stopPropagation();
      finish(true);
    } else if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      finish(false);
    }
  });
  input.addEventListener("blur", () => finish(true));
  nameElement.replaceWith(input);
  requestAnimationFrame(() => {
    input.focus();
    input.select();
  });
}

function renderBarrierList() {
  if (!barrierList) return;
  updateBarrierAdjustControls();
  const barriers = mapSafetyBarriers();
  if (selectedBarrierId && !barriers.some(barrier => barrier.id === selectedBarrierId)) selectedBarrierId = null;
  barrierList.innerHTML = "";
  if (!barriers.length) {
    updateBarrierStatus();
    return;
  }
  for (const barrier of barriers) {
    const item = document.createElement("div");
    item.className = "barrier-item";
    if (barrier.id === selectedBarrierId) item.classList.add("selected");
    const text = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = barrier.label;
    const detail = document.createElement("span");
    detail.textContent = `4 corners, clearance ${barrier.clearance_m.toFixed(2)}, opacity ${(Number(barrier.opacity ?? 0.24) * 100).toFixed(0)}%`;
    text.append(name, detail);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Remove";
    remove.addEventListener("click", event => {
      event.stopPropagation();
      pushWallUndoSnapshot();
      saveSafetyBarriers(barriers.filter(candidate => candidate.id !== barrier.id));
    });
    item.addEventListener("click", () => setSelectedBarrier(barrier.id));
    item.append(text, remove);
    barrierList.appendChild(item);
  }
  updateBarrierStatus();
}

function updateObstacleStatus(message = null, tone = "") {
  if (!obstacleStatus) return;
  obstacleStatus.dataset.tone = tone;
  if (message) {
    obstacleStatus.textContent = message;
    return;
  }
  if (obstacleEditing) {
    const count = obstacleDraft?.points?.length || 0;
    obstacleStatus.textContent = count
      ? `${count} point${count === 1 ? "" : "s"} selected. Pick more points around the object, then Save Object.`
      : "Pick existing COLMAP points around the finite object.";
    return;
  }
  const count = mapSafetyObstacles().length;
  obstacleStatus.textContent = count
    ? `${count} finite object${count === 1 ? "" : "s"} saved. Routes may go over or under them when clearance allows.`
    : "Pick existing COLMAP points around furniture or objects. ATLAS fills a finite 3D safety structure.";
}

function updateObstacleControls() {
  const draftCount = obstacleDraft?.points?.length || 0;
  if (addObstacleButton) addObstacleButton.classList.toggle("active", obstacleEditing);
  if (finishObstacleButton) finishObstacleButton.disabled = !obstacleEditing || draftCount < 2 || barrierSaving;
  if (cancelObstacleButton) cancelObstacleButton.disabled = !obstacleEditing || barrierSaving;
  if (undoObstacleEditButton) undoObstacleEditButton.disabled = barrierSaving || obstacleUndoStack.length === 0;
  if (clearObstaclePointsButton) {
    clearObstaclePointsButton.disabled = barrierSaving || !(draftCount > 0 || selectedObstacleId);
  }
}

function cloneBarrierBank(barriers = mapSafetyBarriers()) {
  return (barriers || []).map(barrierPayloadForSave).filter(barrier => (barrier.corners || []).length >= 4);
}

function cloneObstacleBank(obstacles = mapSafetyObstacles()) {
  return (obstacles || []).map(obstaclePayloadForSave).filter(Boolean);
}

function pushWallUndoSnapshot() {
  const snapshot = cloneBarrierBank();
  wallUndoStack.push(snapshot);
  if (wallUndoStack.length > 32) wallUndoStack.shift();
  updateBarrierAdjustControls();
}

function pushObstacleUndoSnapshot() {
  const snapshot = cloneObstacleBank();
  obstacleUndoStack.push(snapshot);
  if (obstacleUndoStack.length > 32) obstacleUndoStack.shift();
  updateObstacleControls();
}

function restoreWallSnapshot(snapshot) {
  const next = cloneBarrierBank(snapshot);
  stagedSafetyBarrierMapId = currentMapEntry?.id || null;
  stagedSafetyBarriers = next;
  if (currentMapEntry) currentMapEntry.safety_barriers = next;
  const libEntry = (mapLibraryData.maps || []).find(m => m.id === currentMapEntry?.id);
  if (libEntry) libEntry.safety_barriers = next;
  selectedBarrierId = next.some(barrier => barrier.id === selectedBarrierId) ? selectedBarrierId : next[0]?.id || null;
  barrierUnsaved = true;
  plannedMission = null;
  plannedPatrol = null;
  renderMissionCommands([]);
  renderPatrolCommands([]);
  renderBarrierList();
  invalidateStaticLayer();
  updateMissionStatus();
  updatePatrolStatus();
  markBarrierAdjustUnsaved("Undo applied. Press Save Walls to commit the restored wall layout.");
}

function undoWallEdit() {
  const snapshot = wallUndoStack.pop();
  if (!snapshot) return;
  barrierCornerDrag = null;
  barrierTransformDrag = null;
  clearBarrierHover();
  restoreWallSnapshot(snapshot);
  updateBarrierAdjustControls();
}

function undoObstacleEdit() {
  const snapshot = obstacleUndoStack.pop();
  if (!snapshot) return;
  obstaclePointDrag = null;
  obstacleTransformDrag = null;
  obstaclePointHover = null;
  obstacleTransformHover = null;
  selectedObstacleId = snapshot.some(obstacle => obstacle.id === selectedObstacleId) ? selectedObstacleId : snapshot[0]?.id || null;
  saveSafetyObstacles(snapshot);
  updateObstacleControls();
}

function renderObstacleList() {
  if (!obstacleList) return;
  updateObstacleControls();
  const obstacles = mapSafetyObstacles();
  if (selectedObstacleId && !obstacles.some(obstacle => obstacle.id === selectedObstacleId)) selectedObstacleId = null;
  obstacleList.innerHTML = "";
  if (!obstacles.length) {
    updateObstacleStatus();
    return;
  }
  for (const obstacle of obstacles) {
    const item = document.createElement("div");
    item.className = "barrier-item obstacle-item";
    item.dataset.obstacleId = obstacle.id;
    if (obstacle.id === selectedObstacleId) item.classList.add("selected");
    const text = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = obstacle.label;
    name.title = "Double-click to rename, Enter to save";
    name.addEventListener("dblclick", event => {
      event.preventDefault();
      event.stopPropagation();
      beginObstacleRename(obstacle, name);
    });
    const height = obstacle.bounds.max[1] - obstacle.bounds.min[1];
    const detail = document.createElement("span");
    detail.textContent = `${obstacle.points.length} points, height ${height.toFixed(2)}, clearance ${obstacle.clearance_m.toFixed(2)}, opacity ${(Number(obstacle.opacity ?? 0.24) * 100).toFixed(0)}%`;
    text.append(name, detail);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Remove";
    remove.addEventListener("click", event => {
      event.stopPropagation();
      pushObstacleUndoSnapshot();
      saveSafetyObstacles(obstacles.filter(candidate => candidate.id !== obstacle.id));
    });
    const rename = document.createElement("button");
    rename.type = "button";
    rename.textContent = "Rename";
    rename.addEventListener("click", event => {
      event.stopPropagation();
      beginObstacleRename(obstacle, name);
    });
    item.addEventListener("click", () => setSelectedObstacle(obstacle.id, { renderList: false }));
    item.addEventListener("dblclick", event => {
      if (event.target.closest("button") || event.target.closest("input")) return;
      event.preventDefault();
      event.stopPropagation();
      beginObstacleRename(obstacle, name);
    });
    item.append(text, rename, remove);
    obstacleList.appendChild(item);
  }
  updateObstacleStatus();
}

async function saveSafetyBarriers(nextBarriers) {
  if (!currentMapEntry?.id || barrierSaving) return;
  const barriersForSave = normalizeSafetyBarrierBank(nextBarriers);
  const obstaclesForSave = normalizeSafetyObstacleBank(mapSafetyObstacles());
  barrierSaving = true;
  updateBarrierStatus("Saving safety barriers...", "busy");
  try {
    const data = await postJson("/api/map/barriers", {
      map_id: currentMapEntry.id,
      barriers: barriersForSave,
      obstacles: obstaclesForSave,
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
    renderObstacleList();
    invalidateStaticLayer();
    plannedPatrol = null;
    renderPatrolCommands([]);
    if (patrolPoints.length >= 2) validatePatrolPreview(false);
    else updatePatrolStatus();
    updateMissionStatus();
  } catch (err) {
    updateBarrierStatus(`Could not save safety barriers: ${err.message || err}`, "error");
  } finally {
    barrierSaving = false;
    updateBarrierAdjustControls();
    updateObstacleControls();
  }
}

async function saveSafetyObstacles(nextObstacles) {
  if (!currentMapEntry?.id || barrierSaving) return;
  const barriersForSave = normalizeSafetyBarrierBank(mapSafetyBarriers());
  const obstaclesForSave = normalizeSafetyObstacleBank(nextObstacles);
  barrierSaving = true;
  updateObstacleStatus("Saving safety obstacles...", "busy");
  try {
    const data = await postJson("/api/map/barriers", {
      map_id: currentMapEntry.id,
      barriers: barriersForSave,
      obstacles: obstaclesForSave,
    });
    if (data.state?.library) mapLibraryData = data.state.library;
    currentMapEntry = selectedMap() || data.map || currentMapEntry;
    obstacleEditing = false;
    obstacleDraft = null;
    selectedObstacleId = null;
    obstaclePointHover = null;
    obstaclePointDrag = null;
    obstacleTransformHover = null;
    obstacleTransformDrag = null;
    renderObstacleList();
    renderBarrierList();
    invalidateStaticLayer();
    plannedMission = null;
    plannedPatrol = null;
    renderMissionCommands([]);
    renderPatrolCommands([]);
    if (patrolPoints.length >= 2) validatePatrolPreview(false);
    else updatePatrolStatus();
    updateMissionStatus();
  } catch (err) {
    updateObstacleStatus(`Could not save safety obstacles: ${err.message || err}`, "error");
  } finally {
    barrierSaving = false;
    updateObstacleControls();
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
    color: selectedBarrierColor(),
    opacity: selectedBarrierOpacity(),
  });
  selectedBarrierId = next[next.length - 1].id;
  pushWallUndoSnapshot();
  saveSafetyBarriers(next);
  return true;
}

function setSafetyBarrierMode(mode) {
  safetyBarrierMode = mode === "obstacles" ? "obstacles" : "walls";
  safetyTabWallsButton?.classList.toggle("active", safetyBarrierMode === "walls");
  safetyTabObstaclesButton?.classList.toggle("active", safetyBarrierMode === "obstacles");
  safetyTabWallsButton?.setAttribute("aria-selected", String(safetyBarrierMode === "walls"));
  safetyTabObstaclesButton?.setAttribute("aria-selected", String(safetyBarrierMode === "obstacles"));
  wallTools?.classList.toggle("active", safetyBarrierMode === "walls");
  obstacleTools?.classList.toggle("active", safetyBarrierMode === "obstacles");
  if (safetyBarrierMode !== "obstacles") {
    obstacleEditing = false;
    obstacleDraft = null;
    obstaclePointHover = null;
    obstacleTransformHover = null;
    obstaclePointDrag = null;
    obstacleTransformDrag = null;
  }
  if (safetyBarrierMode !== "walls") {
    barrierEditing = false;
    barrierDraft = null;
    addBarrierButton?.classList.remove("active");
  }
  updateObstacleControls();
  updateObstacleStatus();
  updateBarrierAdjustControls();
  updateBarrierStatus();
}

function addObstacleFromPickedPoint(picked) {
  if (!picked?.rxyz) return false;
  if (!obstacleDraft) {
    obstacleDraft = {
      id: `obstacle_${Date.now().toString(36)}`,
      label: `Obstacle ${mapSafetyObstacles().length + 1}`,
      points: [],
      clearance_m: selectedObstacleClearance(),
      color: selectedObstacleColor(),
      opacity: selectedObstacleOpacity(),
    };
  }
  const point = picked.rxyz.slice(0, 3);
  if (obstacleDraft.points.some(existing => norm(sub(existing, point)) < 0.04)) {
    updateObstacleStatus("That point is already part of this obstacle. Pick another visible point.", "error");
    return false;
  }
  obstacleDraft.points.push(point);
  obstacleDraft.clearance_m = selectedObstacleClearance();
  obstacleDraft.color = selectedObstacleColor();
  obstacleDraft.opacity = selectedObstacleOpacity();
  obstacleDraft.bounds = obstacleBoundsFromPoints(obstacleDraft.points, 0);
  updateObstacleControls();
  updateObstacleStatus();
  invalidateStaticLayer();
  return true;
}

function finishObstacleDraft() {
  if (!obstacleDraft || (obstacleDraft.points || []).length < 2) {
    updateObstacleStatus("Pick at least two visible points around the object before saving.", "error");
    return;
  }
  const obstacles = mapSafetyObstacles();
  const payload = obstaclePayloadForSave({
    ...obstacleDraft,
    label: `Obstacle ${obstacles.length + 1}`,
    clearance_m: selectedObstacleClearance(),
    color: selectedObstacleColor(),
    opacity: selectedObstacleOpacity(),
  });
  if (!payload) {
    updateObstacleStatus("Could not form a finite 3D object from those points. Pick a wider point cluster.", "error");
    return;
  }
  selectedObstacleId = payload.id;
  pushObstacleUndoSnapshot();
  saveSafetyObstacles(obstacles.concat(payload));
}

function planMissionPreview() {
  if (!missionTarget?.rxyz) {
    renderMissionCommands([]);
    updateMissionStatus("Pick an existing COLMAP point before planning.");
    return;
  }
  const requestedSpeed = Number(missionSpeedSelect?.value || 0.4);
  const speed = missionCommandSpeed(requestedSpeed);
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
      renderMissionCommands([]);
      updateMissionStatus(`Mission blocked by a safety barrier. ${routePlan.reason || safety.reason}`);
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
    requested_speed: requestedSpeed,
    distance,
    safety,
    pending_current_pose: !currentPoseReady,
    created_at: Date.now(),
  };
  plannedMission.commands = buildMissionCommandPlan(plannedMission);
  renderMissionCommands(plannedMission.commands);
  const distText = distance == null ? "distance pending until first live R,t" : `${distance.toFixed(2)} map units`;
  const clearanceText = safety.nearest
    ? ` Nearest safety clearance: ${safety.nearest.distance.toFixed(2)} map units.`
    : "";
  const actionText = profile?.targetLooksGround ? "horizontal approach above the point, then land" : "horizontal approach, then descend";
  const detourText = routePlan?.detoured ? " with a safety-barrier detour" : "";
  const speedText = requestedSpeed > speed + 1e-6 ? `${speed.toFixed(2)} m/s indoor cap` : `${speed.toFixed(2)} m/s`;
  const gateText = firstLocalizationConfirmed ? "Confirm before any autonomous command." : "Start live localization and confirm first R,t before execution.";
  updateMissionStatus(`Preflight path saved${detourText}: ${actionText} at ${speedText} (${distText}).${clearanceText} ${gateText}`);
  updateFlightControlState();
}

async function startLiveAtlas() {
  const mapId = currentMapEntry?.id || mapLibraryData?.selected_map_id || "default_demo";
  const patrolProfile = selectedLivePatrolProfile();
  const patrolId = String(patrolProfile?.patrol_id || "").trim();
  const phoneIp = (liveAtlasPhoneIp?.value || "").trim();
  const fps = selectedLiveAtlasFps();
  const liveCheckOnly = patrolProfile?.flight_enabled === false;
  if (!phoneIp) {
    uploadStatus.textContent = "Enter the Android phone IP before starting Live ATLAS.";
    return;
  }
  rememberPhoneIp(phoneIp);
  pendingLivePatrolId = patrolId;
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
  resetLiveFrameLockedPlayback();
  liveVideoWaitingForFirstPose = false;
  liveVideoSyncedToFirstPose = false;
  liveAtlasPreviewActive = true;
  clearUploadedVideoPreview();
  setLiveFrameMode(true);
  if (liveFrameView) liveFrameView.removeAttribute("src");
  setLiveFrameStatus("Connecting to Android MSDK stream. Waiting for first live DJI frame...", true);
  setDjiCommandStatus(
    liveCheckOnly
      ? "Live Check started: video and localization are enabled; all flight commands remain locked."
      : "Live localization started. Takeoff is now unlocked; confirm before sending any flight command.",
    "ok",
  );
  updateFlightControlState();
  const patrol = patrolList(currentMapEntry).find(item => item.id === patrolId);
  uploadStatus.textContent = `Starting Live ATLAS on ${currentMapEntry?.title || mapId}${patrol ? ` for ${patrolTitle(patrol)}` : ""}`;
  await loadViewerData(false, currentMapEntry);
  liveReplayWaitingViewPrepared = true;
  showDemo({ resetVideo: false });
  renderReplayTabs();
  try {
    await postJson("/api/drone/live-atlas", {
      map_id: mapId,
      patrol_id: patrolId || null,
      phone_ip: phoneIp,
      fps,
      max_size: 1200,
      view_only: liveCheckOnly,
    });
  } catch (error) {
    liveReplayInFlight = false;
    liveAtlasPreviewActive = false;
    pendingLiveReplayOpen = false;
    pendingLiveReplayMapId = null;
    pendingLivePatrolId = "";
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
  pendingLivePatrolId = "";
  firstLocalizationConfirmed = false;
  plannedMission = null;
  renderMissionCommands([]);
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

function enemyStatusLabel(status) {
  const value = String(status || "not_trained");
  if (value === "needs_videos") return "needs videos";
  if (value === "needs_labels") return "needs labels";
  if (value === "labels_ready") return "labels ready";
  if (value === "dataset_ready") return "dataset ready";
  if (value === "dataset_ready_needs_labels") return "dataset needs labels";
  if (value === "trained") return "trained";
  if (value === "queued") return "training queued";
  if (value === "training") return "training";
  if (value === "training_failed") return "training failed";
  return value.replaceAll("_", " ");
}

function currentEnemyProfile() {
  const enemies = Array.isArray(enemyLibraryData.enemies) ? enemyLibraryData.enemies : [];
  return enemies.find(enemy => enemy.id === selectedEnemyId) || enemies[0] || null;
}

function currentEnemyFrames() {
  const profile = currentEnemyProfile();
  return Array.isArray(profile?.frames) ? profile.frames : [];
}

function currentEnemyFrame() {
  const frames = currentEnemyFrames();
  return frames.find(frame => frame.id === selectedEnemyFrameId) || frames[0] || null;
}

function enemyFrameStats(profile) {
  const frames = Array.isArray(profile?.frames) ? profile.frames : [];
  const labeled = frames.filter(frame => frame.status === "labeled").length;
  const review = frames.filter(frame => frame.status === "review").length;
  const skipped = frames.filter(frame => frame.status === "skipped").length;
  return { total: frames.length, labeled, review, skipped };
}

function setEnemyAnnotationStatus(message) {
  if (enemyAnnotationStatus) enemyAnnotationStatus.textContent = message;
}

function selectEnemyProfile(enemyId, frameId = "") {
  const enemies = Array.isArray(enemyLibraryData.enemies) ? enemyLibraryData.enemies : [];
  const profile = enemies.find(enemy => enemy.id === enemyId) || enemies[0] || null;
  selectedEnemyId = profile?.id || "";
  const frames = Array.isArray(profile?.frames) ? profile.frames : [];
  const frame = frames.find(item => item.id === frameId) || frames[0] || null;
  selectedEnemyFrameId = frame?.id || "";
  enemyBoxDraft = frame?.box ? { ...frame.box } : null;
  loadEnemyAnnotationImage(frame);
  renderEnemyLibrary();
}

function loadEnemyAnnotationImage(frame) {
  enemyAnnotationImageReady = false;
  enemyAnnotationImageFrameId = frame?.id || "";
  enemyCanvasRectCache = null;
  if (!frame?.url) {
    drawEnemyAnnotationCanvas();
    return;
  }
  enemyAnnotationImage = new Image();
  enemyAnnotationImage.onload = () => {
    enemyAnnotationImageReady = true;
    drawEnemyAnnotationCanvas();
  };
  enemyAnnotationImage.onerror = () => {
    enemyAnnotationImageReady = false;
    drawEnemyAnnotationCanvas();
  };
  enemyAnnotationImage.src = `${frame.url}?v=${Date.now()}`;
}

function resizeEnemyAnnotationCanvas() {
  if (!enemyAnnotationCanvas) return;
  const rect = enemyAnnotationCanvas.getBoundingClientRect();
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const width = Math.max(640, Math.round((rect.width || 960) * dpr));
  const height = Math.max(360, Math.round((rect.height || 540) * dpr));
  if (enemyAnnotationCanvas.width !== width || enemyAnnotationCanvas.height !== height) {
    enemyAnnotationCanvas.width = width;
    enemyAnnotationCanvas.height = height;
    enemyCanvasRectCache = null;
  }
}

function enemyImageRect() {
  if (!enemyAnnotationCanvas || !enemyAnnotationImageReady || !enemyAnnotationImage.naturalWidth) return null;
  if (enemyCanvasRectCache) return enemyCanvasRectCache;
  const canvasWidth = enemyAnnotationCanvas.width;
  const canvasHeight = enemyAnnotationCanvas.height;
  const imageWidth = enemyAnnotationImage.naturalWidth;
  const imageHeight = enemyAnnotationImage.naturalHeight;
  const scale = Math.min(canvasWidth / imageWidth, canvasHeight / imageHeight);
  const width = imageWidth * scale;
  const height = imageHeight * scale;
  enemyCanvasRectCache = {
    x: (canvasWidth - width) / 2,
    y: (canvasHeight - height) / 2,
    width,
    height,
  };
  return enemyCanvasRectCache;
}

function enemyCanvasPoint(event) {
  const rect = enemyAnnotationCanvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (enemyAnnotationCanvas.width / rect.width),
    y: (event.clientY - rect.top) * (enemyAnnotationCanvas.height / rect.height),
  };
}

function enemyNormalizedPoint(point) {
  const rect = enemyImageRect();
  if (!rect) return null;
  return {
    x: Math.max(0, Math.min(1, (point.x - rect.x) / rect.width)),
    y: Math.max(0, Math.min(1, (point.y - rect.y) / rect.height)),
  };
}

function enemyBoxFromCorners(start, end) {
  const x1 = Math.max(0, Math.min(1, Math.min(start.x, end.x)));
  const x2 = Math.max(0, Math.min(1, Math.max(start.x, end.x)));
  const y1 = Math.max(0, Math.min(1, Math.min(start.y, end.y)));
  const y2 = Math.max(0, Math.min(1, Math.max(start.y, end.y)));
  const width = x2 - x1;
  const height = y2 - y1;
  if (width < 0.01 || height < 0.01) return null;
  return {
    x_center: x1 + width / 2,
    y_center: y1 + height / 2,
    width,
    height,
  };
}

function drawEnemyAnnotationCanvas() {
  if (!enemyAnnotationCanvas) return;
  resizeEnemyAnnotationCanvas();
  const localCtx = enemyAnnotationCanvas.getContext("2d");
  const w = enemyAnnotationCanvas.width;
  const h = enemyAnnotationCanvas.height;
  localCtx.clearRect(0, 0, w, h);
  localCtx.fillStyle = "#020916";
  localCtx.fillRect(0, 0, w, h);
  localCtx.strokeStyle = "rgba(109, 223, 255, 0.16)";
  localCtx.lineWidth = 1;
  for (let x = 0; x < w; x += 48) {
    localCtx.beginPath();
    localCtx.moveTo(x, 0);
    localCtx.lineTo(x, h);
    localCtx.stroke();
  }
  for (let y = 0; y < h; y += 48) {
    localCtx.beginPath();
    localCtx.moveTo(0, y);
    localCtx.lineTo(w, y);
    localCtx.stroke();
  }
  const frame = currentEnemyFrame();
  if (!frame || !enemyAnnotationImageReady) {
    localCtx.fillStyle = "#a7e8ff";
    localCtx.font = `${Math.max(15, Math.round(w / 58))}px Inter, system-ui, sans-serif`;
    localCtx.textAlign = "center";
    localCtx.fillText(frame ? "Loading calibration frame..." : "Extract frames, then select a frame to annotate.", w / 2, h / 2);
    return;
  }
  const rect = enemyImageRect();
  localCtx.drawImage(enemyAnnotationImage, rect.x, rect.y, rect.width, rect.height);
  const box = enemyBoxDraft || frame.box;
  if (box) {
    const x = rect.x + (box.x_center - box.width / 2) * rect.width;
    const y = rect.y + (box.y_center - box.height / 2) * rect.height;
    const bw = box.width * rect.width;
    const bh = box.height * rect.height;
    const reviewBox = frame.status === "review" && !enemyBoxDraft;
    localCtx.fillStyle = reviewBox ? "rgba(255, 147, 51, 0.13)" : "rgba(255, 209, 74, 0.12)";
    localCtx.fillRect(x, y, bw, bh);
    localCtx.strokeStyle = reviewBox ? "#ff9f43" : "#ffd84e";
    localCtx.lineWidth = Math.max(2, w / 480);
    localCtx.strokeRect(x, y, bw, bh);
    localCtx.fillStyle = "rgba(1, 8, 20, 0.78)";
    localCtx.fillRect(x, Math.max(0, y - 30), 142, 28);
    localCtx.fillStyle = "#fff3a3";
    localCtx.font = `${Math.max(13, Math.round(w / 72))}px Inter, system-ui, sans-serif`;
    localCtx.textAlign = "left";
    localCtx.fillText(reviewBox ? "review match" : "enemy drone", x + 9, Math.max(20, y - 10));
  }
}

function renderEnemyAnnotation() {
  const enemies = Array.isArray(enemyLibraryData.enemies) ? enemyLibraryData.enemies : [];
  if (!selectedEnemyId && enemies.length) selectedEnemyId = enemies[0].id;
  const profile = currentEnemyProfile();
  if (enemyAnnotationProfile) {
    enemyAnnotationProfile.innerHTML = enemies.map(enemy => (
      `<option value="${escapeHtml(enemy.id)}"${enemy.id === selectedEnemyId ? " selected" : ""}>${escapeHtml(enemy.name)}</option>`
    )).join("") || "<option value=\"\">No profiles</option>";
  }
  const frames = currentEnemyFrames();
  const frame = currentEnemyFrame();
  if (profile && !selectedEnemyFrameId && frame) selectedEnemyFrameId = frame.id;
  if (frame && enemyAnnotationImageFrameId !== frame.id) {
    enemyBoxDraft = frame.box ? { ...frame.box } : null;
    loadEnemyAnnotationImage(frame);
  }
  if (enemyFrameStrip) {
    enemyFrameStrip.innerHTML = "";
    if (!profile) {
      enemyFrameStrip.innerHTML = "<div class=\"enemy-empty compact\">No enemy profile selected.</div>";
    } else if (!frames.length) {
      enemyFrameStrip.innerHTML = "<div class=\"enemy-empty compact\">No frames yet. Click Extract Frames.</div>";
    } else {
      frames.forEach((item, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `enemy-frame-thumb ${item.id === selectedEnemyFrameId ? "active" : ""} ${item.status || "unlabeled"}`;
        button.dataset.frameId = item.id;
        button.innerHTML = `
          <img src="${escapeHtml(item.url)}" alt="" loading="lazy" />
          <span>${String(index + 1).padStart(3, "0")}</span>
          <b>${escapeHtml(item.status || "unlabeled")}</b>
        `;
        button.addEventListener("click", () => {
          selectedEnemyFrameId = item.id;
          enemyBoxDraft = item.box ? { ...item.box } : null;
          loadEnemyAnnotationImage(item);
          renderEnemyAnnotation();
        });
        enemyFrameStrip.appendChild(button);
      });
    }
  }
  if (profile) {
    const stats = enemyFrameStats(profile);
    const frameText = frame ? ` · frame ${frames.findIndex(f => f.id === frame.id) + 1}/${frames.length}` : "";
    setEnemyAnnotationStatus(`${profile.name}: ${stats.labeled}/${stats.total} labeled, ${stats.review} review, ${stats.skipped} skipped${frameText}. Draw one tight box, then auto-track the clip.`);
  } else {
    setEnemyAnnotationStatus("Choose an enemy drone profile to extract frames and draw bounding boxes.");
  }
  drawEnemyAnnotationCanvas();
}

function renderEnemyLibrary() {
  if (!enemyDroneList) return;
  const enemies = Array.isArray(enemyLibraryData.enemies) ? enemyLibraryData.enemies : [];
  const modelStatus = enemyStatusLabel(enemyLibraryData.model_status || "not_trained");
  if (enemyModelStatus) enemyModelStatus.textContent = modelStatus;
  if (enemyModelNote) {
    const totalVideos = Number(enemyLibraryData.total_videos || 0);
    const trainedModel = enemyLibraryData.selected_model ? " · live patrol enabled" : "";
    const trainingMessage = enemyLibraryData.training_message ? ` · ${enemyLibraryData.training_message}` : "";
    enemyModelNote.textContent = enemies.length
      ? `${enemies.length} drone profile${enemies.length === 1 ? "" : "s"} · ${totalVideos} calibration clip${totalVideos === 1 ? "" : "s"}${trainedModel}${trainingMessage}`
      : "Upload enemy-drone clips to begin detector calibration.";
  }
  enemyDroneList.innerHTML = "";
  if (!enemies.length) {
    enemyDroneList.innerHTML = `
      <div class="enemy-empty">
        <strong>No enemy drones registered yet.</strong>
        <span>Upload short clips of a target drone from multiple angles to create the first YOLO class.</span>
      </div>
    `;
    renderEnemyAnnotation();
    return;
  }
  if (!selectedEnemyId || !enemies.some(enemy => enemy.id === selectedEnemyId)) selectedEnemyId = enemies[0].id;

  for (const enemy of enemies) {
    const videos = Array.isArray(enemy.videos) ? enemy.videos : [];
    const frameStats = enemyFrameStats(enemy);
    const card = document.createElement("article");
    card.className = `enemy-drone-card${enemy.id === selectedEnemyId ? " selected" : ""}`;
    card.dataset.enemyId = enemy.id;
    card.innerHTML = `
      <div class="enemy-card-main">
        <button type="button" class="enemy-delete" data-enemy-id="${enemy.id}" title="Delete this enemy drone">×</button>
        <div class="enemy-drone-orbit" aria-hidden="true">
          <span></span><span></span><span></span><span></span>
        </div>
        <div>
          <p class="enemy-kicker">${escapeHtml(enemy.class_name || "enemy_drone")}</p>
          <h3 class="enemy-card-title" data-enemy-id="${enemy.id}" title="Double-click to rename">${escapeHtml(enemy.name || "Enemy Drone")}</h3>
          <p>${videos.length} clip${videos.length === 1 ? "" : "s"} · ${frameStats.labeled}/${frameStats.total} labels · ${frameStats.review} review · ${escapeHtml(enemyStatusLabel(enemy.training_status))}</p>
        </div>
      </div>
      <div class="enemy-video-strip">
        ${videos.slice(0, 4).map(video => `
          <a href="${escapeHtml(video.url)}" target="_blank" rel="noreferrer">
            <span>clip</span>
            <strong>${escapeHtml(video.filename)}</strong>
          </a>
        `).join("") || "<span class=\"enemy-no-video\">No videos yet</span>"}
        ${videos.length > 4 ? `<span class="enemy-more">+${videos.length - 4} more</span>` : ""}
      </div>
      <div class="enemy-card-actions">
        <label class="enemy-small-upload">
          Upload More
          <input class="enemy-more-upload" data-enemy-id="${enemy.id}" type="file" accept="video/*" multiple />
        </label>
        <button type="button" class="enemy-extract" data-enemy-id="${enemy.id}">Extract Frames</button>
        <button type="button" class="enemy-prepare" data-enemy-id="${enemy.id}">Prepare Class</button>
        <button type="button" class="enemy-rename" data-enemy-id="${enemy.id}">Rename</button>
      </div>
    `;
    enemyDroneList.appendChild(card);
  }

  for (const card of enemyDroneList.querySelectorAll(".enemy-drone-card")) {
    card.addEventListener("click", event => {
      if (event.target.closest("button, input, label, a")) return;
      selectEnemyProfile(card.dataset.enemyId);
    });
  }
  for (const title of enemyDroneList.querySelectorAll(".enemy-card-title")) {
    title.addEventListener("dblclick", event => {
      event.stopPropagation();
      runUi(() => renameEnemyDrone(title.dataset.enemyId));
    });
  }
  for (const button of enemyDroneList.querySelectorAll(".enemy-rename")) {
    button.addEventListener("click", event => {
      event.stopPropagation();
      runUi(() => renameEnemyDrone(button.dataset.enemyId));
    });
  }
  for (const button of enemyDroneList.querySelectorAll(".enemy-delete")) {
    button.addEventListener("click", event => {
      event.stopPropagation();
      runUi(() => deleteEnemyDrone(button.dataset.enemyId));
    });
  }
  for (const button of enemyDroneList.querySelectorAll(".enemy-prepare")) {
    button.addEventListener("click", event => {
      event.stopPropagation();
      runUi(() => prepareEnemyYoloDataset(button.dataset.enemyId));
    });
  }
  for (const button of enemyDroneList.querySelectorAll(".enemy-extract")) {
    button.addEventListener("click", event => {
      event.stopPropagation();
      runUi(() => extractEnemyFrames(button.dataset.enemyId));
    });
  }
  for (const input of enemyDroneList.querySelectorAll(".enemy-more-upload")) {
    input.addEventListener("change", event => runUi(async () => {
      const files = [...(event.target.files || [])];
      const enemyId = input.dataset.enemyId;
      if (!files.length || !enemyId) return;
      enemyUploadStatus.textContent = `Uploading ${files.length} calibration clip${files.length === 1 ? "" : "s"}...`;
      const data = await uploadVideos("/api/enemy-drone/upload", files, { enemy_id: enemyId });
      enemyLibraryData = data.library || enemyLibraryData;
      input.value = "";
      enemyUploadStatus.textContent = "Calibration clips uploaded.";
      renderEnemyLibrary();
    }));
  }
  renderEnemyAnnotation();
}

async function refreshEnemyLibrary() {
  const resp = await fetch("/api/enemy-drones", { cache: "no-store" });
  if (!resp.ok) throw new Error(`enemy drone library ${resp.status}`);
  enemyLibraryData = await resp.json();
  renderEnemyLibrary();
  const status = String(enemyLibraryData.model_status || "");
  if ((status === "queued" || status === "training") && document.body.classList.contains("show-enemy")) {
    setTimeout(() => runUi(refreshEnemyLibrary), 2500);
  }
  return enemyLibraryData;
}

async function uploadEnemyCalibration() {
  const files = [...(enemyVideoUpload?.files || [])];
  const name = String(enemyNameInput?.value || "").trim();
  if (!files.length) throw new Error("Choose at least one enemy-drone calibration video.");
  if (!name) throw new Error("Give this enemy drone a name before uploading.");
  enemyUploadStatus.textContent = `Uploading ${files.length} calibration clip${files.length === 1 ? "" : "s"} for ${name}...`;
  const data = await uploadVideos("/api/enemy-drone/upload", files, { name });
  enemyLibraryData = data.library || enemyLibraryData;
  enemyVideoUpload.value = "";
  enemyNameInput.value = "";
  enemyUploadStatus.textContent = `${name} added to the detector bank.`;
  renderEnemyLibrary();
}

async function extractEnemyFrames(enemyId = "") {
  const targetId = enemyId || selectedEnemyId;
  if (!targetId) throw new Error("Choose an enemy drone profile first.");
  setEnemyAnnotationStatus("Extracting calibration frames...");
  const data = await postJson("/api/enemy-drone/extract-frames", {
    enemy_id: targetId,
    fps: 2,
    max_frames_per_video: 180,
    max_size: 960,
  });
  enemyLibraryData = data.library || enemyLibraryData;
  selectedEnemyId = targetId;
  const profile = (enemyLibraryData.enemies || []).find(enemy => enemy.id === targetId);
  const firstFrame = (profile?.frames || [])[0];
  selectedEnemyFrameId = firstFrame?.id || "";
  enemyBoxDraft = firstFrame?.box ? { ...firstFrame.box } : null;
  loadEnemyAnnotationImage(firstFrame);
  setEnemyAnnotationStatus(`Extracted ${data.added || 0} new frames. Draw boxes and save labels.`);
  renderEnemyLibrary();
}

async function renameEnemyDrone(enemyId) {
  if (!enemyId) return;
  const enemy = (enemyLibraryData.enemies || []).find(item => item.id === enemyId);
  const nextName = prompt("Enemy drone name", enemy?.name || "Enemy Drone");
  if (nextName === null) return;
  const data = await postJson("/api/enemy-drone/rename", { enemy_id: enemyId, name: nextName });
  enemyLibraryData = data.library || enemyLibraryData;
  renderEnemyLibrary();
}

async function deleteEnemyDrone(enemyId) {
  if (!enemyId) return;
  const enemy = (enemyLibraryData.enemies || []).find(item => item.id === enemyId);
  if (!confirm(`Delete ${enemy?.name || "this enemy drone"} and its calibration videos?`)) return;
  const data = await postJson("/api/enemy-drone/delete", { enemy_id: enemyId });
  enemyLibraryData = data.library || enemyLibraryData;
  renderEnemyLibrary();
}

async function prepareEnemyYoloDataset(enemyId = "") {
  const data = await postJson("/api/enemy-drone/prepare-yolo", { enemy_id: enemyId || "" });
  enemyLibraryData = data.library || enemyLibraryData;
  enemyUploadStatus.textContent = "YOLO dataset prepared from saved frame labels.";
  renderEnemyLibrary();
}

async function trainEnemyYoloModel() {
  const epochs = Math.max(1, Math.min(300, Number(enemyTrainEpochs?.value || 50)));
  const imgsz = Math.max(160, Math.min(1600, Number(enemyTrainImgsz?.value || 640)));
  if (enemyUploadStatus) enemyUploadStatus.textContent = "Queueing YOLO fine-tuning from labeled enemy-drone frames...";
  if (enemyTrainModelButton) enemyTrainModelButton.disabled = true;
  try {
    const data = await postJson("/api/enemy-drone/train-yolo", {
      epochs,
      imgsz,
      batch: 8,
      device: "auto",
      base_model: "yolov8n.pt",
    });
    enemyLibraryData = data.library || enemyLibraryData;
    if (enemyUploadStatus) enemyUploadStatus.textContent = "YOLO fine-tuning queued. This can take a while on CPU.";
    renderEnemyLibrary();
  } finally {
    if (enemyTrainModelButton) enemyTrainModelButton.disabled = false;
  }
}

async function saveEnemyFrameLabel(status = "labeled") {
  const profile = currentEnemyProfile();
  const frame = currentEnemyFrame();
  if (!profile || !frame) throw new Error("Choose a frame first.");
  const box = status === "labeled" ? (enemyBoxDraft || frame.box) : null;
  if (status === "labeled" && !box) throw new Error("Draw a bounding box around the drone first.");
  const data = await postJson("/api/enemy-drone/label-frame", {
    enemy_id: profile.id,
    frame_id: frame.id,
    status,
    box,
  });
  enemyLibraryData = data.library || enemyLibraryData;
  const updatedFrame = currentEnemyFrame();
  enemyBoxDraft = updatedFrame?.box ? { ...updatedFrame.box } : null;
  renderEnemyLibrary();
}

async function autoTrackEnemyLabels() {
  const profile = currentEnemyProfile();
  const frame = currentEnemyFrame();
  if (!profile || !frame) throw new Error("Choose a frame first.");
  const box = enemyBoxDraft || frame.box;
  if (!box) throw new Error("Draw or select a bounding box before auto-tracking.");
  setEnemyAnnotationStatus("Auto-tracking this enemy drone through the calibration clip...");
  const data = await postJson("/api/enemy-drone/track-labels", {
    enemy_id: profile.id,
    frame_id: frame.id,
    box,
    direction: "both",
    accept_threshold: 0.72,
    review_threshold: 0.50,
    search_scale: 3.0,
    max_frames: 160,
    overwrite: false,
  });
  enemyLibraryData = data.library || enemyLibraryData;
  selectedEnemyId = profile.id;
  selectedEnemyFrameId = frame.id;
  const labeled = Number(data.labeled || 0);
  const review = Number(data.review || 0);
  const stopped = Number(data.stopped || 0);
  setEnemyAnnotationStatus(`Auto-track finished: ${labeled} seed label, ${review} boxes awaiting review${stopped ? ", stopped at weak match" : ""}. Auto-tracked boxes are never accepted automatically.`);
  renderEnemyLibrary();
}

function copyPreviousEnemyBox() {
  const frames = currentEnemyFrames();
  const index = frames.findIndex(frame => frame.id === selectedEnemyFrameId);
  for (let i = index - 1; i >= 0; i -= 1) {
    if (frames[i]?.box) {
      enemyBoxDraft = { ...frames[i].box };
      drawEnemyAnnotationCanvas();
      setEnemyAnnotationStatus("Copied previous bounding box. Adjust it if needed, then save.");
      return;
    }
  }
  setEnemyAnnotationStatus("No previous labeled frame has a box to copy.");
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
    const patrolCount = patrolList(entry).length;
    const replayLabel = hasReplay
      ? `${replays.length} path${replays.length === 1 ? "" : "s"}, ${replayPoseCountText(active, counts)} active`
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
          <dt>Patrols</dt><dd>${patrolCount ? `${patrolCount} saved` : "none yet"}</dd>
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
      const fixedReference = currentMapEntry?.kind === "map_copy"
        && (currentMapEntry?.localization_map_id || currentMapEntry?.source_map_id);
      uploadStatus.textContent = fixedReference
        ? `Registering ${files.length} video${files.length === 1 ? "" : "s"} into fixed ${currentMapEntry?.title || mapId} coordinates. Original map will not be changed: ${names}`
        : `Enhancing ${currentMapEntry?.title || mapId} with ${files.length} mapping video${files.length === 1 ? "" : "s"}: ${names}`;
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
  renderSavedPatrols();
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
    btn.innerHTML = `
      <span>${replay.title || "Drone Path"}</span>
      <small>${replayPoseCountText(replay)}</small>
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
    const enhance = document.createElement("button");
    enhance.type = "button";
    enhance.className = "replay-enhance-map";
    enhance.title = "Use this drone path/video to enhance the selected 3D map";
    enhance.textContent = "Add to Map";
    enhance.addEventListener("click", event => {
      event.stopPropagation();
      runUi(() => enhanceMapFromReplay(replay.id));
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
    item.appendChild(enhance);
    item.appendChild(rename);
    item.appendChild(del);
    replayTabList.appendChild(item);
  }
  if (liveReplayInFlight) {
    replayTabList.appendChild(createPendingReplayTab());
  }
}

function makePatrolId() {
  return `patrol_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function patrolTitle(patrol) {
  return String(patrol?.title || "Untitled Patrol");
}

function normalizePatrolMode(value, loopFallback = true) {
  const raw = String(value || "").trim().toLowerCase();
  if (["back-and-forth", "back_and_forth", "back-forth", "pingpong", "ping-pong", "bounce"].includes(raw)) {
    return "back-and-forth";
  }
  if (raw === "circle" || raw === "loop") return "circle";
  return loopFallback ? "circle" : "back-and-forth";
}

function patrolModeFromPatrol(patrol) {
  if (patrol?.patrol_mode || patrol?.mode) {
    return normalizePatrolMode(patrol.patrol_mode || patrol.mode, patrol?.loop !== false);
  }
  return normalizePatrolMode(null, patrol?.loop !== false);
}

function patrolMode() {
  return normalizePatrolMode(patrolModeSelect?.value, patrolLoopInput ? patrolLoopInput.checked : true);
}

function patrolModeLabel(mode = patrolMode()) {
  return normalizePatrolMode(mode) === "back-and-forth" ? "back-and-forth" : "circle";
}

function savedPatrolDescription(patrol) {
  const count = Array.isArray(patrol?.points) ? patrol.points.length : 0;
  const speed = Number(patrol?.speed || 0.10);
  const altitude = Number(patrol?.altitude_m || 1.0);
  const scan = patrol?.scan_mode === "forward" ? "forward scan" : "left/right scan";
  const mode = patrolModeLabel(patrolModeFromPatrol(patrol));
  return `${count} point${count === 1 ? "" : "s"} · ${mode} · ${altitude.toFixed(1)} m · ${speed.toFixed(2)} m/s · ${scan}`;
}

function mapCoordinateLineageIds(entry) {
  const mapsById = new Map((mapLibraryData?.maps || []).map(map => [map.id, map]));
  const lineage = new Set();
  const pending = [String(entry?.id || "")];
  while (pending.length && lineage.size < 32) {
    const mapId = pending.pop();
    if (!mapId || lineage.has(mapId)) continue;
    lineage.add(mapId);
    const map = mapsById.get(mapId);
    if (!map) continue;
    for (const key of ["source_map_id", "localization_map_id", "coordinate_frame_id"]) {
      const parentId = String(map?.[key] || "");
      if (parentId && parentId !== mapId && !lineage.has(parentId)) pending.push(parentId);
    }
  }
  return lineage;
}

function mapsShareCoordinateFrame(left, right) {
  const leftLineage = mapCoordinateLineageIds(left);
  return [...mapCoordinateLineageIds(right)].some(mapId => leftLineage.has(mapId));
}

function hidePatrolImportModal() {
  patrolImportModal?.classList.add("hidden");
}

async function importPatrolFromMap(sourceMapId, patrolId, button) {
  if (!currentMapEntry?.id) return;
  const originalLabel = button?.textContent || "Import";
  if (button) {
    button.disabled = true;
    button.textContent = "Importing…";
  }
  try {
    const data = await postJson("/api/map/patrol/import", {
      target_map_id: currentMapEntry.id,
      source_map_id: sourceMapId,
      patrol_id: patrolId,
    });
    if (data.state?.library) mapLibraryData = data.state.library;
    if (data.map) syncMapEntryInLibrary(data.map);
    else currentMapEntry = selectedMap();
    renderMapLibrary();
    renderSavedPatrols();
    hidePatrolImportModal();
    if (data.patrol) loadPatrolIntoEditor(data.patrol, { selecting: false });
    const importedTitle = patrolTitle(data.patrol);
    uploadStatus.textContent = `Imported patrol: ${importedTitle}`;
    updatePatrolStatus(
      `Imported "${importedTitle}". The source patrol was not changed. Validate it on this map before flight.`,
      "busy",
    );
  } catch (error) {
    if (button) {
      button.disabled = false;
      button.textContent = originalLabel;
    }
    throw error;
  }
}

function renderPatrolImportOptions() {
  if (!patrolImportList) return;
  patrolImportList.innerHTML = "";
  const target = currentMapEntry;
  if (!target) {
    patrolImportList.innerHTML = `<div class="patrol-import-empty">Select a target map first.</div>`;
    return;
  }
  const existingIds = new Set(patrolList(target).map(patrol => patrol.id));
  const compatibleSources = (mapLibraryData?.maps || [])
    .filter(map => map.id !== target.id)
    .filter(map => patrolList(map).length > 0)
    .filter(map => mapsShareCoordinateFrame(target, map));
  const options = compatibleSources.flatMap(map => (
    patrolList(map).map(patrol => ({ map, patrol }))
  ));
  if (!options.length) {
    patrolImportList.innerHTML = `
      <div class="patrol-import-empty">
        No patrols are available from maps that share this map's coordinate frame.
        Import is blocked across unrelated maps because their points would send the drone to incorrect positions.
      </div>
    `;
    return;
  }

  for (const { map, patrol } of options) {
    const alreadyPresent = existingIds.has(patrol.id);
    const item = document.createElement("article");
    item.className = "patrol-import-item";
    const copy = document.createElement("div");
    copy.className = "patrol-import-copy";
    const title = document.createElement("strong");
    title.textContent = patrolTitle(patrol);
    const description = document.createElement("span");
    description.textContent = savedPatrolDescription(patrol);
    const source = document.createElement("small");
    source.textContent = `From ${map.title || map.id}`;
    copy.append(title, description, source);

    const action = document.createElement("button");
    action.type = "button";
    action.className = "patrol-import-action";
    action.textContent = alreadyPresent ? "Imported" : "Import";
    action.disabled = alreadyPresent;
    if (!alreadyPresent) {
      action.addEventListener("click", () => {
        runUi(() => importPatrolFromMap(map.id, patrol.id, action));
      });
    }
    item.append(copy, action);
    patrolImportList.appendChild(item);
  }
}

async function showPatrolImportModal() {
  await refreshMapLibrary();
  currentMapEntry = selectedMap();
  if (patrolImportSubtitle) {
    patrolImportSubtitle.textContent = `Choose a saved patrol to copy into ${currentMapEntry?.title || "the selected map"}.`;
  }
  renderPatrolImportOptions();
  patrolImportModal?.classList.remove("hidden");
}

function syncMapEntryInLibrary(entry) {
  if (!entry?.id) return;
  mapLibraryData.maps = (mapLibraryData.maps || []).map(map => map.id === entry.id ? entry : map);
  currentMapEntry = selectedMap() || entry;
}

function currentPatrolDraftPayload() {
  const name = String(patrolNameInput?.value || "").trim() || `Patrol ${patrolList(currentMapEntry).length + 1}`;
  const mode = patrolMode();
  return {
    id: editingPatrolId || makePatrolId(),
    title: name.slice(0, 80),
    points: patrolPoints.map(point => ({
      rxyz: point.rxyz.slice(0, 3),
      rgb: point.rgb || null,
    })),
    speed: Number(patrolSpeedSelect?.value || 0.10),
    altitude_m: Math.max(0, patrolAltitudeY() - (room?.floorY ?? 0)),
    dwell_s: patrolDwellSeconds(),
    scan_mode: patrolScanMode(),
    patrol_mode: mode,
    loop: mode === "circle",
    created_at: patrolList(currentMapEntry).find(patrol => patrol.id === editingPatrolId)?.created_at || new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
}

function setPatrolControlsFromPatrol(patrol) {
  if (patrolNameInput) patrolNameInput.value = patrolTitle(patrol);
  if (patrolSpeedSelect && patrol?.speed != null) patrolSpeedSelect.value = String(patrol.speed);
  if (patrolAltitudeInput && patrol?.altitude_m != null) patrolAltitudeInput.value = String(Number(patrol.altitude_m).toFixed(1));
  if (patrolDwellSelect && patrol?.dwell_s != null) patrolDwellSelect.value = String(patrol.dwell_s);
  if (patrolScanModeSelect && patrol?.scan_mode) patrolScanModeSelect.value = patrol.scan_mode;
  const mode = patrolModeFromPatrol(patrol);
  if (patrolModeSelect) patrolModeSelect.value = mode;
  if (patrolLoopInput) patrolLoopInput.checked = mode === "circle";
}

function loadPatrolIntoEditor(patrol, options = {}) {
  if (!patrol) return;
  editingPatrolId = patrol.id || null;
  activePatrolId = patrol.id || null;
  patrolPoints = (patrol.points || [])
    .filter(point => Array.isArray(point?.rxyz))
    .map(point => ({ rxyz: point.rxyz.slice(0, 3), rgb: point.rgb || null }));
  setPatrolControlsFromPatrol(patrol);
  plannedPatrol = null;
  renderPatrolCommands([]);
  patrolSelecting = Boolean(options.selecting);
  editPatrolButton?.classList.toggle("active", patrolSelecting);
  patrolControlPanel?.classList.toggle("is-selecting", patrolSelecting);
  invalidateStaticLayer();
  if (!liveLocalizationStarted()) renderLivePatrolSelector(patrol.id);
  renderSavedPatrols();
  updatePatrolStatus(
    options.selecting
      ? `Adjusting "${patrolTitle(patrol)}": drag points or click map points to extend the patrol.`
      : `Loaded "${patrolTitle(patrol)}". Validate to preview with current walls.`,
    options.selecting ? "busy" : "",
  );
}

function newPatrolDraft() {
  editingPatrolId = null;
  activePatrolId = null;
  patrolPoints = [];
  patrolPointSafetyIssues = new Map();
  plannedPatrol = null;
  patrolSelecting = true;
  if (patrolNameInput) patrolNameInput.value = `Patrol ${patrolList(currentMapEntry).length + 1}`;
  if (patrolModeSelect) patrolModeSelect.value = "circle";
  if (patrolLoopInput) patrolLoopInput.checked = true;
  renderPatrolCommands([]);
  editPatrolButton?.classList.add("active");
  patrolControlPanel?.classList.add("is-selecting");
  invalidateStaticLayer();
  renderSavedPatrols();
  updatePatrolStatus("New patrol: click COLMAP points in order. ATLAS will later enter from the nearest point after live localization.", "busy");
}

async function persistPatrols(nextPatrols, statusMessage = "Saved patrols.") {
  if (!currentMapEntry?.id) return null;
  const data = await postJson("/api/map/patrols", {
    map_id: currentMapEntry.id,
    patrols: nextPatrols,
  });
  if (data.state?.library) mapLibraryData = data.state.library;
  if (data.map) syncMapEntryInLibrary(data.map);
  else currentMapEntry = selectedMap();
  renderMapLibrary();
  renderSavedPatrols();
  uploadStatus.textContent = statusMessage;
  return data.map || currentMapEntry;
}

async function savePatrolDraft() {
  if (patrolPoints.length < 2) {
    updatePatrolStatus("Add at least two patrol points before saving.", "error");
    return null;
  }
  const preview = validatePatrolPreview(false);
  if (!preview) return null;
  const payload = currentPatrolDraftPayload();
  const patrols = patrolList(currentMapEntry).filter(patrol => patrol.id !== payload.id);
  patrols.push(payload);
  updatePatrolStatus(`Saving "${payload.title}"...`, "busy");
  try {
    await persistPatrols(patrols, `Saved patrol: ${payload.title}`);
  } catch (error) {
    const reason = error?.message || String(error || "Unknown save error");
    updatePatrolStatus(`Patrol was not saved. ${reason}`, "error");
    if (uploadStatus) uploadStatus.textContent = `Patrol save failed: ${reason}`;
    return null;
  }
  editingPatrolId = payload.id;
  activePatrolId = payload.id;
  updatePatrolStatus(`Saved "${payload.title}". At flight time, ATLAS will replan from the live drone pose to patrol point 1.`, "busy");
  return payload;
}

async function deleteSavedPatrol(patrolId) {
  const patrol = patrolList(currentMapEntry).find(item => item.id === patrolId);
  if (!patrol) return;
  if (!window.confirm(`Delete saved patrol "${patrolTitle(patrol)}"?`)) return;
  const patrols = patrolList(currentMapEntry).filter(item => item.id !== patrolId);
  if (editingPatrolId === patrolId) editingPatrolId = null;
  if (activePatrolId === patrolId) activePatrolId = null;
  await persistPatrols(patrols, `Deleted patrol: ${patrolTitle(patrol)}`);
  if (editingPatrolId == null && activePatrolId == null) {
    patrolPoints = [];
    patrolPointSafetyIssues = new Map();
    plannedPatrol = null;
    renderPatrolCommands([]);
    invalidateStaticLayer();
    updatePatrolStatus();
  }
}

function renderSavedPatrols() {
  if (!savedPatrolList) return;
  renderLivePatrolSelector();
  updateLiveControlSummary();
  const patrols = patrolList(currentMapEntry);
  if (!patrols.length) {
    savedPatrolList.innerHTML = `<div class="saved-patrol-empty">No saved patrols yet. Press New Patrol, mark points on the clean map, then save.</div>`;
    return;
  }
  savedPatrolList.innerHTML = "";
  for (const patrol of patrols) {
    const item = document.createElement("article");
    item.className = `saved-patrol-item${patrol.id === activePatrolId ? " active" : ""}`;
    const recordingThisPatrol = manualPatrolRecording?.status === "recording" && manualPatrolRecording.patrol_id === patrol.id;
    item.innerHTML = `
      <button type="button" class="saved-patrol-main" data-patrol-id="${patrol.id}">
        <strong>${escapeHtml(patrolTitle(patrol))}</strong>
        <span>${escapeHtml(savedPatrolDescription(patrol))}</span>
      </button>
      <div class="saved-patrol-actions">
        <button type="button" data-action="play" data-patrol-id="${patrol.id}">Start ${escapeHtml(patrolTitle(patrol))} · 2 Circles</button>
        ${recordingThisPatrol
          ? `<button type="button" data-action="record" data-patrol-id="${patrol.id}">Finish Teach</button>`
          : `<button type="button" data-action="record" data-patrol-id="${patrol.id}">Teach Full Loop</button>
             <button type="button" data-action="record-missing" data-patrol-id="${patrol.id}">Teach Finish 3→4→1</button>`}
        <button type="button" data-action="adjust" data-patrol-id="${patrol.id}">Adjust</button>
        <button type="button" class="danger-action" data-action="delete" data-patrol-id="${patrol.id}">Delete</button>
      </div>
      <small class="saved-patrol-profile-status" aria-live="polite"></small>
    `;
    item.querySelector(".saved-patrol-main")?.addEventListener("click", () => {
      loadPatrolIntoEditor(patrol, { selecting: false });
      validatePatrolPreview(false);
    });
    for (const button of item.querySelectorAll("[data-action]")) {
      button.addEventListener("click", event => {
        event.stopPropagation();
        const action = button.dataset.action;
        if (action === "adjust") {
          loadPatrolIntoEditor(patrol, { selecting: true });
        } else if (action === "delete") {
          runUi(() => deleteSavedPatrol(patrol.id));
        } else if (action === "play") {
          runUi(() => executeSavedPatrol(patrol.id, "combined"));
        } else if (action === "record") {
          runUi(() => toggleManualPatrolRecording(patrol));
        } else if (action === "record-missing") {
          runUi(() => toggleManualPatrolRecording(patrol, "continuation_3_4_1"));
        }
      });
    }
    savedPatrolList.appendChild(item);
  }
}

async function toggleManualPatrolRecording(patrol, recordingMode = "full_loop") {
  if (manualPatrolRecording?.status === "recording") {
    if (manualPatrolRecording.patrol_id !== patrol.id) {
      updatePatrolStatus(
        `Finish the active recording for "${manualPatrolRecording.patrol_title}" first.`,
        "error",
      );
      return;
    }
    const recordedRoute = manualPatrolRecording.route_label || "1 → 2 → 3 → 4 → 1";
    const ok = window.confirm(
      `Finish recording "${patrolTitle(patrol)}" now?\n\n` +
      `Only finish after completing ${recordedRoute.replaceAll("->", "→")} and hovering at point 1.`
    );
    if (!ok) return;
    updatePatrolStatus("Finishing manual patrol recording and building the reference trajectory...", "busy");
    const data = await postJson("/api/manual-patrol/finish", {
      recording_id: manualPatrolRecording.recording_id,
    });
    manualPatrolRecording = data.recording;
    updatePatrolStatus(
      `Manual patrol saved: ${data.recording.accepted_pose_count} trusted poses and ` +
      `${data.recording.frame_count} synchronized frames. It is ready for guarded replay processing.`,
      "success",
    );
    renderSavedPatrols();
    return;
  }

  if (!firstLocalizationConfirmed) {
    updatePatrolStatus(
      "Start Live Localization and confirm the pose before recording the teach route.",
      "error",
    );
    return;
  }
  const continuationFromPoint3 = recordingMode === "continuation_3_4_1";
  const continuationFromPoint4 = recordingMode === "continuation_4_1";
  const startPoint = continuationFromPoint4 ? 4 : (continuationFromPoint3 ? 3 : 1);
  const routeText = continuationFromPoint4
    ? "4 → 1"
    : (continuationFromPoint3 ? "3 → 4 → 1" : "1 → 2 → 3 → 4 → 1");
  const ok = window.confirm(
    `Start recording "${patrolTitle(patrol)}" at point ${startPoint}?\n\n` +
    `Before continuing, hover at point ${startPoint} and confirm the model matches the real drone. ` +
    `Then fly ${routeText} manually, including every in-place turn. ` +
    "Keep Live Localization running throughout."
  );
  if (!ok) return;
  const data = await postJson("/api/manual-patrol/start", {
    map_id: currentMapEntry?.id,
    patrol_id: patrol.id,
    patrol_title: patrolTitle(patrol),
    recording_mode: recordingMode,
  });
  manualPatrolRecording = data.recording;
  updatePatrolStatus(
    `Recording "${patrolTitle(patrol)}". Fly ${routeText} manually, then press Finish Teach.`,
    "busy",
  );
  renderSavedPatrols();
}

async function executeSavedPatrol(patrolId, patrolStage = "combined") {
  const patrol = patrolList(currentMapEntry).find(item => item.id === patrolId);
  if (!patrol) return;
  loadPatrolIntoEditor(patrol, { selecting: false });
  const boundPatrol = boundLivePatrolId();
  if (boundPatrol && boundPatrol !== patrolId) {
    const reason = `Live Localization is currently bound to another patrol. Stop Live, select ${patrolTitle(patrol)}, and start its localization profile before flight.`;
    updatePatrolStatus(reason, "error");
    setDjiCommandStatus(reason, "error");
    updateFlightControlState();
    return;
  }
  const lockReason = liveMovementLockReason();
  if (lockReason) {
    updatePatrolStatus(lockReason, "error");
    setDjiCommandStatus(lockReason, "error");
    updateFlightControlState();
    return;
  }
  if (!firstLocalizationConfirmed) {
    updatePatrolStatus("Start live localization, take off, and confirm the first TSolve pose before executing this patrol.", "error");
    return;
  }
  if (!guidedMotionArmed()) {
    updatePatrolStatus("Enable guided movement after confirming localization before executing a saved patrol.", "error");
    setDjiCommandStatus("Guided movement is not armed. Keep controller ready, then enable guided movement.", "error");
    updateFlightControlState();
    return;
  }
  const buildResult = patrolStage === "entry"
    ? buildPatrolReturnToStartPlan()
    : (patrolStage === "loop" ? buildPatrolLoopPlan() : buildConnectedPatrolPlan());
  const livePatrol = buildResult.plan;
  if (!livePatrol?.commands?.length) {
    updatePatrolStatus(buildResult.error || "The requested patrol stage has no safe route.", "error");
    return;
  }
  if (patrolStage === "loop") {
    const current = closestPose()?.rcenter;
    const start = livePatrol.route?.[0];
    const startError = current && start ? Math.hypot(current[0] - start[0], current[2] - start[2]) : Infinity;
    if (!Number.isFinite(startError) || startError > 0.24) {
      updatePatrolStatus(
        `The drone is ${Number.isFinite(startError) ? startError.toFixed(2) : "not localized"} m from Point 1. Press Go to Start first; the two-circle patrol will not silently include an entry leg.`,
        "error",
      );
      return;
    }
  }
  const headingNotice = useModelHeadingForFlightInput?.checked
    ? `Initial heading: using the model alignment (${selectedDroneHeadingTrimDeg()} deg). Confirm its nose matches the real drone before continuing.\n`
    : "Initial heading: ATLAS will perform a small forward calibration probe.\n";
  const actionText = patrolStage === "entry"
    ? "fly only to Point 1 and hover"
    : (patrolStage === "loop"
      ? "start exactly at Point 1 and fly the locked 1→2→3→4→1 route for two circles"
      : "fly to Point 1, verify it online, then continue directly through the locked 1→2→3→4→1 route for exactly two circles");
  const ok = window.confirm(
    `${patrolStage === "entry" ? "Go to the start of" : "Start the full"} saved patrol "${patrolTitle(patrol)}"?\n\n` +
    `ATLAS will ${actionText}.\n` +
    "If localization is temporarily lost, the drone will hover and keep relocalizing online instead of timing out.\n" +
    headingNotice +
    `Keep the controller ready and use Hover Now if anything looks wrong.\n\nContinue?`
  );
  if (!ok) return;
  await sendPatrolToBridge(livePatrol, patrolTitle(patrol), { stage: patrolStage });
}

function livePathCreationStage() {
  const msg = liveReplayStageDetail || liveReplayMessage || "TSolve online localization running";
  const processed = Number(poseStreamMeta?.processed_count ?? poseStreamMeta?.stream?.pose_count ?? livePoseStreamCount ?? 0);
  const accepted = Number(poseStreamMeta?.accepted_count ?? poseStreamMeta?.stream?.accepted_pose_count ?? 0);
  const expected = Number(poseStreamMeta?.expected_count ?? poseStreamMeta?.stream?.expected_count ?? 0);
  const mapId = pendingLiveReplayMapId || poseStreamMeta?.stream?.map_id;
  const map = (mapLibraryData?.maps || []).find(m => m.id === mapId);
  const prefix = map && currentMapEntry?.id !== map.id ? `${map.title || "selected map"}: ` : "";
  if (expected > 0) {
    return `${prefix}${msg} (${accepted}/${processed}/${expected} accepted/processed/target)`;
  }
  return `${prefix}${msg} (${accepted}/${processed} accepted/processed)`;
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

async function enhanceMapFromReplay(replayId) {
  if (!currentMapEntry?.id || !replayId) return;
  const replay = replayList(currentMapEntry).find(r => r.id === replayId);
  const pathTitle = replay?.title || "this drone path";
  const mapTitle = currentMapEntry?.title || "this 3D map";
  if (!window.confirm(`Add frames from "${pathTitle}" to "${mapTitle}" and rerun COLMAP? This improves the map offline and can take several minutes.`)) return;
  uploadStatus.textContent = `Enhancing ${mapTitle} from drone path: ${pathTitle}`;
  await postJson("/api/replay/enhance-map", {
    map_id: currentMapEntry.id,
    replay_id: replayId,
  });
  await pollStatus();
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

function buildMiniPreviewRoom(sceneData, entry) {
  const pointRows = sceneData?.points3D || [];
  const cloud = pointRows.map(point => point.xyz).filter(Boolean);
  const cameras = (sceneData?.map_cameras || []).map(camera => camera.center).filter(Boolean);
  const sample = [];
  const stride = Math.max(1, Math.ceil(cloud.length / 7000));
  for (let i = 0; i < cloud.length; i += stride) sample.push(cloud[i]);
  sample.push(...cameras);
  if (!sample.length) return null;

  const origin = [0, 0, 0];
  for (const point of sample) for (let axis = 0; axis < 3; axis++) origin[axis] += point[axis];
  for (let axis = 0; axis < 3; axis++) origin[axis] /= sample.length;

  const covarianceMatrix = covariance(sample, origin);
  const eigenX = powerEigen(covarianceMatrix, [1, 0.2, 0.1]);
  const eigenZ = powerEigen(deflate(covarianceMatrix, eigenX), [0.1, 1, 0.2]);
  const axisX = normalize(eigenX.v);
  const zSign = Number(entry?.display_z_sign ?? -1) < 0 ? -1 : 1;
  const axisZ = mul(normalize(sub(eigenZ.v, mul(axisX, dot(eigenZ.v, axisX)))), zSign);
  let axisY = normalize(cross(axisZ, axisX));
  const rawTransform = xyz => {
    const delta = sub(xyz, origin);
    return [dot(delta, axisX), dot(delta, axisY), dot(delta, axisZ)];
  };
  const pointY = cloud.slice(0, 5000).map(point => rawTransform(point)[1]);
  const cameraY = cameras.map(point => rawTransform(point)[1]);
  if (cameraY.length && median(cameraY) < median(pointY)) axisY = mul(axisY, -1);
  const transform = xyz => {
    const delta = sub(xyz, origin);
    return [dot(delta, axisX), dot(delta, axisY), dot(delta, axisZ)];
  };
  const points = pointRows.map(point => ({ ...point, rxyz: transform(point.xyz) }));
  return { points, bounds: robustPreviewBounds(points.map(point => point.rxyz)) };
}

function drawSceneMiniPreview(canvas, sceneData, entry) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const pctx = canvas.getContext("2d");
  pctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  pctx.clearRect(0, 0, rect.width, rect.height);

  const miniRoom = buildMiniPreviewRoom(sceneData, entry);
  const points = miniRoom?.points || [];
  if (!points.length) {
    drawMapCardPlaceholder(canvas, entry);
    return;
  }

  const bounds = miniRoom.bounds;
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

  const drawPreviewPolygon = (vertices, fill, stroke, lineWidth = 1) => {
    if (!Array.isArray(vertices) || vertices.length < 3) return;
    const projectedVertices = vertices.map(toPreview);
    pctx.beginPath();
    pctx.moveTo(projectedVertices[0][0], projectedVertices[0][1]);
    for (let i = 1; i < projectedVertices.length; i++) {
      pctx.lineTo(projectedVertices[i][0], projectedVertices[i][1]);
    }
    pctx.closePath();
    if (fill) {
      pctx.fillStyle = fill;
      pctx.fill();
    }
    pctx.strokeStyle = stroke;
    pctx.lineWidth = lineWidth;
    pctx.stroke();
  };
  const drawPreviewLine = (a, b, color, lineWidth = 1) => {
    const pa = toPreview(a);
    const pb = toPreview(b);
    pctx.strokeStyle = color;
    pctx.lineWidth = lineWidth;
    pctx.beginPath();
    pctx.moveTo(pa[0], pa[1]);
    pctx.lineTo(pb[0], pb[1]);
    pctx.stroke();
  };

  const skeletonFloor = bounds.mins[1];
  const skeletonCeiling = bounds.maxs[1];
  const skeletonBottom = [
    [bounds.mins[0], skeletonFloor, bounds.mins[2]],
    [bounds.maxs[0], skeletonFloor, bounds.mins[2]],
    [bounds.maxs[0], skeletonFloor, bounds.maxs[2]],
    [bounds.mins[0], skeletonFloor, bounds.maxs[2]],
  ];
  const skeletonTop = skeletonBottom.map(point => [point[0], skeletonCeiling, point[2]]);
  for (let i = 0; i < 4; i++) {
    const next = (i + 1) % 4;
    drawPreviewLine(skeletonBottom[i], skeletonBottom[next], "rgba(65, 190, 255, 0.48)", 1.1);
    drawPreviewLine(skeletonTop[i], skeletonTop[next], "rgba(198, 245, 255, 0.72)", 1.35);
    drawPreviewLine(skeletonBottom[i], skeletonTop[i], "rgba(120, 220, 255, 0.58)", 1);
  }

  for (const barrier of (entry?.safety_barriers || [])) {
    const corners = (barrier?.corners || []).map(asVec3).filter(Boolean);
    if (corners.length < 4) continue;
    const color = sanitizeHexColor(barrier.color, "#cfd8df");
    const opacity = Math.max(0.05, Math.min(0.95, Number(barrier.opacity ?? 0.24)));
    drawPreviewPolygon(
      corners,
      rgbaFromHex(color, opacity),
      rgbaFromHex(color, Math.min(0.92, opacity + 0.42)),
      1.35,
    );
  }

  for (const obstacle of (entry?.safety_obstacles || [])) {
    const prism = obstacleHullPrism(obstacle);
    if (!prism) continue;
    const color = sanitizeHexColor(obstacle.color, "#86dfff");
    const opacity = Math.max(0.05, Math.min(0.95, Number(obstacle.opacity ?? 0.24)));
    const stroke = rgbaFromHex(color, Math.min(0.94, opacity + 0.42));
    drawPreviewPolygon(prism.bottom, rgbaFromHex(color, opacity * 0.72), stroke, 1);
    drawPreviewPolygon(prism.top, rgbaFromHex(color, Math.min(0.72, opacity + 0.08)), stroke, 1.15);
    for (let i = 0; i < prism.bottom.length; i++) {
      const next = (i + 1) % prism.bottom.length;
      drawPreviewPolygon(
        [prism.bottom[i], prism.bottom[next], prism.top[next], prism.top[i]],
        rgbaFromHex(color, opacity * 0.82),
        stroke,
        0.9,
      );
    }
  }

  const projected = [];
  const stride = Math.max(1, Math.ceil(points.length / 2800));
  for (let i = 0; i < points.length; i += stride) {
    const pt = points[i];
    const [x, y, depth, perspective] = toPreview(pt.rxyz || pt.xyz || pt);
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

async function refreshMapLibrary(options = {}) {
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
  if (options.render !== false) renderMapLibrary();
  return mapLibraryData;
}

function buildReplayDisplayPoses(roomPoses, floorY, options = {}) {
  const out = filterReplayPoseTrack(roomPoses, options);
  if (options.applyLanding !== true) return out;

  const goodIdx = out
    .map((p, i) => (isRealPose(p) ? i : -1))
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
    const landY = floorY + 0.035;
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

function assignStablePathHeadings(roomPoses, startGoodIndex = 0) {
  const good = roomPoses.filter(p => isRealPose(p));
  for (let i = Math.max(0, startGoodIndex); i < good.length; i++) {
    const heading = stablePathHeadingAt(good, i);
    if (heading && norm(heading) > 1e-8) good[i].pathHeading = heading;
  }
}

function closestPose() {
  if ((liveReplayInFlight || pendingLiveReplayOpen) && liveCurrentPoseOverride?.rcenter) {
    return liveCurrentPoseOverride;
  }
  const good = room?.poses?.filter(p => isRealPose(p)) || [];
  if (!good.length) return null;
  const t = currentReplayClockTime(good);
  const timed = sortedTimedPoses(good);
  if (timed.length >= 2) {
    if (t <= Number(timed[0].time_sec)) {
      return {
        ...timed[0],
        rheading: timed[0].rheading
          || timed[0].rotationHeading
          || timed[0].pathHeading
          || sub(timed[1].rcenter, timed[0].rcenter),
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
        const interpolatedRawHeading = a.rheading && b.rheading
          ? normalize(lerpVec(a.rheading, b.rheading, u))
          : (nearest.rheading || a.rheading || b.rheading);
        return {
          ...nearest,
          instance_id: `${a.instance_id}->${b.instance_id}`,
          time_sec: t,
          center: lerpOptionalVec(a.center, b.center, u, nearest.center),
          rcenter: lerpVec(a.rcenter, b.rcenter, u),
          rotationYaw,
          rheading: interpolatedRawHeading
            || (Number.isFinite(rotationYaw) ? headingFromYaw(rotationYaw) : null)
            || nearest.rotationHeading
            || nearest.pathHeading
            || a.pathHeading
            || b.pathHeading
            || sub(b.rcenter, a.rcenter),
        };
      }
    }
    const last = timed[timed.length - 1];
    const prev = timed[timed.length - 2];
    return {
      ...last,
      rheading: last.rheading || last.rotationHeading || last.pathHeading || sub(last.rcenter, prev.rcenter),
    };
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
    const maxStep = Math.min(1.35, Math.max(0.42, 0.85 * dt + 0.28));
    return step <= maxStep;
  }
  return step <= 0.75;
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

function obstacleBoxCorners(obstacle, clearance = 0) {
  const bounds = normalizedObstacleBounds(obstacle, clearance);
  if (!bounds) return [];
  const min = bounds.min;
  const max = bounds.max;
  return [
    [min[0], min[1], min[2]],
    [max[0], min[1], min[2]],
    [max[0], min[1], max[2]],
    [min[0], min[1], max[2]],
    [min[0], max[1], min[2]],
    [max[0], max[1], min[2]],
    [max[0], max[1], max[2]],
    [min[0], max[1], max[2]],
  ];
}

function detourCandidatesForObstacle(obstacle, routeY) {
  const clearance = Number(obstacle.clearance_m || 0.35);
  const expanded = normalizedObstacleBounds(obstacle, clearance + 0.45);
  if (!expanded) return [];
  if (routeY < expanded.min[1] || routeY > expanded.max[1]) return [];
  const min = expanded.min;
  const max = expanded.max;
  const corners = [
    [min[0], routeY, min[2]],
    [max[0], routeY, min[2]],
    [max[0], routeY, max[2]],
    [min[0], routeY, max[2]],
  ];
  const center = [(min[0] + max[0]) * 0.5, routeY, (min[2] + max[2]) * 0.5];
  return corners.flatMap(point => {
    const away = normalize([point[0] - center[0], 0, point[2] - center[2]]);
    const sideA = normalize([-away[2], 0, away[0]]);
    return [
      [point[0] + away[0] * 0.35, routeY, point[2] + away[2] * 0.35],
      [point[0] + sideA[0] * 0.35, routeY, point[2] + sideA[2] * 0.35],
      [point[0] - sideA[0] * 0.35, routeY, point[2] - sideA[2] * 0.35],
    ];
  });
}

function candidateRouteGraphNodes(routeY) {
  const nodes = [];
  for (const barrier of mapSafetyBarriers()) {
    for (const candidate of detourCandidatesForBarrier(barrier, routeY)) {
      if (!nodes.some(p => Math.hypot(p[0] - candidate[0], p[2] - candidate[2]) < 0.08)) nodes.push(candidate);
    }
  }
  for (const obstacle of mapSafetyObstacles()) {
    for (const candidate of detourCandidatesForObstacle(obstacle, routeY)) {
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

function roomVerticalFlightLimits() {
  const floor = room?.floorY ?? room?.bounds?.min?.[1] ?? 0;
  const ceiling = room?.bounds?.max?.[1] ?? floor + 2.5;
  return {
    floor,
    ceiling: Math.max(ceiling, floor + 0.8),
  };
}

function routeWithVerticalObjectBypass(start, goal) {
  const obstacles = mapSafetyObstacles();
  if (!obstacles.length) return null;
  const { floor, ceiling } = roomVerticalFlightLimits();
  const yCandidates = [];
  for (const obstacle of obstacles) {
    const bounds = normalizedObstacleBounds(obstacle, 0);
    if (!bounds) continue;
    const clearance = Number(obstacle.clearance_m || 0.35);
    const overY = bounds.max[1] + clearance + 0.22;
    const underY = bounds.min[1] - clearance - 0.22;
    if (overY < ceiling - 0.12) yCandidates.push({ y: overY, mode: "over" });
    if (underY > floor + 0.35) yCandidates.push({ y: underY, mode: "under" });
  }
  yCandidates.sort((a, b) => Math.abs(a.y - start[1]) - Math.abs(b.y - start[1]));
  const unique = [];
  for (const candidate of yCandidates) {
    if (!unique.some(existing => Math.abs(existing.y - candidate.y) < 0.08)) unique.push(candidate);
  }
  for (const candidate of unique.slice(0, 8)) {
    const waypoints = [
      start,
      [start[0], candidate.y, start[2]],
      [goal[0], candidate.y, goal[2]],
      goal,
    ].filter((point, index, arr) => index === 0 || !sameRoutePoint(point, arr[index - 1]));
    const segments = routeSegmentsFromWaypoints(waypoints);
    const safety = missionRouteSafetyCheck(segments);
    if (!safety.blocked) {
      return { waypoints, segments, safety, mode: candidate.mode };
    }
  }
  return null;
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
  if (!baseSafety.blocked || !safetyBlockerCount()) {
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

  const verticalBypass = routeWithVerticalObjectBypass(cur.rcenter, approach);
  if (verticalBypass?.waypoints?.length) {
    const waypoints = sameRoutePoint(verticalBypass.waypoints[verticalBypass.waypoints.length - 1], target)
      ? verticalBypass.waypoints
      : verticalBypass.waypoints.concat([target]);
    const segments = routeSegmentsFromWaypoints(waypoints);
    const safety = missionRouteSafetyCheck(segments);
    if (!safety.blocked) {
      return {
        blocked: false,
        detoured: true,
        vertical_bypass: verticalBypass.mode,
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
    reason: baseSafety.reason || "No clear detour was found around the saved safety barriers.",
  };
}

function missionCommandSpeed(speed = missionSpeedSelect?.value) {
  const raw = Number(speed || 0.10);
  return Math.max(0.05, Math.min(MISSION_AUTONOMY_SPEED_LIMIT_MPS, Number.isFinite(raw) ? raw : 0.10));
}

function patrolCommandSpeed(speed = patrolSpeedSelect?.value) {
  const raw = Number(speed || 0.10);
  return Math.max(0.04, Math.min(PATROL_AUTONOMY_SPEED_LIMIT_MPS, Number.isFinite(raw) ? raw : 0.10));
}

function patrolAltitudeY() {
  const floorY = room?.floorY ?? room?.bounds?.min?.[1] ?? 0;
  const requested = Number(patrolAltitudeInput?.value || takeoffHeightInput?.value || 1.0);
  const height = Math.max(0.3, Math.min(2.0, Number.isFinite(requested) ? requested : 1.0));
  return floorY + height;
}

function patrolDwellSeconds() {
  const dwell = Number(patrolDwellSelect?.value || 2.0);
  return Math.max(0.8, Math.min(8, Number.isFinite(dwell) ? dwell : 2.0));
}

function patrolScanMode() {
  return String(patrolScanModeSelect?.value || "yaw-sweep");
}

function patrolTargetPoint(point) {
  if (!point?.rxyz) return null;
  return [point.rxyz[0], patrolAltitudeY(), point.rxyz[2]];
}

function patrolLegRoute(start, end) {
  if (!start || !end) return { blocked: true, waypoints: [], segments: [], reason: "missing patrol leg endpoint" };
  const direct = [start, end];
  const directSegments = routeSegmentsFromWaypoints(direct);
  const directSafety = missionRouteSafetyCheck(directSegments);
  if (!directSafety.blocked || !safetyBlockerCount()) {
    return {
      blocked: directSafety.blocked,
      waypoints: direct,
      segments: directSegments,
      safety: directSafety,
      detoured: false,
      reason: directSafety.reason,
    };
  }
  const horizontal = shortestSafeHorizontalRoute(start, end, start[1]);
  if (horizontal?.length) {
    const segments = routeSegmentsFromWaypoints(horizontal);
    const safety = missionRouteSafetyCheck(segments);
    if (!safety.blocked) {
      return { blocked: false, waypoints: horizontal, segments, safety, detoured: true };
    }
  }
  const verticalBypass = routeWithVerticalObjectBypass(start, end);
  if (verticalBypass?.waypoints?.length) {
    return {
      blocked: false,
      waypoints: verticalBypass.waypoints,
      segments: verticalBypass.segments,
      safety: verticalBypass.safety,
      detoured: true,
      vertical_bypass: verticalBypass.mode,
    };
  }
  return {
    blocked: true,
    waypoints: direct,
    segments: directSegments,
    safety: directSafety,
    detoured: false,
    reason: directSafety.reason || "No clear detour was found around the saved safety barriers.",
  };
}

function patrolBaseTargets() {
  return patrolPoints.map(patrolTargetPoint).filter(Boolean);
}

function pushUniqueRoutePoint(sequence, point) {
  if (!point) return;
  if (!sequence.length || !sameRoutePoint(sequence[sequence.length - 1], point)) {
    sequence.push(point);
  }
}

function patrolSequenceTargets(targets, entryIndex = 0, mode = patrolMode()) {
  const sequence = [];
  const n = targets.length;
  if (!n) return sequence;
  const entry = Math.max(0, Math.min(n - 1, Number.isFinite(entryIndex) ? entryIndex : 0));
  if (n === 1) return targets.slice();
  if (patrolModeLabel(mode) === "back-and-forth") {
    const forwardFirst = entry < n - 1;
    if (forwardFirst) {
      for (let i = entry; i < n; i++) pushUniqueRoutePoint(sequence, targets[i]);
      for (let i = n - 2; i >= 0; i--) pushUniqueRoutePoint(sequence, targets[i]);
      for (let i = 1; i <= entry; i++) pushUniqueRoutePoint(sequence, targets[i]);
    } else {
      for (let i = entry; i >= 0; i--) pushUniqueRoutePoint(sequence, targets[i]);
      for (let i = 1; i < n; i++) pushUniqueRoutePoint(sequence, targets[i]);
    }
    return sequence;
  }
  for (let offset = 0; offset < n; offset++) {
    pushUniqueRoutePoint(sequence, targets[(entry + offset) % n]);
  }
  pushUniqueRoutePoint(sequence, targets[entry]);
  return sequence;
}

function chooseNearestPatrolEntry(start, targets) {
  let best = null;
  for (let i = 0; i < targets.length; i++) {
    const leg = patrolLegRoute(start, targets[i]);
    const distance = leg.blocked
      ? norm(sub(targets[i], start)) + 1e6
      : routeLengthFromSegments(leg.segments || routeSegmentsFromWaypoints(leg.waypoints || [start, targets[i]]));
    if (!best || distance < best.distance) {
      best = { index: i, leg, distance };
    }
  }
  return best || { index: 0, leg: patrolLegRoute(start, targets[0]), distance: Infinity };
}

function planPatrolRoute(requireCurrentPose = false, options = {}) {
  const targets = patrolBaseTargets();
  if (targets.length < 2) {
    return { blocked: true, reason: "Add at least two patrol points.", route: targets, legs: [], arrival_indices: [] };
  }
  const cur = closestPose();
  if (requireCurrentPose && !cur?.rcenter) {
    return { blocked: true, reason: "No current TSolve pose yet. Start live localization first.", route: targets, legs: [], arrival_indices: [] };
  }

  const mode = patrolMode();
  const loop = mode === "circle";
  const route = [];
  const legs = [];
  const arrivalIndices = [];
  let entryIndex = 0;
  let firstLeg = null;
  let orderedTargets = patrolSequenceTargets(targets, 0, mode);
  let start = cur?.rcenter ? [cur.rcenter[0], patrolAltitudeY(), cur.rcenter[2]] : orderedTargets[0];
  route.push(start);
  if (cur?.rcenter) {
    if (options.entryStrategy === "nearest") {
      const nearest = chooseNearestPatrolEntry(start, targets);
      entryIndex = nearest.index;
      firstLeg = nearest.leg;
      orderedTargets = patrolSequenceTargets(targets, entryIndex, mode);
    } else {
      // A normal operator-started patrol enters at point 1. Automatic
      // post-pursuit recovery explicitly requests the nearest safe point.
      entryIndex = 0;
      firstLeg = patrolLegRoute(start, targets[0]);
      orderedTargets = patrolSequenceTargets(targets, 0, mode);
    }
  }
  const sequence = cur?.rcenter ? orderedTargets : orderedTargets.slice(1);

  for (let goalIndex = 0; goalIndex < sequence.length; goalIndex++) {
    const goal = sequence[goalIndex];
    const leg = goalIndex === 0 && firstLeg ? firstLeg : patrolLegRoute(start, goal);
    legs.push(leg);
    const waypoints = leg.waypoints || [start, goal];
    for (let i = 1; i < waypoints.length; i++) {
      if (!sameRoutePoint(route[route.length - 1], waypoints[i])) route.push(waypoints[i]);
    }
    arrivalIndices.push(route.length - 1);
    if (leg.blocked) {
      return {
        blocked: true,
        reason: leg.reason || leg.safety?.reason || "Patrol route is blocked by a safety barrier.",
        route,
        legs,
        arrival_indices: arrivalIndices,
        patrol_mode: mode,
        loop,
        pending_current_pose: !cur?.rcenter,
        entry_index: entryIndex,
      };
    }
    start = goal;
  }

  const segments = routeSegmentsFromWaypoints(route);
  const safety = missionRouteSafetyCheck(segments);
  return {
    blocked: safety.blocked,
    reason: safety.reason,
    route,
    legs,
    route_segments: segments,
    arrival_indices: arrivalIndices,
    patrol_mode: mode,
    loop,
    entry_index: entryIndex,
    safety,
    detoured: legs.some(leg => leg.detoured),
    distance: routeLengthFromSegments(segments),
    pending_current_pose: !cur?.rcenter,
  };
}

function updatePatrolStatus(message = null, tone = "") {
  if (!patrolStatus) return;
  patrolStatus.dataset.tone = tone;
  if (message) {
    patrolStatus.textContent = message;
    updateFlightControlState();
    return;
  }
  if (patrolSelecting) {
    patrolStatus.textContent = `Editing patrol: click map points to add waypoints. ${patrolPoints.length} point${patrolPoints.length === 1 ? "" : "s"} selected.`;
  } else if (plannedPatrol) {
    const detour = plannedPatrol.detoured ? " Safety-barrier detour active." : "";
    patrolStatus.textContent = `Patrol validated: ${patrolPoints.length} points, ${patrolModeLabel(plannedPatrol.patrol_mode)} mode, ${plannedPatrol.commands?.length || 0} guarded steps, ${patrolCommandSpeed(plannedPatrol.speed).toFixed(2)} m/s.${detour}`;
  } else if (patrolPoints.length) {
    const issues = updatePatrolPointSafetyIssues();
    if (issues.size) {
      const [index, issue] = issues.entries().next().value;
      patrolStatus.dataset.tone = "error";
      patrolStatus.textContent = `Patrol point ${index + 1} is too close to ${issue.label}. Move it farther away or reduce the saved clearance.`;
    } else {
      patrolStatus.textContent = `${patrolPoints.length} patrol point${patrolPoints.length === 1 ? "" : "s"} selected. Validate before saving.`;
    }
  } else {
    patrolStatus.textContent = "Press Edit Patrol, then click 2 or more map points to create a stable scan route.";
  }
  updateFlightControlState();
}

function renderPatrolCommands(commands = plannedPatrol?.commands || []) {
  if (!patrolCommandList) return;
  if (!commands?.length) {
    patrolCommandList.innerHTML = '<div class="mission-command-empty">Validate patrol to preview slow point-to-point and scan commands.</div>';
    return;
  }
  const totalSeconds = commands.reduce((sum, command) => sum + (Number(command.duration_s) || 0), 0);
  const rows = commands.map((command, index) => {
    const duration = Number(command.duration_s);
    const durationText = Number.isFinite(duration) && duration > 0 ? `${duration.toFixed(1)}s` : "gate";
    return `
      <article class="mission-command-item">
        <span class="mission-command-index">${index + 1}</span>
        <div>
          <strong>${escapeHtml(command.title)}</strong>
          <p>${escapeHtml(command.detail)}</p>
          <em>${escapeHtml(command.safety || "guarded")} · ${escapeHtml(durationText)}</em>
        </div>
      </article>
    `;
  }).join("");
  patrolCommandList.innerHTML = `
    <div class="mission-command-summary">
      <strong>Patrol command preview</strong>
      <span>${commands.length} guarded steps · ${patrolModeLabel(plannedPatrol?.patrol_mode)} · ~${totalSeconds.toFixed(1)}s · TSolve-gated slow patrol</span>
    </div>
    ${rows}
  `;
}

function validatePatrolPreview(requireCurrentPose = false, options = {}) {
  if (patrolPoints.length < 2) {
    plannedPatrol = null;
    renderPatrolCommands([]);
    updatePatrolStatus("Add at least two patrol points.", "error");
    return null;
  }
  const pointIssues = updatePatrolPointSafetyIssues();
  if (pointIssues.size) {
    const [index, issue] = pointIssues.entries().next().value;
    plannedPatrol = null;
    renderPatrolCommands([]);
    updatePatrolStatus(
      `Patrol point ${index + 1} is too close to ${issue.label}. ${issue.reason} Move it farther from obstacles or adjust the safety distance.`,
      "error",
    );
    invalidateStaticLayer();
    return null;
  }
  const plan = planPatrolRoute(requireCurrentPose, options);
  if (plan.blocked) {
    plannedPatrol = null;
    renderPatrolCommands([]);
    updatePatrolStatus(`Patrol blocked by a safety barrier. ${plan.reason || "Move points or adjust barriers."}`, "error");
    return null;
  }
  plannedPatrol = {
    type: "patrol",
    points: patrolPoints.map(point => ({ rxyz: point.rxyz.slice(0, 3), rgb: point.rgb || null })),
    route: plan.route,
    route_segments: plan.route_segments,
    arrival_indices: plan.arrival_indices,
    legs: plan.legs,
    patrol_mode: plan.patrol_mode,
    loop: plan.loop,
    detoured: plan.detoured,
    distance: plan.distance,
    speed: patrolCommandSpeed(),
    requested_speed: Number(patrolSpeedSelect?.value || 0.10),
    altitude_y: patrolAltitudeY(),
    altitude_m: Math.max(0, patrolAltitudeY() - (room?.floorY ?? 0)),
    dwell_s: patrolDwellSeconds(),
    scan_mode: patrolScanMode(),
    safety: plan.safety,
    pending_current_pose: plan.pending_current_pose,
    entry_index: plan.entry_index,
    title: String(patrolNameInput?.value || "").trim() || "Room Patrol",
    created_at: Date.now(),
  };
  plannedPatrol.commands = buildPatrolCommandPlan(plannedPatrol);
  renderPatrolCommands(plannedPatrol.commands);
  const currentGate = plan.pending_current_pose
    ? "Saved/offline preview starts at point 1; live execution will also enter at point 1."
    : `Live entry point: patrol point ${Number(plan.entry_index || 0) + 1}.`;
  updatePatrolStatus(
    `Patrol validated: ${patrolPoints.length} points, ${patrolModeLabel(plannedPatrol.patrol_mode)} mode, ${plannedPatrol.distance?.toFixed(2) || "pending"} map units, ${plannedPatrol.speed.toFixed(2)} m/s. ${currentGate}`,
    "busy",
  );
  return plannedPatrol;
}

function routeHeadingDeg(a, b) {
  if (!a || !b) return 0;
  return Math.atan2(b[2] - a[2], b[0] - a[0]) * 180 / Math.PI;
}

function headingDegFromVector(heading) {
  if (!heading || norm([heading[0] || 0, 0, heading[2] || 0]) < 1e-8) return null;
  return Math.atan2(heading[2], heading[0]) * 180 / Math.PI;
}

function currentDroneHeadingDeg() {
  const cur = closestPose();
  if (!cur) return null;
  const heading = headingForPose(cur);
  const deg = headingDegFromVector(heading);
  return Number.isFinite(deg) ? deg : null;
}

function signedHeadingDeltaDeg(fromDeg, toDeg) {
  let delta = Number(toDeg) - Number(fromDeg || 0);
  while (delta > 180) delta -= 360;
  while (delta < -180) delta += 360;
  return delta;
}

function formatCommandPoint(point) {
  if (!point) return "-";
  return point.map(v => Number(v).toFixed(2)).join(", ");
}

function buildMissionCommandPlan(mission = plannedMission) {
  if (!mission?.route?.length || mission.route.length < 2) return [];
  const speed = missionCommandSpeed(mission.speed);
  const requestedSpeed = Number(mission.requested_speed || mission.speed || speed);
  const speedCapped = Number.isFinite(requestedSpeed) && requestedSpeed > speed + 1e-6;
  const profile = mission.profile || missionLandingProfile(mission.target)?.mode;
  const commands = [
    {
      type: "gate",
      title: "User gate",
      detail: "Use only after takeoff, first TSolve R,t is visible, and localization is confirmed.",
      safety: "manual-confirm",
    },
  ];

  let previousHeading = currentDroneHeadingDeg();
  for (let i = 1; i < mission.route.length; i++) {
    const a = mission.route[i - 1];
    const b = mission.route[i];
    const d = norm(sub(b, a));
    if (d <= 1e-5) continue;
    const heading = routeHeadingDeg(a, b);
    const yawDelta = previousHeading == null ? 0 : signedHeadingDeltaDeg(previousHeading, heading);
    const isFinalVertical = Math.abs(a[0] - b[0]) < 1e-5
      && Math.abs(a[2] - b[2]) < 1e-5
      && Math.abs(a[1] - b[1]) > 1e-5;
    if (isFinalVertical) {
      commands.push({
        type: profile === "horizontal-approach-then-land" ? "land" : "descend",
        title: profile === "horizontal-approach-then-land" ? "Land" : "Vertical descend",
        from: a,
        to: b,
        distance: d,
        detail: profile === "horizontal-approach-then-land"
          ? `Land above destination ${formatCommandPoint(b)}.`
          : `Descend slowly to selected height at ${formatCommandPoint(b)}.`,
        duration_s: Math.max(2.0, d / Math.max(0.05, speed * 0.5)),
        safety: "vertical-only",
      });
      continue;
    }

    commands.push({
      type: "yaw",
      title: `Yaw to segment ${i}`,
      from: a,
      to: b,
      heading_deg: heading,
      reference_heading_deg: previousHeading,
      yaw_delta_deg: yawDelta,
      detail: previousHeading == null
        ? `Face ${heading.toFixed(1)} degrees toward waypoint ${i}. Current TSolve heading was unavailable.`
        : `Turn ${yawDelta.toFixed(1)} degrees from current TSolve heading ${previousHeading.toFixed(1)} to face waypoint ${i}.`,
      duration_s: 1.0,
      safety: "slow-yaw",
    });
    commands.push({
      type: "cruise",
      title: `Slow cruise ${i}`,
      from: a,
      to: b,
      heading_deg: heading,
      speed_mps: speed,
      detail: `Move ${d.toFixed(2)} map units to ${formatCommandPoint(b)} at ${speed.toFixed(2)} m/s.`,
      distance: d,
      duration_s: d / speed,
      safety: speedCapped ? `speed capped from ${requestedSpeed.toFixed(2)} m/s` : "indoor-speed",
    });
    previousHeading = heading;
    commands.push({
      type: "hover",
      title: "Hover and re-localize",
      detail: "Stop, hold position, wait for TSolve R,t, and only then continue.",
      duration_s: MISSION_RELOCALIZE_HOVER_SECONDS,
      safety: "pose-check",
    });
  }
  return commands;
}

function buildPatrolCommandPlan(patrol = plannedPatrol) {
  if (!patrol?.route?.length || patrol.route.length < 2) return [];
  const speed = patrolCommandSpeed(patrol.speed);
  const requestedSpeed = Number(patrol.requested_speed || patrol.speed || speed);
  const speedCapped = Number.isFinite(requestedSpeed) && requestedSpeed > speed + 1e-6;
  const arrivalSet = new Set((patrol.arrival_indices || []).map(Number));
  const commands = [
    {
      type: "gate",
      title: "Patrol gate",
      detail: "Use only after takeoff, first TSolve R,t is visible, localization is confirmed, and the path is visually checked.",
      safety: "manual-confirm",
    },
  ];

  let previousHeading = currentDroneHeadingDeg();
  let patrolStopIndex = 0;
  for (let i = 1; i < patrol.route.length; i++) {
    const a = patrol.route[i - 1];
    const b = patrol.route[i];
    const d = norm(sub(b, a));
    if (d <= 1e-5) continue;
    const heading = routeHeadingDeg(a, b);
    const yawDelta = previousHeading == null ? 0 : signedHeadingDeltaDeg(previousHeading, heading);
    commands.push({
      type: "yaw",
      title: `Patrol yaw ${i}`,
      from: a,
      to: b,
      heading_deg: heading,
      reference_heading_deg: previousHeading,
      yaw_delta_deg: yawDelta,
      detail: previousHeading == null
        ? `Face ${heading.toFixed(1)} degrees toward patrol segment ${i}.`
        : `Turn ${yawDelta.toFixed(1)} degrees to face patrol segment ${i}.`,
      duration_s: 1.2,
      safety: "slow-yaw",
    });
    commands.push({
      type: "cruise",
      title: `Patrol cruise ${i}`,
      from: a,
      to: b,
      heading_deg: heading,
      speed_mps: speed,
      detail: `Move ${d.toFixed(2)} map units at ${speed.toFixed(2)} m/s.`,
      distance: d,
      duration_s: d / speed,
      safety: speedCapped ? `speed capped from ${requestedSpeed.toFixed(2)} m/s` : "patrol-speed",
    });
    previousHeading = heading;
    commands.push({
      type: "hover",
      title: "Hover and re-localize",
      detail: "Stop, hold position, wait for a fresh TSolve R,t, and only then continue.",
      duration_s: MISSION_RELOCALIZE_HOVER_SECONDS,
      safety: "pose-check",
    });

    if (arrivalSet.has(i)) {
      patrolStopIndex += 1;
      commands.push({
        type: "hover",
        title: `Patrol point ${patrolStopIndex}`,
        point_index: patrolStopIndex,
        at: b,
        detail: `Hold at patrol point ${patrolStopIndex} for ${Number(patrol.dwell_s || 2).toFixed(1)} seconds.`,
        duration_s: Number(patrol.dwell_s || 2),
        safety: "scan-hold",
      });
      if (patrol.scan_mode === "yaw-sweep") {
        commands.push({
          type: "yaw",
          title: `Scan left ${patrolStopIndex}`,
          at: b,
          yaw_delta_deg: -PATROL_SCAN_YAW_DEG,
          detail: `Slowly rotate gimbal/body view ${PATROL_SCAN_YAW_DEG} degrees left to scan the area.`,
          duration_s: 1.4,
          safety: "scan-yaw",
        });
        commands.push({
          type: "hover",
          title: `Left scan hold ${patrolStopIndex}`,
          at: b,
          detail: "Hold for visual scan and TSolve pose refresh.",
          duration_s: Math.max(0.8, Number(patrol.dwell_s || 2) * 0.5),
          safety: "scan-hold",
        });
        commands.push({
          type: "yaw",
          title: `Scan right ${patrolStopIndex}`,
          at: b,
          yaw_delta_deg: PATROL_SCAN_YAW_DEG * 2,
          detail: `Sweep ${PATROL_SCAN_YAW_DEG} degrees right from forward to inspect the opposite side.`,
          duration_s: 2.2,
          safety: "scan-yaw",
        });
        commands.push({
          type: "hover",
          title: `Right scan hold ${patrolStopIndex}`,
          at: b,
          detail: "Hold for visual scan and TSolve pose refresh.",
          duration_s: Math.max(0.8, Number(patrol.dwell_s || 2) * 0.5),
          safety: "scan-hold",
        });
        commands.push({
          type: "yaw",
          title: `Return forward ${patrolStopIndex}`,
          at: b,
          yaw_delta_deg: -PATROL_SCAN_YAW_DEG,
          detail: "Return to the patrol travel heading before continuing.",
          duration_s: 1.4,
          safety: "scan-yaw",
        });
      }
    }
  }
  return commands;
}

async function sendPatrolToBridge(patrol = plannedPatrol, patrolName = "saved patrol", options = {}) {
  if (!patrol?.commands?.length) {
    updatePatrolStatus("Patrol plan has no executable steps. Validate again before executing.", "error");
    return;
  }
  const mode = patrolModeLabel(patrol.patrol_mode || (patrol.loop ? "circle" : "back-and-forth"));
  const commandCount = patrol.commands.length;
  activeExecutionPatrolRoute = {
    title: patrolName,
    route: (patrol.route || []).map(point => point.slice(0, 3)),
    arrival_indices: (patrol.arrival_indices || []).map(Number),
    patrol_mode: mode,
    detoured: Boolean(patrol.detoured),
    started_at: Date.now(),
  };
  invalidateStaticLayer();
  render();
  updatePatrolStatus(`Sending "${patrolName}" with ${commandCount} guarded steps to the live DJI bridge...`, "busy");
  try {
    const result = await sendDjiFlightCommand("mission", {
      mission: {
        client_safety_version: 3,
        guided_enabled: true,
        patrol: true,
        patrol_stage: options.stage || patrol.patrol_stage || "combined",
        pose_max_age_seconds: 2.5,
        pose_recovery_seconds: 8.0,
        continuous_relocalization: true,
        pulse_seconds: 0.30,
        smooth_continuous_cruise: true,
        cruise_window_seconds: 0.55,
        cruise_pose_watchdog_seconds: 0.65,
        max_forward_rc: 0.022,
        max_lateral_rc: 0.010,
        allow_lateral_rc: false,
        allow_axis_auto_calibration: false,
        axis_probe_rc: 0.018,
        axis_probe_seconds: 0.45,
        max_yaw_rc: 0.050,
        max_scan_yaw_rc: 0.025,
        allow_patrol_scan_yaw: false,
        alignment_grace_seconds: 35.0,
        max_vertical_rc: 0.018,
        max_step_seconds: 2.0,
        max_cruise_seconds: 120.0,
        max_pose_step_map_units: 0.30,
        max_pose_step_hard_map_units: 0.55,
        cross_track_recovery_start_map_units: 0.30,
        max_cross_track_map_units: 0.80,
        arrival_radius_map_units: 0.24,
        arrival_deadband_map_units: 0.14,
        target_frame: "atlas_room",
        map_id: currentMapEntry?.id || null,
        map_title: currentMapEntry?.title || null,
        replay_id: activeReplay(currentMapEntry)?.id || null,
        patrol_id: editingPatrolId || activePatrolId || null,
        patrol_title: patrolName,
        entry_index: patrol.entry_index ?? 0,
        points: patrol.points,
        route: patrol.route,
        route_segments: patrol.route_segments,
        arrival_indices: patrol.arrival_indices,
        patrol_mode: mode,
        loop: mode === "circle",
        speed: patrol.speed,
        altitude_y: patrol.altitude_y,
        altitude_m: patrol.altitude_m,
        dwell_s: patrol.dwell_s,
        scan_mode: patrol.scan_mode,
        commands: patrol.commands,
        safety_barriers: mapSafetyBarriers(),
        safety_obstacles: mapSafetyObstacles(),
        safety_motion_buffer_m: FLIGHT_SAFETY_PULSE_BUFFER_M,
        barrier_clearance_m: selectedBarrierClearance(),
        obstacle_clearance_m: selectedObstacleClearance(),
        heading_trim_deg: 0,
        operator_heading_calibrated: Boolean(useModelHeadingForFlightInput?.checked),
        initial_body_heading_offset_deg: -selectedDroneHeadingTrimDeg(),
        initial_pose_offset_room: initialPoseOffsetRoom.slice(0, 3),
        confirmed_at: new Date().toISOString(),
      },
    });
    const bridgeMessage = result.result?.message || result.message || "Patrol packet queued.";
    const pendingMessage = `${bridgeMessage} ${commandCount} guarded patrol steps are visible below.`;
    updatePatrolStatus(pendingMessage, "busy");
    setDjiCommandStatus(pendingMessage, "busy");
    activePatrolExecutionContext = {
      mapId: currentMapEntry?.id || null,
      patrolId: editingPatrolId || activePatrolId || null,
      patrolName,
      commandId: String(result.command_id || ""),
      autoResumed: Boolean(options.autoResume),
      startedAt: Date.now(),
    };
    return result;
  } catch (error) {
    activeExecutionPatrolRoute = null;
    activePatrolExecutionContext = null;
    invalidateStaticLayer();
    updatePatrolStatus(`Patrol command failed: ${error.message}`, "error");
    setDjiCommandStatus(`Patrol command failed: ${error.message}`, "error");
    return null;
  }
}

async function resumeInterruptedPatrolAfterPursuit(pursuitCommandId, pursuitResult) {
  if (enemyPatrolResumeInFlight || handledEnemyPursuitResumeIds.has(pursuitCommandId)) return;
  handledEnemyPursuitResumeIds.add(pursuitCommandId);
  if (handledEnemyPursuitResumeIds.size > 24) {
    const oldest = handledEnemyPursuitResumeIds.values().next().value;
    handledEnemyPursuitResumeIds.delete(oldest);
  }
  const context = enemyPursuitResumeContext;
  enemyPursuitResumeContext = null;
  enemyTargetSuppressedUntilClear = true;
  if (!context) {
    updateEnemyResponseStatus(
      `Pursuit complete: holding at estimated ${Number(pursuitResult.final_clearance_m || pursuitResult.stop_clearance_m || 0).toFixed(2)} m. No running patrol was interrupted, so no automatic route was started.`,
      "ok",
    );
    return;
  }
  enemyPatrolResumeInFlight = true;
  try {
    if (context.mapId !== currentMapEntry?.id) {
      throw new Error("the selected map changed after the patrol was interrupted");
    }
    if (!guidedMotionArmed()) {
      throw new Error("guided movement is no longer armed");
    }
    const savedPatrol = patrolList(currentMapEntry).find(patrol => patrol.id === context.patrolId);
    if (!savedPatrol) {
      throw new Error("the interrupted saved patrol is no longer available");
    }
    loadPatrolIntoEditor(savedPatrol, { selecting: false });
    const resumePlan = validatePatrolPreview(true, { entryStrategy: "nearest" });
    if (!resumePlan?.commands?.length) {
      throw new Error("no safe route exists from the current TSolve pose to the nearest patrol point");
    }
    const entryPoint = Number(resumePlan.entry_index || 0) + 1;
    updateEnemyResponseStatus(
      `Interception complete. Rejoining "${context.patrolName}" at nearest safe point ${entryPoint}; saved wall/obstacle clearance plus ${FLIGHT_SAFETY_PULSE_BUFFER_M.toFixed(2)} m remains active.`,
      "busy",
    );
    const queued = await sendPatrolToBridge(resumePlan, context.patrolName, { autoResume: true });
    if (!queued) throw new Error("the DJI bridge did not queue the patrol rejoin mission");
  } catch (error) {
    activePatrolExecutionContext = null;
    updateEnemyResponseStatus(
      `Interception complete, but automatic patrol rejoin was blocked: ${error.message || error}. Drone remains in hover.`,
      "error",
    );
    setDjiCommandStatus(`Patrol rejoin blocked: ${error.message || error}. Drone remains in hover.`, "error");
  } finally {
    enemyPatrolResumeInFlight = false;
    updateEnemyResponseControls();
  }
}

function buildPatrolReturnToStartPlan() {
  const cur = closestPose();
  const firstTarget = patrolPoints[0] ? patrolTargetPoint(patrolPoints[0]) : null;
  if (!cur?.rcenter) return { error: "No current TSolve R,t pose is available for return planning." };
  if (!firstTarget) return { error: "No patrol start point is defined." };
  const start = [cur.rcenter[0], patrolAltitudeY(), cur.rcenter[2]];
  const leg = patrolLegRoute(start, firstTarget);
  if (leg.blocked) {
    return { error: leg.reason || leg.safety?.reason || "Return path to patrol start is blocked." };
  }
  const route = leg.waypoints || [start, firstTarget];
  const segments = routeSegmentsFromWaypoints(route);
  const safety = missionRouteSafetyCheck(segments);
  if (safety.blocked) {
    return { error: safety.reason || "Return path violates the saved safety clearance." };
  }
  const plan = {
    type: "patrol-return",
    patrol_stage: "entry",
    points: [{ rxyz: firstTarget }],
    route,
    route_segments: segments,
    arrival_indices: [route.length - 1],
    patrol_mode: "back-and-forth",
    loop: false,
    detoured: leg.detoured,
    distance: routeLengthFromSegments(segments),
    speed: patrolCommandSpeed(),
    requested_speed: Number(patrolSpeedSelect?.value || 0.10),
    altitude_y: patrolAltitudeY(),
    altitude_m: Math.max(0, patrolAltitudeY() - (room?.floorY ?? 0)),
    dwell_s: 2.0,
    scan_mode: "none",
    safety,
    pending_current_pose: false,
    entry_index: 0,
    title: "Return to patrol start",
    created_at: Date.now(),
  };
  plan.commands = buildPatrolCommandPlan(plan).concat([{
    type: "hover",
    title: "Manual landing gate",
    at: firstTarget,
    detail: "Hold over patrol start. Press Land only after visually confirming the landing area is clear.",
    duration_s: 4.0,
    safety: "manual-land-confirm",
  }]);
  return { plan };
}

function buildPatrolLoopPlan() {
  const targets = patrolBaseTargets();
  if (targets.length < 2) return { error: "No complete patrol loop is defined." };
  const ordered = patrolSequenceTargets(targets, 0, "circle");
  const route = [ordered[0]];
  const arrivalIndices = [];
  const legs = [];
  for (let index = 1; index < ordered.length; index++) {
    const leg = patrolLegRoute(ordered[index - 1], ordered[index]);
    if (leg.blocked || leg.detoured) {
      return {
        error: leg.reason || "The locked visual patrol requires its direct recorded loop geometry.",
      };
    }
    legs.push(leg);
    const waypoints = leg.waypoints || [ordered[index - 1], ordered[index]];
    for (let waypoint = 1; waypoint < waypoints.length; waypoint++) {
      if (!sameRoutePoint(route[route.length - 1], waypoints[waypoint])) {
        route.push(waypoints[waypoint]);
      }
    }
    arrivalIndices.push(route.length - 1);
  }
  const segments = routeSegmentsFromWaypoints(route);
  const safety = missionRouteSafetyCheck(segments);
  if (safety.blocked) return { error: safety.reason || "The locked patrol loop is blocked." };
  const plan = {
    type: "patrol",
    patrol_stage: "loop",
    points: patrolPoints.map(point => ({ rxyz: patrolTargetPoint(point), rgb: point.rgb || null })),
    route,
    route_segments: segments,
    arrival_indices: arrivalIndices,
    legs,
    patrol_mode: "circle",
    loop: true,
    detoured: false,
    distance: routeLengthFromSegments(segments),
    speed: patrolCommandSpeed(),
    requested_speed: Number(patrolSpeedSelect?.value || 0.10),
    altitude_y: patrolAltitudeY(),
    altitude_m: Math.max(0, patrolAltitudeY() - (room?.floorY ?? 0)),
    dwell_s: patrolDwellSeconds(),
    scan_mode: "none",
    safety,
    pending_current_pose: false,
    entry_index: 0,
    title: String(patrolNameInput?.value || "").trim() || "Room Patrol",
    created_at: Date.now(),
  };
  plan.commands = buildPatrolCommandPlan(plan);
  return { plan };
}

function buildConnectedPatrolPlan() {
  const entryResult = buildPatrolReturnToStartPlan();
  if (!entryResult.plan) return entryResult;
  const loopResult = buildPatrolLoopPlan();
  if (!loopResult.plan) return loopResult;

  const entry = entryResult.plan;
  const loop = loopResult.plan;
  const entryJoinIndex = entry.route.length - 1;
  const route = entry.route.concat(loop.route.slice(1));
  const routeSegments = routeSegmentsFromWaypoints(route);
  const safety = missionRouteSafetyCheck(routeSegments);
  if (safety.blocked) {
    return { error: safety.reason || "The connected entry and patrol loop are blocked." };
  }

  // The entry gate runs once. The manual landing hold belongs to the old
  // standalone Go-to-Start workflow and must not interrupt the connected
  // mission. Every entry cruise is marked so the bridge requires the tighter
  // Point-1 arrival radius before it begins the repeatable loop body.
  const entryCommands = entry.commands
    .filter(command => command.title !== "Manual landing gate")
    .map(command => ({
      ...command,
      patrol_stage: "entry",
      title: command.type === "gate" ? command.title : `Entry · ${command.title}`,
    }));
  const loopCommands = loop.commands
    .filter(command => command.type !== "gate")
    .map(command => ({ ...command, patrol_stage: "loop" }));

  return {
    plan: {
      ...loop,
      type: "patrol-connected",
      patrol_stage: "combined",
      route,
      route_segments: routeSegments,
      arrival_indices: [
        entryJoinIndex,
        ...loop.arrival_indices.map(index => entryJoinIndex + Number(index)),
      ],
      legs: [...(entry.legs || []), ...(loop.legs || [])],
      loop: true,
      detoured: Boolean(entry.detoured),
      distance: routeLengthFromSegments(routeSegments),
      safety,
      loop_start_route_index: entryJoinIndex,
      title: `${loop.title} · connected entry + 2 circles`,
      commands: entryCommands.concat(loopCommands),
    },
  };
}

function renderMissionCommands(commands = plannedMission?.commands || []) {
  if (!missionCommandList) return;
  if (!commands?.length) {
    missionCommandList.innerHTML = '<div class="mission-command-empty">Plan a path to preview guarded slow movement commands.</div>';
    return;
  }
  const totalSeconds = commands.reduce((sum, command) => sum + (Number(command.duration_s) || 0), 0);
  const rows = commands.map((command, index) => {
    const duration = Number(command.duration_s);
    const durationText = Number.isFinite(duration) && duration > 0 ? `${duration.toFixed(1)}s` : "gate";
    return `
      <article class="mission-command-item">
        <span class="mission-command-index">${index + 1}</span>
        <div>
          <strong>${escapeHtml(command.title)}</strong>
          <p>${escapeHtml(command.detail)}</p>
          <em>${escapeHtml(command.safety || "guarded")} · ${escapeHtml(durationText)}</em>
        </div>
      </article>
    `;
  }).join("");
  missionCommandList.innerHTML = `
    <div class="mission-command-summary">
      <strong>Slow command preview</strong>
      <span>${commands.length} guarded steps · ~${totalSeconds.toFixed(1)}s · lateral movement locked until virtual-stick follower is verified</span>
    </div>
    ${rows}
  `;
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
  const detoured = plannedMission?.detoured ? " Safety-barrier detour active." : "";
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

function patrolPointHitInfo(clientX, clientY) {
  if (!patrolPoints.length || !patrolSelecting) return null;
  const rect = canvas.getBoundingClientRect();
  const sx = clientX - rect.left;
  const sy = clientY - rect.top;
  for (let i = patrolPoints.length - 1; i >= 0; i--) {
    const point = patrolTargetPoint(patrolPoints[i]) || patrolPoints[i].rxyz;
    if (!point) continue;
    const [x, y] = project(point);
    const deleteX = x + 14;
    const deleteY = y - 14;
    const ddx = deleteX - sx;
    const ddy = deleteY - sy;
    if (ddx * ddx + ddy * ddy <= 12 * 12) {
      return { index: i, deleteHit: true };
    }
    const dx = x - sx;
    const dy = y - sy;
    if (dx * dx + dy * dy <= 24 * 24) {
      return { index: i, deleteHit: false };
    }
  }
  return null;
}

function patrolPointHit(clientX, clientY) {
  const hit = patrolPointHitInfo(clientX, clientY);
  return hit && !hit.deleteHit ? hit.index : -1;
}

function deletePatrolPoint(index) {
  if (index < 0 || index >= patrolPoints.length) return;
  patrolPoints.splice(index, 1);
  patrolPointHover = null;
  patrolDraggingIndex = -1;
  patrolPointSafetyIssues = new Map();
  plannedPatrol = null;
  renderPatrolCommands([]);
  updatePatrolStatus(`Deleted patrol point ${index + 1}. Validate again before saving or flying.`, patrolPoints.length >= 2 ? "busy" : "");
  updateFlightControlState();
  invalidateStaticLayer();
}

function updatePatrolPointHover(clientX, clientY) {
  const next = patrolPointHitInfo(clientX, clientY);
  const previousKey = patrolPointHover ? `${patrolPointHover.index}:${patrolPointHover.deleteHit}` : "";
  const nextKey = next ? `${next.index}:${next.deleteHit}` : "";
  if (previousKey !== nextKey) {
    patrolPointHover = next;
    markFastInteraction(120);
  }
  if (next) {
    canvas.style.cursor = next.deleteHit ? "pointer" : "grab";
  }
}

function updateMissionTargetFromPointer(clientX, clientY) {
  if (!missionTarget?.rxyz) return false;
  const next = screenToViewEditPlane(clientX, clientY, missionTarget.rxyz);
  if (!next) return false;
  missionTarget = { rxyz: next, rgb: missionTarget.rgb || null };
  plannedMission = null;
  renderMissionCommands([]);
  updateMissionStatus();
  return true;
}

function updatePatrolPointFromPointer(clientX, clientY) {
  if (patrolDraggingIndex < 0 || !patrolPoints[patrolDraggingIndex]?.rxyz) return false;
  const anchor = patrolTargetPoint(patrolPoints[patrolDraggingIndex]) || patrolPoints[patrolDraggingIndex].rxyz;
  const next = screenToViewEditPlane(clientX, clientY, anchor);
  if (!next) return false;
  patrolPoints[patrolDraggingIndex] = {
    ...patrolPoints[patrolDraggingIndex],
    rxyz: [next[0], patrolPoints[patrolDraggingIndex].rxyz[1], next[2]],
  };
  plannedPatrol = null;
  renderPatrolCommands([]);
  updatePatrolStatus();
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

function drawPatrolPointMarker(rxyz, index, active = false, invalid = false) {
  const [x, y] = project(rxyz);
  const hover = patrolPointHover?.index === index;
  ctx.save();
  ctx.shadowColor = invalid
    ? "rgba(255, 76, 108, 0.95)"
    : (active ? "rgba(255, 232, 104, 0.95)" : "rgba(102, 219, 255, 0.72)");
  ctx.shadowBlur = active ? 18 : 11;
  ctx.fillStyle = invalid
    ? "rgba(255, 54, 92, 0.78)"
    : (active ? "rgba(255, 213, 73, 0.78)" : "rgba(50, 195, 255, 0.62)");
  ctx.strokeStyle = invalid ? "rgba(255, 218, 226, 0.98)" : "rgba(225, 250, 255, 0.95)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, active ? 10 : 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "rgba(2, 11, 22, 0.92)";
  ctx.font = "800 11px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(String(index + 1), x, y + 0.5);
  if (hover && patrolSelecting) {
    const hx = x + 14;
    const hy = y - 14;
    ctx.shadowColor = "rgba(255, 78, 110, 0.95)";
    ctx.shadowBlur = 12;
    ctx.fillStyle = "rgba(101, 10, 31, 0.92)";
    ctx.strokeStyle = "rgba(255, 218, 226, 0.98)";
    ctx.lineWidth = 1.7;
    ctx.beginPath();
    ctx.arc(hx, hy, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.strokeStyle = "rgba(255, 235, 240, 0.98)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(hx - 3.5, hy - 3.5);
    ctx.lineTo(hx + 3.5, hy + 3.5);
    ctx.moveTo(hx + 3.5, hy - 3.5);
    ctx.lineTo(hx - 3.5, hy + 3.5);
    ctx.stroke();
  }
  ctx.restore();
}

function drawPatrolScanSweep(rxyz) {
  const [x, y] = project(rxyz);
  ctx.save();
  ctx.strokeStyle = "rgba(111, 225, 255, 0.34)";
  ctx.lineWidth = 1.4;
  ctx.setLineDash([4, 8]);
  for (const angle of [-0.75, 0, 0.75]) {
    const endpoint = [rxyz[0] + Math.cos(view.yaw + angle) * 0.55, rxyz[1], rxyz[2] + Math.sin(view.yaw + angle) * 0.55];
    const [ex, ey] = project(endpoint);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(ex, ey);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.restore();
}

function patrolCoverageSamples() {
  const targets = patrolBaseTargets();
  if (targets.length < 2 || !room?.displayPoints?.length) return [];
  const cacheKey = targets.map(point => point.map(value => Number(value).toFixed(3)).join(",")).join("|");
  if (patrolCoverageCache.room === room && patrolCoverageCache.key === cacheKey) {
    return patrolCoverageCache.samples;
  }
  const route = targets.concat([targets[0]]);
  const spacing = Math.max(0.12, Math.min(0.30, (room.bounds?.radius || 2) * 0.035));
  const samples = [];
  for (let segment = 0; segment < route.length - 1; segment++) {
    const a = route[segment];
    const b = route[segment + 1];
    const distance = norm(sub(b, a));
    const steps = Math.max(1, Math.ceil(distance / spacing));
    for (let step = 0; step < steps; step++) {
      samples.push({
        point: lerpVec(a, b, step / steps),
        segment,
        atTurn: step === 0,
      });
    }
  }
  samples.push({ point: route[route.length - 1], segment: route.length - 2, atTurn: true });

  const cellSize = Math.max(0.24, Math.min(0.55, (room.bounds?.radius || 2) * 0.065));
  const searchRadius = cellSize * 2.2;
  const cells = new Map();
  const stride = Math.max(1, Math.ceil(room.displayPoints.length / 90000));
  for (let i = 0; i < room.displayPoints.length; i += stride) {
    const point = room.displayPoints[i].rxyz;
    const key = `${Math.floor(point[0] / cellSize)},${Math.floor(point[2] / cellSize)}`;
    cells.set(key, (cells.get(key) || 0) + 1);
  }
  const cellReach = Math.ceil(searchRadius / cellSize);
  for (const sample of samples) {
    const cx = Math.floor(sample.point[0] / cellSize);
    const cz = Math.floor(sample.point[2] / cellSize);
    let count = 0;
    for (let dx = -cellReach; dx <= cellReach; dx++) {
      for (let dz = -cellReach; dz <= cellReach; dz++) {
        count += cells.get(`${cx + dx},${cz + dz}`) || 0;
      }
    }
    sample.density = count;
  }
  const densities = samples.map(sample => sample.density).sort((a, b) => a - b);
  const medianDensity = densities[Math.floor(densities.length * 0.5)] || 1;
  const lowFloor = Math.max(18, medianDensity * 0.38);
  const marginalFloor = Math.max(38, medianDensity * 0.68);
  for (const sample of samples) {
    sample.risk = sample.density < lowFloor ? "high" : (sample.density < marginalFloor ? "marginal" : "good");
  }
  patrolCoverageCache = { room, key: cacheKey, samples };
  return samples;
}

function drawPatrolCoverageRisk() {
  if (!view.showCoverageRisk) return;
  const samples = patrolCoverageSamples();
  if (samples.length < 2) return;
  const colors = {
    high: "rgba(255, 62, 92, 0.92)",
    marginal: "rgba(255, 184, 58, 0.82)",
  };
  for (let i = 0; i < samples.length - 1; i++) {
    const risk = samples[i].risk === "high" || samples[i + 1].risk === "high"
      ? "high"
      : (samples[i].risk === "marginal" || samples[i + 1].risk === "marginal" ? "marginal" : "good");
    if (risk === "good") continue;
    drawLine(samples[i].point, samples[i + 1].point, colors[risk], risk === "high" ? 8 : 5);
  }
  for (const sample of samples) {
    if (sample.risk === "good") continue;
    const [x, y] = project(sample.point);
    ctx.save();
    ctx.fillStyle = sample.risk === "high" ? "rgba(255, 52, 82, 0.20)" : "rgba(255, 184, 58, 0.14)";
    ctx.strokeStyle = colors[sample.risk];
    ctx.lineWidth = sample.risk === "high" ? 1.7 : 1.2;
    ctx.beginPath();
    ctx.arc(x, y, sample.atTurn ? 13 : 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
  for (const sample of samples.filter(candidate => candidate.atTurn)) {
    if (sample.risk === "good") continue;
    drawLabel(
      sample.point,
      sample.risk === "high" ? "low map coverage" : "marginal coverage",
      sample.risk === "high" ? "#ff879d" : "#ffd080",
    );
  }
}

function drawPatrolMission(cur = null) {
  const previewTargets = patrolBaseTargets();
  const activeRoute = activeExecutionPatrolRoute?.route?.length ? activeExecutionPatrolRoute.route : null;
  if (activeRoute?.length >= 2) {
    drawPolyline(activeRoute, "rgba(255, 142, 48, 0.96)", 3.4, [11, 5]);
    const arrivals = new Set((activeExecutionPatrolRoute.arrival_indices || []).map(Number));
    for (let i = 1; i < activeRoute.length - 1; i++) {
      drawRouteMarker(activeRoute[i], arrivals.has(i) ? 9 : 5);
    }
    drawLabel(activeRoute[activeRoute.length - 1], activeExecutionPatrolRoute.title || "active patrol", "#ffd99c");
  }
  const route = plannedPatrol?.route?.length ? plannedPatrol.route : (activeRoute ? [] : previewTargets);
  if (route.length >= 2) {
    drawPolyline(route, plannedPatrol?.detoured ? "rgba(255, 220, 85, 0.86)" : "rgba(88, 211, 255, 0.84)", 2.4, [7, 7]);
    if (plannedPatrol?.route?.length) {
      for (let i = 1; i < plannedPatrol.route.length - 1; i++) {
        drawRouteMarker(plannedPatrol.route[i], plannedPatrol.arrival_indices?.includes(i) ? 8 : 5);
      }
    }
  } else if (cur?.rcenter && previewTargets.length === 1) {
    drawLine([cur.rcenter[0], patrolAltitudeY(), cur.rcenter[2]], previewTargets[0], "rgba(88, 211, 255, 0.54)", 1.8, [5, 8]);
  }
  for (let i = 0; i < patrolPoints.length; i++) {
    const target = patrolTargetPoint(patrolPoints[i]);
    if (!target) continue;
    const issue = patrolPointSafetyIssues.get(i) || pointSafetyIssue(target);
    drawPatrolScanSweep(target);
    drawPatrolPointMarker(target, i, patrolDraggingIndex === i, Boolean(issue));
    if (issue) drawLabel(target, "too close", "#ff9eb4");
  }
  if (patrolPoints.length) {
    const lastTarget = patrolTargetPoint(patrolPoints[patrolPoints.length - 1]);
    if (lastTarget) drawLabel(lastTarget, patrolSelecting ? "add next patrol point" : "patrol", "#8eeeff");
  }
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

function drawBarrierPanelFill(corners, barrier = null) {
  const projected = corners.map(p => project(p));
  const color = sanitizeHexColor(barrier?.color, "#cfd8df");
  const opacity = Math.max(0.05, Math.min(0.95, Number(barrier?.opacity ?? 0.24)));
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(projected[0][0], projected[0][1]);
  for (let i = 1; i < projected.length; i++) ctx.lineTo(projected[i][0], projected[i][1]);
  ctx.closePath();
  ctx.shadowColor = rgbaFromHex(color, Math.min(0.28, opacity + 0.08));
  ctx.shadowBlur = 8;
  ctx.fillStyle = rgbaFromHex(color, opacity);
  ctx.strokeStyle = rgbaFromHex(color, Math.min(0.88, opacity + 0.36));
  ctx.lineWidth = barrier?.id === selectedBarrierId ? 2.4 : 1.7;
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
  ctx.fillStyle = "rgba(238, 242, 246, 0.08)";
  ctx.strokeStyle = "rgba(4, 8, 12, 0.10)";
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
    const color = sanitizeHexColor(barrier.color, "#cfd8df");
    const opacity = Math.max(0.05, Math.min(0.95, Number(barrier.opacity ?? 0.24)));
    const selected = barrier.id === selectedBarrierId;
    drawBarrierPanelFill(corners, barrier);
    drawLine(corners[0], corners[1], rgbaFromHex(color, selected ? 0.92 : Math.min(0.85, opacity + 0.42)), selected ? 3.2 : 2.5);
    for (const [aIdx, bIdx] of edges.slice(1)) {
      drawLine(corners[aIdx], corners[bIdx], rgbaFromHex(color, selected ? 0.78 : Math.min(0.72, opacity + 0.30)), selected ? 2.1 : 1.6);
    }
    drawLine(corners[0], corners[2], rgbaFromHex(color, Math.min(0.32, opacity + 0.08)), 0.8, [5, 7]);
    drawLine(corners[1], corners[3], rgbaFromHex(color, Math.min(0.32, opacity + 0.08)), 0.8, [5, 7]);

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

function drawObstacleFace(corners, face, fill, stroke = "rgba(136, 226, 255, 0.42)") {
  const projected = face.map(index => project(corners[index]));
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(projected[0][0], projected[0][1]);
  for (let i = 1; i < projected.length; i++) ctx.lineTo(projected[i][0], projected[i][1]);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1.1;
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function obstacleCenter(obstacle) {
  const bounds = normalizedObstacleBounds(obstacle, 0);
  if (!bounds) return null;
  return [
    (bounds.min[0] + bounds.max[0]) * 0.5,
    (bounds.min[1] + bounds.max[1]) * 0.5,
    (bounds.min[2] + bounds.max[2]) * 0.5,
  ];
}

function convexHullXZ(points) {
  const clean = (points || [])
    .map(asVec3)
    .filter(Boolean)
    .sort((a, b) => (a[0] - b[0]) || (a[2] - b[2]));
  const unique = [];
  for (const point of clean) {
    if (!unique.some(existing => Math.hypot(existing[0] - point[0], existing[2] - point[2]) < 1e-5)) {
      unique.push(point);
    }
  }
  if (unique.length <= 2) return unique;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[0] - o[0]);
  const lower = [];
  for (const point of unique) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) lower.pop();
    lower.push(point);
  }
  const upper = [];
  for (let i = unique.length - 1; i >= 0; i--) {
    const point = unique[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) upper.pop();
    upper.push(point);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function obstacleHullPrism(obstacle) {
  const bounds = normalizedObstacleBounds(obstacle, 0);
  if (!bounds) return null;
  let hull = convexHullXZ(obstacle.points || []);
  if (hull.length < 3) {
    const corners = obstacleBoxCorners(obstacle, 0);
    if (corners.length < 8) return null;
    hull = [corners[0], corners[1], corners[2], corners[3]];
  }
  const floorY = bounds.min[1];
  const topY = bounds.max[1];
  const bottom = hull.map(point => [point[0], floorY, point[2]]);
  const top = hull.map(point => [point[0], topY, point[2]]);
  return { bottom, top };
}

function drawPolygonFace(points, fill, stroke = "rgba(136, 226, 255, 0.42)", lineWidth = 1.1) {
  if (!points?.length) return;
  const projected = points.map(point => project(point));
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(projected[0][0], projected[0][1]);
  for (let i = 1; i < projected.length; i++) ctx.lineTo(projected[i][0], projected[i][1]);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth;
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawObstaclePointHandle(point, obstacleId, pointIndex, draft = false) {
  const [x, y] = project(point);
  const hover = obstaclePointHover
    && obstaclePointHover.obstacleId === obstacleId
    && obstaclePointHover.pointIndex === pointIndex;
  const active = obstaclePointDrag
    && obstaclePointDrag.obstacleId === obstacleId
    && obstaclePointDrag.pointIndex === pointIndex;
  ctx.save();
  ctx.shadowColor = active ? "rgba(255, 228, 94, 0.95)" : (hover ? "rgba(255, 126, 148, 0.85)" : "rgba(120, 230, 255, 0.42)");
  ctx.shadowBlur = active ? 16 : (hover ? 12 : 6);
  ctx.fillStyle = draft ? "rgba(111, 238, 255, 0.82)" : "rgba(218, 232, 240, 0.76)";
  ctx.strokeStyle = hover ? "rgba(255, 118, 145, 0.95)" : "rgba(226, 250, 255, 0.86)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(x, y, active ? 6.5 : 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  if (hover) {
    ctx.shadowColor = "rgba(255, 70, 110, 0.9)";
    ctx.shadowBlur = 10;
    ctx.strokeStyle = "rgba(255, 92, 128, 0.98)";
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.moveTo(x + 8, y - 18);
    ctx.lineTo(x + 20, y - 6);
    ctx.moveTo(x + 20, y - 18);
    ctx.lineTo(x + 8, y - 6);
    ctx.stroke();
  }
  ctx.restore();
}

function obstacleRotateHandlePoint(obstacle, center = obstacleCenter(obstacle)) {
  if (!center) return null;
  const points = obstacle?.points || [];
  const radius = points.reduce(
    (best, point) => Math.max(best, Math.hypot(point[0] - center[0], point[2] - center[2])),
    Math.max(0.28, Math.min(1.4, (room?.bounds?.radius || 2) * 0.10)),
  );
  const offset = Math.max(0.24, Math.min(1.4, radius + (room?.bounds?.radius || 2) * 0.045));
  return [center[0], center[1], center[2] + offset];
}

function drawObstacleTransformHandles(obstacle, center) {
  if (safetyBarrierMode !== "obstacles" || obstacle.draft || !center) return;
  const moveActive = obstacleTransformDrag?.obstacleId === obstacle.id && obstacleTransformDrag?.type === "move";
  const rotateActive = obstacleTransformDrag?.obstacleId === obstacle.id && obstacleTransformDrag?.type === "rotate";
  const moveHover = obstacleTransformHover?.obstacleId === obstacle.id && obstacleTransformHover?.type === "move";
  const rotateHover = obstacleTransformHover?.obstacleId === obstacle.id && obstacleTransformHover?.type === "rotate";
  const selected = obstacle.id === selectedObstacleId;
  if (!selected && !moveActive && !rotateActive && !moveHover && !rotateHover) return;
  const rotateHandle = obstacleRotateHandlePoint(obstacle, center);
  const [cx, cy] = project(center);
  const [rx, ry] = rotateHandle ? project(rotateHandle) : [cx, cy];

  ctx.save();
  if (view.mode !== "side" && rotateHandle) {
    ctx.strokeStyle = rotateActive || rotateHover ? "rgba(180, 238, 255, 0.96)" : "rgba(140, 205, 226, 0.58)";
    ctx.lineWidth = rotateActive || rotateHover ? 2.2 : 1.4;
    ctx.setLineDash([4, 5]);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(rx, ry);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.shadowColor = moveActive || moveHover ? "rgba(180, 238, 255, 0.95)" : "rgba(116, 211, 244, 0.72)";
  ctx.shadowBlur = moveActive || moveHover ? 18 : 10;
  ctx.fillStyle = moveActive || moveHover ? "rgba(224, 250, 255, 0.96)" : "rgba(120, 214, 244, 0.86)";
  ctx.strokeStyle = "rgba(236, 252, 255, 0.96)";
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.arc(cx, cy, 7.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  if (view.mode !== "side" && rotateHandle) {
    ctx.shadowColor = rotateActive || rotateHover ? "rgba(180, 238, 255, 0.96)" : "rgba(116, 211, 244, 0.70)";
    ctx.shadowBlur = rotateActive || rotateHover ? 18 : 10;
    ctx.fillStyle = rotateActive || rotateHover ? "rgba(224, 250, 255, 0.96)" : "rgba(82, 172, 205, 0.88)";
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

function drawSafetyObstacles() {
  const obstacles = mapSafetyObstacles();
  const draft = obstacleDraft?.points?.length
    ? [{
        id: "draft_obstacle",
        label: "New Object",
        points: obstacleDraft.points,
        bounds: obstacleBoundsFromPoints(obstacleDraft.points, 0),
        clearance_m: selectedObstacleClearance(),
        color: selectedObstacleColor(),
        opacity: selectedObstacleOpacity(),
        draft: true,
      }]
    : [];
  for (const obstacle of obstacles.concat(draft)) {
    const prism = obstacleHullPrism(obstacle);
    if (!prism) continue;
    const color = sanitizeHexColor(obstacle.color, obstacle.draft ? "#86dfff" : "#cfd8df");
    const opacity = Math.max(0.05, Math.min(0.95, Number(obstacle.opacity ?? 0.24)));
    const fill = rgbaFromHex(color, obstacle.draft ? Math.min(0.68, opacity + 0.10) : opacity);
    const stroke = rgbaFromHex(color, obstacle.draft ? 0.90 : Math.min(0.92, opacity + 0.34));
    drawPolygonFace(prism.bottom, fill, stroke, obstacle.draft ? 1.6 : 1.1);
    drawPolygonFace(prism.top, rgbaFromHex(color, Math.min(0.72, opacity + 0.08)), stroke, obstacle.draft ? 1.7 : 1.2);
    for (let i = 0; i < prism.bottom.length; i++) {
      const j = (i + 1) % prism.bottom.length;
      drawPolygonFace([prism.bottom[i], prism.bottom[j], prism.top[j], prism.top[i]], fill, stroke, obstacle.draft ? 1.4 : 1.0);
      drawLine(prism.bottom[i], prism.bottom[j], stroke, obstacle.draft ? 1.6 : 1.0);
      drawLine(prism.top[i], prism.top[j], stroke, obstacle.draft ? 1.8 : 1.2);
      drawLine(prism.bottom[i], prism.top[i], stroke, obstacle.draft ? 1.3 : 0.9);
    }
    for (let i = 0; i < (obstacle.points || []).length; i++) {
      drawObstaclePointHandle(obstacle.points[i], obstacle.id, i, Boolean(obstacle.draft));
    }
    const center = obstacleCenter(obstacle);
    if (center) {
      drawLabel(center, obstacle.label || "object", obstacle.draft ? "#a9f7ff" : "#d8e3ea");
      drawObstacleTransformHandles(obstacle, center);
    }
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

function editableObstacleItems() {
  const items = mapSafetyObstacles().map(obstacle => ({ ...obstacle, draft: false }));
  if (obstacleDraft?.points?.length) {
    items.push({
      id: "draft_obstacle",
      label: "New Object",
      points: obstacleDraft.points,
      bounds: obstacleBoundsFromPoints(obstacleDraft.points, 0),
      clearance_m: selectedObstacleClearance(),
      color: selectedObstacleColor(),
      opacity: selectedObstacleOpacity(),
      draft: true,
    });
  }
  return items;
}

function obstaclePointHit(clientX, clientY) {
  if (safetyBarrierMode !== "obstacles") return null;
  const rect = canvas.getBoundingClientRect();
  const sx = clientX - rect.left;
  const sy = clientY - rect.top;
  const items = editableObstacleItems();
  for (let oi = items.length - 1; oi >= 0; oi--) {
    const obstacle = items[oi];
    const points = obstacle.points || [];
    for (let pi = points.length - 1; pi >= 0; pi--) {
      const [x, y] = project(points[pi]);
      const xDx = (x + 14) - sx;
      const xDy = (y - 12) - sy;
      if (xDx * xDx + xDy * xDy <= 11 * 11) {
        return { obstacleId: obstacle.id, pointIndex: pi, draft: Boolean(obstacle.draft), deleteHit: true };
      }
      const dx = x - sx;
      const dy = y - sy;
      if (dx * dx + dy * dy <= 18 * 18) {
        return { obstacleId: obstacle.id, pointIndex: pi, draft: Boolean(obstacle.draft), deleteHit: false };
      }
    }
  }
  return null;
}

function obstacleTransformHit(clientX, clientY) {
  if (safetyBarrierMode !== "obstacles" || obstacleEditing) return null;
  const rect = canvas.getBoundingClientRect();
  const sx = clientX - rect.left;
  const sy = clientY - rect.top;
  const obstacles = mapSafetyObstacles();
  for (let oi = obstacles.length - 1; oi >= 0; oi--) {
    const obstacle = obstacles[oi];
    const center = obstacleCenter(obstacle);
    if (!center) continue;
    const rotateHandle = obstacleRotateHandlePoint(obstacle, center);
    if (view.mode !== "side" && rotateHandle) {
      const [rx, ry] = project(rotateHandle);
      const rdx = rx - sx;
      const rdy = ry - sy;
      if (rdx * rdx + rdy * rdy <= 20 * 20) {
        return { type: "rotate", obstacleId: obstacle.id };
      }
    }
    const [cx, cy] = project(center);
    const cdx = cx - sx;
    const cdy = cy - sy;
    if (cdx * cdx + cdy * cdy <= 22 * 22) {
      return { type: "move", obstacleId: obstacle.id };
    }
  }
  return null;
}

function obstacleHitKey(hit) {
  if (!hit) return "";
  return `${hit.obstacleId}:${hit.pointIndex}:${hit.deleteHit ? "delete" : "drag"}`;
}

function obstacleTransformKey(hit) {
  if (!hit) return "";
  return `${hit.type}:${hit.obstacleId}`;
}

function updateObstaclePointHover(clientX, clientY) {
  if (safetyBarrierMode !== "obstacles" || obstaclePointDrag || obstacleTransformDrag) {
    if (obstaclePointHover || obstacleTransformHover) {
      obstaclePointHover = null;
      obstacleTransformHover = null;
      markFastInteraction(90);
    }
    return;
  }
  const hit = obstaclePointHit(clientX, clientY);
  const transform = hit ? null : obstacleTransformHit(clientX, clientY);
  if (
    obstacleHitKey(hit) !== obstacleHitKey(obstaclePointHover)
    || obstacleTransformKey(transform) !== obstacleTransformKey(obstacleTransformHover)
  ) {
    obstaclePointHover = hit;
    obstacleTransformHover = transform;
    markFastInteraction(90);
  }
  if (hit) canvas.style.cursor = hit.deleteHit ? "pointer" : "grab";
  else if (transform) canvas.style.cursor = "grab";
  else if (!barrierCornerHover && !barrierTransformHover) canvas.style.cursor = missionSelecting || patrolSelecting ? "crosshair" : "";
}

function replaceObstacleInCurrentMap(obstacleId, updater) {
  const next = mapSafetyObstacles().map(obstacle => (
    obstacle.id === obstacleId ? obstaclePayloadForSave(updater(obstacle)) : obstaclePayloadForSave(obstacle)
  )).filter(Boolean);
  if (currentMapEntry) currentMapEntry.safety_obstacles = next;
  const libEntry = (mapLibraryData.maps || []).find(m => m.id === currentMapEntry?.id);
  if (libEntry) libEntry.safety_obstacles = next;
  plannedMission = null;
  plannedPatrol = null;
  renderMissionCommands([]);
  renderPatrolCommands([]);
  invalidateStaticLayer();
  return next;
}

function deleteObstaclePoint(hit) {
  if (!hit) return false;
  if (hit.draft) {
    if (!obstacleDraft?.points?.length) return false;
    obstacleDraft.points.splice(hit.pointIndex, 1);
    obstacleDraft.bounds = obstacleBoundsFromPoints(obstacleDraft.points, 0);
    obstaclePointHover = null;
    updateObstacleControls();
    updateObstacleStatus();
    invalidateStaticLayer();
    return true;
  }
  const obstacles = mapSafetyObstacles();
  const obstacle = obstacles.find(candidate => candidate.id === hit.obstacleId);
  if (!obstacle) return false;
  const nextPoints = obstacle.points.filter((_, index) => index !== hit.pointIndex);
  if (nextPoints.length < 2) {
    const ok = window.confirm(`${obstacle.label} needs at least two points. Delete the whole object?`);
    if (!ok) return false;
    saveSafetyObstacles(obstacles.filter(candidate => candidate.id !== hit.obstacleId));
    selectedObstacleId = null;
    obstaclePointHover = null;
    return true;
  }
  const next = obstacles.map(candidate => (
    candidate.id === hit.obstacleId
      ? obstaclePayloadForSave({ ...candidate, points: nextPoints, bounds: obstacleBoundsFromPoints(nextPoints, 0) })
      : obstaclePayloadForSave(candidate)
  )).filter(Boolean);
  saveSafetyObstacles(next);
  obstaclePointHover = null;
  return true;
}

function updateObstaclePointFromPointer(clientX, clientY) {
  if (!obstaclePointDrag) return false;
  const source = obstaclePointDrag.draft
    ? obstacleDraft
    : mapSafetyObstacles().find(obstacle => obstacle.id === obstaclePointDrag.obstacleId);
  const anchor = source?.points?.[obstaclePointDrag.pointIndex];
  if (!anchor) return false;
  const nextPoint = screenToViewEditPlane(clientX, clientY, anchor);
  if (!nextPoint) return false;
  obstacleDragMoved = true;
  if (obstaclePointDrag.draft) {
    obstacleDraft.points[obstaclePointDrag.pointIndex] = nextPoint;
    obstacleDraft.bounds = obstacleBoundsFromPoints(obstacleDraft.points, 0);
  } else {
    replaceObstacleInCurrentMap(obstaclePointDrag.obstacleId, obstacle => {
      const points = obstacle.points.map((point, index) => index === obstaclePointDrag.pointIndex ? nextPoint : point);
      return { ...obstacle, points, bounds: obstacleBoundsFromPoints(points, 0) };
    });
  }
  updateObstacleStatus("Object point adjusted. Drag more points or save is applied on release.", "busy");
  return true;
}

function saveDraggedObstaclePoint() {
  if (!obstaclePointDrag) return;
  const hit = obstaclePointDrag;
  obstaclePointDrag = null;
  obstaclePointHover = null;
  obstacleClickSuppress = true;
  if (hit.draft) {
    updateObstacleControls();
    updateObstacleStatus();
    invalidateStaticLayer();
    return;
  }
  if (obstacleDragMoved) {
    saveSafetyObstacles(mapSafetyObstacles());
    obstacleDragMoved = false;
  } else {
    setSelectedObstacle(hit.obstacleId);
  }
}

function startObstacleTransformDrag(hit, clientX, clientY) {
  const obstacle = mapSafetyObstacles().find(candidate => candidate.id === hit.obstacleId);
  const points = (obstacle?.points || []).map(point => point.slice(0, 3));
  const center = obstacleCenter(obstacle);
  if (!obstacle || points.length < 2 || !center) return false;
  if (hit.type === "move") {
    const pointer = screenToViewEditPlane(clientX, clientY, center);
    if (!pointer) return false;
    obstacleTransformDrag = {
      type: "move",
      obstacleId: hit.obstacleId,
      startPoints: points,
      center,
      startPointer: pointer,
    };
    obstacleDragMoved = false;
    return true;
  }
  const rotateHandle = obstacleRotateHandlePoint(obstacle, center);
  const pointer = screenToViewEditPlane(clientX, clientY, rotateHandle || center) || rotateHandle;
  if (!pointer) return false;
  obstacleTransformDrag = {
    type: "rotate",
    obstacleId: hit.obstacleId,
    startPoints: points,
    center,
    startAngle: Math.atan2(pointer[2] - center[2], pointer[0] - center[0]),
  };
  obstacleDragMoved = false;
  return true;
}

function updateObstacleTransformFromPointer(clientX, clientY) {
  if (!obstacleTransformDrag) return false;
  const drag = obstacleTransformDrag;
  let nextPoints = drag.startPoints.map(point => point.slice(0, 3));
  if (drag.type === "move") {
    const pointer = screenToViewEditPlane(clientX, clientY, drag.center);
    if (!pointer) return false;
    const delta = [
      pointer[0] - drag.startPointer[0],
      pointer[1] - drag.startPointer[1],
      pointer[2] - drag.startPointer[2],
    ];
    nextPoints = nextPoints.map(point => [point[0] + delta[0], point[1] + delta[1], point[2] + delta[2]]);
  } else if (drag.type === "rotate") {
    const pointer = screenToViewEditPlane(clientX, clientY, obstacleRotateHandlePoint({ points: drag.startPoints }, drag.center) || drag.center);
    if (!pointer) return false;
    const angle = Math.atan2(pointer[2] - drag.center[2], pointer[0] - drag.center[0]);
    const delta = angle - drag.startAngle;
    const cos = Math.cos(delta);
    const sin = Math.sin(delta);
    nextPoints = nextPoints.map(point => {
      const dx = point[0] - drag.center[0];
      const dz = point[2] - drag.center[2];
      return [
        drag.center[0] + cos * dx - sin * dz,
        point[1],
        drag.center[2] + sin * dx + cos * dz,
      ];
    });
  }
  replaceObstacleInCurrentMap(drag.obstacleId, obstacle => ({
    ...obstacle,
    points: nextPoints,
    bounds: obstacleBoundsFromPoints(nextPoints, 0),
  }));
  obstacleDragMoved = true;
  updateObstacleStatus(`${drag.type === "move" ? "Object moved" : "Object rotated"}. Release to save.`, "busy");
  updateMissionStatus();
  return true;
}

function saveDraggedObstacleTransform() {
  if (!obstacleTransformDrag) return;
  const hit = obstacleTransformDrag;
  obstacleTransformDrag = null;
  obstacleTransformHover = null;
  obstacleClickSuppress = true;
  if (obstacleDragMoved) {
    obstacleDragMoved = false;
    selectedObstacleId = hit.obstacleId;
    saveSafetyObstacles(mapSafetyObstacles());
  } else {
    setSelectedObstacle(hit.obstacleId);
  }
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
  canvas.style.cursor = missionSelecting || patrolSelecting ? "crosshair" : "";
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
    color: sanitizeHexColor(barrier.color || selectedBarrierColor(), "#cfd8df"),
    opacity: Math.max(0.05, Math.min(0.95, Number(barrier.opacity ?? selectedBarrierOpacity()))),
    created_at: barrier.created_at,
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
  plannedPatrol = null;
  renderMissionCommands([]);
  renderPatrolCommands([]);
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
  const box = room?.structureBox;
  if (box?.bottom?.length === 4) {
    const bottom = box.bottom;
    const floorY = bottom.reduce((sum, p) => sum + p[1], 0) / bottom.length;
    const u0 = bottom[0];
    const u1 = bottom[1];
    const v1 = bottom[3];
    const uSpan = norm(sub(u1, u0));
    const vSpan = norm(sub(v1, u0));
    const step = Math.max(0.25, Math.pow(10, Math.floor(Math.log10(Math.max(uSpan, vSpan) / 8))));
    const uCount = Math.max(2, Math.ceil(uSpan / step));
    const vCount = Math.max(2, Math.ceil(vSpan / step));

    for (let i = 0; i <= uCount; i++) {
      const t = i / uCount;
      const a = lerpVec(bottom[0], bottom[1], t);
      const b = lerpVec(bottom[3], bottom[2], t);
      a[1] = floorY;
      b[1] = floorY;
      const major = i % 5 === 0 || i === uCount;
      drawLine(a, b, major ? "rgba(75,205,255,0.16)" : "rgba(75,205,255,0.07)", major ? 1.2 : 0.8);
    }
    for (let i = 0; i <= vCount; i++) {
      const t = i / vCount;
      const a = lerpVec(bottom[0], bottom[3], t);
      const b = lerpVec(bottom[1], bottom[2], t);
      a[1] = floorY;
      b[1] = floorY;
      const major = i % 5 === 0 || i === vCount;
      drawLine(a, b, major ? "rgba(75,205,255,0.16)" : "rgba(75,205,255,0.07)", major ? 1.2 : 0.8);
    }

    const xStart = lerpVec(bottom[0], bottom[3], 0.08);
    const xEnd = lerpVec(bottom[1], bottom[2], 0.08);
    const zStart = lerpVec(bottom[0], bottom[1], 0.08);
    const zEnd = lerpVec(bottom[3], bottom[2], 0.08);
    xStart[1] = floorY; xEnd[1] = floorY;
    zStart[1] = floorY; zEnd[1] = floorY;
    drawLine(xStart, xEnd, "rgba(255,105,140,0.82)", 2);
    drawLine(zStart, zEnd, "rgba(91,169,255,0.82)", 2);
    drawLine(bottom[0], [bottom[0][0], floorY + Math.max(0.2, (room.bounds.max[1] - room.bounds.min[1]) * 0.2), bottom[0][2]], "rgba(105,218,255,0.82)", 2);
    drawLabel(lerpVec(xStart, xEnd, 0.72), "room X", "#ff9db5");
    drawLabel(lerpVec(zStart, zEnd, 0.72), "room Z", "#95c7ff");
    return;
  }

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
  const box = room?.structureBox;
  const b = room?.structureBounds || room?.bounds;
  if (!box && !b) return;
  let bottom = box?.bottom;
  let top = box?.top;
  if (!bottom || !top) {
    const floorY = Math.max(room.floorY, b.min[1]);
    const ceilingY = Math.max(floorY + 0.24, b.max[1]);
    const x0 = b.min[0], x1 = b.max[0];
    const z0 = b.min[2], z1 = b.max[2];
    bottom = [
      [x0, floorY, z0],
      [x1, floorY, z0],
      [x1, floorY, z1],
      [x0, floorY, z1],
    ];
    top = bottom.map(p => [p[0], ceilingY, p[2]]);
  }
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
  const good = room?.poses?.filter(p => isRealPose(p)) || [];
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
  const good = room.poses.filter(p => isRealPose(p));
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

function isPatrolPlanningMode() {
  return Boolean(
    patrolSelecting ||
    editingPatrolId ||
    activePatrolId ||
    patrolPoints.length ||
    plannedPatrol ||
    activeExecutionPatrolRoute
  );
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
  // Flight calibration and visualization must share the same TSolve heading.
  // rotationHeading is a replay-display alignment and is not used by bridge
  // control, so it must never override a real room-frame rheading here.
  if (cur?.rheading && norm(cur.rheading) > 1e-8) return cur.rheading;
  if (cur?.rotationHeading && norm(cur.rotationHeading) > 1e-8) return cur.rotationHeading;
  if (cur?.pathHeading && norm(cur.pathHeading) > 1e-8) return cur.pathHeading;
  const good = room.poses.filter(p => isRealPose(p));
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
  return normalize(rotateHorizontalHeading(heading, DRONE_VISUAL_YAW_OFFSET + selectedDroneHeadingTrimRad()));
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
    liveRouteRenderingActive() ? "live-route" : (room?.poses?.length || 0),
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
    isPatrolPlanningMode() ? "patrol-plan" : "replay-route",
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
      if (!isPatrolPlanningMode() && !liveRouteRenderingActive()) drawPath();
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

  const good = room.poses.filter(p => isRealPose(p));
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

  const trustedCur = closestPose();
  const cur = liveRouteRenderingActive()
    ? currentLiveDisplayPose(trustedCur)
    : trustedCur;
  currentRenderedPose = cur || null;
  if (view.mode === "drone" && cur?.rcenter) {
    centerViewOn(cur.rcenter, 0.50, 0.56, true);
  }

  drawStaticLayer(rect, dpr);
  if (!isPatrolPlanningMode() && liveRouteRenderingActive()) drawPath();
  drawSafetyBarriers();
  drawSafetyObstacles();
  drawPatrolCoverageRisk();
  drawPatrolMission(cur);
  drawMissionTarget(cur);

  if (replayFramePlaybackEnabled) {
    const framePose = replayFramePoseAt(currentReplayClockTime(room?.poses || []));
    if (framePose) updateReplayFrameViewForPose(framePose);
  }

  if (cur && cur.rcenter) {
    if (!replayFramePlaybackEnabled) updateReplayFrameViewForPose(cur);
    drawRouteMarker(cur.rcenter, 12);
    if (window.directDroneOverlayInstalled === false && !window.directDroneModelReady) {
      drawDroneIcon(cur.rcenter, headingForPose(cur));
    }
    drawLabel(cur.rcenter, "drone", "#ffd5df");
    poseTime.textContent = cur.time_sec == null ? cur.instance_id : `${Number(cur.time_sec).toFixed(2)} s`;
    poseTotal.textContent = cur.total_ms == null ? "-" : `${Number(cur.total_ms).toFixed(2)} ms`;
    poseAction.textContent = cur.stages_ms?.ysolve_static_action_double_ms == null ? "-" : `${Number(cur.stages_ms.ysolve_static_action_double_ms).toFixed(2)} ms`;
    poseRoot.textContent = cur.stages_ms?.ysolve_static_root_total_ms == null ? "-" : `${Number(cur.stages_ms.ysolve_static_root_total_ms).toFixed(2)} ms`;
    poseCenter.textContent = formatVector(cur.rcenter || cur.center);
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
  stopPoseClockPlayback();
  replayFrameHoldTimeSec = null;
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
  const accepted = Number(quality.accepted ?? poses.filter(p => p.success && !p.held_pose).length ?? 0);
  const held = Number(replay?.counts?.held ?? poses.filter(p => p.held_pose).length ?? 0);
  const failed = Number(replay?.counts?.failed ?? poses.filter(p => p.success === false).length ?? 0);
  const frameCount = Number(replay?.counts?.frames ?? poses.length ?? 0);
  const qualityNotes = [held ? `${held} held` : "", failed ? `${failed} failed` : ""].filter(Boolean).join(", ");
  const taughtBaselineReplay = ["route_constrained_taught_baseline", "simulated_live_patrol_baseline"].includes(replay?.kind);
  const replayLine = liveReplayInFlight
    ? `Live TSolve initializing${accepted ? `: ${accepted} accepted` : ""}`
    : (taughtBaselineReplay
      ? `${accepted}/${frameCount || poses.length} validated recorded-frame poses`
    : (poses.length
      ? `${accepted}/${frameCount || poses.length} real TSolve R,t updates${qualityNotes ? ` (${qualityNotes})` : ""}`
      : "No TSolve live replay yet"));
  const streamLine = liveReplayInFlight
    ? "Live replay processing: waiting for exported R,t stream"
    : (taughtBaselineReplay
      ? "Captured DJI frames drive a live-timed corrected-pose simulation"
      : (poses.length ? "MP4 stream replay drives pose time" : "Upload drone video to localize online"));
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
    const firstPose = firstPlayableReplayPose();
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
  renderSavedPatrols();
  renderBarrierList();
  renderObstacleList();
  renderStartPreview();
}

function fleetSelectedDrone() {
  return (fleetData.drones || []).find(item => item.id === selectedFleetDroneId) || null;
}

function setFleetStatus(message, tone = "") {
  if (!fleetDispatchStatus) return;
  fleetDispatchStatus.textContent = message;
  fleetDispatchStatus.classList.toggle("error", tone === "error");
  fleetDispatchStatus.classList.toggle("ok", tone === "ok");
}

function fillFleetSelect(select, items, selectedValue, placeholder, labeler) {
  if (!select) return;
  const previous = selectedValue || select.value;
  select.replaceChildren();
  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = placeholder;
  select.appendChild(blank);
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = labeler(item);
    select.appendChild(option);
  }
  if ([...select.options].some(option => option.value === previous)) select.value = previous;
}

function refreshFleetPatrolOptions() {
  const map = (fleetData.maps || []).find(item => item.id === fleetMapSelect?.value);
  fillFleetSelect(
    fleetPatrolSelect,
    map?.patrols || [],
    fleetPatrolSelect?.value,
    map ? "Choose saved patrol" : "Choose a map first",
    patrol => `${patrol.title} · ${patrol.points} points`,
  );
}

function formatFleetEventTime(value) {
  const date = new Date(Number(value || 0) * 1000);
  if (!Number.isFinite(date.getTime())) return "—";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function fleetSessionLabel(session) {
  if (!session) return "Offline";
  if (session.status === "error") return "Attention";
  if (session.status === "stopping") return "Stopping";
  if (session.patrol_running) return "Patrolling";
  if (session.status === "running" && session.localization_ready) return "Ready";
  if (session.status === "running" || session.status === "queued") return "Connecting";
  return "Offline";
}

function fleetControlAvailability(session) {
  const active = ["queued", "running", "stopping"].includes(session?.status);
  const bridgeState = session?.bridge_status || "offline";
  const ready = active && Boolean(session?.localization_ready) && bridgeState === "streaming";
  const controlPending = Boolean(session?.control_pending);
  return {
    active,
    bridgeState,
    ready,
    takeoff: ready && !session?.airborne && !session?.takeoff_pending,
    startPatrol: ready && Boolean(session?.airborne) && !session?.patrol_running && !session?.patrol_pending && !controlPending,
    stopPatrol: active && Boolean(session?.airborne),
    land: active && Boolean(session?.airborne) && !session?.patrol_running && !session?.land_pending && !controlPending,
    endSession: active,
  };
}

function createFleetOverviewCard(droneId) {
  const card = document.createElement("article");
  card.className = "fleet-overview-card";
  card.dataset.fleetDroneId = droneId;
  card.innerHTML = `
    <div class="fleet-overview-card-head">
      <div><h3 data-field="name">Drone</h3><p data-field="assignment">No assignment</p></div>
      <div class="fleet-state-badge" data-field="badge">Offline</div>
    </div>
    <div class="fleet-overview-map-window">
      <iframe data-field="map" title="" loading="eager" hidden></iframe>
      <div class="fleet-overview-map-placeholder"><span>⌁</span><small>Dispatch this drone to open its live 3D map</small></div>
      <div class="fleet-overview-video-bar"><span data-field="bridge">Offline</span><strong data-field="frames">0 frames</strong></div>
    </div>
    <div class="fleet-overview-metrics">
      <div><span>Localization</span><strong data-field="localization">Waiting</strong></div>
      <div><span>Accepted</span><strong data-field="poses">0 poses</strong></div>
      <div><span>Patrol</span><strong data-field="patrol">—</strong></div>
      <div><span>Last action</span><strong data-field="action">None</strong></div>
    </div>
    <div class="fleet-overview-recent"><strong>Latest</strong><span data-field="event">Session created.</span></div>
    <div class="fleet-overview-controls" aria-label="Drone quick controls">
      <button type="button" data-action="focus">Inspect</button>
      <button type="button" data-action="takeoff">Take Off</button>
      <button type="button" data-action="start_patrol">Start Patrol</button>
      <button type="button" data-action="stop_patrol">Stop / Hover</button>
      <button type="button" data-action="land">Land</button>
      <button type="button" data-action="end_session">End Session</button>
    </div>
  `;
  for (const button of card.querySelectorAll("button[data-action]")) {
    button.addEventListener("click", () => {
      const action = button.dataset.action;
      if (action === "focus") {
        selectedFleetDroneId = droneId;
        if (fleetDroneSelect) fleetDroneSelect.value = droneId;
        renderFleetSwitcher();
        renderFleetFocus();
        fleetFocus?.scrollIntoView({ behavior: "smooth", block: "start" });
      } else if (action === "end_session") {
        runUi(() => endFleetSession(droneId));
      } else {
        runUi(() => fleetControl(action, droneId));
      }
    });
  }
  return card;
}

function updateFleetOverviewCard(card, drone) {
  const session = drone.session || {};
  const state = fleetControlAvailability(session);
  const label = fleetSessionLabel(session);
  const needsAttention = session.status === "error" || session.last_control_status === "error";
  card.style.setProperty("--fleet-color", drone.color || "#56ddff");
  card.classList.toggle("attention", needsAttention);
  card.querySelector('[data-field="name"]').textContent = drone.name;
  card.querySelector('[data-field="assignment"]').textContent = `${session.map_title || "Map"} · ${session.patrol_title || "Patrol"} · ${drone.phone_ip}`;
  const badge = card.querySelector('[data-field="badge"]');
  badge.textContent = needsAttention ? "Attention" : label;
  badge.className = `fleet-state-badge${needsAttention ? " error" : (state.active ? " running" : "")}`;
  card.querySelector('[data-field="bridge"]').textContent = String(state.bridgeState).replaceAll("_", " ");
  card.querySelector('[data-field="frames"]').textContent = `${Number(session.frames_saved || 0).toLocaleString()} frames`;
  card.querySelector('[data-field="localization"]').textContent = session.localization_ready ? "Valid" : (session.stage ? String(session.stage).replaceAll("_", " ") : "Waiting");
  card.querySelector('[data-field="poses"]').textContent = `${Number(session.accepted_pose_count || 0).toLocaleString()} poses`;
  card.querySelector('[data-field="patrol"]').textContent = session.patrol_title || "—";
  card.querySelector('[data-field="action"]').textContent = session.last_command ? String(session.last_command).replaceAll("_", " ") : "None";
  const mapFrame = card.querySelector('[data-field="map"]');
  if (session.map_id) {
    const nextSrc = `index.html?fleet-embed=1&fleet-drone=${encodeURIComponent(drone.id)}`;
    if (mapFrame.dataset.fleetSrc !== nextSrc) {
      mapFrame.dataset.fleetSrc = nextSrc;
      mapFrame.src = nextSrc;
    }
    mapFrame.title = `${drone.name} live 3D map and TSolve statistics`;
    mapFrame.hidden = false;
  } else {
    mapFrame.removeAttribute("src");
    mapFrame.removeAttribute("data-fleet-src");
    mapFrame.hidden = true;
  }
  const latestEvent = (session.events || []).at(-1);
  card.querySelector('[data-field="event"]').textContent = latestEvent?.message || session.message || "Waiting for the first operational event.";
  card.querySelector('[data-action="takeoff"]').disabled = !state.takeoff;
  card.querySelector('[data-action="start_patrol"]').disabled = !state.startPatrol;
  card.querySelector('[data-action="stop_patrol"]').disabled = !state.stopPatrol;
  card.querySelector('[data-action="land"]').disabled = !state.land;
  card.querySelector('[data-action="end_session"]').disabled = !state.endSession;
}

function renderFleetOverview() {
  if (!fleetOverviewGrid) return;
  const dispatched = (fleetData.drones || [])
    .filter(drone => Boolean(drone.session))
    .sort((a, b) => Number(["queued", "running", "stopping"].includes(b.session?.status)) - Number(["queued", "running", "stopping"].includes(a.session?.status)));
  if (fleetOverviewCount) {
    const activeCount = dispatched.filter(drone => ["queued", "running", "stopping"].includes(drone.session?.status)).length;
    fleetOverviewCount.textContent = `${dispatched.length} dispatched · ${activeCount} active`;
  }
  if (!dispatched.length) {
    fleetOverviewGrid.innerHTML = '<div class="fleet-overview-empty"><span>⌁</span><strong>No dispatched drones</strong><p>Every active drone will appear here at the same time.</p></div>';
    return;
  }
  const wantedIds = new Set(dispatched.map(drone => drone.id));
  for (const existing of fleetOverviewGrid.querySelectorAll("[data-fleet-drone-id]")) {
    if (!wantedIds.has(existing.dataset.fleetDroneId)) existing.remove();
  }
  fleetOverviewGrid.querySelector(".fleet-overview-empty")?.remove();
  for (const drone of dispatched) {
    let card = fleetOverviewGrid.querySelector(`[data-fleet-drone-id="${CSS.escape(drone.id)}"]`);
    if (!card) card = createFleetOverviewCard(drone.id);
    updateFleetOverviewCard(card, drone);
    fleetOverviewGrid.appendChild(card);
  }
}

function renderFleetSwitcher() {
  if (!fleetSwitcher) return;
  fleetSwitcher.replaceChildren();
  for (const drone of fleetData.drones || []) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `fleet-switch${drone.id === selectedFleetDroneId ? " active" : ""}`;
    button.style.setProperty("--fleet-color", drone.color || "#56ddff");
    const dot = document.createElement("span");
    dot.className = "fleet-switch-dot";
    const copy = document.createElement("span");
    const name = document.createElement("strong");
    name.textContent = drone.name;
    const state = document.createElement("small");
    state.textContent = fleetSessionLabel(drone.session);
    copy.append(name, state);
    button.append(dot, copy);
    button.addEventListener("click", () => {
      selectedFleetDroneId = drone.id;
      if (fleetDroneSelect) fleetDroneSelect.value = drone.id;
      renderFleet();
    });
    fleetSwitcher.appendChild(button);
  }
}

function renderFleetFocus() {
  const drone = fleetSelectedDrone();
  if (!fleetFocus) return;
  fleetFocus.classList.toggle("empty", !drone);
  if (!drone) return;
  const session = drone.session;
  const label = fleetSessionLabel(session);
  fleetFocusName.textContent = drone.name;
  fleetFocusOverline.textContent = `${drone.phone_ip} · dedicated Android link`;
  fleetFocusAssignment.textContent = session
    ? `${session.map_title || "Map"} · ${session.patrol_title || "Patrol"}`
    : "No active assignment";
  fleetFocusBadge.textContent = label;
  fleetFocusBadge.className = `fleet-state-badge${session?.status === "error" ? " error" : (session?.status === "running" || session?.status === "queued" || session?.status === "stopping" ? " running" : "")}`;
  const bridgeState = session?.bridge_status || "offline";
  fleetLiveIndicator.textContent = bridgeState;
  fleetLiveFrameCount.textContent = `${Number(session?.frames_saved || 0).toLocaleString()} frames`;
  fleetMetricLocalization.textContent = session?.localization_ready ? "Valid" : (session?.stage ? String(session.stage).replaceAll("_", " ") : "Waiting");
  fleetMetricPoses.textContent = Number(session?.accepted_pose_count || 0).toLocaleString();
  fleetMetricMap.textContent = session?.map_title || "—";
  fleetMetricPatrol.textContent = session?.patrol_title || "—";
  fleetMetricBridge.textContent = String(bridgeState).replaceAll("_", " ");
  fleetMetricAction.textContent = session?.last_command ? String(session.last_command).replaceAll("_", " ") : "None";
  if (fleetLivePreview) {
    if (session?.live_preview_url && bridgeState !== "offline") {
      fleetLivePreview.src = `${session.live_preview_url}?t=${Math.floor(Number(session.bridge_updated_at || Date.now() / 1000) * 2)}`;
      fleetLivePreview.hidden = false;
    } else {
      fleetLivePreview.removeAttribute("src");
      fleetLivePreview.hidden = true;
    }
  }
  const availability = fleetControlAvailability(session);
  if (fleetTakeoffButton) fleetTakeoffButton.disabled = !availability.takeoff;
  if (fleetStartPatrolButton) fleetStartPatrolButton.disabled = !availability.startPatrol;
  if (fleetHoverButton) fleetHoverButton.disabled = !availability.stopPatrol;
  if (fleetLandButton) fleetLandButton.disabled = !availability.land;
  if (fleetEndSessionButton) fleetEndSessionButton.disabled = !availability.endSession;
  const events = [...(session?.events || [])].reverse();
  fleetLogCount.textContent = `${events.length} event${events.length === 1 ? "" : "s"}`;
  fleetSmartLog.replaceChildren();
  if (!events.length) {
    const empty = document.createElement("p");
    empty.className = "fleet-log-empty";
    empty.textContent = "No session events yet. Dispatch this drone to begin its operational timeline.";
    fleetSmartLog.appendChild(empty);
  } else {
    for (const event of events) {
      const row = document.createElement("div");
      row.className = `fleet-log-event ${event.level || "info"}`;
      const time = document.createElement("time");
      time.textContent = formatFleetEventTime(event.created_at || event.updated_at);
      const dot = document.createElement("i");
      const message = document.createElement("span");
      message.textContent = event.message;
      row.append(time, dot, message);
      fleetSmartLog.appendChild(row);
    }
  }
}

function renderFleet() {
  const summary = fleetData.summary || {};
  if (fleetSummaryRegistered) fleetSummaryRegistered.textContent = Number(summary.registered || 0);
  if (fleetSummaryActive) fleetSummaryActive.textContent = Number(summary.active || 0);
  if (fleetSummaryAirborne) fleetSummaryAirborne.textContent = Number(summary.airborne || 0);
  if (fleetSummaryAttention) fleetSummaryAttention.textContent = Number(summary.attention || 0);
  if (!selectedFleetDroneId && fleetData.drones?.length) selectedFleetDroneId = fleetData.drones[0].id;
  if (selectedFleetDroneId && !(fleetData.drones || []).some(item => item.id === selectedFleetDroneId)) {
    selectedFleetDroneId = fleetData.drones?.[0]?.id || "";
  }
  fillFleetSelect(fleetDroneSelect, fleetData.drones || [], fleetDroneSelect?.value || selectedFleetDroneId, "Choose drone endpoint", drone => `${drone.name} · ${drone.phone_ip}`);
  fillFleetSelect(fleetMapSelect, fleetData.maps || [], fleetMapSelect?.value, "Choose patrol map", map => `${map.title}${map.has_geofence ? "" : " · geofence required"}`);
  refreshFleetPatrolOptions();
  renderFleetOverview();
  renderFleetSwitcher();
  renderFleetFocus();
  if (fleetStopAllButton) fleetStopAllButton.disabled = Number(summary.active || 0) === 0;
}

function scheduleFleetEmbedMesh() {
  if (!fleetEmbedMode) return;
  let attempts = 0;
  const reveal = () => {
    attempts += 1;
    if (window.ATLAS_MAP_MESH?.isVisible?.()) return;
    const button = document.getElementById("toggle-mesh");
    if (window.ATLAS_MAP_MESH?.isAvailable?.() && button && !button.disabled) {
      button.click();
      return;
    }
    if (attempts < 80) window.setTimeout(reveal, 100);
  };
  window.setTimeout(reveal, 0);
}

async function refreshFleetEmbed() {
  if (!fleetEmbedMode || fleetEmbedPollBusy) return;
  fleetEmbedPollBusy = true;
  try {
    const response = await fetch("/api/fleet", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not load fleet state.");
    const drone = (data.drones || []).find(item => item.id === fleetEmbedDroneId);
    const session = drone?.session || null;
    fleetEmbedSession = session;
    document.body.dataset.fleetEmbedDrone = fleetEmbedDroneId;
    document.body.dataset.fleetEmbedStatus = session?.status || "waiting";
    if (!drone || !session?.map_id) {
      liveReplayMessage = drone
        ? `Waiting for ${drone.name} to be assigned to a map.`
        : "This fleet drone is not registered.";
      if (stats) stats.textContent = liveReplayMessage;
      return;
    }

    const entry = (mapLibraryData.maps || []).find(item => item.id === session.map_id);
    if (!entry) throw new Error(`Assigned map ${session.map_id} is not available.`);
    const assignmentKey = `${drone.id}:${session.map_id}:${session.replay_id || session.created_at || "session"}`;
    liveReplayInFlight = true;
    pendingLiveReplayOpen = true;
    pendingLiveReplayMapId = session.map_id;
    liveAtlasPreviewActive = true;
    liveReplayMessage = session.message || `${drone.name} fleet localization is ${session.stage || session.status}.`;
    liveReplayStageDetail = `${session.map_title || entry.title} · ${session.patrol_title || "Patrol"} · ${session.stage || session.status}`;

    if (assignmentKey !== fleetEmbedAssignmentKey || currentMapEntry?.id !== entry.id || !scene || !room) {
      fleetEmbedAssignmentKey = assignmentKey;
      livePoseStreamKey = "";
      livePoseStreamCount = 0;
      poses = [];
      poseStreamMeta = null;
      liveCurrentPoseOverride = null;
      resetLiveFrameLockedPlayback();
      mapLibraryData.selected_map_id = entry.id;
      currentMapEntry = entry;
      await loadViewerData(true, entry);
      showDemo({ push: false, resetVideo: false });
      scheduleFleetEmbedMesh();
    }
  } catch (error) {
    liveReplayMessage = error?.message || String(error);
    if (stats) stats.textContent = liveReplayMessage;
  } finally {
    fleetEmbedPollBusy = false;
  }
}

async function refreshFleet() {
  const response = await fetch("/api/fleet", { cache: "no-store" });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Could not load fleet state.");
  fleetData = data;
  renderFleet();
  return data;
}

function scheduleFleetPoll() {
  window.clearTimeout(fleetPollTimer);
  if (currentScreen !== "fleet") return;
  fleetPollTimer = window.setTimeout(async () => {
    try {
      await refreshFleet();
      if (fleetDispatchStatus?.classList.contains("error")) setFleetStatus("");
    } catch (error) {
      setFleetStatus(error.message, "error");
    }
    scheduleFleetPoll();
  }, 1500);
}

async function saveFleetDrone() {
  const name = fleetDroneNameInput?.value.trim() || "";
  const phoneIp = fleetDroneIpInput?.value.trim() || "";
  if (!name || !phoneIp) throw new Error("Enter both a display name and the Android phone IP.");
  setFleetStatus("Saving independent drone endpoint...");
  const data = await postJson("/api/fleet/drone", { name, phone_ip: phoneIp });
  fleetData = { ...fleetData, ...data.fleet };
  selectedFleetDroneId = data.drone.id;
  if (fleetDroneNameInput) fleetDroneNameInput.value = "";
  if (fleetDroneIpInput) fleetDroneIpInput.value = "";
  await refreshFleet();
  setFleetStatus(`${data.drone.name} saved at ${data.drone.phone_ip}.`, "ok");
}

async function dispatchFleetDrone() {
  const droneId = fleetDroneSelect?.value || "";
  const mapId = fleetMapSelect?.value || "";
  const patrolId = fleetPatrolSelect?.value || "";
  if (!droneId || !mapId || !patrolId) throw new Error("Choose a drone, map, and saved patrol first.");
  const drone = (fleetData.drones || []).find(item => item.id === droneId);
  const map = (fleetData.maps || []).find(item => item.id === mapId);
  if (!map?.has_geofence) throw new Error("This map needs a saved closed-wall geofence before dispatch.");
  const ok = window.confirm(`Connect ${drone?.name || "this drone"} to ${map.title} and prepare the selected patrol?\n\nThis starts its independent live stream and localization. It will not take off until you press Take Off.`);
  if (!ok) return;
  setFleetStatus("Starting isolated DJI bridge and TSolve session...");
  await postJson("/api/fleet/dispatch", {
    drone_id: droneId,
    map_id: mapId,
    patrol_id: patrolId,
    fps: Number(fleetFpsSelect?.value || 5),
  });
  selectedFleetDroneId = droneId;
  await refreshFleet();
  setFleetStatus(`${drone?.name || "Drone"} is connecting. Wait for Localization: Valid before takeoff.`, "ok");
}

async function fleetControl(action, droneId = selectedFleetDroneId) {
  const drone = (fleetData.drones || []).find(item => item.id === droneId);
  if (!drone?.session) throw new Error("Select an active drone session first.");
  if (action === "start_patrol") {
    const ok = window.confirm(`Start ${drone.session.patrol_title} on ${drone.name}?\n\nThe drone will enter at the nearest patrol point and use the saved geofence and obstacle clearances.`);
    if (!ok) return;
  }
  if (action === "land") {
    const ok = window.confirm(`Land ${drone.name} now?`);
    if (!ok) return;
  }
  setFleetStatus(`Sending ${action.replaceAll("_", " ")} to ${drone.name}...`);
  await postJson("/api/fleet/control", { drone_id: drone.id, action, height_m: 1.0 });
  await refreshFleet();
  setFleetStatus(`${action.replaceAll("_", " ")} queued for ${drone.name}.`, "ok");
}

async function endFleetSession(droneId = selectedFleetDroneId) {
  const drone = (fleetData.drones || []).find(item => item.id === droneId);
  if (!drone?.session) return;
  const ok = window.confirm(`End ${drone.name}'s live session?\n\nATLAS will command hover, stop localization, and save the path. Land first if the drone is airborne.`);
  if (!ok) return;
  await postJson("/api/fleet/stop", { drone_id: drone.id });
  await refreshFleet();
  setFleetStatus(`${drone.name} is stopping and saving its path.`, "ok");
}

async function stopAllFleetSessions() {
  const active = Number(fleetData.summary?.active || 0);
  if (!active) return;
  const ok = window.confirm(`Stop all ${active} active fleet sessions?\n\nEach bridge receives an emergency hover before its localization path is closed.`);
  if (!ok) return;
  await postJson("/api/fleet/stop", {});
  await refreshFleet();
  setFleetStatus("Stop-all sent. Every active drone is entering neutral hover and saving its path.", "ok");
}

async function init() {
  await refreshMapLibrary({ render: !fleetEmbedMode });
  if (fleetEmbedMode) {
    renderStarted = true;
    showDemo({ push: false, resetVideo: false });
    render();
    await refreshFleetEmbed();
    return;
  }
  await refreshEnemyLibrary();
  await refreshFleet();
  await loadViewerData(true);
  renderStarted = true;
  updateNavState();
  setupLiveControlDrag();
  setLiveControlPinned(storedLiveControlPinned());
  updateFlightControlState();
  updatePatrolStatus();
  renderBarrierList();
  renderObstacleList();
  renderSavedPatrols();
  render();
  if (launchParams.get("patrol-view") === "1") {
    showDemo({ push: false, resetVideo: false });
  }
}

function screenTitle(screen) {
  if (screen === "modal") return "Create Map";
  if (screen === "enemy") return "Enemy Drone Lab";
  if (screen === "fleet") return "Fleet Monitor";
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
  window.clearTimeout(fleetPollTimer);
  document.body.classList.remove("show-start", "show-enemy", "show-fleet");
  document.body.classList.add("show-demo");
  mapModal?.classList.add("hidden");
  demoApp?.setAttribute("aria-hidden", "false");
  startPage?.setAttribute("aria-hidden", "true");
  enemyPage?.setAttribute("aria-hidden", "true");
  fleetPage?.setAttribute("aria-hidden", "true");
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
  window.clearTimeout(fleetPollTimer);
  video.pause();
  document.body.classList.remove("show-demo", "show-enemy", "show-fleet");
  document.body.classList.add("show-start");
  mapModal?.classList.add("hidden");
  demoApp?.setAttribute("aria-hidden", "true");
  startPage?.setAttribute("aria-hidden", "false");
  enemyPage?.setAttribute("aria-hidden", "true");
  fleetPage?.setAttribute("aria-hidden", "true");
  currentScreen = "start";
  updateNavState();
  renderMapLibrary();
  renderStartPreview();
}

function showEnemyLab(options = {}) {
  if (options.push !== false) rememberScreen("enemy");
  window.clearTimeout(fleetPollTimer);
  video.pause();
  document.body.classList.remove("show-start", "show-demo", "show-fleet");
  document.body.classList.add("show-enemy");
  mapModal?.classList.add("hidden");
  demoApp?.setAttribute("aria-hidden", "true");
  startPage?.setAttribute("aria-hidden", "true");
  enemyPage?.setAttribute("aria-hidden", "false");
  fleetPage?.setAttribute("aria-hidden", "true");
  currentScreen = "enemy";
  updateNavState();
  runUi(refreshEnemyLibrary);
}

function showFleetMonitor(options = {}) {
  if (options.push !== false) rememberScreen("fleet");
  video.pause();
  document.body.classList.remove("show-start", "show-demo", "show-enemy");
  document.body.classList.add("show-fleet");
  mapModal?.classList.add("hidden");
  demoApp?.setAttribute("aria-hidden", "true");
  startPage?.setAttribute("aria-hidden", "true");
  enemyPage?.setAttribute("aria-hidden", "true");
  fleetPage?.setAttribute("aria-hidden", "false");
  currentScreen = "fleet";
  updateNavState();
  runUi(refreshFleet);
  scheduleFleetPoll();
}

async function selectMap(mapId, openAfter = false) {
  const data = await postJson("/api/map/select", { map_id: mapId });
  if (data.state?.library) mapLibraryData = data.state.library;
  else await refreshMapLibrary();
  currentMapEntry = selectedMap();
  patrolPoints = [];
  patrolPointSafetyIssues = new Map();
  plannedPatrol = null;
  activeExecutionPatrolRoute = null;
  activePatrolExecutionContext = null;
  interruptedPatrolExecutionContext = null;
  enemyPursuitResumeContext = null;
  enemyTargetSuppressedUntilClear = false;
  patrolSelecting = false;
  patrolDraggingIndex = -1;
  editingPatrolId = null;
  activePatrolId = null;
  renderPatrolCommands([]);
  editPatrolButton?.classList.remove("active");
  patrolControlPanel?.classList.remove("is-selecting");
  renderMapLibrary();
  renderReplayTabs();
  renderSavedPatrols();
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
    setSelectedBarrier(cornerHit.barrierId);
    pushWallUndoSnapshot();
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
    setSelectedBarrier(transformHit.barrierId);
    pushWallUndoSnapshot();
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
  const obstacleHit = obstaclePointHit(e.clientX, e.clientY);
  if (obstacleHit?.deleteHit) {
    obstacleClickSuppress = true;
    if (!obstacleHit.draft) pushObstacleUndoSnapshot();
    deleteObstaclePoint(obstacleHit);
    markFastInteraction(140);
    return;
  }
  if (obstacleHit) {
    setSelectedObstacle(obstacleHit.obstacleId === "draft_obstacle" ? null : obstacleHit.obstacleId);
    if (!obstacleHit.draft) pushObstacleUndoSnapshot();
    obstaclePointDrag = obstacleHit;
    obstaclePointHover = obstacleHit;
    obstacleDragMoved = false;
    dragging = false;
    missionDraggingTarget = false;
    canvas.style.cursor = "grabbing";
    markFastInteraction(120);
    updateObstacleStatus("Drag this object point. Top view edits X/Z; side views edit visible horizontal/height plane.", "busy");
    return;
  }
  const obstacleTransformHitResult = obstacleTransformHit(e.clientX, e.clientY);
  if (obstacleTransformHitResult && startObstacleTransformDrag(obstacleTransformHitResult, e.clientX, e.clientY)) {
    setSelectedObstacle(obstacleTransformHitResult.obstacleId);
    pushObstacleUndoSnapshot();
    obstaclePointHover = null;
    obstacleTransformHover = obstacleTransformHitResult;
    dragging = false;
    missionDraggingTarget = false;
    canvas.style.cursor = "grabbing";
    markFastInteraction(120);
    updateObstacleStatus(
      obstacleTransformHitResult.type === "move"
        ? "Drag the center handle to move this object."
        : "Drag the diamond handle to rotate this object around the vertical axis.",
      "busy",
    );
    return;
  }
  if (barrierEditing) {
    dragging = false;
    markFastInteraction(120);
    return;
  }
  const patrolHit = patrolPointHitInfo(e.clientX, e.clientY);
  if (patrolHit?.deleteHit) {
    deletePatrolPoint(patrolHit.index);
    patrolDragMoved = true;
    dragging = false;
    missionDraggingTarget = false;
    markFastInteraction(160);
    return;
  }
  if (patrolHit && patrolHit.index >= 0) {
    patrolDraggingIndex = patrolHit.index;
    patrolDragMoved = false;
    dragging = false;
    missionDraggingTarget = false;
    markFastInteraction(120);
    updatePatrolStatus(`Drag patrol point ${patrolHit.index + 1}. Top view edits X/Z; side views edit horizontal/height view plane.`, "busy");
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
  if (obstaclePointDrag) {
    saveDraggedObstaclePoint();
  }
  if (obstacleTransformDrag) {
    saveDraggedObstacleTransform();
  }
  if (missionDraggingTarget) {
    missionDraggingTarget = false;
    updateMissionStatus();
  }
  if (patrolDraggingIndex >= 0) {
    patrolDraggingIndex = -1;
    updatePatrolStatus();
  }
  dragging = false;
  canvas.style.cursor = barrierCornerHover || barrierTransformHover || obstaclePointHover || obstacleTransformHover || patrolPointHover
    ? "grab"
    : (missionSelecting || patrolSelecting ? "crosshair" : "");
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
  if (obstaclePointDrag) {
    markFastInteraction(120);
    updateObstaclePointFromPointer(e.clientX, e.clientY);
    return;
  }
  if (obstacleTransformDrag) {
    markFastInteraction(120);
    updateObstacleTransformFromPointer(e.clientX, e.clientY);
    return;
  }
  if (missionDraggingTarget) {
    missionDragMoved = true;
    markFastInteraction(120);
    updateMissionTargetFromPointer(e.clientX, e.clientY);
    return;
  }
  if (patrolDraggingIndex >= 0) {
    patrolDragMoved = true;
    markFastInteraction(120);
    updatePatrolPointFromPointer(e.clientX, e.clientY);
    return;
  }
  updateBarrierHover(e.clientX, e.clientY);
  updateObstaclePointHover(e.clientX, e.clientY);
  updatePatrolPointHover(e.clientX, e.clientY);
  if (!dragging) return;
  markFastInteraction(220);
  view.yaw += (e.clientX - last.x) * 0.006;
  view.pitch += (e.clientY - last.y) * 0.006;
  view.pitch = Math.max(-1.55, Math.min(1.35, view.pitch));
  last = { x: e.clientX, y: e.clientY };
});
canvas.addEventListener("mouseleave", () => {
  if (!barrierCornerDrag && !barrierTransformDrag) clearBarrierHover();
  if (!obstaclePointDrag && !obstacleTransformDrag) {
    obstaclePointHover = null;
    obstacleTransformHover = null;
  }
  if (patrolDraggingIndex < 0) patrolPointHover = null;
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
  if (obstacleClickSuppress) {
    obstacleClickSuppress = false;
    return;
  }
  if (barrierDragMoved) {
    barrierDragMoved = false;
    return;
  }
  if (obstacleDragMoved) {
    obstacleDragMoved = false;
    return;
  }
  if (missionDragMoved) {
    missionDragMoved = false;
    return;
  }
  if (patrolDragMoved) {
    patrolDragMoved = false;
    return;
  }
  if (initialPositionSelecting) {
    const picked = nearestVisibleMapPoint(e.clientX, e.clientY);
    if (!picked?.rxyz || !liveCurrentPoseOverride?.rcenter) {
      setDjiCommandStatus("No visible map point under the cursor. Choose a point at the drone's actual position.", "error");
      return;
    }
    const rawCenter = liveCurrentPoseOverride.rcenter.map(
      (value, index) => value - Number(initialPoseOffsetRoom[index] || 0),
    );
    const candidate = [
      Number(picked.rxyz[0]) - Number(rawCenter[0]),
      0,
      Number(picked.rxyz[2]) - Number(rawCenter[2]),
    ];
    const magnitude = Math.hypot(candidate[0], candidate[2]);
    if (!Number.isFinite(magnitude) || magnitude > 1.0) {
      setDjiCommandStatus(
        `Position correction ${Number.isFinite(magnitude) ? magnitude.toFixed(2) : "invalid"} is too large. Re-localize instead of forcing the pose.`,
        "error",
      );
      return;
    }
    initialPoseOffsetRoom = candidate;
    liveCurrentPoseOverride = correctedLivePose({ ...liveCurrentPoseOverride, rcenter: rawCenter });
    initialPositionSelecting = false;
    canvas.style.cursor = "";
    updateInitialPositionControls();
    invalidateStaticLayer();
    if (missionTarget?.rxyz) planMissionPreview();
    if (patrolPoints.length >= 2) validatePatrolPreview(false);
    setDjiCommandStatus(
      `Initial position corrected by ${magnitude.toFixed(2)} map units. Flight control will use the same correction.`,
      "ok",
    );
    return;
  }
  if (obstacleEditing) {
    const picked = nearestVisibleMapPoint(e.clientX, e.clientY);
    if (!picked?.rxyz) {
      updateObstacleStatus("No visible map point under the cursor. Pick a point on the object surface.", "error");
      return;
    }
    addObstacleFromPickedPoint(picked);
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
  if (patrolSelecting) {
    const picked = nearestVisibleMapPoint(e.clientX, e.clientY);
    if (!picked?.rxyz) {
      updatePatrolStatus("No visible map point under the cursor. Try a denser point area.", "error");
      return;
    }
    patrolPoints.push({ rxyz: picked.rxyz, rgb: picked.rgb || null });
    plannedPatrol = null;
    renderPatrolCommands([]);
    updatePatrolStatus();
    updateFlightControlState();
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
  renderMissionCommands([]);
  missionSelecting = false;
  selectTargetButton?.classList.remove("active");
  updateMissionStatus();
});

document.getElementById("open-demo")?.addEventListener("click", showDemo);
document.querySelector(".map-card.selected")?.addEventListener("dblclick", showDemo);
document.getElementById("back-library").addEventListener("click", showLibrary);
navBack?.addEventListener("click", goBack);
atlasHome?.addEventListener("click", goHome);
enemyLabButton?.addEventListener("click", () => showEnemyLab());
fleetMonitorButton?.addEventListener("click", () => showFleetMonitor());
fleetMapSelect?.addEventListener("change", refreshFleetPatrolOptions);
fleetDroneSelect?.addEventListener("change", () => {
  selectedFleetDroneId = fleetDroneSelect.value;
  renderFleet();
});
fleetSaveDroneButton?.addEventListener("click", () => runUi(saveFleetDrone));
fleetDispatchButton?.addEventListener("click", () => runUi(dispatchFleetDrone));
fleetStopAllButton?.addEventListener("click", () => runUi(stopAllFleetSessions));
fleetTakeoffButton?.addEventListener("click", () => runUi(() => fleetControl("takeoff")));
fleetStartPatrolButton?.addEventListener("click", () => runUi(() => fleetControl("start_patrol")));
fleetHoverButton?.addEventListener("click", () => runUi(() => fleetControl("stop_patrol")));
fleetLandButton?.addEventListener("click", () => runUi(() => fleetControl("land")));
fleetEndSessionButton?.addEventListener("click", () => runUi(() => endFleetSession()));
enemyRefreshButton?.addEventListener("click", () => runUi(refreshEnemyLibrary));
enemyUploadSubmit?.addEventListener("click", () => runUi(uploadEnemyCalibration));
enemyPrepareAllButton?.addEventListener("click", () => runUi(() => prepareEnemyYoloDataset("")));
enemyTrainModelButton?.addEventListener("click", () => runUi(trainEnemyYoloModel));
enemyVideoUpload?.addEventListener("change", () => {
  const files = [...(enemyVideoUpload.files || [])];
  if (!enemyUploadStatus) return;
  enemyUploadStatus.textContent = files.length
    ? `${files.length} calibration video${files.length === 1 ? "" : "s"} selected.`
    : "No upload selected.";
});
enemyAnnotationProfile?.addEventListener("change", () => {
  selectEnemyProfile(enemyAnnotationProfile.value);
});
enemyExtractFramesButton?.addEventListener("click", () => runUi(() => extractEnemyFrames(selectedEnemyId)));
enemySaveBoxButton?.addEventListener("click", () => runUi(() => saveEnemyFrameLabel("labeled")));
enemyTrackBoxButton?.addEventListener("click", () => runUi(autoTrackEnemyLabels));
enemyNegativeFrameButton?.addEventListener("click", () => runUi(() => saveEnemyFrameLabel("negative")));
enemySkipFrameButton?.addEventListener("click", () => runUi(() => saveEnemyFrameLabel("skipped")));
enemyClearBoxButton?.addEventListener("click", () => {
  enemyBoxDraft = null;
  drawEnemyAnnotationCanvas();
});
enemyConfirmLockButton?.addEventListener("click", () => runUi(confirmEnemyLockOn));
enemyStartPursuitButton?.addEventListener("click", () => runUi(confirmEnemyPursuit));
enemyClearAlertButton?.addEventListener("click", clearEnemyAlert);
enemySaveRangeSampleButton?.addEventListener("click", () => runUi(saveEnemyRangeSample));
enemyValidateRangeButton?.addEventListener("click", () => runUi(validateEnemyRangeCalibration));
enemyResetRangeButton?.addEventListener("click", () => runUi(resetEnemyRangeCalibration));
enemyStopClearanceInput?.addEventListener("change", () => {
  enemyStopClearanceInput.value = selectedEnemyStopClearanceM().toFixed(2);
  updateEnemyResponseControls();
});
enemyCopyPrevBoxButton?.addEventListener("click", copyPreviousEnemyBox);
enemyAnnotationCanvas?.addEventListener("pointerdown", event => {
  const start = enemyNormalizedPoint(enemyCanvasPoint(event));
  if (!start) return;
  enemyBoxDrag = { start, end: start };
  enemyAnnotationCanvas.setPointerCapture(event.pointerId);
});
enemyAnnotationCanvas?.addEventListener("pointermove", event => {
  if (!enemyBoxDrag) return;
  const end = enemyNormalizedPoint(enemyCanvasPoint(event));
  if (!end) return;
  enemyBoxDrag.end = end;
  enemyBoxDraft = enemyBoxFromCorners(enemyBoxDrag.start, end);
  drawEnemyAnnotationCanvas();
});
enemyAnnotationCanvas?.addEventListener("pointerup", event => {
  if (enemyAnnotationCanvas.hasPointerCapture(event.pointerId)) {
    enemyAnnotationCanvas.releasePointerCapture(event.pointerId);
  }
  enemyBoxDrag = null;
  drawEnemyAnnotationCanvas();
});
enemyAnnotationCanvas?.addEventListener("pointercancel", () => {
  enemyBoxDrag = null;
});
document.getElementById("start").addEventListener("click", event => {
  event.preventDefault();
  playCurrentReplay();
});
simulateLivePathButton?.addEventListener("click", () => runUi(startSimulatedBaselineLive));
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
  obstacleEditing = false;
  obstacleDraft = null;
  patrolSelecting = false;
  addBarrierButton?.classList.remove("active");
  addObstacleButton?.classList.remove("active");
  editPatrolButton?.classList.remove("active");
  patrolControlPanel?.classList.remove("is-selecting");
  if (cancelBarrierButton) cancelBarrierButton.disabled = true;
  updateBarrierStatus();
  updateObstacleStatus();
  missionSelecting = !missionSelecting;
  selectTargetButton.classList.toggle("active", missionSelecting);
  updateMissionStatus();
});
clearTargetButton?.addEventListener("click", () => {
  missionTarget = null;
  plannedMission = null;
  renderMissionCommands([]);
  missionSelecting = false;
  selectTargetButton?.classList.remove("active");
  updateMissionStatus();
});
planMissionButton?.addEventListener("click", planMissionPreview);
startMissionButton?.addEventListener("click", async () => {
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
  const lockReason = liveMovementLockReason();
  if (lockReason) {
    updateMissionStatus(lockReason);
    setDjiCommandStatus(lockReason, "error");
    updateFlightControlState();
    return;
  }
  if (!guidedMotionArmed()) {
    updateMissionStatus("Enable guided movement after confirming the first TSolve pose before sending movement commands.");
    setDjiCommandStatus("Guided movement is not armed. Confirm localization, then enable guided movement.", "error");
    updateFlightControlState();
    return;
  }
  planMissionPreview();
  const safety = plannedMission?.route_segments?.length
    ? missionRouteSafetyCheck(plannedMission.route_segments)
    : missionBarrierCheck(missionTarget.rxyz);
  if (safety.blocked) {
    plannedMission = null;
    renderMissionCommands([]);
    updateMissionStatus(`Mission blocked by a safety barrier. ${safety.reason}`);
    updateFlightControlState();
    return;
  }
  plannedMission.commands = buildMissionCommandPlan(plannedMission);
  renderMissionCommands(plannedMission.commands);
  const commandCount = plannedMission.commands?.length || 0;
  if (!commandCount) {
    updateMissionStatus("Mission plan has no executable steps. Re-plan after selecting a destination.");
    return;
  }
  const speed = missionCommandSpeed(plannedMission.speed || missionSpeedSelect?.value);
  const ok = window.confirm(
    `Send a guarded indoor mission to the DJI bridge?\n\n` +
    `ATLAS will use in-place yaw corrections and a guarded slow forward cruise while fresh poses keep arriving.\n` +
    `Patrol speed: ${speed.toFixed(2)} m/s.\n` +
    `Keep the physical controller ready. Hover Now remains available.\n\nContinue?`
  );
  if (!ok) return;
  updateMissionStatus(`Mission confirmed. Sending ${commandCount} guarded steps to the live DJI bridge...`);
  try {
    const result = await sendDjiFlightCommand("mission", {
      mission: {
        client_safety_version: 2,
        guided_enabled: true,
        pose_max_age_seconds: 2.5,
        pose_recovery_seconds: 45.0,
        pulse_seconds: 0.30,
        smooth_continuous_cruise: true,
        cruise_window_seconds: 0.55,
        cruise_pose_watchdog_seconds: 0.65,
        max_forward_rc: 0.022,
        max_lateral_rc: 0.010,
        allow_lateral_rc: false,
        allow_axis_auto_calibration: false,
        axis_probe_rc: 0.018,
        axis_probe_seconds: 0.45,
        max_yaw_rc: 0.050,
        max_scan_yaw_rc: 0.025,
        alignment_grace_seconds: 35.0,
        max_vertical_rc: 0.018,
        max_step_seconds: 2.0,
        max_cruise_seconds: 120.0,
        max_pose_step_map_units: 0.30,
        arrival_radius_map_units: 0.24,
        arrival_deadband_map_units: 0.14,
        map_id: currentMapEntry?.id || null,
        map_title: currentMapEntry?.title || null,
        replay_id: activeReplay(currentMapEntry)?.id || null,
        target: plannedMission.target,
        approach: plannedMission.approach,
        route: plannedMission.route,
        route_segments: plannedMission.route_segments,
        speed: plannedMission.speed,
        profile: plannedMission.profile,
        commands: plannedMission.commands,
        safety_barriers: mapSafetyBarriers(),
        safety_obstacles: mapSafetyObstacles(),
        barrier_clearance_m: selectedBarrierClearance(),
        heading_trim_deg: 0,
        operator_heading_calibrated: Boolean(useModelHeadingForFlightInput?.checked),
        initial_body_heading_offset_deg: -selectedDroneHeadingTrimDeg(),
        initial_pose_offset_room: initialPoseOffsetRoom.slice(0, 3),
        confirmed_at: new Date().toISOString(),
      },
    });
    const bridgeMessage = result.result?.message || result.message || "Mission packet queued.";
    const pendingMessage = `${bridgeMessage} ${commandCount} guarded steps are visible below. Waiting for DJI bridge acknowledgement.`;
    updateMissionStatus(pendingMessage);
    setDjiCommandStatus(pendingMessage, "busy");
  } catch (error) {
    updateMissionStatus(`Mission command failed: ${error.message}`);
    setDjiCommandStatus(`Mission command failed: ${error.message}`, "error");
  }
});
editPatrolButton?.addEventListener("click", () => {
  barrierEditing = false;
  barrierDraft = null;
  obstacleEditing = false;
  obstacleDraft = null;
  missionSelecting = false;
  addBarrierButton?.classList.remove("active");
  addObstacleButton?.classList.remove("active");
  selectTargetButton?.classList.remove("active");
  if (cancelBarrierButton) cancelBarrierButton.disabled = true;
  patrolSelecting = !patrolSelecting;
  editPatrolButton.classList.toggle("active", patrolSelecting);
  patrolControlPanel?.classList.toggle("is-selecting", patrolSelecting);
  updateBarrierStatus();
  updateObstacleStatus();
  updateMissionStatus();
  updatePatrolStatus();
  invalidateStaticLayer();
});
newPatrolButton?.addEventListener("click", () => {
  barrierEditing = false;
  barrierDraft = null;
  obstacleEditing = false;
  obstacleDraft = null;
  missionSelecting = false;
  addBarrierButton?.classList.remove("active");
  addObstacleButton?.classList.remove("active");
  selectTargetButton?.classList.remove("active");
  if (cancelBarrierButton) cancelBarrierButton.disabled = true;
  newPatrolDraft();
});
clearPatrolButton?.addEventListener("click", () => {
  patrolPoints = [];
  patrolPointSafetyIssues = new Map();
  plannedPatrol = null;
  editingPatrolId = null;
  activePatrolId = null;
  patrolSelecting = false;
  patrolDraggingIndex = -1;
  editPatrolButton?.classList.remove("active");
  patrolControlPanel?.classList.remove("is-selecting");
  renderPatrolCommands([]);
  renderSavedPatrols();
  invalidateStaticLayer();
  updatePatrolStatus();
});
validatePatrolButton?.addEventListener("click", () => {
  validatePatrolPreview(false);
});
startPatrolButton?.addEventListener("click", async () => {
  await savePatrolDraft();
});
stopPatrolButton?.addEventListener("click", async () => {
  if (!liveLocalizationStarted()) {
    updatePatrolStatus("Start live localization before using Stop Patrol.", "error");
    return;
  }
  try {
    setDjiCommandStatus("Stopping patrol and sending immediate hover...", "busy");
    await sendDjiFlightCommand("hover", { patrol_stop: true, emergency_stop: true });
    activeExecutionPatrolRoute = null;
    activePatrolExecutionContext = null;
    interruptedPatrolExecutionContext = null;
    enemyPursuitResumeContext = null;
    invalidateStaticLayer();
    updatePatrolStatus("Patrol stopped. Drone is holding position.", "busy");
    setDjiCommandStatus("Patrol stopped. Immediate hover sent.", "busy");
  } catch (error) {
    setDjiCommandStatus(`Stop hover failed: ${error.message || error}`, "error");
  }
});
[patrolSpeedSelect, patrolAltitudeInput, patrolDwellSelect, patrolScanModeSelect, patrolModeSelect, patrolLoopInput].forEach(control => {
  control?.addEventListener("change", () => {
    plannedPatrol = null;
    renderPatrolCommands([]);
    if (patrolPoints.length >= 2) validatePatrolPreview(false);
    else updatePatrolStatus();
  });
});
addBarrierButton?.addEventListener("click", () => {
  if (barrierUnsaved) {
    updateBarrierStatus("Save or discard the staged wall edits before adding a new wall.", "error");
    return;
  }
  barrierEditing = true;
  barrierAdjusting = false;
  barrierDraft = null;
  obstacleEditing = false;
  obstacleDraft = null;
  missionSelecting = false;
  patrolSelecting = false;
  plannedMission = null;
  plannedPatrol = null;
  renderMissionCommands([]);
  renderPatrolCommands([]);
  selectTargetButton?.classList.remove("active");
  editPatrolButton?.classList.remove("active");
  patrolControlPanel?.classList.remove("is-selecting");
  addBarrierButton.classList.add("active");
  addObstacleButton?.classList.remove("active");
  if (cancelBarrierButton) cancelBarrierButton.disabled = false;
  updateBarrierAdjustControls();
  updateObstacleControls();
  updateBarrierStatus();
  updateObstacleStatus();
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
undoWallEditButton?.addEventListener("click", undoWallEdit);
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
  pushWallUndoSnapshot();
  plannedMission = null;
  plannedPatrol = null;
  renderMissionCommands([]);
  renderPatrolCommands([]);
  saveSafetyBarriers([]);
});
barrierClearanceInput?.addEventListener("change", () => {
  if (!selectedBarrierId) return;
  pushWallUndoSnapshot();
  updateSelectedBarrierPatch({ clearance_m: selectedBarrierClearance() });
});
barrierColorInput?.addEventListener("change", () => {
  if (!selectedBarrierId) return;
  pushWallUndoSnapshot();
  updateSelectedBarrierPatch({ color: selectedBarrierColor() });
});
barrierOpacityInput?.addEventListener("input", () => {
  if (!selectedBarrierId) return;
  updateSelectedBarrierPatch({ opacity: selectedBarrierOpacity() }, false);
});
barrierOpacityInput?.addEventListener("change", () => {
  if (!selectedBarrierId) return;
  pushWallUndoSnapshot();
  updateSelectedBarrierPatch({ opacity: selectedBarrierOpacity() }, true);
});
safetyTabWallsButton?.addEventListener("click", () => setSafetyBarrierMode("walls"));
safetyTabObstaclesButton?.addEventListener("click", () => setSafetyBarrierMode("obstacles"));
addObstacleButton?.addEventListener("click", () => {
  setSafetyBarrierMode("obstacles");
  obstacleEditing = true;
  obstacleDraft = {
    id: `obstacle_${Date.now().toString(36)}`,
    label: `Obstacle ${mapSafetyObstacles().length + 1}`,
    points: [],
    clearance_m: selectedObstacleClearance(),
    color: selectedObstacleColor(),
    opacity: selectedObstacleOpacity(),
  };
  barrierEditing = false;
  barrierDraft = null;
  barrierAdjusting = false;
  missionSelecting = false;
  patrolSelecting = false;
  plannedMission = null;
  plannedPatrol = null;
  renderMissionCommands([]);
  renderPatrolCommands([]);
  selectTargetButton?.classList.remove("active");
  editPatrolButton?.classList.remove("active");
  addBarrierButton?.classList.remove("active");
  patrolControlPanel?.classList.remove("is-selecting");
  updateObstacleControls();
  updateObstacleStatus();
  updateBarrierStatus();
  updateMissionStatus();
});
finishObstacleButton?.addEventListener("click", finishObstacleDraft);
undoObstacleEditButton?.addEventListener("click", undoObstacleEdit);
cancelObstacleButton?.addEventListener("click", () => {
  obstacleEditing = false;
  obstacleDraft = null;
  updateObstacleControls();
  updateObstacleStatus();
  invalidateStaticLayer();
});
clearObstaclesButton?.addEventListener("click", () => {
  if (!mapSafetyObstacles().length) return;
  if (!window.confirm("Clear all finite safety objects for this 3D map?")) return;
  pushObstacleUndoSnapshot();
  plannedMission = null;
  plannedPatrol = null;
  renderMissionCommands([]);
  renderPatrolCommands([]);
  saveSafetyObstacles([]);
});
obstacleClearanceInput?.addEventListener("change", () => {
  if (obstacleDraft) {
    obstacleDraft.clearance_m = selectedObstacleClearance();
    updateObstacleStatus();
    invalidateStaticLayer();
    return;
  }
  if (selectedObstacleId) {
    pushObstacleUndoSnapshot();
    updateSelectedObstaclePatch({ clearance_m: selectedObstacleClearance() });
  }
});
obstacleColorInput?.addEventListener("change", () => {
  if (obstacleDraft) {
    obstacleDraft.color = selectedObstacleColor();
    invalidateStaticLayer();
    return;
  }
  if (selectedObstacleId) {
    pushObstacleUndoSnapshot();
    updateSelectedObstaclePatch({ color: selectedObstacleColor() });
  }
});
obstacleOpacityInput?.addEventListener("input", () => {
  if (obstacleDraft) {
    obstacleDraft.opacity = selectedObstacleOpacity();
    invalidateStaticLayer();
    return;
  }
  if (selectedObstacleId) updateSelectedObstaclePatch({ opacity: selectedObstacleOpacity() }, false);
});
obstacleOpacityInput?.addEventListener("change", () => {
  if (obstacleDraft) {
    obstacleDraft.opacity = selectedObstacleOpacity();
    invalidateStaticLayer();
    return;
  }
  if (selectedObstacleId) {
    pushObstacleUndoSnapshot();
    updateSelectedObstaclePatch({ opacity: selectedObstacleOpacity() }, true);
  }
});
clearObstaclePointsButton?.addEventListener("click", () => {
  if (obstacleDraft) {
    obstacleDraft.points = [];
    obstacleDraft.bounds = null;
    updateObstacleControls();
    updateObstacleStatus();
    invalidateStaticLayer();
    return;
  }
  if (!selectedObstacleId) return;
  const obstacle = mapSafetyObstacles().find(candidate => candidate.id === selectedObstacleId);
  if (!obstacle) return;
  if (!window.confirm(`Clear all points from ${obstacle.label}? This removes the object.`)) return;
  pushObstacleUndoSnapshot();
  const next = mapSafetyObstacles().filter(candidate => candidate.id !== selectedObstacleId);
  selectedObstacleId = null;
  saveSafetyObstacles(next);
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
djiEmergencyHoverButton?.addEventListener("click", async () => {
  if (!liveLocalizationStarted()) {
    setDjiCommandStatus("Start live localization first so hover is routed through the live bridge.", "error");
    return;
  }
  try {
    setDjiCommandStatus("Sending emergency hover...", "busy");
    await sendDjiFlightCommand("hover", { emergency_stop: true });
    setDjiCommandStatus("Emergency hover sent. Mission movement should pause immediately.", "ok");
  } catch (err) {
    setDjiCommandStatus(`Hover failed: ${err.message || err}`, "error");
  }
});
guidedMotionEnable?.addEventListener("change", () => {
  if (guidedMotionEnable.checked) {
    setDjiCommandStatus("Guided movement armed. Confirm Mission will send tiny TSolve-gated pulses.", "busy");
  } else {
    setDjiCommandStatus("Guided movement disarmed. Mission commands are locked.", "");
  }
  updateFlightControlState();
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
  if (patrolPoints.length >= 2) validatePatrolPreview(false);
  else updatePatrolStatus();
  updateFlightControlState();
});
correctInitialPositionButton?.addEventListener("click", () => {
  if (!liveCurrentPoseOverride?.rcenter) {
    setDjiCommandStatus("Wait for the first visible localization before correcting its position.", "error");
    return;
  }
  initialPositionSelecting = !initialPositionSelecting;
  missionSelecting = false;
  patrolSelecting = false;
  updateInitialPositionControls();
  canvas.style.cursor = initialPositionSelecting ? "crosshair" : "";
  setDjiCommandStatus(
    initialPositionSelecting
      ? "Click a visible map point at the drone's actual horizontal position."
      : "Initial position correction cancelled.",
    initialPositionSelecting ? "busy" : "",
  );
});
resetInitialPositionButton?.addEventListener("click", () => {
  if (liveCurrentPoseOverride?.rcenter) {
    liveCurrentPoseOverride.rcenter = liveCurrentPoseOverride.rcenter.map(
      (value, index) => value - Number(initialPoseOffsetRoom[index] || 0),
    );
  }
  initialPoseOffsetRoom = [0, 0, 0];
  initialPositionSelecting = false;
  updateInitialPositionControls();
  invalidateStaticLayer();
  setDjiCommandStatus("Manual position correction reset. Reconfirm the displayed pose before movement.", "busy");
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
toggleCoverageRiskButton?.addEventListener("click", () => {
  view.showCoverageRisk = !view.showCoverageRisk;
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
  const fallback = document.body.classList.contains("show-demo")
    ? "demo"
    : (document.body.classList.contains("show-enemy")
      ? "enemy"
      : (document.body.classList.contains("show-fleet") ? "fleet" : "start"));
  const target = options.pop === false ? fallback : (screenHistory.pop() || fallback);
  if (target === "demo") showDemo({ push: false });
  else if (target === "enemy") showEnemyLab({ push: false });
  else if (target === "fleet") showFleetMonitor({ push: false });
  else showLibrary({ push: false });
}

function goBack() {
  const target = screenHistory.pop();
  if (target === "demo") showDemo({ push: false });
  else if (target === "modal") openMapModal({ push: false });
  else if (target === "enemy") showEnemyLab({ push: false });
  else if (target === "fleet") showFleetMonitor({ push: false });
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
    if (liveControlIsPinned()) return;
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
    if (currentScreen === "fleet") setFleetStatus(message, "error");
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
	      latestDjiLiveStatus = null;
	      setDjiLiveText("offline", "No DJI bridge status yet.");
	      updateFlightControlState();
	      return;
	    }
	    const status = await resp.json();
	    latestDjiLiveStatus = status;
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
    const control = status.last_control;
    if (control && control.id) {
      const result = control.result || {};
      const progress = control.progress || {};
      const progressAnchor = Array.isArray(progress.position_anchor)
        ? progress.position_anchor.slice(0, 3).map(Number)
        : null;
      const rotationPositionLocked = Boolean(
        control.command === "mission" &&
        control.status === "running" &&
        progress.translation_locked &&
        progressAnchor?.length === 3 &&
        progressAnchor.every(Number.isFinite)
      );
      liveRotationPositionAnchor = rotationPositionLocked ? progressAnchor : null;
      if (rotationPositionLocked && liveCurrentPoseOverride?.rcenter) {
        liveCurrentPoseOverride = {
          ...liveCurrentPoseOverride,
          rawRotationRcenter: liveCurrentPoseOverride.rcenter,
          rcenter: progressAnchor,
          rotationPositionLocked: true,
        };
      }
      const pursuitControl = Boolean(
        control.progress?.enemy_pursuit ||
        result.enemy_pursuit ||
        (enemyPursuitCommandId && control.id === enemyPursuitCommandId)
      );
      if (pursuitControl) {
        if (control.status === "running") {
          enemyPursuitInFlight = true;
          enemyPursuitCommandId = String(control.id);
          const progress = control.progress || {};
          updateEnemyResponseStatus(progress.message || control.message || "Guarded pursuit is running.", "busy");
        } else {
          enemyPursuitInFlight = false;
          enemyPursuitCommandId = "";
          const pursuitResult = control.result || {};
          if (pursuitResult.reached) {
            updateEnemyResponseStatus(
              enemyPursuitResumeContext
                ? `Pursuit complete at estimated ${Number(pursuitResult.final_clearance_m || pursuitResult.stop_clearance_m || 0).toFixed(2)} m. Preparing the nearest safe patrol rejoin...`
                : `Pursuit complete: holding at estimated ${Number(pursuitResult.final_clearance_m || pursuitResult.stop_clearance_m || 0).toFixed(2)} m clearance.`,
              "ok",
            );
            void resumeInterruptedPatrolAfterPursuit(String(control.id), pursuitResult);
          } else {
            enemyPursuitResumeContext = null;
            updateEnemyResponseStatus(
              `Pursuit stopped in hover: ${pursuitResult.abort_reason || control.error || pursuitResult.message || "safety gate ended the pursuit"}. Patrol is not resumed after a failed safety gate.`,
              "error",
            );
          }
        }
        updateEnemyResponseControls();
      }
      if (
        activePatrolExecutionContext?.commandId === String(control.id) &&
        control.status !== "running"
      ) {
        activePatrolExecutionContext = null;
      }
      const controlStatusKey = JSON.stringify({
        id: control.id,
        ok: control.ok,
        status: control.status || "",
        error: control.error || "",
        updated_at: control.updated_at || "",
        aborted: result.aborted || false,
        abort_reason: result.abort_reason || "",
        executed_count: Array.isArray(result.executed) ? result.executed.length : 0,
        skipped_count: Array.isArray(result.skipped) ? result.skipped.length : 0,
        pulses: Number(result.executed_pulses || 0),
      });
      if (controlStatusKey !== lastDjiControlStatusKey) {
        lastDjiControlStatusKey = controlStatusKey;
        lastDjiControlStatusId = control.id;
        if (pendingDjiControlAck?.id === control.id) pendingDjiControlAck = null;
      if (control.command === "mission") {
        if (control.status === "running") {
          const runningMessage = control.message || "Mission accepted by DJI bridge and is executing.";
          setDjiCommandStatus(runningMessage, "busy");
          updateMissionStatus(runningMessage);
        } else {
          const result = control.result || {};
	          const skipped = Array.isArray(result.skipped) ? result.skipped.length : 0;
	          const executed = Array.isArray(result.executed) ? result.executed.length : 0;
	          const aborted = result.aborted ? ` Aborted: ${result.abort_reason || "safety gate stopped the mission"}.` : "";
	          const pulses = Number(result.executed_pulses || 0);
	          const rcCounts = result.rc_summary?.pulse_counts || {};
	          const rcText = result.rc_summary
	            ? ` RC pulses: forward ${Number(rcCounts.forward || 0)}, yaw ${Number(rcCounts.yaw || 0)}, vertical ${Number(rcCounts.vertical || 0)}.`
	            : "";
	          const settings = result.guided_settings || {};
	          const settingText = settings.max_forward_rc
	            ? ` Max RC fwd ${Number(settings.max_forward_rc).toFixed(2)}, yaw ${Number(settings.max_yaw_rc || 0).toFixed(2)}.`
	            : "";
	          const failedReason = control.ok === false
	            ? (control.error || result.abort_reason || result.message || "mission did not execute")
	            : "";
	          const missionMessage = failedReason
	            ? `Mission did not execute: ${failedReason}.`
	            : result.physical_motion_locked
	            ? `Mission reached bridge: ${executed} hover hold(s) executed; ${skipped} yaw/cruise/landing step(s) are safety-locked until closed-loop TSolve flight is enabled.`
	            : `Mission reached bridge: ${executed} guarded step(s), ${pulses} RC pulse(s) executed.${rcText}${settingText}${aborted}`;
          setDjiCommandStatus(missionMessage, failedReason || result.physical_motion_locked || result.aborted ? "busy" : "ok");
          updateMissionStatus(missionMessage);
        }
      } else if (control.ok === false) {
        setDjiCommandStatus(`${control.command || "Command"} failed: ${control.error || "unknown error"}`, "error");
      } else {
        setDjiCommandStatus(`${control.command || "Command"} completed on DJI bridge.`, "ok");
      }
      }
    } else {
      liveRotationPositionAnchor = null;
    }
	    updateFlightControlState();
	    if (
      ["stopped", "cancelled", "error"].includes(String(state).toLowerCase()) &&
      liveFrameMode &&
      !liveReplayInFlight &&
      !replayFramePlaybackEnabled &&
      !replayUsesCapturedFrames()
    ) {
      setLiveFrameMode(false);
      stopPoseClockPlayback();
    }
	  } catch (error) {
	    latestDjiLiveStatus = null;
	    setDjiLiveText("offline", "Live DJI status is not reachable.");
	    updateFlightControlState();
	  }
	}

function diagnosticCount(value) {
  if (value === null || value === undefined || value === "") return "-";
  const count = Number(value);
  return Number.isFinite(count) && count >= 0 ? Math.round(count).toLocaleString() : "-";
}

function diagnosticTiming(value, total) {
  if (value === null || value === undefined || value === "") return "-";
  const milliseconds = Number(value);
  if (!Number.isFinite(milliseconds) || milliseconds < 0) return "-";
  const totalMilliseconds = Number(total);
  const percentage = Number.isFinite(totalMilliseconds) && totalMilliseconds > 0
    ? ` · ${(100 * milliseconds / totalMilliseconds).toFixed(0)}%`
    : "";
  return `${milliseconds.toFixed(milliseconds >= 100 ? 0 : 2)} ms${percentage}`;
}

function updateLocalizationDiagnostics(stream, payload) {
  if (!diagState) return;
  const diagnostic =
    payload?.latest_localization_diagnostics ||
    payload?.stream?.latest_localization_diagnostics ||
    stream?.latest_localization_diagnostics;
  if (!diagnostic) {
    diagState.textContent = "waiting";
    diagState.className = "localization-diagnostics-state";
    diagFrameMethod.textContent = "Waiting for a localization frame";
    diagReason.textContent = "No diagnostics received yet.";
    return;
  }

  const features = diagnostic.features || {};
  const timings = diagnostic.timings_ms || {};
  const total = timings.total_frame_ms;
  const frame = Number(diagnostic.frame_index);
  const frameLabel = Number.isFinite(frame) ? `Frame ${Math.round(frame).toLocaleString()}` : "Current frame";
  const method = String(diagnostic.method || "localization").replaceAll("_", " ");
  const accepted = diagnostic.accepted === true;
  diagState.textContent = accepted ? "accepted" : "held / rejected";
  diagState.className = `localization-diagnostics-state ${accepted ? "ok" : "warning"}`;
  diagFrameMethod.textContent = `${frameLabel} · ${method}`;
  diagReason.textContent = diagnostic.reason || (accepted ? "Pose accepted." : "No acceptance reason reported.");

  diagExtracted.textContent = diagnosticCount(features.extracted);
  diagMatched.textContent = diagnosticCount(features.matched);
  diagFlowInput.textContent = diagnosticCount(features.flow_input);
  diagTracked.textContent = diagnosticCount(features.tracked);
  diagPnp.textContent = diagnosticCount(features.pnp_inliers);
  diagSelected.textContent = diagnosticCount(features.selected);
  diagPruned.textContent = diagnosticCount(features.pruned);

  diagTotalMs.textContent = diagnosticTiming(total, null);
  diagFrameLoadMs.textContent = diagnosticTiming(timings.frame_load_ms, total);
  diagHeadingFlowMs.textContent = diagnosticTiming(timings.heading_flow_ms, total);
  diagFeatureMs.textContent = diagnosticTiming(timings.feature_extract_ms, total);
  diagMatchMs.textContent = diagnosticTiming(timings.match_ms, total);
  diagRegisterMs.textContent = diagnosticTiming(timings.register_ms, total);
  diagFlowMs.textContent = diagnosticTiming(timings.optical_flow_ms, total);
  diagVisualRouteMs.textContent = diagnosticTiming(timings.visual_route_ms, total);
  diagVisualHeadingMs.textContent = diagnosticTiming(timings.visual_heading_ms, total);
  diagRouteLogicMs.textContent = diagnosticTiming(timings.route_logic_ms, total);
  diagLocalRecoveryMs.textContent = diagnosticTiming(timings.local_recovery_ms, total);
  diagCaseBuildMs.textContent = diagnosticTiming(timings.case_build_ms, total);
  diagCaseOutputMs.textContent = diagnosticTiming(timings.case_output_ms, total);
  diagTsolveMs.textContent = diagnosticTiming(timings.tsolve_ms, total);
  diagBackgroundApplyMs.textContent = diagnosticTiming(timings.background_apply_ms, total);
  diagPoseUpdateMs.textContent = diagnosticTiming(timings.pose_update_ms, total);
  diagStreamPublishMs.textContent = diagnosticTiming(timings.stream_publish_ms, total);
  diagPaceMs.textContent = diagnosticTiming(timings.pace_wait_ms, total);
  diagBackgroundWorkerMs.textContent = diagnosticTiming(timings.background_worker_ms, null);
  diagOtherMs.textContent = diagnosticTiming(timings.other_ms, total);
}

function updateLivePoseStats(stream, payload) {
  updateLocalizationDiagnostics(stream, payload);
  if (!stats || !scene || !room) return;
  const title = currentMapEntry?.title || "Selected map";
  const scanLine = displayPointSummaryLine();
  const processed = Number(payload?.processed_count ?? stream?.pose_count ?? poses.length ?? 0);
  const expected = Number(payload?.expected_count ?? stream?.expected_count ?? 0);
  const renderable = (room.poses || []).filter(
    pose => pose?.rcenter && (pose.success !== false || pose.held_pose)
  ).length;
  const simulatedLive = Boolean(stream?.simulated_live || payload?.stream?.simulated_live);
  const displayed = Math.min(liveFrameLockedDisplayedCount, renderable);
  const catchupSkipped = Math.min(liveFrameLockedDroppedCount, renderable);
  const buffered = liveFrameLockedQueue.length;
  const catchupText = catchupSkipped ? `; ${catchupSkipped} late frame${catchupSkipped === 1 ? "" : "s"} skipped to stay live` : "";
  const countLine = simulatedLive
    ? (expected > 0
      ? `Live-clock simulation: ${displayed}/${renderable}/${processed}/${expected} displayed/renderable/processed/target; ${buffered} buffered${catchupText}`
      : `Live-clock simulation: ${displayed}/${renderable}/${processed} displayed/renderable/processed; ${buffered} buffered${catchupText}`)
    : (expected > 0
      ? `Live-clock TSolve: ${displayed}/${renderable}/${processed}/${expected} displayed/renderable/processed/target; ${buffered} buffered${catchupText}`
      : `Live-clock TSolve: ${displayed}/${renderable}/${processed} displayed/renderable/processed; ${buffered} buffered${catchupText}`);
  const sourceLine = mapSourceLine();
  stats.innerHTML = `${title}<br>${scene.points3D.length} COLMAP map points<br>${scanLine}${scene.map_cameras.length} map cameras<br>${countLine}<br>${liveReplayMessage || "online self-localization active"}<br>${sourceLine}`;
}

function liveRouteRenderingActive() {
  return Boolean(
    liveReplayInFlight ||
    pendingLiveReplayOpen ||
    liveReplayCompletionPending ||
    poseStreamMeta?.complete === false
  );
}

async function loadLiveReplayPartial(stream = null) {
  if (!liveReplayInFlight && !pendingLiveReplayOpen && !liveReplayCompletionPending) return false;
  if (!scene) return false;
  const resp = fleetEmbedMode
    ? await fetch(
      `/api/fleet/live-replay?drone_id=${encodeURIComponent(fleetEmbedDroneId)}&after=${Math.max(0, livePoseStreamCount)}&t=${Date.now()}`,
      { cache: "no-store" }
    )
    : await fetch(
      `/api/live-replay?after=${Math.max(0, livePoseStreamCount)}&t=${Date.now()}`,
      { cache: "no-store" }
    );
  if (!resp.ok && resp.status !== 404) return false;
  const payload = await resp.json().catch(() => null);
  if (!payload?.ok || !Array.isArray(payload.poses)) return false;

  const previousProcessed = livePoseStreamCount;
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
  const incomingPoses = payload.poses;
  const deltaStart = Number(payload.delta_start);
  const canAppendDelta = Boolean(
    payload.delta === true &&
    payload.delta_reset !== true &&
    Number.isFinite(deltaStart) &&
    deltaStart === previousProcessed
  );
  poses = canAppendDelta ? poses.concat(incomingPoses) : incomingPoses;
  poseStreamMeta = { ...payload, poses };
  const expected = Number(payload.expected_count ?? payload.stream?.expected_count ?? 0);
  const firstLatency = Number(stream?.first_pose_latency_seconds ?? payload.stream?.first_pose_latency_seconds);
  const latencyLine = Number.isFinite(firstLatency)
    ? `; first R,t in ${firstLatency.toFixed(1)} s`
    : "";
  const simulatedLive = Boolean(stream?.simulated_live || payload?.stream?.simulated_live);
  liveReplayStageDetail = simulatedLive
    ? (expected > 0
      ? `Simulated live input: ${processed}/${expected} frame poses received`
      : `Simulated live input: ${processed} frame poses received`)
    : (expected > 0
      ? `TSolve R,t stream: ${processed}/${expected} frame updates${latencyLine}`
      : `TSolve R,t stream: ${processed} frame updates${latencyLine}`);
  if (!updateLiveRoomPoseStream(poses)) room = buildRoomFrame();
  enqueueLiveFrameLockedPoses(room.poses, poseStreamMeta, payload.stream || stream);
  ensureLiveStreamVideoSource(payload.stream || stream);
  // Held poses deliberately preserve the last trusted position while carrying
  // fresh optical yaw. They are safe and necessary for a live in-place turn.
  const latestPose = latestLivePoseForDisplay(room.poses);
  const correctedLatestPose = latestPose ? correctedLivePose(latestPose) : null;
  liveCurrentPoseOverride = correctedLatestPose;
  if (liveReplayInFlight) {
    if (latestPose && !liveVideoSyncedToFirstPose) {
      liveVideoSyncedToFirstPose = true;
      liveVideoWaitingForFirstPose = false;
    }
  } else if (latestPose) {
    syncUploadedVideoToLatestPose(room.poses);
  } else {
    syncUploadedVideoToProcessingFrame(payload);
  }
  updateLivePoseStats(stream || payload.stream, poseStreamMeta);
  updateFlightControlState();
  return true;
}

async function pollLivePoseStream() {
  if (livePosePollBusy) return;
  if (!liveReplayInFlight && !pendingLiveReplayOpen) return;
  livePosePollBusy = true;
  try {
    await loadLiveReplayPartial(fleetEmbedSession || poseStreamMeta?.stream || null);
  } finally {
    livePosePollBusy = false;
  }
}

async function pollStatus(compact = false) {
  if (liveStatusPollBusy) return;
  liveStatusPollBusy = true;
  try {
    const resp = await fetch(compact ? "/api/status?compact=1" : "/api/status", { cache: "no-store" });
    if (!resp.ok) throw new Error(`status ${resp.status}`);
    const state = await resp.json();
    const compactTerminalTransition = Boolean(
      compact &&
      (
        (["done", "error", "cancelled", "failed"].includes(state.map?.status) && state.map?.status !== lastMapStatus) ||
        (["done", "error", "cancelled", "failed"].includes(state.drone?.status) && state.drone?.status !== lastDroneStatus)
      )
    );
    if (compactTerminalTransition && !state.library) {
      const libraryResp = await fetch("/api/maps", { cache: "no-store" });
      if (libraryResp.ok) state.library = await libraryResp.json();
    }
    const backendRecording = state.manual_patrol_recording || null;
    const recordingChanged = JSON.stringify(backendRecording) !== JSON.stringify(manualPatrolRecording);
    manualPatrolRecording = backendRecording;
    if (recordingChanged) renderSavedPatrols();
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
    const activeJobStates = new Set(["queued", "running", "stopping"]);
    const liveJobStates = new Set(["queued", "running", "stopping"]);
    liveReplayInFlight = liveJobStates.has(state.drone?.status);
    if (liveReplayInFlight && !liveReplayStartedAt) liveReplayStartedAt = performance.now();
    liveReplayMessage = state.drone?.message || liveReplayMessage;
    if (state.drone?.status === "queued" || state.drone?.status === "running" || state.drone?.status === "stopping") {
      const stream = state.drone?.live_stream || {};
      if (stream.map_id) pendingLiveReplayMapId = stream.map_id;
      const poseCount = Number(stream.pose_count ?? livePoseStreamCount ?? 0);
      const acceptedPoseCount = Number(stream.accepted_pose_count ?? 0);
      const expectedCount = Number(stream.expected_count ?? poseStreamMeta?.expected_count ?? 0);
      liveReplayStageDetail = expectedCount > 0
        ? `${liveReplayMessage || "TSolve online localization running"} · ${acceptedPoseCount}/${poseCount}/${expectedCount} accepted/processed/target`
        : `${liveReplayMessage || "TSolve online localization running"} · ${acceptedPoseCount}/${poseCount} accepted/processed`;
    } else if (state.drone?.status === "done") {
      liveReplayStageDetail = state.drone?.message || "Live TSolve path ready";
    } else if (state.drone?.status === "error") {
      liveReplayStageDetail = state.drone?.message || "Live TSolve path failed";
    }
    if ((currentScreen === "demo" || liveReplayInFlight) && state.drone?.status !== lastDroneStatus) {
      renderReplayTabs();
    }
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
    if (droneDoneNow) liveReplayCompletionPending = true;
    const droneErroredNow = state.drone?.status === "error" && pendingLiveReplayOpen;
    const droneCancelledNow = ["cancelled", "failed"].includes(state.drone?.status) && pendingLiveReplayOpen;
    const droneBackendResetNow = (
      state.drone?.status === "idle" &&
      pendingLiveReplayOpen &&
      ["queued", "running", "stopping"].includes(previousDroneStatus)
    );
    lastMapStatus = state.map?.status || null;
    lastDroneStatus = state.drone?.status || null;
    if (liveReplayInFlight && renderStarted && !liveReplayWaitingViewPrepared) {
      const streamMapId = String(state.drone?.live_stream?.map_id || pendingLiveReplayMapId || "");
      const canReuseLoadedMap = Boolean(
        scene && room && currentMapEntry?.id &&
        (!streamMapId || String(currentMapEntry.id) === streamMapId)
      );
      if (canReuseLoadedMap) {
        poses = [];
        poseStreamMeta = null;
        room.poses = [];
        resetLiveFrameLockedPlayback();
      } else {
        await loadViewerData(false, currentMapEntry);
        renderStartPreview();
      }
      liveReplayWaitingViewPrepared = true;
    }
    if (liveReplayInFlight && currentScreen !== "demo" && renderStarted) {
      showDemo();
    }
    if (liveReplayInFlight && scene && room && !poses.length) {
      const title = currentMapEntry?.title || "Selected map";
      const scanLine = displayPointSummaryLine();
      stats.innerHTML = `${title}<br>${scene.points3D.length} COLMAP map points<br>${scanLine}${scene.map_cameras.length} map cameras<br>Live TSolve initializing<br>${liveReplayMessage}`;
    }
    const dronePlaybackDoneNow = liveReplayCompletionPending && liveFrameLockedPlaybackDrained();
    const completedLiveFramePose = dronePlaybackDoneNow && liveFrameLockedDisplayedPose
      ? { ...liveFrameLockedDisplayedPose }
      : null;
    if ((mapDoneNow || dronePlaybackDoneNow) && renderStarted) {
      await loadViewerData(Boolean(dronePlaybackDoneNow), currentMapEntry);
      renderReplayTabs();
      renderStartPreview();
      if (completedLiveFramePose && Number.isFinite(Number(completedLiveFramePose.time_sec))) {
        replayFrameHoldTimeSec = Number(completedLiveFramePose.time_sec);
        const heldFrame = replayFramePoseAt(replayFrameHoldTimeSec);
        if (heldFrame) updateReplayFrameViewForPose(heldFrame, { force: true });
      }
      if (dronePlaybackDoneNow && pendingLiveReplayOpen && poses.length) {
        if (state.drone?.live_stream?.live_atlas) liveAtlasPreviewActive = false;
        liveReplayInFlight = false;
        liveCurrentPoseOverride = null;
        liveReplayWaitingViewPrepared = false;
        liveReplayStartedAt = 0;
        pendingLiveReplayOpen = false;
        pendingLiveReplayMapId = null;
        resetLiveFrameLockedPlayback();
        uploadStatus.textContent = `Live TSolve replay ready: ${currentMapEntry?.title || "selected map"}`;
        showDemo();
      } else if (dronePlaybackDoneNow && pendingLiveReplayOpen) {
        if (state.drone?.live_stream?.live_atlas) setLiveFrameMode(false);
        if (state.drone?.live_stream?.live_atlas) liveAtlasPreviewActive = false;
        liveReplayInFlight = false;
        liveCurrentPoseOverride = null;
        liveReplayWaitingViewPrepared = false;
        liveReplayStartedAt = 0;
        pendingLiveReplayOpen = false;
        pendingLiveReplayMapId = null;
        resetLiveFrameLockedPlayback();
        uploadStatus.textContent = "Live replay finished, but no TSolve poses were produced for this video.";
      } else if (dronePlaybackDoneNow) {
        liveReplayInFlight = false;
        liveCurrentPoseOverride = null;
        liveReplayWaitingViewPrepared = false;
        liveReplayStartedAt = 0;
        resetLiveFrameLockedPlayback();
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
    if (droneCancelledNow) {
      if (state.drone?.live_stream?.live_atlas) setLiveFrameMode(false);
      liveAtlasPreviewActive = false;
      liveReplayInFlight = false;
      liveCurrentPoseOverride = null;
      liveReplayWaitingViewPrepared = false;
      liveReplayStartedAt = 0;
      pendingLiveReplayOpen = false;
      pendingLiveReplayMapId = null;
      uploadStatus.textContent = state.drone?.message || "Live path creation stopped.";
      renderReplayTabs();
    }
    if (droneBackendResetNow) {
      setLiveFrameMode(false);
      liveAtlasPreviewActive = false;
      liveReplayInFlight = false;
      liveCurrentPoseOverride = null;
      liveReplayWaitingViewPrepared = false;
      liveReplayCompletionPending = false;
      liveReplayStartedAt = 0;
      pendingLiveReplayOpen = false;
      pendingLiveReplayMapId = null;
      resetLiveFrameLockedPlayback();
      uploadStatus.textContent = "Live localization was reset. ATLAS is ready to start a new session.";
      renderReplayTabs();
    }
  } catch {
    mapStatus.textContent = "Map: local backend not connected";
    droneStatus.textContent = "Drone replay: start scripts/atlas_app_server.py";
  } finally {
    liveStatusPollBusy = false;
  }
}

async function pollEnemyLiveDetections() {
  if (!enemyLiveDetectorState && !enemyLiveDetection) return;
  if (!enemyDetectionEnabled()) {
    renderEnemyDetectionDisabled();
    return;
  }
  try {
    const resp = await fetch(`public/live_dji/enemy_detections.json?t=${Date.now()}`, { cache: "no-store" });
    if (!resp.ok) {
      if (enemyLiveDetectorState) enemyLiveDetectorState.textContent = "No detector status yet.";
      if (enemyLiveDetection) {
        enemyLiveDetection.textContent = enemyLibraryData.selected_model
          ? "Waiting for the live bridge to start enemy detection."
          : "Train a YOLO model in Enemy Drone Lab before patrol detection.";
        enemyLiveDetection.dataset.tone = "";
      }
      return;
    }
    const payload = await resp.json();
    if (!enemyDetectionIsFresh(payload)) {
      enemyDetectionHistory = [];
      if (enemyLiveDetectorState) enemyLiveDetectorState.textContent = "stale";
      if (enemyLiveDetection) {
        enemyLiveDetection.textContent = "Detector result is stale; no flight response will be sent.";
        enemyLiveDetection.dataset.tone = "error";
      }
      return;
    }
    const status = String(payload.status || "unknown").replaceAll("_", " ");
    if (enemyLiveDetectorState) enemyLiveDetectorState.textContent = status;
    const detections = Array.isArray(payload.detections) ? payload.detections : [];
    if (enemyLiveDetection) {
      if (payload.status === "detected" && detections.length) {
        const best = detections.slice().sort((a, b) => Number(b.confidence || 0) - Number(a.confidence || 0))[0];
        enemyLiveDetection.textContent = `${detections.length} target candidate${detections.length === 1 ? "" : "s"} · ${best.class_name || "enemy drone"} ${(Number(best.confidence || 0) * 100).toFixed(0)}%`;
        enemyLiveDetection.dataset.tone = "alert";
        if (enemyTargetSuppressedUntilClear) {
          enemyLiveDetection.textContent += " · intercepted; patrol recovery will not re-chase until target leaves view";
          return;
        }
        const guardedBest = bestEnemyDetection(detections);
        const frameKey = String(payload.frame || payload.updated_at || "");
        if (guardedBest && !enemyDetectionHistory.some(item => item.frame === frameKey)) {
          enemyDetectionHistory.push({
            frame: frameKey,
            className: String(guardedBest.class_name || ""),
            target: guardedBest,
            payload,
          });
          enemyDetectionHistory = enemyDetectionHistory.slice(-ENEMY_CONFIRM_WINDOW);
        }
        if (guardedBest) {
          const className = String(guardedBest.class_name || "");
          const matches = enemyDetectionHistory.filter(item => item.className === className);
          if (matches.length >= ENEMY_CONFIRM_HITS) {
            const confirmed = matches[matches.length - 1];
            await pauseForEnemyDetection(confirmed.payload, confirmed.target);
          } else if (!enemyPursuitInFlight) {
            updateEnemyResponseStatus(
              `Possible enemy drone ${matches.length}/${ENEMY_CONFIRM_HITS} confirmations. Patrol response remains gated.`,
              "busy"
            );
          }
        }
      } else if (payload.status === "clear" || payload.status === "ready") {
        const clearFrame = String(payload.frame || payload.updated_at || "");
        if (!enemyDetectionHistory.some(item => item.frame === clearFrame)) {
          enemyDetectionHistory.push({ frame: clearFrame, className: "", target: null, payload });
          enemyDetectionHistory = enemyDetectionHistory.slice(-ENEMY_CONFIRM_WINDOW);
        }
        if (
          enemyTargetSuppressedUntilClear &&
          enemyDetectionHistory.slice(-ENEMY_CONFIRM_HITS).filter(item => !item.target).length >= ENEMY_CONFIRM_HITS
        ) {
          enemyTargetSuppressedUntilClear = false;
          clearEnemyAlert();
          enemyDetectionHistory = [];
          updateEnemyResponseStatus("Intercepted target has left view. Automatic enemy detection is armed again.", "ok");
        }
        enemyLiveDetection.textContent = payload.message || "No enemy drone detected.";
        enemyLiveDetection.dataset.tone = "ok";
        if (!enemyAlertState.active) updateEnemyResponseControls();
      } else {
        enemyLiveDetection.textContent = payload.message || "Enemy detector is not active.";
        enemyLiveDetection.dataset.tone = payload.status === "error" ? "error" : "";
        if (!enemyAlertState.active) updateEnemyResponseControls();
      }
    }
  } catch {
    if (enemyLiveDetectorState) enemyLiveDetectorState.textContent = "Detector status unavailable.";
  }
}

setEnemyDetectionEnabled(storedEnemyDetectionEnabled(), { persist: false });
void syncEnemyDetectionRuntime(enemyDetectionEnabled());
if (!fleetEmbedMode) {
  setInterval(pollDjiLivePreview, 1000);
  pollDjiLivePreview();
  setInterval(pollEnemyLiveDetections, 1000);
  pollEnemyLiveDetections();
}

document.getElementById("create-map").addEventListener("click", openMapModal);
document.getElementById("close-map-modal").addEventListener("click", closeMapModal);
mapModal?.addEventListener("click", event => {
  if (event.target === mapModal) closeMapModal();
});
document.getElementById("close-video-library-modal")?.addEventListener("click", hideVideoLibrary);
videoLibraryModal?.addEventListener("click", event => {
  if (event.target === videoLibraryModal) hideVideoLibrary();
});
document.getElementById("close-patrol-import-modal")?.addEventListener("click", hidePatrolImportModal);
patrolImportModal?.addEventListener("click", event => {
  if (event.target === patrolImportModal) hidePatrolImportModal();
});
importPatrolButton?.addEventListener("click", () => runUi(showPatrolImportModal));
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
liveAtlasPhoneIp?.addEventListener("change", () => rememberPhoneIp(liveAtlasPhoneIp.value));
savePhoneIpButton?.addEventListener("click", () => {
  rememberPhoneIp(liveAtlasPhoneIp?.value);
  updateLiveControlSummary();
});
phoneIpSelect?.addEventListener("change", () => {
  if (liveAtlasPhoneIp) liveAtlasPhoneIp.value = phoneIpSelect.value;
  rememberPhoneIp(phoneIpSelect.value);
});
liveAtlasFps?.addEventListener("change", updateLiveControlSummary);
liveAtlasPatrolSelect?.addEventListener("change", () => {
  const patrolId = String(liveAtlasPatrolSelect.value || "").trim();
  const patrol = patrolList(currentMapEntry).find(item => item.id === patrolId);
  if (patrol && !liveLocalizationStarted()) loadPatrolIntoEditor(patrol, { selecting: false });
  updateLiveControlSummary();
});
enemyDetectionEnabledInput?.addEventListener("change", () => {
  if (!enemyDetectionEnabledInput.checked && enemyPursuitInFlight) {
    enemyDetectionEnabledInput.checked = true;
    updateEnemyResponseStatus("Stop the active guarded pursuit with Hover Now before disabling enemy detection.", "error");
    return;
  }
  setEnemyDetectionEnabled(enemyDetectionEnabledInput.checked);
  void syncEnemyDetectionRuntime(enemyDetectionEnabledInput.checked);
  if (enemyDetectionEnabledInput.checked) void pollEnemyLiveDetections();
});
pinLiveControlButton?.addEventListener("click", event => {
  event.preventDefault();
  event.stopPropagation();
  setLiveControlPinned(!liveControlIsPinned());
});
liveLocalizationControl?.addEventListener("toggle", () => {
  syncLiveControlCollapsedState();
  if (renderStarted) render();
});
function applyDroneHeadingAlignment() {
  try {
    localStorage.setItem(DRONE_HEADING_TRIM_STORAGE_KEY, String(selectedDroneHeadingTrimDeg()));
  } catch {
    // Heading trim still works for this session when local storage is blocked.
  }
  if (plannedMission) {
    plannedMission.commands = buildMissionCommandPlan(plannedMission);
    renderMissionCommands(plannedMission.commands);
  }
  if (plannedPatrol) {
    plannedPatrol.commands = buildPatrolCommandPlan(plannedPatrol);
    renderPatrolCommands(plannedPatrol.commands);
  }
  if (droneHeadingTrimValue) {
    const value = selectedDroneHeadingTrimDeg();
    droneHeadingTrimValue.textContent = `${value > 0 ? "+" : ""}${value} deg`;
  }
}
droneHeadingTrimSelect?.addEventListener("input", applyDroneHeadingAlignment);
droneHeadingTrimSelect?.addEventListener("change", applyDroneHeadingAlignment);
window.addEventListener("resize", renderStartPreview);
renderPhoneIpOptions();
renderDroneHeadingTrim();
updateLiveControlSummary();
setupLiveControlSections();
if (fleetEmbedMode) {
  setInterval(() => refreshFleetEmbed(), 1000);
} else {
  setInterval(() => pollStatus(true), 2000);
}
// Fetch only newly published observations at the restored 10-FPS ceiling. Status is
// intentionally kept on its slower interval because it contains maps and job
// logs; using it as the pose clock made the model visibly pause and jump.
if (fleetEmbedMode) {
  setInterval(() => {
    pollLivePoseStream();
  }, 100);
} else {
setInterval(() => {
  pollLivePoseStream();
}, 100);
}
setInterval(() => {
  if (liveRouteRenderingActive()) advanceLiveFrameLockedPlayback();
}, 25);

window.TSOLVE_VIEWER = {
  getCurrentPose: () => liveRouteRenderingActive()
    ? currentLiveDisplayPose(liveCurrentPoseOverride)
    : (currentRenderedPose || closestPose()),
  getHeadingForPose: pose => headingForPose(pose),
  projectRoomPoint: rxyz => project(rxyz),
  projectRoomPointToViewport: rxyz => projectToViewport(rxyz),
  getRoom: () => room,
  getView: () => view,
  getCurrentMapEntry: () => currentMapEntry,
  getMapLibrary: () => mapLibraryData,
  getDroneModel: () => droneModel,
  getDroneHeadingTrimRad: () => selectedDroneHeadingTrimRad(),
  // Frame-locked evidence must use the exact localized heading for that frame.
  useDroneYawSmoothing: () => false,
  isMapInteractionBusy: () => Boolean(
    barrierEditing || obstacleEditing || missionSelecting || patrolSelecting || initialPositionSelecting ||
    barrierCornerDrag || barrierTransformDrag || obstaclePointDrag || obstacleTransformDrag ||
    missionDraggingTarget || patrolDraggingIndex >= 0
  ),
};

init().catch(err => {
  stats.textContent = `failed to load viewer data: ${err}`;
  console.error(err);
});
