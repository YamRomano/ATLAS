import * as THREE from "./vendor/three.module.js";
import { GLTFLoader } from "./vendor/GLTFLoader.js";

const host = document.getElementById("viewport");
const status = document.getElementById("status");
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setClearColor(0x071827, 0);
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x071827, 0.02);
const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 300);
const orbit = { target: new THREE.Vector3(), yaw: -0.75, pitch: 0.42, distance: 16 };
let home = { target: orbit.target.clone(), yaw: orbit.yaw, pitch: orbit.pitch, distance: orbit.distance };

scene.add(new THREE.GridHelper(30, 30, 0x5fbde9, 0x1d5270));

function placeCamera() {
  const cp = Math.cos(orbit.pitch);
  camera.position.set(
    orbit.target.x + orbit.distance * cp * Math.sin(orbit.yaw),
    orbit.target.y + orbit.distance * Math.sin(orbit.pitch),
    orbit.target.z + orbit.distance * cp * Math.cos(orbit.yaw),
  );
  camera.lookAt(orbit.target);
}

function resize() {
  const width = innerWidth;
  const height = innerHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(1, height);
  camera.updateProjectionMatrix();
}

let dragging = false;
let previousX = 0;
let previousY = 0;
renderer.domElement.addEventListener("pointerdown", (event) => {
  dragging = true;
  previousX = event.clientX;
  previousY = event.clientY;
  renderer.domElement.setPointerCapture(event.pointerId);
});
renderer.domElement.addEventListener("pointermove", (event) => {
  if (!dragging) return;
  orbit.yaw -= (event.clientX - previousX) * 0.006;
  orbit.pitch = THREE.MathUtils.clamp(orbit.pitch + (event.clientY - previousY) * 0.006, -1.45, 1.45);
  previousX = event.clientX;
  previousY = event.clientY;
});
renderer.domElement.addEventListener("pointerup", () => { dragging = false; });
renderer.domElement.addEventListener("wheel", (event) => {
  event.preventDefault();
  orbit.distance = THREE.MathUtils.clamp(orbit.distance * Math.exp(event.deltaY * 0.001), 0.25, 120);
}, { passive: false });
renderer.domElement.addEventListener("dblclick", () => {
  orbit.target.copy(home.target);
  orbit.yaw = home.yaw;
  orbit.pitch = home.pitch;
  orbit.distance = home.distance;
});

new GLTFLoader().load(
  "./public/camera_path_lab/scene_pilot_fused_preview.glb",
  (gltf) => {
    let count = 0;
    gltf.scene.traverse((node) => {
      if (!node.isPoints) return;
      count += node.geometry.getAttribute("position")?.count || 0;
      node.material = new THREE.PointsMaterial({
        size: 0.045,
        sizeAttenuation: true,
        vertexColors: true,
        transparent: true,
        opacity: 0.78,
        depthWrite: false,
        toneMapped: false,
      });
    });
    scene.add(gltf.scene);
    const bounds = new THREE.Box3().setFromObject(gltf.scene);
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    orbit.target.copy(center);
    orbit.distance = Math.max(3, size.length() * 0.78);
    home = { target: orbit.target.clone(), yaw: orbit.yaw, pitch: orbit.pitch, distance: orbit.distance };
    status.textContent = `${count.toLocaleString()} colored preview points`;
  },
  undefined,
  (error) => {
    console.error(error);
    status.textContent = "Preview failed to load";
  },
);

addEventListener("resize", resize);
resize();
renderer.setAnimationLoop(() => {
  placeCamera();
  renderer.render(scene, camera);
});
