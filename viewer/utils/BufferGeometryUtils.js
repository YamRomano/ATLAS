export function toTrianglesDrawMode(geometry) {
  // The DJI Mini 3 Pro GLB uses ordinary triangle primitives. GLTFLoader imports
  // this helper for rare triangle-strip/fan primitives, so a pass-through keeps
  // the local loader self-contained without pulling the full three examples tree.
  return geometry;
}
