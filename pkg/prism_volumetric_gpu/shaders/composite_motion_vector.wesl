// Composite-motion-vector twin: evaluates on-device the cloud screen-space
// motion vector composition, mirroring the CPU golden
// `composite_motion_vector` (in
// `prism_render_architecture::volumetric::temporal`).
//
// The temporal reprojection (design section 10) drives the cloud history
// resample with a screen-space motion vector that is the pure sum of two
// sources:
//   result = cloud_advection + camera_motion
// where `cloud_advection` is the wind-driven cloud drift projected to screen
// space and `camera_motion` is the camera-induced screen motion. Keeping it a
// pure vec2 sum makes the reprojection deterministic and trivially testable.
// The CPU golden owns that algebra; this kernel is the on-device twin that runs
// one thread per query.
//
// It uses only the portable core-WGSL subset - a vec2 add - so it runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard screen-space motion-vector composition; no Unreal Engine
// source or derived code.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One motion-vector query: the cloud advection velocity and the camera-induced
// screen motion, each a screen-space vec2. Matches `GpuQuery` in
// `src/composite_motion_vector.rs`.
struct Query {
    advection_x: f32,
    advection_y: f32,
    camera_x: f32,
    camera_y: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<vec4<f32>>;

@compute @workgroup_size(64)
fn composite_motion_vector_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let advection = vec2<f32>(q.advection_x, q.advection_y);
    let camera = vec2<f32>(q.camera_x, q.camera_y);
    let motion = advection + camera;
    results[idx] = vec4<f32>(motion.x, motion.y, 0.0, 0.0);
}
