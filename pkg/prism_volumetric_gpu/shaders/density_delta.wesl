// Carve-brush density-delta twin: evaluates on-device the signed density delta
// a spherical negative-density brush applies at a world position, mirroring the
// CPU golden `density_delta` (in
// `prism_render_architecture::volumetric::coupling`).
//
// Two-way scene coupling (design section 9d) lets aircraft and projectiles
// punch cavities and wakes into the cloud density field. The brush is a purely
// negative source: the delta is `-strength` at the brush centre and eases
// smoothly to `0` at (and beyond) the brush radius via a `smoothstep` falloff,
// so the result lies in `-1..=0`. Applied through the saturating `apply_carve`
// choke point it can only ever remove density, never manufacturing a negative
// final density or transmittance. Out-of-range radii (`<= 0`) are floored to
// `EPS` and strengths are saturated, so the kernel never divides by zero. The
// CPU golden `density_delta` owns that math; this kernel is the on-device twin
// that runs one thread per query and reproduces the same value.
//
// The distance uses `sqrt` (mirroring `Vec3::distance` == `length_squared().sqrt()`)
// and the falloff mirrors the reference's hand-rolled `math::smoothstep`
// bit-for-bit (the collapsed-edge EPS guard and the `t*t*(3-2t)` Hermite
// polynomial), NOT the device-native `smoothstep`, so the twin stays bit-close
// to the reference and the parity test can assert a tight tolerance.
//
// It uses only the portable core-WGSL subset - sqrt, clamp and multiply/add -
// so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard spherical falloff density carving; no Unreal Engine
// source or derived code.

const EPS: f32 = 1e-6;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One carve query: the sampled world position plus the brush parameters
// (centre, radius, strength). Matches `GpuQuery` in `src/density_delta.rs`;
// two 16-byte rows keep the struct 16-byte aligned.
struct Query {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    radius: f32,
    center_x: f32,
    center_y: f32,
    center_z: f32,
    strength: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

fn saturate_s(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Hermite smoothstep between `edge0` and `edge1`, clamped to `[0, 1]`,
// mirroring `math::smoothstep`: collapsed edges degrade to a hard step at
// `edge0` instead of dividing by zero.
fn smoothstep_c(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if (abs(span) < EPS) {
        if (x < edge0) {
            return 0.0;
        }
        return 1.0;
    }
    let t = saturate_s((x - edge0) / span);
    return t * t * (3.0 - 2.0 * t);
}

// Euclidean distance, mirroring `Vec3::distance` == `length_squared().sqrt()`.
fn vec3_distance(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) -> f32 {
    let dx = ax - bx;
    let dy = ay - by;
    let dz = az - bz;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

// Signed density delta the brush applies at the sampled position, mirroring
// `density_delta`. Result lies in `-1..=0`.
fn density_delta(q: Query) -> f32 {
    let radius = max(q.radius, EPS);
    let strength = saturate_s(q.strength);
    let dist = vec3_distance(q.pos_x, q.pos_y, q.pos_z, q.center_x, q.center_y, q.center_z);
    let falloff = 1.0 - smoothstep_c(0.0, radius, dist);
    return -strength * falloff;
}

@compute @workgroup_size(64)
fn density_delta_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    results[idx] = density_delta(queries[idx]);
}
