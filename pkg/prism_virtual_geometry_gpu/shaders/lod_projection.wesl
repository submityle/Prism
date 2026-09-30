// Prism virtual-geometry LOD screen-space-error twin: projects each cluster
// level's object-space geometric error to screen pixels at the current view
// distance, mirroring the CPU golden
// `LodProjection::projected_error_pixels` (in
// `prism_render_architecture::virtual_geometry`).
//
// The screen-space-error LOD selector displays the coarsest level whose
// projected error still fits the pixel budget. The projection itself is the
// hot inner term: `max(error, 0) * focal_length_pixels / max(distance, eps)`.
// This kernel is the on-device twin of that closed form - one thread per query
// evaluates the same expression so the selector's projection term is validated
// against the reference rather than merely compiled. Selection policy
// (hysteresis, prefetch, level walk) stays on the CPU golden; only the pure
// projection scalar is mirrored here.
//
// Numerical model: the value is `(m * f) / d` with `m = max(error, 0)`,
// `f = focal_length_pixels`, `d = max(distance, eps)`. `max` is exact and the
// multiply is correctly rounded, but a GPU divide by a non-power-of-two divisor
// is only guaranteed correctly rounded to within one ULP (Metal implements it
// as a refined reciprocal), so the twin matches the CPU golden within one ULP
// in general and bit-for-bit whenever `d` is an exact power of two (the divide
// degenerates to exponent scaling). `eps` is `f32::EPSILON = 2^-23`, written as
// the exact hex float `0x1p-23` so the CPU and GPU clamp thresholds coincide to
// the bit. The kernel uses only the portable core-WGSL subset - no optional
// feature - so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard perspective screen-space-error projection for LOD
// selection; no Unreal Engine source or derived code.

// f32::EPSILON = 2^-23, exactly representable.
const EPSILON: f32 = 0x1p-23;

struct Params {
    // Number of queries in `inputs` / outputs in `pixels`.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One projection query. `12`-byte stride matching the `bytemuck` upload struct.
struct LodProjectionQuery {
    // Object-space geometric deviation bound of the level, in world units.
    geometric_error: f32,
    // View-space distance to the cluster, in world units.
    view_distance: f32,
    // Perspective factor: half viewport height / tan(0.5 * vertical_fov).
    focal_length_pixels: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<LodProjectionQuery>;
@group(0) @binding(2) var<storage, read_write> pixels: array<f32>;

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }

    let q = inputs[i];
    // distance = view_distance.max(f32::EPSILON) - mirrors the CPU clamp.
    let distance = max(q.view_distance, EPSILON);
    // error.max(0) * focal / distance: one multiply then one divide, fma-immune.
    pixels[i] = max(q.geometric_error, 0.0) * q.focal_length_pixels / distance;
}
