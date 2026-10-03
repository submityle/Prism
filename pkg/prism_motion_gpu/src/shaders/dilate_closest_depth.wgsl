// Closest-depth motion-vector dilation (deterministic neighborhood gather).
//
// One invocation produces exactly one output pixel. The center pixel seeds the
// search; the kernel then scans the clamped (2*radius+1) square neighborhood in
// row-major order (ny outer, nx inner) and replaces the incumbent only when a
// neighbor is *strictly* nearer the camera under the active depth ordering.
// Equal depths therefore keep the incumbent, and because the scan order is
// fixed the chosen winner is fully deterministic.
//
// The output velocity is always a whole-vector copy of some input velocity, and
// the only derived quantity is the depth comparison, so no floating-point
// rounding ever enters the written value: the device output equals the CPU
// golden `dilate_closest_depth` bit-for-bit (exact equality, not tolerance).
//
// `order_flag == 0` is SmallerIsCloser (classic forward-Z, 0 = near); any other
// value is LargerIsCloser (reversed-Z, 1 = near).
//
// Provenance: standard temporal-AA closest-depth dilation. No Unreal Engine
// source or derived code.

struct Params {
    // Field width in pixels.
    width: u32,
    // Field height in pixels.
    height: u32,
    // Square neighborhood radius in pixels (0 copies the input).
    radius: u32,
    // 0 => SmallerIsCloser; otherwise LargerIsCloser.
    order_flag: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Row-major input velocity field (element (x, y) at y * width + x).
@group(0) @binding(1) var<storage, read> velocity: array<vec2<f32>>;
// Row-major input depth field, same dimensions as `velocity`.
@group(0) @binding(2) var<storage, read> depth: array<f32>;
// Row-major output velocity field.
@group(0) @binding(3) var<storage, read_write> dst: array<vec2<f32>>;

// Returns true when `candidate` is strictly nearer the camera than `reference`.
fn is_closer(candidate: f32, reference: f32) -> bool {
    return select(candidate > reference, candidate < reference, params.order_flag == 0u);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    let width = params.width;
    let center = y * width + x;

    // Seed with the center pixel so a fully tied neighborhood is a no-op copy.
    var best_depth = depth[center];
    var best_velocity = velocity[center];

    // Clamped square bounds with saturating low edges (unsigned-safe).
    let x_lo = select(0u, x - params.radius, x >= params.radius);
    let x_hi = min(x + params.radius, width - 1u);
    let y_lo = select(0u, y - params.radius, y >= params.radius);
    let y_hi = min(y + params.radius, params.height - 1u);

    for (var ny = y_lo; ny <= y_hi; ny = ny + 1u) {
        for (var nx = x_lo; nx <= x_hi; nx = nx + 1u) {
            let idx = ny * width + nx;
            let candidate_depth = depth[idx];
            if (is_closer(candidate_depth, best_depth)) {
                best_depth = candidate_depth;
                best_velocity = velocity[idx];
            }
        }
    }

    dst[center] = best_velocity;
}
