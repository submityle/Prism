// McGuire NeighborMax: expand a TileMax field to its 3x3 neighborhood maximum.
//
// One invocation produces exactly one output tile holding the largest-magnitude
// velocity among itself and its (clamped) eight neighbors. Neighbors are scanned
// in the identical row-major order as the CPU golden (ny outer, nx inner) and
// folded with "keep the larger magnitude, ties keep the incumbent" (strict `>`),
// seeded at the zero vector, so the chosen winner is fully deterministic.
//
// The written value is always a whole-vector copy of some input tile velocity;
// the only derived quantity is the squared length `x*x + y*y` in the golden's
// operation order, so the device output equals the golden `neighbor_max`
// bit-for-bit (exact equality, not tolerance).
//
// Provenance: Morgan McGuire et al., "A Reconstruction Filter for Plausible
// Motion Blur" (I3D 2012), NeighborMax stage. No Unreal Engine source or
// derived code.

struct Params {
    // Number of tile columns.
    tiles_x: u32,
    // Number of tile rows.
    tiles_y: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Row-major coarse per-tile input field.
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
// Row-major coarse per-tile output field.
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;

// Squared length in the golden's operation order (no fused multiply-add).
fn len_sq(v: vec2<f32>) -> f32 {
    return v.x * v.x + v.y * v.y;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tx = gid.x;
    let ty = gid.y;
    if (tx >= params.tiles_x || ty >= params.tiles_y) {
        return;
    }

    let tx_lo = select(0u, tx - 1u, tx >= 1u);
    let tx_hi = min(tx + 1u, params.tiles_x - 1u);
    let ty_lo = select(0u, ty - 1u, ty >= 1u);
    let ty_hi = min(ty + 1u, params.tiles_y - 1u);

    var acc = vec2<f32>(0.0, 0.0);
    for (var ny = ty_lo; ny <= ty_hi; ny = ny + 1u) {
        for (var nx = tx_lo; nx <= tx_hi; nx = nx + 1u) {
            let v = src[ny * params.tiles_x + nx];
            if (len_sq(v) > len_sq(acc)) {
                acc = v;
            }
        }
    }

    dst[ty * params.tiles_x + tx] = acc;
}
