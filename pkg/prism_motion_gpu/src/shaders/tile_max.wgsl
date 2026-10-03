// McGuire TileMax: full-resolution velocity field -> per-tile maximum-magnitude.
//
// One invocation produces exactly one output tile. It scans its tile's pixels
// in the identical row-major order as the CPU golden (y outer, x inner),
// folding with "keep the larger magnitude, ties keep the incumbent": a pixel
// replaces the accumulator only when its squared length is *strictly* greater.
// The accumulator seeds at the zero vector, so an all-zero tile stays zero and
// the first-seen maximum wins any tie.
//
// The written value is always a whole-vector copy of some input velocity; the
// only derived quantity is the squared length `x*x + y*y`, computed in the same
// operation order as the golden `Vec2::length_squared` (two muls + one add, no
// fused multiply-add), so the device output equals the golden `tile_max`
// bit-for-bit (exact equality, not tolerance).
//
// Partial edge tiles clamp their high bound to the field dimensions, matching
// the golden `div_ceil` tiling.
//
// Provenance: Morgan McGuire et al., "A Reconstruction Filter for Plausible
// Motion Blur" (I3D 2012), TileMax stage. No Unreal Engine source or derived
// code.

struct Params {
    // Full-resolution field width in pixels.
    width: u32,
    // Full-resolution field height in pixels.
    height: u32,
    // Tile edge length in pixels.
    tile_size: u32,
    // Number of tile columns (width.div_ceil(tile_size)).
    tiles_x: u32,
    // Number of tile rows (height.div_ceil(tile_size)).
    tiles_y: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Row-major full-resolution velocity field.
@group(0) @binding(1) var<storage, read> velocity: array<vec2<f32>>;
// Row-major coarse per-tile output field.
@group(0) @binding(2) var<storage, read_write> tiles: array<vec2<f32>>;

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

    let x_lo = tx * params.tile_size;
    let y_lo = ty * params.tile_size;
    let x_hi = min(x_lo + params.tile_size, params.width);
    let y_hi = min(y_lo + params.tile_size, params.height);

    var acc = vec2<f32>(0.0, 0.0);
    for (var y = y_lo; y < y_hi; y = y + 1u) {
        for (var x = x_lo; x < x_hi; x = x + 1u) {
            let v = velocity[y * params.width + x];
            if (len_sq(v) > len_sq(acc)) {
                acc = v;
            }
        }
    }

    tiles[ty * params.tiles_x + tx] = acc;
}
