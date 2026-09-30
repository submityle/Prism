// Worley (cellular) noise twin: computes on-device the [0, 1]-normalised
// distance to the nearest jittered feature point, mirroring the CPU golden
// `worley_3d` (in `prism_render_architecture::volumetric::noise`).
//
// The cloud density field's billowy clump form and high-frequency detail
// erosion come from Worley noise (design section 4, Nubis-style modelling):
// each lattice cell holds one feature point jittered inside its unit cube by a
// deterministic integer hash; the 3x3x3 neighbourhood around the sample cell is
// searched for the true nearest point, and the resulting distance is saturated
// into [0, 1]. The CPU golden `worley_3d` owns that math; this kernel is the
// on-device twin that runs one thread per sample point and returns the same
// value.
//
// The feature-point selection is pure integer work (FNV-1a byte mixing plus an
// xorshift-multiply avalanche, then a top-24-bit unit map) and WGSL unsigned
// integers wrap on overflow exactly like Rust's `wrapping_mul`/`^`/`>>`, so the
// jittered points are bit-identical to the reference. Only the final distance
// (a `sqrt` of a sum of squares) can differ, and only by a legal multiply-add
// contraction. It uses only the portable core-WGSL subset - no exp, pow or
// optional feature - so this twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Worley cellular noise (Worley 1996); no Unreal Engine
// source or derived code.

// FNV-1a 32-bit offset basis and prime, the standard constants.
const FNV_OFFSET_BASIS: u32 = 0x811c9dc5u;
const FNV_PRIME: u32 = 0x01000193u;

// Decorrelation seeds for the feature point's y and z offsets, matching the
// `WORLEY_SEED_Y` / `WORLEY_SEED_Z` constants in the reference.
const WORLEY_SEED_Y: u32 = 0x68e31da4u;
const WORLEY_SEED_Z: u32 = 0xb5279a45u;

struct Params {
    // Number of valid entries in `queries` / outputs in `results`.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One noise query. `16`-byte stride matching the `bytemuck` upload struct.
struct Query {
    x: f32,
    y: f32,
    z: f32,
    seed: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

// Clamps `x` into `[0, 1]`, mirroring `math::saturate`.
fn saturate_v(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Mixes one 32-bit word into the running FNV-1a hash, byte by byte, mirroring
// `fnv_word`.
fn fnv_word(hash: u32, word: u32) -> u32 {
    var h = hash;
    var shift = 0u;
    while (shift < 32u) {
        h = h ^ ((word >> shift) & 0xFFu);
        h = h * FNV_PRIME;
        shift = shift + 8u;
    }
    return h;
}

// Deterministic integer hash of an integer lattice point and a seed, mirroring
// `hash_cell` (FNV-1a mixing plus an xorshift-multiply avalanche).
fn hash_cell(ix: i32, iy: i32, iz: i32, seed: u32) -> u32 {
    var h = FNV_OFFSET_BASIS;
    h = fnv_word(h, seed);
    h = fnv_word(h, bitcast<u32>(ix));
    h = fnv_word(h, bitcast<u32>(iy));
    h = fnv_word(h, bitcast<u32>(iz));
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    h = h * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return h;
}

// Maps a 32-bit hash to [0, 1) using its top twenty-four bits, mirroring
// `hash_to_unit`. 2^24 == 16777216.
fn hash_to_unit(h: u32) -> f32 {
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

// Distance-squared to the nearest jittered feature point in the 3x3x3
// neighbourhood, mirroring the search loop in `worley_3d`.
fn worley_min_sq(px: f32, py: f32, pz: f32, seed: u32) -> f32 {
    let xi = floor(px);
    let yi = floor(py);
    let zi = floor(pz);
    let x0 = i32(xi);
    let y0 = i32(yi);
    let z0 = i32(zi);
    let fx = px - xi;
    let fy = py - yi;
    let fz = pz - zi;
    var min_sq = 3.4028235e38;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let cx = x0 + dx;
                let cy = y0 + dy;
                let cz = z0 + dz;
                let ox = hash_to_unit(hash_cell(cx, cy, cz, seed));
                let oy = hash_to_unit(hash_cell(cx, cy, cz, seed ^ WORLEY_SEED_Y));
                let oz = hash_to_unit(hash_cell(cx, cy, cz, seed ^ WORLEY_SEED_Z));
                let fpx = f32(dx) + ox - fx;
                let fpy = f32(dy) + oy - fy;
                let fpz = f32(dz) + oz - fz;
                let d2 = fpx * fpx + fpy * fpy + fpz * fpz;
                min_sq = min(min_sq, d2);
            }
        }
    }
    return min_sq;
}

@compute @workgroup_size(64)
fn worley(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = saturate_v(sqrt(worley_min_sq(q.x, q.y, q.z, q.seed)));
}
