// Surfel-allocation (atlas-addressing) producer kernel: the on-device twin of
// the CPU golden `SurfelAtlas` addressing in
// `prism_render_shading::gi::surface_cache::atlas`.
//
// A Lumen-style surface cache stores every surfel's directional outgoing
// radiance in a shared octahedral atlas. Before any radiance can be written the
// producer must resolve, per surfel, *where* in the atlas its sample lands: the
// surfel's square tile and, inside that tile, the octahedral texel for the
// sample direction. This kernel runs one thread per allocation request and
// writes that resolved global atlas slot, mirroring the CPU golden exactly:
//
//   capacity = tiles_per_row * tile_rows
//   invalid                              when surfel_id >= capacity
//   col = surfel_id % tiles_per_row
//   row = surfel_id / tiles_per_row
//   origin = (col, row) * tile_resolution
//   texel  = origin + uv_to_local(dir_to_oct(dir), tile_resolution)
//
// The addressing is integer-exact apart from the octahedral `dir_to_oct` fold,
// which is a closed-form sequence of abs / add / reciprocal / multiply / floor
// in the portable core-WGSL subset (no transcendental, no optional device
// feature), so it runs unmodified on Metal, Vulkan and DX12 and a CPU mirror
// reproduces the same texel bit-for-bit.
//
// Provenance: standard octahedral surfel-atlas addressing; no Unreal Engine
// source or derived code.

// Dispatch parameters. 16-byte `repr(C)` matching `GpuSurfelAllocParams` in
// `src/gi/surface_cache/gpu/abi.rs`.
struct Params {
    count: u32,
    tiles_per_row: u32,
    tile_rows: u32,
    tile_resolution: u32,
}

// One allocation request: the surfel id plus the sample direction as three
// scalars (kept scalar, not `vec3`, so the std430 stride stays 16 bytes with a
// 4-byte alignment). Matches `GpuSurfelAllocRequest`.
struct Request {
    surfel_id: u32,
    dir_x: f32,
    dir_y: f32,
    dir_z: f32,
}

// Resolved atlas slot for a request. 20-byte `repr(C)` matching
// `GpuSurfelAllocSlot`; `flags` bit 0 is the valid flag.
struct Slot {
    texel_x: u32,
    texel_y: u32,
    tile_col: u32,
    tile_row: u32,
    flags: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> requests: array<Request>;
@group(0) @binding(2) var<storage, read_write> slots: array<Slot>;

// Smallest positive normal `f32`; guards the octahedral L1 reciprocal against a
// zero-length direction (which maps to the +z pole at the square centre).
const OCT_MIN_POSITIVE: f32 = 1.17549435e-38;

// Branchless sign that maps zero to +1 (the GLSL `signNotZero` helper); keeps
// the octahedral fold continuous across the seam.
fn sign_not_zero(value: f32) -> f32 {
    if (value >= 0.0) {
        return 1.0;
    }
    return -1.0;
}

// Encodes a direction into octahedral UV in `[0, 1]^2`, mirroring the CPU
// golden `dir_to_oct` op-for-op.
fn dir_to_oct(dir: vec3<f32>) -> vec2<f32> {
    let l1 = abs(dir.x) + abs(dir.y) + abs(dir.z);
    if (l1 <= OCT_MIN_POSITIVE) {
        return vec2<f32>(0.5, 0.5);
    }
    let inv = 1.0 / l1;
    var px = dir.x * inv;
    var py = dir.y * inv;
    if (dir.z < 0.0) {
        let fx = (1.0 - abs(py)) * sign_not_zero(px);
        let fy = (1.0 - abs(px)) * sign_not_zero(py);
        px = fx;
        py = fy;
    }
    return vec2<f32>(px * 0.5 + 0.5, py * 0.5 + 0.5);
}

// Clamp a (floored, non-negative) texel index into `[0, max_idx]`, mirroring the
// CPU golden `clamp_index`. `!(value > 0.0)` collapses both non-positive and
// `NaN` inputs to zero.
fn clamp_index(value: f32, max_idx: u32) -> u32 {
    if (!(value > 0.0)) {
        return 0u;
    }
    return min(u32(value), max_idx);
}

// Nearest tile-local texel for an octahedral UV, mirroring the CPU golden
// `uv_to_local`.
fn uv_to_local(uv: vec2<f32>, res: u32) -> vec2<u32> {
    let res_f = f32(res);
    let fx = floor(clamp(uv.x, 0.0, 1.0) * res_f);
    let fy = floor(clamp(uv.y, 0.0, 1.0) * res_f);
    let max_idx = res - 1u;
    return vec2<u32>(clamp_index(fx, max_idx), clamp_index(fy, max_idx));
}

@compute @workgroup_size(64)
fn surfel_alloc_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let req = requests[idx];

    var slot: Slot;
    slot.texel_x = 0u;
    slot.texel_y = 0u;
    slot.tile_col = 0u;
    slot.tile_row = 0u;
    slot.flags = 0u;

    let capacity = params.tiles_per_row * params.tile_rows;
    if (req.surfel_id >= capacity) {
        slots[idx] = slot;
        return;
    }

    let col = req.surfel_id % params.tiles_per_row;
    let row = req.surfel_id / params.tiles_per_row;
    let origin_x = col * params.tile_resolution;
    let origin_y = row * params.tile_resolution;

    let dir = vec3<f32>(req.dir_x, req.dir_y, req.dir_z);
    let uv = dir_to_oct(dir);
    let local = uv_to_local(uv, params.tile_resolution);

    slot.texel_x = origin_x + local.x;
    slot.texel_y = origin_y + local.y;
    slot.tile_col = col;
    slot.tile_row = row;
    slot.flags = 1u;
    slots[idx] = slot;
}
