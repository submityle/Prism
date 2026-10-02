// Surfel-atlas decode (gather) producer kernel: the on-device twin of the CPU
// golden `SurfelAtlas::texel_to_dir` in
// `prism_render_shading::gi::surface_cache::atlas`.
//
// A Lumen-style surface cache stores each surfel's directional outgoing
// radiance on an octahedral map inside the surfel's own atlas tile. When the
// cache is *sampled* during shading the reverse mapping is needed: given a
// surfel id and a global atlas texel, which unit world direction does that
// texel centre represent? This kernel answers that per request — resolve the
// surfel's tile, reject texels outside it, then octahedrally decode the
// tile-local texel centre back to a direction. It is the exact inverse of the
// allocation kernel `surfel_alloc.wesl`.
//
// One thread per request. Only `+ - * /`, `abs`, `sqrt`, `floor`, `min`,
// `select`, `normalize` and comparisons are used — the portable core-WGSL
// subset — so it runs unmodified on Metal, Vulkan and DX12. The octahedral
// decode is the same closed form the golden `oct_to_dir` evaluates, so the CPU
// mirror reproduces it bit-for-bit (the final `normalize` is the one true-
// machine op whose last bit may differ on device, which is acceptable).
//
// Provenance: standard octahedral surfel-atlas addressing; no Unreal Engine
// source or derived code.

// Dispatch parameters. 16-byte `repr(C)` matching `GpuSurfelDecodeParams`.
struct DecodeParams {
    count: u32,
    tiles_per_row: u32,
    tile_rows: u32,
    tile_resolution: u32,
}

// One decode request. 12-byte `repr(C)` matching `GpuSurfelDecodeRequest`.
struct DecodeRequest {
    surfel_id: u32,
    texel_x: u32,
    texel_y: u32,
}

// One decoded direction plus the valid flag. 16-byte `repr(C)` matching
// `GpuSurfelDecodeResult`.
struct DecodeResult {
    dir_x: f32,
    dir_y: f32,
    dir_z: f32,
    flags: u32,
}

@group(0) @binding(0) var<uniform> params: DecodeParams;
@group(0) @binding(1) var<storage, read> requests: array<DecodeRequest>;
@group(0) @binding(2) var<storage, read_write> results: array<DecodeResult>;

// Valid-result bit, matching `SURFEL_DECODE_FLAG_VALID`.
const DECODE_FLAG_VALID: u32 = 1u;

// Branchless sign mapping zero to `+1` (GLSL `signNotZero`); keeps the
// octahedral fold continuous across the seam. Mirrors `sign_not_zero` in the
// golden.
fn sign_not_zero(value: f32) -> f32 {
    return select(-1.0, 1.0, value >= 0.0);
}

// Decode an octahedral `UV` in `[0, 1]^2` to a unit direction, the closed-form
// inverse the golden `oct_to_dir` evaluates.
fn oct_to_dir(uv: vec2<f32>) -> vec3<f32> {
    let e = uv * 2.0 - vec2<f32>(1.0, 1.0);
    let z = 1.0 - abs(e.x) - abs(e.y);
    var x = e.x;
    var y = e.y;
    if (z < 0.0) {
        x = (1.0 - abs(e.y)) * sign_not_zero(e.x);
        y = (1.0 - abs(e.x)) * sign_not_zero(e.y);
    }
    return normalize(vec3<f32>(x, y, z));
}

// Unit direction stored at a tile-local texel centre, mirroring the golden
// `SurfelAtlas::local_texel_to_dir`.
fn local_texel_to_dir(lx: u32, ly: u32, resolution: u32) -> vec3<f32> {
    let res = f32(resolution);
    let max_idx = resolution - 1u;
    let cx = (f32(min(lx, max_idx)) + 0.5) / res;
    let cy = (f32(min(ly, max_idx)) + 0.5) / res;
    return oct_to_dir(vec2<f32>(cx, cy));
}

@compute @workgroup_size(64)
fn surfel_decode_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let request = requests[idx];

    var result: DecodeResult;
    result.dir_x = 0.0;
    result.dir_y = 0.0;
    result.dir_z = 0.0;
    result.flags = 0u;

    let capacity = params.tiles_per_row * params.tile_rows;
    if (request.surfel_id >= capacity) {
        results[idx] = result;
        return;
    }

    let res = params.tile_resolution;
    let col = request.surfel_id % params.tiles_per_row;
    let row = request.surfel_id / params.tiles_per_row;
    let origin_x = col * res;
    let origin_y = row * res;

    if (request.texel_x < origin_x || request.texel_y < origin_y) {
        results[idx] = result;
        return;
    }
    let lx = request.texel_x - origin_x;
    let ly = request.texel_y - origin_y;
    if (lx >= res || ly >= res) {
        results[idx] = result;
        return;
    }

    let dir = local_texel_to_dir(lx, ly, res);
    result.dir_x = dir.x;
    result.dir_y = dir.y;
    result.dir_z = dir.z;
    result.flags = DECODE_FLAG_VALID;
    results[idx] = result;
}
