// Stateless hash / RNG twin: evaluates the module's integer avalanche hash and
// its unit-float wrapper on-device, mirroring the CPU goldens `hash_u32` and
// `rng_unit` (in `prism_render_architecture::volumetric::reference`).
//
// `hash_u32` is a Wang-style all-wrapping scramble that turns a counter into a
// well-distributed u32; `rng_unit(x)` maps `hash_u32(x)` to `[0, 1)` using the
// top 24 bits so the value lands exactly on the f32 mantissa grid and never
// rounds up to 1.0. Both are pure and deterministic — the fixed-seed
// reproducibility contract of design section 16.
//
// WGSL u32 arithmetic is defined to wrap on overflow, exactly matching the CPU
// golden's `wrapping_mul` / `wrapping_add`, so the bit pattern is reproduced
// exactly. The kernel uses only the portable core-WGSL subset, so it runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Wang-style integer hash; no Unreal Engine source or
// derived code.

const MUL: u32 = 0x27d4eb2du;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One hash query: the input counter/seed. Matches `GpuQuery` in
// `src/hash_rng.rs`.
struct Query {
    x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One hash result: the raw hashed u32 and its unit-float mapping. Matches
// `GpuSample` in `src/hash_rng.rs`.
struct Res {
    hash: u32,
    unit: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

fn hash_u32(x: u32) -> u32 {
    var h = (x ^ 61u) * MUL;
    h ^= x >> 16u;
    h = h + (h << 3u);
    h ^= h >> 4u;
    h = h * MUL;
    h ^= h >> 15u;
    return h;
}

fn unit_from_u32(bits: u32) -> f32 {
    return f32(bits >> 8u) * (1.0 / 16777216.0);
}

@compute @workgroup_size(64)
fn hash_rng_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let h = hash_u32(queries[idx].x);
    results[idx] = Res(h, unit_from_u32(h));
}
