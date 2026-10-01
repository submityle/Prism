// Raw integer-hash twin for the film-grain pass: evaluates the CPU gold
// standard `hash_u32` on-device, one thread per seed, mirroring
// `prism_render_architecture::particle::film_grain::hash_u32`.
//
// `hash_u32` is an xor-shift and odd-constant-multiply avalanche finalizer with
// strong bit diffusion. WGSL u32 arithmetic is defined to wrap on overflow,
// exactly matching the CPU golden's `wrapping_mul`, so this kernel reproduces
// the hash bit for bit (not merely close); the parity test asserts bit-exact
// equality. The kernel is the portable core-WGSL integer subset, so it runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: hand-rolled integer avalanche hash plus wgpu compute dispatch; no
// Unreal Engine source or derived code.

// Odd multiply constants of the `hash_u32` finalizer (`MIX_A`, `MIX_B`).
const MIX_A: u32 = 0x7feb352du;
const MIX_B: u32 = 0x846ca68bu;

// Uniform parameters: the seed count plus three pad words. Matches `HashParams`
// in `src/film_grain.rs`.
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> seeds: array<u32>;
@group(0) @binding(2) var<storage, read_write> hashes: array<u32>;

// Pure u32 avalanche hash: xor-shift and odd-constant-multiply finalizer. u32
// multiply wraps in WGSL, matching the CPU golden `wrapping_mul`, so the output
// bit pattern is reproduced exactly.
fn hash_u32(seed: u32) -> u32 {
    var h = seed;
    h ^= h >> 16u;
    h = h * MIX_A;
    h ^= h >> 15u;
    h = h * MIX_B;
    h ^= h >> 16u;
    return h;
}

@compute @workgroup_size(64)
fn film_grain_hash_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    hashes[idx] = hash_u32(seeds[idx]);
}
