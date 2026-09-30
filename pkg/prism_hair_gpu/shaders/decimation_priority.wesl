// Importance-weighted decimation ranking-key twin: evaluates on-device the same
// per-strand priority the CPU golden
// `prism_render_architecture::hair::decimation::decimation_priority` computes,
// one thread per strand.
//
// Continuous density LOD keeps fewer render strands as a groom recedes, and to
// stay pop-free the kept set at a low count must nest inside the kept set at a
// higher count. The reference builds that nested order by ranking every strand
// by a scalar priority = importance + jitter * (hash - 0.5): the importance
// biases long/curly/authored strands to survive to the lowest counts (the
// density-LOD bias UE5 Groom and HairWorks apply), while a per-strand hash
// jitter in [-0.5, 0.5) * jitter decorrelates the ranking spatially so heavy
// thinning removes strands evenly instead of carving bald patches. The host
// then sorts these priorities (descending, ties by ascending index) into the
// decimation order; this kernel is only the per-strand key evaluation, the part
// that maps cleanly to one GPU thread per strand.
//
// Determinism is the whole point, so the jitter draws from the same hand-written
// `splitmix64` integer hash the golden uses, reproduced here bit-for-bit with a
// `vec2<u32>` (lo, hi) emulation because baseline WGSL has no native 64-bit
// integer, at the same fixed `DECIMATION_JITTER_KEY` sub-key. The hash is
// integer-exact; only the final `importance + jitter * h` combine is a single
// multiply-add a GPU may fuse, so CPU and GPU agree to within the documented fma
// tolerance rather than bit-for-bit. At `jitter == 0` the key collapses to the
// importance itself and the twin is exact.
//
// Buffer layout: `params` carries the jitter amplitude and the strand count;
// `seeds[i]` and `importances[i]` are strand `i`'s per-strand hash seed and
// already-folded importance; `priorities` receives one f32 ranking key per
// strand at index `i`.
//
// Provenance: standard importance-weighted stochastic LOD decimation ranking
// plus wgpu compute dispatch; no Unreal Engine source or derived code.

// Hash sub-key salt for the per-strand decimation jitter, bit-for-bit the CPU
// `DECIMATION_JITTER_KEY` so the hash draw matches the reference exactly.
const DECIMATION_JITTER_KEY: u32 = 0x00DEC1AAu;

struct Params {
    // Peak decorrelation amplitude added to importance (the CPU `jitter`).
    jitter: f32,
    // Number of strands, bounding the dispatch.
    strand_count: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per-strand hash seed (mirrors the CPU `seeds` slice).
@group(0) @binding(1) var<storage, read> seeds: array<u32>;
// Per-strand already-folded importance in [0, 1] (mirrors `importances`).
@group(0) @binding(2) var<storage, read> importances: array<f32>;
// Per-strand ranking priority the host then sorts into the decimation order.
@group(0) @binding(3) var<storage, read_write> priorities: array<f32>;

// ---------------------------------------------------------------------------
// 64-bit integer emulation for the `splitmix64` hash (baseline WGSL is 32-bit).
// A `u64` is a `vec2<u32>` = (lo, hi). Only the operations the finalizer needs
// are provided: add, xor, right shift, and the low-64-bit product.
// ---------------------------------------------------------------------------

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    var carry = 0u;
    if lo < a.x {
        carry = 1u;
    }
    let hi = a.y + b.y + carry;
    return vec2<u32>(lo, hi);
}

fn u64_xor(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x ^ b.x, a.y ^ b.y);
}

// Right shift by `n` in `0..64`. The branches keep every 32-bit shift amount
// strictly below the 32-bit width so the result is well defined.
fn u64_shr(a: vec2<u32>, n: u32) -> vec2<u32> {
    if n == 0u {
        return a;
    }
    if n < 32u {
        let lo = (a.x >> n) | (a.y << (32u - n));
        let hi = a.y >> n;
        return vec2<u32>(lo, hi);
    }
    return vec2<u32>(a.y >> (n - 32u), 0u);
}

// Full 64-bit product of two 32-bit values, via 16-bit-half schoolbook mul.
fn mul_u32_wide(x: u32, y: u32) -> vec2<u32> {
    let xl = x & 0xFFFFu;
    let xh = x >> 16u;
    let yl = y & 0xFFFFu;
    let yh = y >> 16u;

    let ll = xl * yl;
    let lh = xl * yh;
    let hl = xh * yl;
    let hh = xh * yh;

    let carry0 = (ll >> 16u) + (lh & 0xFFFFu) + (hl & 0xFFFFu);
    let lo = (ll & 0xFFFFu) | ((carry0 & 0xFFFFu) << 16u);
    let hi = (carry0 >> 16u) + (lh >> 16u) + (hl >> 16u) + hh;
    return vec2<u32>(lo, hi);
}

// Low 64 bits of the product of two 64-bit values (wrapping, mirrors Rust
// `wrapping_mul`).
fn u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let p00 = mul_u32_wide(a.x, b.x);
    let mid = a.x * b.y + a.y * b.x;
    return vec2<u32>(p00.x, p00.y + mid);
}

// Full 64-bit `splitmix64` finalizer over the packed `(seed, key)` word,
// bit-for-bit the CPU `hash_bits`.
fn hash_bits(seed: u32, key: u32) -> vec2<u32> {
    // z = (seed << 32 | key) + 0x9E3779B97F4A7C15
    var z = u64_add(vec2<u32>(key, seed), vec2<u32>(0x7F4A7C15u, 0x9E3779B9u));
    // z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = u64_mul(u64_xor(z, u64_shr(z, 30u)), vec2<u32>(0x1CE4E5B9u, 0xBF58476Du));
    // z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = u64_mul(u64_xor(z, u64_shr(z, 27u)), vec2<u32>(0x133111EBu, 0x94D049BBu));
    // z = z ^ (z >> 31)
    return u64_xor(z, u64_shr(z, 31u));
}

// Maps `(seed, key)` to a unit float in `[0, 1)` from the top 24 bits (mirrors
// the CPU `hash_to_unit`; `>> 40` keeps the high 24 bits in the low word).
fn hash_to_unit(seed: u32, key: u32) -> f32 {
    let bits = u64_shr(hash_bits(seed, key), 40u);
    return f32(bits.x) / 16777216.0;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.strand_count {
        return;
    }
    // Symmetric per-strand jitter in [-0.5, 0.5), added to the importance so the
    // host-side descending sort spreads the thinning while preserving the coarse
    // importance ordering.
    let h = hash_to_unit(seeds[gid.x], DECIMATION_JITTER_KEY) - 0.5;
    priorities[gid.x] = importances[gid.x] + params.jitter * h;
}
