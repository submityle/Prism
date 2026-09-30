// Continuous-LOD screen-door dither twin: evaluates on-device the same
// per-strand keep/drop decision the CPU golden
// `prism_render_architecture::hair::transition::strand_survives_dither` makes,
// one thread per strand.
//
// When a groom cross-fades between two LOD tiers, popping is hidden by
// dissolving the finer representation strand-by-strand rather than switching all
// at once: each strand hashes to a stable value in [0, 1) and keeps drawing as
// the finer tier while that hash is at least the cross-fade `blend`. At
// blend == 0 every strand survives (fully finer); as blend rises toward 1 a
// deterministic, growing subset drops out so the kept fraction is (1 - blend);
// at blend == 1 no strand survives because the hash is always < 1. The `seed`
// scopes the pattern per groom so different grooms dissolve independently.
//
// Determinism is the whole point, so the only entropy source is the same
// hand-written `splitmix64` integer hash the golden uses, reproduced here
// bit-for-bit with a `vec2<u32>` (lo, hi) emulation because baseline WGSL has no
// native 64-bit integer. The comparison is a pure integer-derived float against
// `blend` with no transcendental and no fused multiply-add, so the device path
// is bit-identical to the CPU reference and the parity test asserts exact
// equality of the survive flags rather than a tolerance.
//
// Buffer layout: `params` carries the seed, strand count and cross-fade blend;
// `survive` receives one u32 per strand (1 = keeps drawing as the finer tier,
// 0 = dropped) at index `strand_index`, which is simply the invocation id.
//
// Provenance: standard stable-hash screen-door / stochastic-LOD dither plus
// wgpu compute dispatch; no Unreal Engine source or derived code.

struct Params {
    // Per-groom hash scope, matching the CPU `seed`.
    seed: u32,
    // Number of strands, bounding the dispatch.
    strand_count: u32,
    // Cross-fade position in [0, 1]; out-of-range values saturate.
    blend: f32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per-strand survive flag: 1 keeps drawing as the finer tier, 0 is dropped.
@group(0) @binding(1) var<storage, read_write> survive: array<u32>;

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
    // Saturate out-of-range blend, then survive while the strand's stable hash
    // is at or above the cross-fade position (kept fraction = 1 - blend).
    let b = clamp(params.blend, 0.0, 1.0);
    survive[gid.x] = select(0u, 1u, hash_to_unit(params.seed, gid.x) >= b);
}
