//! GPU per-strand LOD cross-fade dither mask for hair.
//!
//! Compute twin of the CPU golden reference
//! `prism_render_architecture::hair::transition::strand_survives_dither`. The
//! LOD ladder snaps a groom to one tier at a hard coverage threshold; snapping
//! the whole groom at once *pops* (strands appear/vanish in one frame). Like
//! `UE5` Groom and AMD `TressFX` continuous-LOD, the fix is a deterministic
//! per-strand screen-door dither: across the cross-fade band a stable
//! `(1 - blend)` fraction of the finer tier's strands is kept and the rest are
//! dropped, so the switch dissolves imperceptibly instead of flickering (design
//! §4/§11 "无 LOD pop").
//!
//! This kernel is the per-strand half of that resolve, run on-device during
//! culling/visibility: one invocation owns one render strand and writes a keep
//! mask (`1` = draw the finer tier for this strand, `0` = drop it and let the
//! coarser proxy cover it). The per-group cross-fade band → `blend` scalar
//! decision (`resolve_hair_lod_transition`) stays a tiny host computation and is
//! fed in through `params.blend`; only the fan-out over thousands of strands
//! belongs on the GPU.
//!
//! The survival test reuses the same `splitmix64`-style `hash_to_unit` as the
//! guide interpolator (`hair_interp.wesl`) so the two stages agree on per-strand
//! identity, and it is bit-for-bit the CPU golden: `hash_to_unit(seed, index) >=
//! clamp(blend, 0, 1)`. At `blend == 0` every strand survives; at `blend == 1`
//! none do (the unit hash is always `< 1`). The 64-bit `splitmix64` finalizer is
//! emulated with a `vec2<u32>` because baseline WGSL is 32-bit, identical to the
//! interpolation kernel. The sandbox has no GPU, so this shader is validated by
//! naga compilation only (see `shader_tests`).

/// Cross-fade dither parameters. `blend` runs `0..=1` (fully finer tier → fully
/// coarser proxy); `seed` salts the per-strand hash so distinct grooms dissolve
/// on independent patterns; `strand_count` bounds the dispatch.
struct HairDitherParams {
    seed: u32,
    strand_count: u32,
    blend: f32,
}

// Per-strand keep mask: `1` keeps the finer tier for this strand, `0` drops it.
@group(0) @binding(0) var<storage, read_write> out_keep: array<u32>;

var<immediate> params: HairDitherParams;

// ---------------------------------------------------------------------------
// 64-bit integer emulation for the `splitmix64` hash (baseline WGSL is 32-bit).
// A `u64` is a `vec2<u32>` = (lo, hi). Only the operations the finalizer needs
// are provided: add, xor, right shift, and the low-64-bit product. Copied
// bit-for-bit from `hair_interp.wesl` so both hair kernels share one identity.
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
// bit-for-bit the CPU `interpolation::hash_bits`.
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
// the CPU `interpolation::hash_to_unit`; `>> 40` keeps the high 24 bits in the
// low word).
fn hash_to_unit(seed: u32, key: u32) -> f32 {
    let bits = u64_shr(hash_bits(seed, key), 40u);
    return f32(bits.x) / 16777216.0;
}

@compute @workgroup_size(64)
fn hair_lod_dither(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= params.strand_count {
        return;
    }

    // Mirror of `strand_survives_dither`: clamp the blend, then survive while the
    // per-strand unit hash is at or above it, so the kept fraction is
    // `(1 - blend)`. `blend == 1` keeps nothing (the hash is always `< 1`).
    let b = clamp(params.blend, 0.0, 1.0);
    let survives = hash_to_unit(params.seed, index) >= b;
    out_keep[index] = select(0u, 1u, survives);
}
