//! GPU guide-to-render strand interpolation kernel (shared, style-agnostic base).
//!
//! Compute twin of the CPU golden reference
//! `prism_render_architecture::hair::interpolation::interpolate_render_strand`.
//! The sandbox has no GPU, so this shader is validated by naga compilation only
//! (see `shader_tests`); it is the GPU-driven *interpolate* stage (design §10.3)
//! that expands a groom's sparse simulated *guide* strands into the many *render*
//! strands that actually draw, mirroring UE5 Groom, AMD TressFX and NVIDIA
//! HairWorks at the algorithm level without reusing their code.
//!
//! One invocation owns one render strand. It reads that strand's binding (up to
//! four nearest guides with barycentric-style weights, a root UV, a seed and a
//! flat output offset), then applies the exact same deterministic transform
//! chain as the CPU golden, in the identical order so the GPU path stays a
//! faithful bit-twin:
//!
//! 1. weighted blend of the corresponding guide control points,
//! 2. length jitter (scaling every point away from the root),
//! 3. clump pull toward the representative guide, growing toward the tip,
//! 4. curl (a seed-stable helix framed on the clumped tangent), and
//! 5. per-point position jitter,
//!
//! with curl and jitter both ramping from zero at the root to full at the tip.
//!
//! Determinism is the whole point: the only entropy source is a hand-written
//! `splitmix64` integer hash, reproduced here bit-for-bit with a `vec2<u32>`
//! (lo, hi) emulation because baseline WGSL has no native 64-bit integer, so a
//! given seed always yields the same render strand and the CPU golden tests
//! validate this kernel's arithmetic. `f32::sin` is likewise banned (not
//! bit-reproducible across platforms), so curl uses the same range-reduced
//! Taylor `sin_turns` the CPU golden uses.
//!
//! Buffer layout: `guide_points` is the flat pool of all guide control points
//! (`xyz` used, root first); `guide_ranges[g]` slices it for guide `g`.
//! `bindings[i]` carries render strand `i`'s influencing guide indices, weights,
//! root UV, seed and `out_offset`. Each invocation writes `len` control points
//! (`xyz` position, `w` = 1.0) into `out_points` starting at `out_offset`, where
//! `len` is the shortest contributing guide's control-point count so mismatched
//! guide resolutions never index out of bounds.

/// Number of guide strands that influence one render strand (mirrors the CPU
/// `GUIDE_INFLUENCE_COUNT`).
const GUIDE_INFLUENCE_COUNT: u32 = 4u;

/// Upper bound on control points per strand, sizing the per-invocation scratch
/// slab so the kernel needs no dynamic allocation. Well beyond typical groom
/// segment counts; callers keep guide resolutions within this.
const MAX_HAIR_STRAND_POINTS: u32 = 256u;

/// `f32::EPSILON`, the zero-length guard used by `normalize_or`.
const F32_EPSILON: f32 = 1.1920929e-7;

// Mathematical constants at f32 precision (baseline WGSL has no `PI` builtin);
// each literal rounds to the same f32 bits as the CPU `core::f32::consts` value.
const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;
const FRAC_PI_2: f32 = 1.5707963267949;

// Purpose salts keep the different deterministic draws from one strand seed
// statistically independent (mirrors the CPU `*_SALT` constants).
const CLUMP_ID_SALT: u32 = 0x00C10107u;
const CURL_PHASE_SALT: u32 = 0x00C01201u;
const LENGTH_JITTER_SALT: u32 = 0x001E4671u;
const POSITION_JITTER_SALT: u32 = 0x00505A17u;

/// Flat range of one guide strand's control points in `guide_points`.
struct HairGuideRange {
    offset: u32,
    count: u32,
}

/// One render strand's binding to its influencing guide strands. Mirrors the
/// CPU `RenderStrandBinding` plus the flat `out_offset` this twin writes to.
struct HairRenderBinding {
    // Guide indices, nearest first; out-of-range entries are ignored.
    guides: vec4<u32>,
    // Per-guide weights, re-normalized over the valid, positive entries.
    weights: vec4<f32>,
    // Root surface UV, preserved for downstream stages (carried, not consumed).
    root_uv: vec2<f32>,
    // Deterministic per-strand seed for clumping, curl phase and jitter.
    seed: u32,
    // First `out_points` slot this strand writes.
    out_offset: u32,
}

/// Tunable parameters shared by every render strand of a groom. Mirrors the CPU
/// `InterpolationParams` (already sanitized by the caller) plus the strand count
/// that bounds the dispatch.
struct HairInterpParams {
    // Number of clump buckets (`0`/`1` => one clump).
    clump_count: u32,
    // How strongly the tip is pulled to its clump guide, in `0..=1`.
    clump_strength: f32,
    // Curl oscillations along the strand length (non-negative).
    curl_frequency: f32,
    // Peak curl displacement at the tip in world units (non-negative).
    curl_amplitude: f32,
    // Peak per-point positional jitter in world units, in `0..=1`.
    position_jitter: f32,
    // Fractional strand-length randomization in `0..=1`.
    length_jitter: f32,
    // Number of render strands, bounding the dispatch.
    strand_count: u32,
}

// Flat pool of all guide control points (`xyz` used, root first).
@group(0) @binding(0) var<storage, read> guide_points: array<vec4<f32>>;
// Per-guide slice descriptor into `guide_points`.
@group(0) @binding(1) var<storage, read> guide_ranges: array<HairGuideRange>;
// Per render strand binding.
@group(0) @binding(2) var<storage, read> bindings: array<HairRenderBinding>;
// Output control points (`xyz` = position, `w` = 1.0), flat at `out_offset`.
@group(0) @binding(3) var<storage, read_write> out_points: array<vec4<f32>>;

var<immediate> params: HairInterpParams;

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

// Signed hash in `[-1, 1)` for symmetric jitter. `unit * 2` is exact (power of
// two), so this matches the CPU `mul_add(2.0, -1.0)` bit-for-bit.
fn hash_to_signed(seed: u32, key: u32) -> f32 {
    return hash_to_unit(seed, key) * 2.0 - 1.0;
}

// Stable sub-key from a purpose salt plus two coordinates (mirrors the CPU
// `sub_key`; u32 arithmetic wraps as required).
fn sub_key(salt: u32, a: u32, b: u32) -> u32 {
    return salt + a * 0x27D4EB2Fu + b * 0x165667B1u;
}

// Round half away from zero, matching Rust `f32::round` (WGSL `round` is
// ties-to-even, so it is not used here).
fn round_away(v: f32) -> f32 {
    return sign(v) * floor(abs(v) + 0.5);
}

// Deterministic sine (the CPU `sin_turns`): range-reduce to `[-PI/2, PI/2]`,
// then a 9th-order Horner Taylor. `f32::sin` is banned for reproducibility.
fn sin_turns(x: f32) -> f32 {
    var a = x - round_away(x / TAU) * TAU;
    if a > FRAC_PI_2 {
        a = PI - a;
    } else if a < -FRAC_PI_2 {
        a = -PI - a;
    }
    let x2 = a * a;
    let poly = ((((x2 * (1.0 / 362880.0) - 1.0 / 5040.0) * x2 + 1.0 / 120.0) * x2 - 1.0 / 6.0) * x2 + 1.0);
    return a * poly;
}

// Unit vector, or `fallback` when `v` is (near) zero length (mirrors the CPU
// `Vec3::normalize_or`).
fn normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if len_sq <= F32_EPSILON {
        return fallback;
    }
    return v * (1.0 / sqrt(len_sq));
}

// Normalized strand parameter `t` in `0..=1` for point `i` of `len`.
fn strand_param(i: u32, len: u32) -> f32 {
    if len > 1u {
        return f32(i) / f32(len - 1u);
    }
    return 0.0;
}

// Stable clump bucket in `0..clump_count` (mirrors the CPU `clump_bucket`).
fn clump_bucket(seed: u32, clump_count: u32) -> u32 {
    if clump_count <= 1u {
        return 0u;
    }
    let scaled = hash_to_unit(seed, CLUMP_ID_SALT) * f32(clump_count);
    return min(u32(floor(scaled)), clump_count - 1u);
}

struct Basis {
    n1: vec3<f32>,
    n2: vec3<f32>,
}

// Two orthonormal vectors spanning the plane normal to `tangent` (mirrors the
// CPU `orthonormal_basis`).
fn orthonormal_basis(tangent: vec3<f32>) -> Basis {
    var helper = vec3<f32>(1.0, 0.0, 0.0);
    if abs(tangent.x) >= 0.9 {
        helper = vec3<f32>(0.0, 1.0, 0.0);
    }
    let n1 = normalize_or(cross(tangent, helper), vec3<f32>(0.0, 0.0, 1.0));
    let n2 = normalize_or(cross(tangent, n1), vec3<f32>(1.0, 0.0, 0.0));
    return Basis(n1, n2);
}

@compute @workgroup_size(64)
fn hair_interpolate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let strand = gid.x;
    if strand >= params.strand_count {
        return;
    }
    let binding = bindings[strand];
    let guide_count = arrayLength(&guide_ranges);

    // Gather in-range, positive-weight, non-empty guides and their total weight.
    var contrib_index: array<u32, 4>;
    var contrib_weight: array<f32, 4>;
    var contrib_len = 0u;
    var weight_sum = 0.0;
    var rep = 0u;
    var rep_weight = -1.0;
    var min_len = 0xFFFFFFFFu;

    for (var k = 0u; k < GUIDE_INFLUENCE_COUNT; k = k + 1u) {
        let raw = max(binding.weights[k], 0.0);
        if raw <= 0.0 {
            continue;
        }
        let index = binding.guides[k];
        if index >= guide_count {
            continue;
        }
        let count = guide_ranges[index].count;
        if count == 0u {
            continue;
        }
        contrib_index[contrib_len] = index;
        contrib_weight[contrib_len] = raw;
        contrib_len = contrib_len + 1u;
        weight_sum = weight_sum + raw;
        if count < min_len {
            min_len = count;
        }
        if raw > rep_weight {
            rep_weight = raw;
            rep = index;
        }
    }

    if contrib_len == 0u || weight_sum <= 0.0 {
        return;
    }

    var len = min_len;
    if len > MAX_HAIR_STRAND_POINTS {
        len = MAX_HAIR_STRAND_POINTS;
    }
    let inv_sum = 1.0 / weight_sum;
    let rep_off = guide_ranges[rep].offset;

    // Per-invocation scratch for the clumped polyline (curl frames off this).
    var shaped: array<vec3<f32>, MAX_HAIR_STRAND_POINTS>;

    // Pass 1: weighted blend, length jitter relative to the root, then clump.
    let length_factor =
        1.0 + hash_to_signed(binding.seed, LENGTH_JITTER_SALT) * params.length_jitter;

    // Blend + length jitter.
    var root = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < len; i = i + 1u) {
        var point = vec3<f32>(0.0, 0.0, 0.0);
        for (var c = 0u; c < contrib_len; c = c + 1u) {
            let off = guide_ranges[contrib_index[c]].offset;
            let gp = guide_points[off + i].xyz;
            point = point + gp * (contrib_weight[c] * inv_sum);
        }
        if i == 0u {
            root = point;
        }
        point = root + (point - root) * length_factor;
        shaped[i] = point;
    }

    // Clump pull toward the representative guide, increasing toward the tip.
    if params.clump_strength > 0.0 {
        for (var i = 0u; i < len; i = i + 1u) {
            let t = strand_param(i, len);
            let pull = params.clump_strength * t;
            let rep_point = guide_points[rep_off + i].xyz;
            shaped[i] = shaped[i] + (rep_point - shaped[i]) * pull;
        }
    }

    // Curl phase is stable per strand and per clump bucket.
    let clump_id = clump_bucket(binding.seed, params.clump_count);
    let clump_seed = binding.seed ^ (clump_id * 0x9E3779B1u);
    let phase = hash_to_unit(clump_seed, CURL_PHASE_SALT) * TAU;

    // Pass 2: add curl (from the clumped tangent) and positional jitter, write.
    for (var i = 0u; i < len; i = i + 1u) {
        let t = strand_param(i, len);
        var point = shaped[i];

        if params.curl_amplitude > 0.0 {
            // Forward-difference tangent, root/tip clamped (CPU `strand_tangent`).
            var raw_tangent = vec3<f32>(0.0, 1.0, 0.0);
            if len > 1u {
                if i + 1u < len {
                    raw_tangent = shaped[i + 1u] - shaped[i];
                } else {
                    raw_tangent = shaped[i] - shaped[i - 1u];
                }
            }
            let tangent = normalize_or(raw_tangent, vec3<f32>(0.0, 1.0, 0.0));
            let basis = orthonormal_basis(tangent);
            let angle = params.curl_frequency * t * TAU + phase;
            let amplitude = params.curl_amplitude * t;
            let offset =
                (basis.n1 * sin_turns(angle) + basis.n2 * sin_turns(angle + FRAC_PI_2)) * amplitude;
            point = point + offset;
        }

        if params.position_jitter > 0.0 {
            let jx = hash_to_signed(binding.seed, sub_key(POSITION_JITTER_SALT, i, 0u));
            let jy = hash_to_signed(binding.seed, sub_key(POSITION_JITTER_SALT, i, 1u));
            let jz = hash_to_signed(binding.seed, sub_key(POSITION_JITTER_SALT, i, 2u));
            point = point + vec3<f32>(jx, jy, jz) * (params.position_jitter * t);
        }

        out_points[binding.out_offset + i] = vec4<f32>(point, 1.0);
    }
}
