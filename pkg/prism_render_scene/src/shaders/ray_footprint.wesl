//! Ray-cone footprint / texture-`LOD` (mip) kernel: the real-device twin of the
//! `CPU` golden arithmetic in
//! `prism_render_architecture::ray_scene::footprint`
//! (`RayFootprint::projected_width` / `texel_span` / `mip_level` / `mip_floor`
//! and the transcendental-free `log2_linear`).
//!
//! A primary or reflection ray is modeled as a cone whose radius grows linearly
//! with travel distance; where it strikes a surface the cross-section defines
//! the shading footprint, and the ratio of that footprint to the surface texel
//! size selects a texture mip level. The kernel reproduces the golden math
//! bit-for-bit: the same non-finite/negative sanitization to zero, the same
//! `width = cone_width + hit_distance * cone_spread_angle`, the same
//! `span = width / texel` guard, and the same repeated-halving/doubling
//! `log2_linear` approximation (exact at powers of two, monotonic, no `log2`).
//! A green `raytrace::shader_tests` proves the source parses and type-checks
//! exactly as on device; `raytrace::footprint_gpu_tests` diffs the dispatched
//! results against the `CPU` golden per record.
//!
//! Buffer contract (kept byte-identical to `abi.rs`, which pins these strides):
//!
//! * `footprints` : `FOOTPRINT_WORDS` (4) `u32` per record — `cone_width` (0),
//!                  `cone_spread_angle` (1, a slope), `hit_distance` (2),
//!                  `texel_world_size` (3), each an `f32` `to_bits` pattern.
//! * `results`    : `RESULT_WORDS` (4) `u32` per record — `projected_width` (0),
//!                  `texel_span` (1), `mip_level` (2) as `f32` bit patterns, and
//!                  `mip_floor` (3) as a raw `u32`.

const FOOTPRINT_WORDS: u32 = 4u;
const RESULT_WORDS: u32 = 4u;

/// `f32` `+inf` bit pattern, used to detect non-finite inputs without an
/// `isFinite` builtin (`abs(v) == +inf` catches `±inf`; `v != v` catches `NaN`).
const POSITIVE_INF_BITS: u32 = 2139095040u;

/// Per-dispatch uniform mirroring `GpuFootprintParams` in `abi.rs`.
struct FootprintParams {
    footprint_count: u32,
    max_mip: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> footprints: array<u32>;
@group(0) @binding(1) var<storage, read_write> results: array<u32>;
@group(0) @binding(2) var<uniform> params: FootprintParams;

/// Replaces non-finite (`NaN`/`±inf`) or negative values with zero, mirroring
/// the golden `sanitize_nonneg`.
fn sanitize_nonneg(value: f32) -> f32 {
    let inf = bitcast<f32>(POSITIVE_INF_BITS);
    let is_nan = value != value;
    let is_inf = abs(value) == inf;
    if (is_nan || is_inf || value < 0.0) {
        return 0.0;
    }
    return value;
}

/// Transcendental-free `log2(ratio)` approximation. The integer part is found
/// by repeated halving/doubling into `[1, 2)` and the fractional part uses the
/// linear approximation `frac ~= mantissa - 1`. Bit-for-bit mirror of the golden
/// `log2_linear`: exact at powers of two, monotonic, and mapping non-finite or
/// non-positive inputs to `0.0`. Only exact power-of-two scalings and integer
/// additions are used, so it evaluates identically to the `CPU` reference.
fn log2_linear(ratio: f32) -> f32 {
    let inf = bitcast<f32>(POSITIVE_INF_BITS);
    let is_nan = ratio != ratio;
    let is_inf = abs(ratio) == inf;
    if (is_nan || is_inf || ratio <= 0.0) {
        return 0.0;
    }
    var mantissa = ratio;
    var exponent = 0.0;
    loop {
        if (mantissa < 2.0) {
            break;
        }
        mantissa = mantissa * 0.5;
        exponent = exponent + 1.0;
    }
    loop {
        if (mantissa >= 1.0) {
            break;
        }
        mantissa = mantissa * 2.0;
        exponent = exponent - 1.0;
    }
    return exponent + (mantissa - 1.0);
}

@compute @workgroup_size(64)
fn ray_footprint(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.footprint_count) {
        return;
    }

    let base = index * FOOTPRINT_WORDS;
    let cone_width = sanitize_nonneg(bitcast<f32>(footprints[base + 0u]));
    let cone_spread = sanitize_nonneg(bitcast<f32>(footprints[base + 1u]));
    let hit_distance = sanitize_nonneg(bitcast<f32>(footprints[base + 2u]));
    let texel = sanitize_nonneg(bitcast<f32>(footprints[base + 3u]));

    // `projected_width = cone_width + hit_distance * cone_spread_angle`.
    let projected_width = cone_width + hit_distance * cone_spread;

    // `texel_span = width / texel`, guarding a non-positive texel to zero.
    var span = 0.0;
    if (texel > 0.0) {
        span = projected_width / texel;
    }

    // Continuous mip clamped to `[0, max_mip]`, then floored to a bucket.
    let raw = log2_linear(span);
    let ceiling = f32(params.max_mip);
    let mip_level = clamp(raw, 0.0, ceiling);
    let mip_floor = u32(floor(mip_level));

    let out = index * RESULT_WORDS;
    results[out + 0u] = bitcast<u32>(projected_width);
    results[out + 1u] = bitcast<u32>(span);
    results[out + 2u] = bitcast<u32>(mip_level);
    results[out + 3u] = mip_floor;
}
