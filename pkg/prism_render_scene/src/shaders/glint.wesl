//! Open-lobe procedural glint / sparkle BRDF — discrete microfacet
//! scintillation helpers; no bindings or entry points.
//!
//! GPU twin of `pkg/prism_render_shading/src/glint.rs`. A smooth statistical
//! NDF assumes infinitely many microfacets per footprint; real flake, snow,
//! glitter and frost expose a FINITE count, only a few of which align with the
//! half-vector at any instant, producing view-dependent twinkling glints.
//!
//! Following Zirr & Kaplanyan 2016 (`Real-Time Rendering of Procedural
//! Multiscale Materials`) and Jakob et al. 2014 (`Discrete Stochastic
//! Microfacet Models`), the smooth NDF `D(h)` is replaced by a counting
//! estimator over a footprint of `N` microfacets: the probability a single
//! facet reflects into a half-vector bin of solid angle `dw` is
//! `p = clamp(D(h)*dw, 0, 1)`, so the aligned count is `k ~ Poisson(N*p)` and
//! the discrete NDF is `D_glint = k / (N*dw)`. This is UNBIASED
//! (`E[k] = N*p = N*D*dw` so `E[D_glint] = D`), so the footprint ensemble
//! carries exactly the smooth lobe's energy — no energy injection — while a
//! single evaluation is a sparse sparkle. As `lambda = N*p` grows the relative
//! variance `1/lambda` vanishes and the discrete NDF converges to the smooth
//! one.
//!
//! Determinism uses a counter-based hash RNG (`glint_hash_u32`,
//! `glint_uniform_at`) seeded by a quantised surface cell crossed with a
//! quantised half-vector bin (`glint_seed`): identical inputs always yield
//! identical glints. Keep arithmetic order, clamps and floors aligned with the
//! CPU golden; a glint strength of zero is an EXACT identity bypass returning
//! the plain GGX NDF. The discrete estimator is unbiased only while
//! `D(h)*dw <= 1`; the probability clamp keeps the result bounded (never NaN)
//! outside that regime.

/// `core::f32::consts::PI` rounded to f32, byte-identical to the CPU golden.
const GLINT_PI: f32 = 3.1415927;
/// Smallest GGX roughness / denominator floor.
const GLINT_MIN_ALPHA: f32 = 1.0e-4;
/// Smallest incidence cosine evaluated.
const GLINT_MIN_COS: f32 = 1.0e-4;
/// Smallest positive normal f32, matching the CPU `f32::MIN_POSITIVE` guard.
const GLINT_MIN_POSITIVE: f32 = 1.1754944e-38;
/// Mean count below which the exact Knuth Poisson sampler is used.
const GLINT_KNUTH_MAX_LAMBDA: f32 = 20.0;
/// Hard iteration cap for the Knuth Poisson loop.
const GLINT_POISSON_MAX_ITERS: u32 = 2048u;
/// Fixed salt mixed into every `glint_seed`.
const GLINT_SEED_SALT: u32 = 0x5eed1357u;

/// Scalar finite predicate, matching the shared isfinite idiom.
fn glint_is_finite(x: f32) -> bool {
    return (x - x) == 0.0;
}

/// `triple32` integer avalanche hash (Chris Wellons); reproduces the CPU golden
/// bit-for-bit under WGSL wrapping u32 arithmetic.
fn glint_hash_u32(x_in: u32) -> u32 {
    var x = x_in;
    x = x ^ (x >> 17u);
    x = x * 0xed5ad4bbu;
    x = x ^ (x >> 11u);
    x = x * 0xac4c1b51u;
    x = x ^ (x >> 15u);
    x = x * 0x31848babu;
    x = x ^ (x >> 14u);
    return x;
}

/// Order-sensitive hash combine, folding `value` into a running `seed`.
fn glint_hash_combine(seed: u32, value: u32) -> u32 {
    return glint_hash_u32(seed + 0x9e3779b9u + glint_hash_u32(value));
}

/// Maps a hashed u32 to a uniform float in `[0, 1)` using the top 24 bits.
fn glint_uniform_from_hash(h: u32) -> f32 {
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

/// The `counter`-th uniform draw of the stream seeded by `seed`.
fn glint_uniform_at(seed: u32, counter: u32) -> f32 {
    return glint_uniform_from_hash(glint_hash_combine(seed, counter));
}

/// Quantises a continuous coordinate to an integer cell index, reinterpreted as
/// u32 so negative coordinates hash cleanly. Clamped into the i32 range first so
/// the cast is well-defined (matches the CPU `as i32 as u32`).
fn glint_quantize_axis(coord: f32, cells_per_unit: f32) -> u32 {
    let scaled = coord * max(cells_per_unit, 0.0);
    let f = floor(scaled);
    let clamped = clamp(f, -2147483600.0, 2147483600.0);
    return bitcast<u32>(i32(clamped));
}

/// Half-vector slope along one tangent axis, `h_axis / h_n`, with `h_n` floored
/// at `GLINT_MIN_COS`.
fn glint_half_slope(h_axis: f32, h_n: f32) -> f32 {
    return h_axis / max(h_n, GLINT_MIN_COS);
}

/// Builds the deterministic sparkle seed for a footprint cell crossed with a
/// half-vector bin and a multiscale `level`.
fn glint_seed(
    cell_u: f32,
    cell_v: f32,
    cells_per_unit: f32,
    slope_x: f32,
    slope_y: f32,
    slope_bins_per_unit: f32,
    level: u32,
) -> u32 {
    let cu = glint_quantize_axis(cell_u, cells_per_unit);
    let cv = glint_quantize_axis(cell_v, cells_per_unit);
    let sx = glint_quantize_axis(slope_x, slope_bins_per_unit);
    let sy = glint_quantize_axis(slope_y, slope_bins_per_unit);
    var s = glint_hash_combine(GLINT_SEED_SALT, cu);
    s = glint_hash_combine(s, cv);
    s = glint_hash_combine(s, sx);
    s = glint_hash_combine(s, sy);
    return glint_hash_combine(s, level);
}

/// Solid angle of a circular half-vector query bin of angular radius
/// `half_width` radians, `dw = 2*pi*(1 - cos(half_width))`, floored positive.
fn glint_bin_solid_angle(half_width: f32) -> f32 {
    let hw = clamp(half_width, 0.0, GLINT_PI);
    let c = cos(hw);
    return max(2.0 * GLINT_PI * (1.0 - c), GLINT_MIN_POSITIVE);
}

/// Isotropic Trowbridge-Reitz (GGX) normal-distribution function; peaks at
/// `n.h = 1` with value `1 / (pi * alpha^2)`.
fn glint_ggx_ndf(n_dot_h: f32, alpha: f32) -> f32 {
    let a = max(alpha, GLINT_MIN_ALPHA);
    let a2 = a * a;
    let c = clamp(n_dot_h, 0.0, 1.0);
    let d = c * c * (a2 - 1.0) + 1.0;
    let denom = GLINT_PI * d * d;
    if (denom > GLINT_MIN_POSITIVE) {
        return max(a2 / denom, 0.0);
    }
    return 0.0;
}

/// Smith `G1` masking-shadowing term for isotropic GGX via the exact Lambda;
/// in `(0, 1]`.
fn glint_smith_g1(n_dot_v: f32, alpha: f32) -> f32 {
    let a = max(alpha, GLINT_MIN_ALPHA);
    let c = clamp(n_dot_v, GLINT_MIN_COS, 1.0);
    let cos2 = c * c;
    let tan2 = (1.0 - cos2) / cos2;
    let lambda = 0.5 * (-1.0 + sqrt(max(1.0 + a * a * tan2, 0.0)));
    return clamp(1.0 / (1.0 + lambda), 0.0, 1.0);
}

/// Height-correlated Smith masking-shadowing `G2` for isotropic GGX; in
/// `[0, 1]`.
fn glint_smith_g2_height_correlated(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a = max(alpha, GLINT_MIN_ALPHA);
    let a2 = a * a;
    let nv = clamp(n_dot_v, GLINT_MIN_COS, 1.0);
    let nl = clamp(n_dot_l, GLINT_MIN_COS, 1.0);
    let ggx_v = nl * sqrt(max(nv * nv * (1.0 - a2) + a2, 0.0));
    let ggx_l = nv * sqrt(max(nl * nl * (1.0 - a2) + a2, 0.0));
    let denom = ggx_v + ggx_l;
    if (denom > GLINT_MIN_POSITIVE) {
        return clamp(2.0 * nl * nv / denom, 0.0, 1.0);
    }
    return 0.0;
}

/// Schlick scalar Fresnel `F = f0 + (1 - f0)*(1 - cos)^5`, clamped to `[0, 1]`.
fn glint_fresnel_schlick(f0: f32, cos_theta: f32) -> f32 {
    let f = clamp(f0, 0.0, 1.0);
    let c = clamp(1.0 - clamp(cos_theta, 0.0, 1.0), 0.0, 1.0);
    let c2 = c * c;
    let c5 = c2 * c2 * c;
    return clamp(f + (1.0 - f) * c5, 0.0, 1.0);
}

/// Effective microfacet count inside a footprint, `density * area`, floored at
/// one facet.
fn glint_microfacet_count(density_per_area: f32, footprint_area: f32) -> f32 {
    return max(max(density_per_area, 0.0) * max(footprint_area, 0.0), 1.0);
}

/// Exact Poisson sampler via Knuth's multiplicative algorithm; best for small
/// `lambda`. Hard-capped at `GLINT_POISSON_MAX_ITERS`.
fn glint_poisson_knuth(lambda: f32, seed: u32) -> f32 {
    let l = max(lambda, 0.0);
    let limit = exp(-l);
    var product = 1.0;
    var k = 0u;
    loop {
        k = k + 1u;
        product = product * glint_uniform_at(seed, k);
        if (!(product > limit) || k >= GLINT_POISSON_MAX_ITERS) {
            break;
        }
    }
    return f32(k - 1u);
}

/// Gaussian approximation to a Poisson draw via a Box-Muller pair; stable for
/// large `lambda`.
fn glint_poisson_gaussian(lambda: f32, seed: u32) -> f32 {
    let l = max(lambda, 0.0);
    let u1 = max(glint_uniform_at(seed, 1u), GLINT_MIN_POSITIVE);
    let u2 = glint_uniform_at(seed, 2u);
    let z = sqrt(max(-2.0 * log(u1), 0.0)) * cos(2.0 * GLINT_PI * u2);
    let k = l + z * sqrt(l);
    return max(round(k), 0.0);
}

/// Draws `k ~ Poisson(lambda)`, switching from Knuth to the Gaussian
/// approximation at `GLINT_KNUTH_MAX_LAMBDA`. Returns `0` for a non-positive
/// mean.
fn glint_poisson_sample(lambda: f32, seed: u32) -> f32 {
    let l = max(lambda, 0.0);
    if (l <= 0.0) {
        return 0.0;
    }
    if (l < GLINT_KNUTH_MAX_LAMBDA) {
        return glint_poisson_knuth(l, seed);
    }
    return glint_poisson_gaussian(l, seed);
}

/// Discrete (glinty) GGX NDF estimator `k / (N*dw)`, unbiased for `D*dw <= 1`.
fn glint_ndf(
    n_dot_h: f32,
    alpha: f32,
    microfacet_count: f32,
    bin_solid_angle: f32,
    seed: u32,
) -> f32 {
    let d_smooth = glint_ggx_ndf(n_dot_h, alpha);
    let n = max(microfacet_count, 1.0);
    let dw = max(bin_solid_angle, GLINT_MIN_POSITIVE);
    let p = clamp(d_smooth * dw, 0.0, 1.0);
    let lambda = max(n * p, 0.0);
    let k = glint_poisson_sample(lambda, seed);
    return max(k / (n * dw), 0.0);
}

/// Blends the smooth and discrete NDFs by `strength` in `[0, 1]`; `strength == 0`
/// is an exact identity bypass returning the plain GGX NDF.
fn glint_ndf_blended(
    n_dot_h: f32,
    alpha: f32,
    microfacet_count: f32,
    bin_solid_angle: f32,
    strength: f32,
    seed: u32,
) -> f32 {
    let s = clamp(strength, 0.0, 1.0);
    let d_smooth = glint_ggx_ndf(n_dot_h, alpha);
    if (s <= 0.0) {
        return d_smooth;
    }
    let d_glint = glint_ndf(n_dot_h, alpha, microfacet_count, bin_solid_angle, seed);
    return d_smooth + (d_glint - d_smooth) * s;
}

/// Full specular micro-BRDF with the glinty NDF,
/// `f_r = D_glint * G2 * F / (4 * NoV * NoL)`; non-negative and finite, with the
/// grazing denominator guarded. `strength == 0` reproduces the smooth GGX lobe.
fn glint_specular_brdf(
    n_dot_v: f32,
    n_dot_l: f32,
    n_dot_h: f32,
    alpha: f32,
    fresnel: f32,
    microfacet_count: f32,
    bin_solid_angle: f32,
    strength: f32,
    seed: u32,
) -> f32 {
    let nv = clamp(n_dot_v, GLINT_MIN_COS, 1.0);
    let nl = clamp(n_dot_l, GLINT_MIN_COS, 1.0);
    let d = glint_ndf_blended(n_dot_h, alpha, microfacet_count, bin_solid_angle, strength, seed);
    let g = glint_smith_g2_height_correlated(nv, nl, alpha);
    let f = clamp(fresnel, 0.0, 1.0);
    let denom = 4.0 * nv * nl;
    if (denom > GLINT_MIN_POSITIVE) {
        return max(d * g * f / denom, 0.0);
    }
    return 0.0;
}
