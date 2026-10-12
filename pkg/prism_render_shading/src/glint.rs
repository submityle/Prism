//! Open-lobe procedural *glint* / sparkle BRDF — discrete microfacet
//! scintillation (CPU golden).
//!
//! A perfectly smooth statistical normal-distribution function (NDF) models a
//! surface patch as containing *infinitely many* microfacets, so the specular
//! lobe is a continuous, temporally stable highlight.  Real metallic flake,
//! snow, sanded car paint, glittery fabric, and frosted glass instead expose a
//! *finite* number of mirror-like facets inside each pixel footprint; only a
//! handful happen to be oriented toward the half-vector at any instant, which
//! is what produces the characteristic twinkling *glints* that shift as the
//! eye, the light, or the surface move.
//!
//! Following Zirr & Kaplanyan, *Real-Time Rendering of Procedural Multiscale
//! Materials* (I3D 2016) — and the earlier stochastic-glint framework of Jakob
//! et al., *Discrete Stochastic Microfacet Models* (SIGGRAPH 2014) — this
//! module replaces the smooth NDF `D(h)` by a **discrete, counting** estimator.
//! Inside a spatial footprint containing `N` microfacets, the probability that
//! any single facet reflects into a small half-vector query bin of solid angle
//! `dω` is
//!
//! ```text
//! p = clamp(D(h) * dω, 0, 1)
//! ```
//!
//! so the number `k` of facets that actually align is a binomially distributed
//! random variable with mean `λ = N * p`, which for the large `N`, small `p`
//! regime of a pixel footprint is accurately modelled by a Poisson law
//! `k ~ Poisson(λ)`.  The discrete NDF estimator is then
//!
//! ```text
//! D_glint(h) = k / (N * dω)
//! ```
//!
//! which is **unbiased**: `E[k] = λ = N * p = N * D(h) * dω`, hence
//! `E[D_glint] = D(h)`.  Averaged over the footprint ensemble the glint lobe
//! carries *exactly* the same energy as the smooth lobe — no energy is injected
//! or lost — while any single evaluation is a sparse, bright, view-dependent
//! sparkle.  As `λ` grows (large footprint, low roughness detail, or a wide
//! bin) the relative variance `1 / λ` vanishes and `D_glint → D`, so the model
//! degrades gracefully to the ordinary smooth microfacet BRDF.
//!
//! Determinism is achieved **without hidden state**: the count `k` is drawn
//! from a counter-based hash RNG ([`hash_u32`] / [`uniform_at`]) seeded by a
//! quantised surface cell crossed with a quantised half-vector bin
//! ([`glint_seed`]).  Every function is therefore a pure, allocation-free
//! function of its inputs: identical inputs always yield identical glints,
//! which is what makes the sparkle *stable* under re-rendering yet *animated*
//! under motion.
//!
//! # Conventions
//! * Math via `bevy_math`; transcendentals via `bevy_math::ops` (never the
//!   `f32` inherent `exp`/`ln`/`cos`), square roots via the `f32::sqrt` method,
//!   matching the sibling thin-film golden.
//! * All roughnesses are clamped to `[MIN_ALPHA, 1]`, all cosines to
//!   `[0, 1]` (incidence cosines to `[MIN_COS, 1]`), and every counting /
//!   probability quantity is clamped into its physical range, so degenerate or
//!   non-finite inputs can never divide by zero or produce `NaN`.
//! * Every reflectance / NDF value is non-negative and finite; the full BRDF is
//!   additionally guarded against the grazing `4·NoL·NoV` singularity.
//! * A glint *strength* of zero is an exact identity bypass: the stochastic
//!   path is skipped entirely and the smooth GGX NDF is returned bit-for-bit,
//!   so a material with the feature disabled is indistinguishable from the
//!   plain specular lobe.
//! * The discrete estimator is unbiased only while `D(h) * dω <= 1`; a caller
//!   must therefore size the query bin `dω` smaller than `1 / D_peak`.  The
//!   probability clamp keeps the result bounded (never `NaN`) outside that
//!   regime, trading a little bias for stability, exactly as the real-time
//!   literature prescribes.

use bevy_math::ops;

/// `core::f32::consts::PI` rounded to `f32`, byte-identical to the GPU twin.
pub const GLINT_PI: f32 = core::f32::consts::PI;

/// Smallest GGX roughness / denominator floor; keeps the closed forms
/// well-conditioned for mirror-smooth inputs.
pub const MIN_ALPHA: f32 = 1.0e-4;

/// Smallest incidence cosine evaluated; keeps grazing angles off the Smith and
/// `4·NoL·NoV` denominators.
pub const MIN_COS: f32 = 1.0e-4;

/// Smallest positive normal `f32`, matching the `f32::MIN_POSITIVE` guard used
/// across the shading goldens.
pub const MIN_POSITIVE: f32 = f32::MIN_POSITIVE;

/// Mean count below which the exact Knuth Poisson sampler is used; at or above
/// it the stable Gaussian approximation takes over (see [`poisson_sample`]).
pub const POISSON_KNUTH_MAX_LAMBDA: f32 = 20.0;

/// Hard iteration cap for the Knuth Poisson loop, bounding worst-case cost even
/// for a pathological `λ`; the sampler only runs it below
/// [`POISSON_KNUTH_MAX_LAMBDA`], so the cap is never the common path.
pub const POISSON_MAX_ITERS: u32 = 2048;

/// Fixed salt mixed into every [`glint_seed`] so the sparkle field does not
/// correlate with an all-zero cell.
pub const GLINT_SEED_SALT: u32 = 0x5eed_1357;

/// Scalar finite predicate, matching the shared `is_finite` idiom used by the
/// GPU twins (`(x - x) == 0`).
#[inline]
pub fn glint_is_finite(x: f32) -> bool {
    x.is_finite()
}

/// `triple32` integer avalanche hash (Chris Wellons).  A high-quality,
/// fully invertible-grade bit mix with excellent diffusion, chosen because it
/// reproduces bit-for-bit under WGSL's wrapping `u32` arithmetic.
#[inline]
pub fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 17;
    x = x.wrapping_mul(0xed5a_d4bb);
    x ^= x >> 11;
    x = x.wrapping_mul(0xac4c_1b51);
    x ^= x >> 15;
    x = x.wrapping_mul(0x3184_8bab);
    x ^= x >> 14;
    x
}

/// Order-sensitive hash combine, folding `value` into a running `seed`.
#[inline]
pub fn hash_combine(seed: u32, value: u32) -> u32 {
    // `0x9e3779b9` is the golden-ratio constant; adding it de-correlates the
    // combine from a plain XOR when either operand is zero.
    hash_u32(seed.wrapping_add(0x9e37_79b9).wrapping_add(hash_u32(value)))
}

/// Maps a hashed `u32` to a uniform float in `[0, 1)` using the top 24 bits, so
/// every result is exactly representable in `f32` (no rounding to `1.0`).
#[inline]
pub fn uniform_from_hash(h: u32) -> f32 {
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// The `counter`-th uniform draw of the stream seeded by `seed`; a pure,
/// stateless counter-based RNG (same `(seed, counter)` always yields the same
/// value).
#[inline]
pub fn uniform_at(seed: u32, counter: u32) -> f32 {
    uniform_from_hash(hash_combine(seed, counter))
}

/// Quantises a continuous coordinate to an integer cell index, reinterpreted as
/// `u32` so negative coordinates hash cleanly.  The float is clamped into the
/// `i32` range before the cast so the conversion is always well-defined and
/// matches the GPU `bitcast<u32>(i32(...))`.
#[inline]
pub fn quantize_axis(coord: f32, cells_per_unit: f32) -> u32 {
    let scaled = coord * cells_per_unit.max(0.0);
    let f = ops::floor(scaled);
    let clamped = f.clamp(-2_147_483_600.0, 2_147_483_600.0);
    (clamped as i32) as u32
}

/// Half-vector slope along one tangent axis, `h_axis / h_n`, with the normal
/// component floored at [`MIN_COS`] so a grazing half-vector cannot blow the
/// slope up to infinity.
#[inline]
pub fn glint_half_slope(h_axis: f32, h_n: f32) -> f32 {
    h_axis / h_n.max(MIN_COS)
}

/// Builds the deterministic sparkle seed for a footprint cell crossed with a
/// half-vector bin and a multiscale `level`.
///
/// `cell_u` / `cell_v` are surface-space coordinates (e.g. UVs or world
/// position projected onto the tangent frame) quantised at `cells_per_unit`
/// resolution; `slope_x` / `slope_y` are the half-vector slopes (see
/// [`glint_half_slope`]) quantised at `slope_bins_per_unit` resolution.  Mixing
/// space *and* direction is what makes a given facet twinkle on only for the
/// narrow range of views and lights that align it with the eye.
#[inline]
pub fn glint_seed(
    cell_u: f32,
    cell_v: f32,
    cells_per_unit: f32,
    slope_x: f32,
    slope_y: f32,
    slope_bins_per_unit: f32,
    level: u32,
) -> u32 {
    let cu = quantize_axis(cell_u, cells_per_unit);
    let cv = quantize_axis(cell_v, cells_per_unit);
    let sx = quantize_axis(slope_x, slope_bins_per_unit);
    let sy = quantize_axis(slope_y, slope_bins_per_unit);
    let mut s = hash_combine(GLINT_SEED_SALT, cu);
    s = hash_combine(s, cv);
    s = hash_combine(s, sx);
    s = hash_combine(s, sy);
    hash_combine(s, level)
}

/// Solid angle of a circular half-vector query bin of angular radius
/// `half_width` radians, `dω = 2·π·(1 − cos(half_width))`.
///
/// This is the acceptance window the discrete estimator counts facets within;
/// a narrower bin resolves sharper, sparser glints, a wider bin averages toward
/// the smooth lobe.  Floored at [`MIN_POSITIVE`] so it is always a usable
/// divisor.
#[inline]
pub fn glint_bin_solid_angle(half_width: f32) -> f32 {
    let hw = half_width.clamp(0.0, GLINT_PI);
    // Solid angle of a spherical cap, evaluated through the half-angle identity
    // `1 - cos(hw) = 2 sin²(hw/2)` rather than literally. Glint bins are narrow
    // (a few milliradians), where the literal form subtracts two nearly equal
    // numbers and loses most of the significant digits: at `hw = 0.01` the
    // result is ~5e-5, so one ULP of `cos` is a *relative* error of ~1e-3. The
    // identity keeps the full precision with one `sin`, and it is what lets the
    // CPU golden and the GPU twin agree to ~1e-7 instead of ~1e-4.
    let half = hw * 0.5;
    let s = ops::sin(half);
    (4.0 * GLINT_PI * s * s).max(MIN_POSITIVE)
}

/// Isotropic Trowbridge-Reitz (GGX) normal-distribution function.
///
/// ```text
/// D = alpha^2 / (pi * ((n·h)^2 * (alpha^2 - 1) + 1)^2)
/// ```
///
/// Returns a non-negative density that peaks at `n·h = 1` with value
/// `1 / (pi * alpha^2)`.  Roughness is floored at [`MIN_ALPHA`] and `n·h`
/// clamped to `[0, 1]`.
#[inline]
pub fn ggx_ndf(n_dot_h: f32, alpha: f32) -> f32 {
    let a = alpha.max(MIN_ALPHA);
    let a2 = a * a;
    let c = n_dot_h.clamp(0.0, 1.0);
    let d = c * c * (a2 - 1.0) + 1.0;
    let denom = GLINT_PI * d * d;
    if denom > MIN_POSITIVE {
        (a2 / denom).max(0.0)
    } else {
        0.0
    }
}

/// Smith `G1` masking-shadowing term for isotropic GGX via the exact `Lambda`,
///
/// ```text
/// Lambda = (-1 + sqrt(1 + alpha^2 * tan^2(theta))) / 2
/// G1     = 1 / (1 + Lambda)
/// ```
///
/// in `(0, 1]`, tending to `1` for a mirror-smooth surface or head-on
/// direction and falling off at grazing angles.
#[inline]
pub fn smith_g1(n_dot_v: f32, alpha: f32) -> f32 {
    let a = alpha.max(MIN_ALPHA);
    let c = n_dot_v.clamp(MIN_COS, 1.0);
    let cos2 = c * c;
    let tan2 = (1.0 - cos2) / cos2;
    let lambda = 0.5 * (-1.0 + (1.0 + a * a * tan2).max(0.0).sqrt());
    (1.0 / (1.0 + lambda)).clamp(0.0, 1.0)
}

/// Height-correlated Smith masking-shadowing `G2` for isotropic GGX,
///
/// ```text
/// G2 = 2·NoL·NoV / (NoL·sqrt(NoV^2·(1-a^2)+a^2) + NoV·sqrt(NoL^2·(1-a^2)+a^2))
/// ```
///
/// in `[0, 1]`; the standard pairing for the smooth specular lobe and reused
/// unchanged by the glint lobe (only the NDF becomes discrete).
#[inline]
pub fn smith_g2_height_correlated(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a = alpha.max(MIN_ALPHA);
    let a2 = a * a;
    let nv = n_dot_v.clamp(MIN_COS, 1.0);
    let nl = n_dot_l.clamp(MIN_COS, 1.0);
    let ggx_v = nl * (nv * nv * (1.0 - a2) + a2).max(0.0).sqrt();
    let ggx_l = nv * (nl * nl * (1.0 - a2) + a2).max(0.0).sqrt();
    let denom = ggx_v + ggx_l;
    if denom > MIN_POSITIVE {
        (2.0 * nl * nv / denom).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Schlick's scalar Fresnel approximation `F = f0 + (1 - f0)·(1 - cos)^5`,
/// clamped to `[0, 1]`.  Evaluate per channel for coloured conductors.
#[inline]
pub fn fresnel_schlick_scalar(f0: f32, cos_theta: f32) -> f32 {
    let f = f0.clamp(0.0, 1.0);
    let c = (1.0 - cos_theta.clamp(0.0, 1.0)).clamp(0.0, 1.0);
    let c2 = c * c;
    let c5 = c2 * c2 * c;
    (f + (1.0 - f) * c5).clamp(0.0, 1.0)
}

/// Effective microfacet count inside a footprint, `density · area`, floored at
/// one facet so a vanishing footprint still describes a single mirror.
#[inline]
pub fn glint_microfacet_count(density_per_area: f32, footprint_area: f32) -> f32 {
    (density_per_area.max(0.0) * footprint_area.max(0.0)).max(1.0)
}

/// Exact Poisson sampler via Knuth's multiplicative algorithm.
///
/// Draws `k ~ Poisson(lambda)` by multiplying uniforms until the running
/// product drops below `e^{-lambda}`.  The uniform stream is the deterministic
/// counter RNG seeded by `seed`, and the loop is hard-capped at
/// [`POISSON_MAX_ITERS`] so a pathological input can never spin forever.  Best
/// for small `lambda`; see [`poisson_sample`] for the regime switch.
#[inline]
pub fn poisson_knuth(lambda: f32, seed: u32) -> f32 {
    let l = lambda.max(0.0);
    let limit = ops::exp(-l);
    let mut product = 1.0_f32;
    let mut k: u32 = 0;
    loop {
        k += 1;
        product *= uniform_at(seed, k);
        if !(product > limit) || k >= POISSON_MAX_ITERS {
            break;
        }
    }
    (k - 1) as f32
}

/// Gaussian approximation to a Poisson draw, `k ≈ round(lambda + z·sqrt(lambda))`
/// with `z` a standard normal from a Box-Muller pair of the counter RNG.
///
/// Accurate and numerically stable for large `lambda`, where Knuth's loop would
/// otherwise run `~lambda` iterations.  The result is rounded to an integer
/// count and floored at zero.
#[inline]
pub fn poisson_gaussian(lambda: f32, seed: u32) -> f32 {
    let l = lambda.max(0.0);
    let u1 = uniform_at(seed, 1).max(MIN_POSITIVE);
    let u2 = uniform_at(seed, 2);
    let z = (-2.0 * ops::ln(u1)).max(0.0).sqrt() * ops::cos(2.0 * GLINT_PI * u2);
    let k = l + z * l.sqrt();
    ops::round(k).max(0.0)
}

/// Draws an integer microfacet count `k ~ Poisson(lambda)`, switching from the
/// exact Knuth sampler to the stable Gaussian approximation at
/// [`POISSON_KNUTH_MAX_LAMBDA`].  Returns `0` for a non-positive mean.
#[inline]
pub fn poisson_sample(lambda: f32, seed: u32) -> f32 {
    let l = lambda.max(0.0);
    if l <= 0.0 {
        return 0.0;
    }
    if l < POISSON_KNUTH_MAX_LAMBDA {
        poisson_knuth(l, seed)
    } else {
        poisson_gaussian(l, seed)
    }
}

/// Discrete (glinty) GGX NDF estimator.
///
/// Counts how many of the `microfacet_count` facets in the footprint reflect
/// into the half-vector bin of solid angle `bin_solid_angle`, drawn as
/// `k ~ Poisson(N · clamp(D·dω, 0, 1))`, and returns the unbiased density
/// `k / (N · dω)`.  In expectation this equals [`ggx_ndf`]; a single evaluation
/// is the sparse sparkle.  All inputs are floored into their valid ranges.
#[inline]
pub fn glint_ndf(
    n_dot_h: f32,
    alpha: f32,
    microfacet_count: f32,
    bin_solid_angle: f32,
    seed: u32,
) -> f32 {
    glint_ndf_from_density(ggx_ndf(n_dot_h, alpha), microfacet_count, bin_solid_angle, seed)
}

/// Discrete estimator over an **arbitrary** smooth density.
///
/// `glint_ndf` bakes in the isotropic GGX NDF, but the über path's specular lobe
/// can be anisotropic (`openpbr-anisotropy-formula`), and the counting model is
/// agnostic to *which* `D` it samples: the expected number of facets aligned
/// with the query bin is `N · D(h) · dω` for whatever `D` the surface's facet
/// orientation distribution has. Taking `d_smooth` as an argument keeps the
/// estimator **unbiased for the lobe it re-shapes** instead of silently
/// replacing an anisotropic distribution with an isotropic one.
#[inline]
pub fn glint_ndf_from_density(
    d_smooth: f32,
    microfacet_count: f32,
    bin_solid_angle: f32,
    seed: u32,
) -> f32 {
    let n = microfacet_count.max(1.0);
    let dw = bin_solid_angle.max(MIN_POSITIVE);
    let p = (d_smooth * dw).clamp(0.0, 1.0);
    let lambda = (n * p).max(0.0);
    let k = poisson_sample(lambda, seed);
    (k / (n * dw)).max(0.0)
}

/// Blends the smooth and discrete NDFs by `strength ∈ [0, 1]`.
///
/// `strength == 0` is an exact identity bypass returning the plain [`ggx_ndf`]
/// (the stochastic path is never entered); `strength == 1` is the full glint.
/// Because the discrete estimator is unbiased, the ensemble mean of the blend
/// is the smooth NDF for every `strength`, so no energy is added or removed.
#[inline]
pub fn glint_ndf_blended(
    n_dot_h: f32,
    alpha: f32,
    microfacet_count: f32,
    bin_solid_angle: f32,
    strength: f32,
    seed: u32,
) -> f32 {
    glint_ndf_blended_from_density(
        ggx_ndf(n_dot_h, alpha),
        microfacet_count,
        bin_solid_angle,
        strength,
        seed,
    )
}

/// [`glint_ndf_blended`] over an arbitrary smooth density (see
/// [`glint_ndf_from_density`]).
#[inline]
pub fn glint_ndf_blended_from_density(
    d_smooth: f32,
    microfacet_count: f32,
    bin_solid_angle: f32,
    strength: f32,
    seed: u32,
) -> f32 {
    let s = strength.clamp(0.0, 1.0);
    if s <= 0.0 {
        return d_smooth;
    }
    let d_glint = glint_ndf_from_density(d_smooth, microfacet_count, bin_solid_angle, seed);
    d_smooth + (d_glint - d_smooth) * s
}

/// Full specular micro-BRDF with the glinty NDF,
///
/// ```text
/// f_r = D_glint * G2 * F / (4 * NoV * NoL)
/// ```
///
/// Pass a per-channel [`fresnel_schlick_scalar`] value for `fresnel`.  Returns
/// a non-negative, finite reflectance; the grazing `4·NoV·NoL` denominator is
/// guarded.  With `strength == 0` this is bit-for-bit the ordinary smooth GGX
/// specular lobe, so the feature is a zero-cost opt-in.
#[inline]
pub fn glint_specular_brdf(
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
    let nv = n_dot_v.clamp(MIN_COS, 1.0);
    let nl = n_dot_l.clamp(MIN_COS, 1.0);
    let d = glint_ndf_blended(n_dot_h, alpha, microfacet_count, bin_solid_angle, strength, seed);
    let g = smith_g2_height_correlated(nv, nl, alpha);
    let f = fresnel.clamp(0.0, 1.0);
    let denom = 4.0 * nv * nl;
    if denom > MIN_POSITIVE {
        (d * g * f / denom).max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-6;


    #[test]
    fn the_explicit_delta_form_matches_the_blend_helper_bit_for_bit() {
        // §17.4 (P3-e) needs to scale glint's *delta* by `1/p`, which the blend
        // helper cannot express (it clamps its strength to `[0, 1]`). The
        // explicit form must nevertheless be bit-for-bit the helper on the
        // full-quality range, or the amortised path would perturb pixels that
        // never opted into it.
        for n_dot_h in [0.2_f32, 0.55, 0.9, 1.0] {
            for alpha in [0.05_f32, 0.2, 0.6] {
                for strength in [0.0_f32, 0.25, 0.5, 1.0] {
                    for seed in [0u32, 7, 0xdead_beef] {
                        let d_smooth = ggx_ndf(n_dot_h, alpha);
                        let d_glint = glint_ndf_from_density(d_smooth, 64.0, 0.02, seed);
                        let explicit = d_smooth + (d_glint - d_smooth) * strength;
                        let helper =
                            glint_ndf_blended(n_dot_h, alpha, 64.0, 0.02, strength, seed);
                        assert_eq!(explicit, helper, "h={n_dot_h} a={alpha} s={strength}");
                    }
                }
            }
        }
    }

    #[test]
    fn hash_is_deterministic_and_diffuses() {
        // Pure function: identical input -> identical output.
        assert_eq!(hash_u32(0), hash_u32(0));
        assert_eq!(hash_u32(123_456), hash_u32(123_456));
        // Adjacent inputs must not collide and must differ in many bits.
        let a = hash_u32(0);
        let b = hash_u32(1);
        assert_ne!(a, b);
        let diff_bits = (a ^ b).count_ones();
        assert!(diff_bits >= 8, "weak avalanche: {diff_bits} bits");
        // triple32 fixes zero (0 -> 0); the seed salt in `glint_seed`/`hash_combine`
        // is what de-correlates an all-zero cell, so a non-zero probe must map
        // somewhere non-trivial.
        assert_ne!(hash_u32(1), 0);
        assert_ne!(hash_u32(u32::MAX), 0);
    }

    #[test]
    fn hash_combine_is_order_sensitive() {
        assert_eq!(hash_combine(7, 9), hash_combine(7, 9));
        assert_ne!(hash_combine(7, 9), hash_combine(9, 7));
    }

    #[test]
    fn uniforms_are_in_unit_interval() {
        for i in 0..100_000u32 {
            let u = uniform_from_hash(hash_u32(i));
            assert!((0.0..1.0).contains(&u), "u={u} out of [0,1)");
        }
    }

    #[test]
    fn uniform_stream_is_well_distributed() {
        // Mean of a long uniform stream should sit near 0.5.
        let mut sum = 0.0_f64;
        let n = 200_000u32;
        for i in 0..n {
            sum += uniform_at(0xabcd_1234, i) as f64;
        }
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 5.0e-3, "mean={mean}");
    }

    #[test]
    fn quantize_axis_floors_and_wraps_negatives() {
        assert_eq!(quantize_axis(0.0, 4.0), 0);
        assert_eq!(quantize_axis(0.9, 1.0), 0);
        assert_eq!(quantize_axis(1.1, 1.0), 1);
        assert_eq!(quantize_axis(2.5, 2.0), 5);
        // -0.1 * 1 floors to -1 -> bit pattern of i32(-1) == u32::MAX.
        assert_eq!(quantize_axis(-0.1, 1.0), u32::MAX);
        // Non-finite / zero resolution are well-defined (no panic, no NaN cast).
        assert_eq!(quantize_axis(5.0, 0.0), 0);
    }

    #[test]
    fn half_slope_is_bounded_for_grazing_half_vectors() {
        let s = glint_half_slope(1.0, 0.0);
        assert!(s.is_finite());
        assert!((s - 1.0 / MIN_COS).abs() < 1.0, "s={s}");
        assert_eq!(glint_half_slope(0.0, 1.0), 0.0);
    }

    #[test]
    fn glint_seed_is_deterministic_and_sensitive() {
        let base = glint_seed(1.0, 2.0, 8.0, 0.1, 0.2, 64.0, 0);
        assert_eq!(base, glint_seed(1.0, 2.0, 8.0, 0.1, 0.2, 64.0, 0));
        // Perturbations are large enough to cross a quantization bin so each
        // distinct cell/direction/level reshuffles the sparkle.
        assert_ne!(base, glint_seed(1.5, 2.0, 8.0, 0.1, 0.2, 64.0, 0));
        assert_ne!(base, glint_seed(1.0, 2.5, 8.0, 0.1, 0.2, 64.0, 0));
        assert_ne!(base, glint_seed(1.0, 2.0, 8.0, 0.5, 0.2, 64.0, 0));
        assert_ne!(base, glint_seed(1.0, 2.0, 8.0, 0.1, 0.6, 64.0, 0));
        assert_ne!(base, glint_seed(1.0, 2.0, 8.0, 0.1, 0.2, 64.0, 1));
    }

    #[test]
    fn bin_solid_angle_is_monotonic_and_bounded() {
        let mut prev = 0.0;
        for i in 0..=64 {
            let hw = i as f32 / 64.0 * GLINT_PI;
            let dw = glint_bin_solid_angle(hw);
            assert!(dw.is_finite() && dw > 0.0);
            assert!(dw <= 4.0 * GLINT_PI + 1.0e-3, "dw={dw}");
            assert!(dw >= prev - 1.0e-6, "non-monotonic: {prev} -> {dw}");
            prev = dw;
        }
        // Zero radius collapses to the positive floor, not literally zero.
        assert_eq!(glint_bin_solid_angle(0.0), MIN_POSITIVE);
        // Non-finite input is clamped, never propagates NaN.
        assert!(glint_bin_solid_angle(f32::NAN).is_finite());
    }

    #[test]
    fn ggx_peak_and_bounds() {
        // Peak at n.h = 1 equals 1 / (pi * alpha^2).
        for &a in &[0.05_f32, 0.2, 0.5, 1.0] {
            let peak = ggx_ndf(1.0, a);
            let expect = 1.0 / (GLINT_PI * a * a);
            assert!((peak - expect).abs() <= expect * 1.0e-4, "a={a}");
        }
        // For alpha < 1 the GGX density rises monotonically toward the peak at
        // n.h = 1, so sweeping n.h upward must be non-decreasing.
        let mut prev = f32::NEG_INFINITY;
        for i in 0..=20 {
            let c = i as f32 / 20.0;
            let d = ggx_ndf(c, 0.3);
            assert!(d.is_finite() && d >= 0.0);
            assert!(d >= prev - EPS, "non-monotonic at c={c}: {prev}->{d}");
            prev = d;
        }
    }

    #[test]
    fn ggx_handles_degenerate_inputs() {
        assert!(ggx_ndf(f32::NAN, 0.3).is_finite());
        assert!(ggx_ndf(0.5, 0.0).is_finite());
        assert!(ggx_ndf(0.5, -1.0).is_finite());
        assert!(ggx_ndf(2.0, 0.3) >= 0.0);
    }

    #[test]
    fn smith_terms_are_in_unit_range() {
        for i in 1..=20 {
            let c = i as f32 / 20.0;
            for &a in &[0.05_f32, 0.3, 0.8] {
                let g1 = smith_g1(c, a);
                assert!((0.0..=1.0).contains(&g1), "g1={g1}");
                let g2 = smith_g2_height_correlated(c, c, a);
                assert!((0.0..=1.0).contains(&g2), "g2={g2}");
            }
        }
        // Head-on, mirror-smooth -> essentially unmasked.
        assert!(smith_g1(1.0, MIN_ALPHA) > 0.999);
        assert!(smith_g2_height_correlated(1.0, 1.0, MIN_ALPHA) > 0.999);
    }

    #[test]
    fn fresnel_schlick_endpoints() {
        // At normal incidence F == f0; at grazing F -> 1.
        assert!((fresnel_schlick_scalar(0.04, 1.0) - 0.04).abs() < EPS);
        assert!(fresnel_schlick_scalar(0.04, 0.0) > 0.999);
        // Monotonic in angle and always bounded.
        let mut prev = fresnel_schlick_scalar(0.04, 1.0);
        for i in 0..=20 {
            let c = 1.0 - i as f32 / 20.0;
            let f = fresnel_schlick_scalar(0.04, c);
            assert!((0.0..=1.0).contains(&f));
            assert!(f >= prev - EPS, "non-monotonic");
            prev = f;
        }
    }

    #[test]
    fn microfacet_count_is_floored_at_one() {
        assert_eq!(glint_microfacet_count(0.0, 100.0), 1.0);
        assert_eq!(glint_microfacet_count(-5.0, 100.0), 1.0);
        assert_eq!(glint_microfacet_count(10.0, -1.0), 1.0);
        assert_eq!(glint_microfacet_count(2.0, 50.0), 100.0);
    }

    #[test]
    fn poisson_is_deterministic_and_integral() {
        for &lambda in &[0.0_f32, 0.5, 3.0, 8.0, 25.0, 100.0] {
            let a = poisson_sample(lambda, 42);
            let b = poisson_sample(lambda, 42);
            assert_eq!(a, b, "non-deterministic at lambda={lambda}");
            assert!(a >= 0.0 && a.fract() == 0.0, "non-integral k={a}");
            assert!(a.is_finite());
        }
        assert_eq!(poisson_sample(0.0, 7), 0.0);
        assert_eq!(poisson_sample(-3.0, 7), 0.0);
    }

    #[test]
    fn poisson_mean_matches_lambda_knuth_regime() {
        // Exact-sampler regime: Monte-Carlo mean must converge to lambda.
        for &lambda in &[1.0_f32, 3.0, 7.0, 15.0] {
            let m = 60_000u32;
            let mut sum = 0.0_f64;
            for i in 0..m {
                sum += poisson_knuth(lambda, hash_u32(i.wrapping_mul(2_654_435_761))) as f64;
            }
            let mean = sum / m as f64;
            assert!(
                (mean - lambda as f64).abs() < 0.1,
                "lambda={lambda} mean={mean}"
            );
        }
    }

    #[test]
    fn poisson_mean_matches_lambda_gaussian_regime() {
        // Large-lambda approximation: mean and variance both near lambda.
        for &lambda in &[30.0_f32, 80.0, 200.0] {
            let m = 60_000u32;
            let mut sum = 0.0_f64;
            let mut sum_sq = 0.0_f64;
            for i in 0..m {
                let k = poisson_gaussian(lambda, hash_u32(i.wrapping_mul(40_503))) as f64;
                sum += k;
                sum_sq += k * k;
            }
            let mean = sum / m as f64;
            let var = sum_sq / m as f64 - mean * mean;
            assert!((mean - lambda as f64).abs() < 0.6, "lambda={lambda} mean={mean}");
            // Poisson variance == lambda; the Gaussian match should be close.
            assert!(
                (var - lambda as f64).abs() < 0.2 * lambda as f64,
                "lambda={lambda} var={var}"
            );
        }
    }

    #[test]
    fn glint_ndf_is_unbiased_estimator_of_ggx() {
        // Ensemble mean of the discrete NDF must equal the smooth NDF.
        let alpha = 0.3;
        let n_dot_h = 0.985;
        let n = 5_000.0;
        let dw = 1.0e-3; // keep D*dw < 1 so the estimator is unbiased
        let d_smooth = ggx_ndf(n_dot_h, alpha);
        assert!(d_smooth * dw < 1.0, "test bin too wide");
        let m = 80_000u32;
        let mut sum = 0.0_f64;
        for i in 0..m {
            let seed = hash_u32(i.wrapping_mul(0x9e37_79b1) ^ 0x1234_abcd);
            sum += glint_ndf(n_dot_h, alpha, n, dw, seed) as f64;
        }
        let mean = sum / m as f64;
        assert!(
            (mean - d_smooth as f64).abs() < d_smooth as f64 * 0.03,
            "mean={mean} d_smooth={d_smooth}"
        );
    }

    #[test]
    fn glint_values_are_quantized_multiples() {
        // Every single evaluation is k / (N * dw) for an integer k.
        let n = 2_000.0;
        let dw = 2.0e-3;
        let quantum = 1.0 / (n * dw);
        for i in 0..256u32 {
            let v = glint_ndf(0.97, 0.4, n, dw, hash_u32(i));
            let k = v / quantum;
            assert!((k - k.round()).abs() < 1.0e-3, "v={v} not a multiple");
        }
    }

    #[test]
    fn glint_converges_to_smooth_for_large_lambda() {
        // Large count => low relative variance => discrete ~ smooth.
        let alpha = 0.5;
        let n_dot_h = 0.9;
        let n = 2_000_000.0;
        let dw = 5.0e-3;
        let d_smooth = ggx_ndf(n_dot_h, alpha);
        assert!(d_smooth * dw < 1.0);
        let mut max_rel = 0.0_f32;
        for i in 0..64u32 {
            let v = glint_ndf(n_dot_h, alpha, n, dw, hash_u32(i.wrapping_add(999)));
            let rel = ((v - d_smooth) / d_smooth).abs();
            max_rel = max_rel.max(rel);
        }
        assert!(max_rel < 0.05, "glint not converging: max_rel={max_rel}");
    }

    #[test]
    fn strength_zero_is_exact_identity_bypass() {
        // strength == 0 must return the smooth NDF bit-for-bit, independent of
        // the stochastic parameters and seed.
        for &c in &[0.1_f32, 0.5, 0.9, 1.0] {
            for &a in &[0.05_f32, 0.3, 0.9] {
                let smooth = ggx_ndf(c, a);
                let blended = glint_ndf_blended(c, a, 10.0, 1.0e-3, 0.0, 0xdead_beef);
                assert_eq!(smooth, blended, "c={c} a={a}");
            }
        }
    }

    #[test]
    fn strength_is_clamped() {
        // Out-of-range strengths clamp to [0,1]; negative behaves like 0.
        let smooth = ggx_ndf(0.9, 0.3);
        assert_eq!(
            glint_ndf_blended(0.9, 0.3, 10.0, 1.0e-3, -5.0, 7),
            smooth
        );
        let full = glint_ndf_blended(0.9, 0.3, 10.0, 1.0e-3, 1.0, 7);
        let over = glint_ndf_blended(0.9, 0.3, 10.0, 1.0e-3, 5.0, 7);
        assert_eq!(full, over);
    }

    #[test]
    fn blended_mean_is_energy_preserving_for_any_strength() {
        // E[blend] == d_smooth for every strength, because the glint term is
        // unbiased: lerp(d, E[d_glint], s) == lerp(d, d, s) == d.
        let alpha = 0.35;
        let n_dot_h = 0.98;
        let n = 4_000.0;
        let dw = 1.0e-3;
        let d_smooth = ggx_ndf(n_dot_h, alpha);
        for &s in &[0.25_f32, 0.5, 0.75, 1.0] {
            let m = 60_000u32;
            let mut sum = 0.0_f64;
            for i in 0..m {
                let seed = hash_u32(i ^ 0x55aa_1100);
                sum += glint_ndf_blended(n_dot_h, alpha, n, dw, s, seed) as f64;
            }
            let mean = sum / m as f64;
            assert!(
                (mean - d_smooth as f64).abs() < d_smooth as f64 * 0.03,
                "s={s} mean={mean} d_smooth={d_smooth}"
            );
        }
    }

    #[test]
    fn specular_brdf_is_finite_nonnegative_and_bypasses() {
        // Wide sweep: always finite and non-negative.
        for ci in 1..=10 {
            let nv = ci as f32 / 10.0;
            for li in 1..=10 {
                let nl = li as f32 / 10.0;
                for &a in &[0.1_f32, 0.4, 0.9] {
                    let f = glint_specular_brdf(
                        nv, nl, 0.97, a, 0.04, 500.0, 1.0e-3, 1.0, 12345,
                    );
                    assert!(f.is_finite() && f >= 0.0, "f={f}");
                }
            }
        }
        // strength == 0 reproduces the smooth GGX specular lobe exactly.
        let nv = 0.8;
        let nl = 0.6;
        let noh = 0.95;
        let a = 0.3;
        let fr = 0.04;
        let d = ggx_ndf(noh, a);
        let g = smith_g2_height_correlated(nv, nl, a);
        let reference = d * g * fr / (4.0 * nv * nl);
        let got = glint_specular_brdf(nv, nl, noh, a, fr, 100.0, 1.0e-3, 0.0, 99);
        assert!((got - reference).abs() <= reference * 1.0e-5, "got={got} ref={reference}");
    }

    #[test]
    fn specular_brdf_mean_matches_smooth_lobe() {
        // Energy check on the full lobe: averaging the glinty BRDF over seeds
        // must reproduce the smooth microfacet BRDF (no energy injection).
        let nv = 0.7;
        let nl = 0.65;
        let noh = 0.985;
        let a = 0.3;
        let fr = 0.04;
        let n = 5_000.0;
        let dw = 1.0e-3;
        let d = ggx_ndf(noh, a);
        let g = smith_g2_height_correlated(nv, nl, a);
        let reference = (d * g * fr / (4.0 * nv * nl)) as f64;
        let m = 80_000u32;
        let mut sum = 0.0_f64;
        for i in 0..m {
            let seed = hash_u32(i.wrapping_mul(2_246_822_519));
            sum += glint_specular_brdf(nv, nl, noh, a, fr, n, dw, 1.0, seed) as f64;
        }
        let mean = sum / m as f64;
        assert!(
            (mean - reference).abs() < reference * 0.03,
            "mean={mean} ref={reference}"
        );
    }

    #[test]
    fn no_nans_from_adversarial_inputs() {
        let bad = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0, 0.0, 1.0e30];
        for &c in &bad {
            for &a in &bad {
                let d = glint_ndf_blended(c, a, 10.0, 1.0e-3, 0.5, 1);
                assert!(d.is_finite() && d >= 0.0, "ndf d={d}");
                let f = glint_specular_brdf(c, c, c, a, c, 10.0, 1.0e-3, 0.5, 1);
                assert!(f.is_finite() && f >= 0.0, "brdf f={f}");
            }
        }
    }

    #[test]
    fn grazing_denominator_is_guarded() {
        // NoV or NoL driven to the floor must not blow the BRDF up to inf/NaN.
        let f = glint_specular_brdf(0.0, 0.0, 1.0, 0.2, 1.0, 100.0, 1.0e-3, 1.0, 3);
        assert!(f.is_finite() && f >= 0.0, "f={f}");
    }
}
