//! Deterministic volumetric scattering phase functions and `octave`-scatter
//! weights for the volumetric cloud subsystem (design sections 5 and 7).
//!
//! Cloud single scattering is driven by an anisotropic phase function that
//! biases light toward the forward direction (the silver-lining and glory
//! response). This module supplies the `CPU` reference implementations of the
//! phases the ray-march and multi-scatter stages consume:
//!
//! - `Henyey-Greenstein` (`HG`): the classic single-parameter anisotropic
//!   phase, normalized so its `4*PI` solid-angle integral is one.
//! - `dual-lobe` `HG`: a convex blend of a sharp forward lobe and a softer
//!   backward lobe, matching measured `Mie` water-droplet phases far better
//!   than a single lobe while staying normalized.
//! - `Draine` (`Jendersie` and `d'Eon` 2023) and the `HG-Draine` blend: a
//!   size-parameter phase that reproduces the forward `Mie` peak more
//!   faithfully than `HG`; the `Draine` shape parameter drives the peak.
//! - `powder`: the Nubis dark-edge approximation that fakes the multiple-
//!   scattering darkening near dense cloud boundaries.
//! - `octave` scattering (Wrenninge style): a geometric decay of the
//!   scattering coefficient, extinction coefficient, and phase eccentricity
//!   across successive scattering octaves, a cheap multiple-scattering
//!   approximation whose energy only ever decreases per octave.
//!
//! Every function here is a pure, deterministic function of its arguments:
//! identical inputs always produce bit-identical outputs, nothing depends on
//! wall-clock time or a random generator, and out-of-range arguments are
//! clamped rather than allowed to `panic` or produce `NaN`.
//!
//! The crate determinism policy allows only `sqrt` among the `f32` float
//! intrinsics, so this module never calls `exp` / `pow` / `ln` / `sin` /
//! `cos` directly. The `^1.5` denominators of the phase functions are formed
//! as `d * sqrt(d)` (pure `sqrt`), integer octave powers are formed by
//! repeated multiplication, and the `powder` exponential uses the shared
//! [`exp_approx`] helper. The `GPU` `WESL` kernels evaluate the identical
//! algebra with native intrinsics; this `CPU` reference exists so the numeric
//! properties (`4*PI` normalization, `powder` monotonicity, `octave` energy
//! decay) can be unit-tested in the sandbox where there is no `GPU`.

use super::math::{clamp, exp_approx, lerp, saturate, EPS, PI};

/// Reciprocal of `4*PI`, the normalization constant of an isotropic phase.
///
/// A normalized phase integrates to one over the `4*PI` steradians of the
/// sphere, so the isotropic phase equals this everywhere and every anisotropic
/// phase here carries it as its leading factor.
const INV_FOUR_PI: f32 = 1.0 / (4.0 * PI);

/// Largest anisotropy magnitude accepted by the phase functions.
///
/// The `Henyey-Greenstein` denominator `1 + g^2 - 2*g*u` collapses to zero at
/// `|g| = 1`, `u = sign(g)`; clamping `g` just inside the unit interval keeps
/// the phase finite and the reference bit-reproducible.
const MAX_ABS_G: f32 = 0.999;

/// Default forward-lobe anisotropy for the `dual-lobe` cloud phase.
///
/// A sharp forward lobe reproduces the silver lining of back-lit clouds.
pub const DEFAULT_FORWARD_G: f32 = 0.8;

/// Default backward-lobe anisotropy for the `dual-lobe` cloud phase.
///
/// A mild backward lobe fills in the ambient wrap-around light.
pub const DEFAULT_BACKWARD_G: f32 = -0.3;

/// Default forward/backward mix for the `dual-lobe` cloud phase.
///
/// `1` selects the forward lobe, `0` selects the backward lobe; the cloud
/// default leans forward.
pub const DEFAULT_LOBE_BLEND: f32 = 0.6;

/// Default `Draine` shape parameter controlling the forward-peak sharpness.
///
/// Larger values sharpen the forward `Mie` peak; `0` reduces `Draine` to
/// `Henyey-Greenstein`.
pub const DEFAULT_DRAINE_ALPHA: f32 = 1.0;

/// Default weight of the `HG` term inside the `HG-Draine` blend.
///
/// `1` selects pure `HG`, `0` selects pure `Draine`.
pub const DEFAULT_HG_DRAINE_WEIGHT: f32 = 0.5;

/// Default `powder` strength for the dark-edge self-shadow approximation.
pub const DEFAULT_POWDER_STRENGTH: f32 = 1.0;

/// The isotropic phase value `1 / (4*PI)`.
///
/// Scattering with no directional preference distributes energy uniformly over
/// the sphere; this is both the `g = 0` limit of every phase here and the
/// low-quality fallback of the degrade chain (design section 7b).
#[must_use]
pub fn isotropic_phase() -> f32 {
    INV_FOUR_PI
}

/// Evaluates the normalized `Henyey-Greenstein` phase for scattering cosine
/// `cos_theta` and anisotropy `g`.
///
/// `cos_theta` is the cosine of the angle between the incoming and outgoing
/// directions (`+1` is pure forward scatter, `-1` pure back scatter) and is
/// clamped to `[-1, 1]`. `g` in `(-1, 1)` is the asymmetry parameter
/// (positive is forward-biased) and is clamped to `[-MAX_ABS_G, MAX_ABS_G]`.
/// The result is normalized so `integral p dOmega = 1` over the sphere, i.e.
/// `integral over [-1, 1] of 2*PI*p d(cos_theta) = 1`; at `g = 0` it equals
/// the isotropic `1 / (4*PI)`.
#[must_use]
pub fn hg_phase(cos_theta: f32, g: f32) -> f32 {
    let u = clamp(cos_theta, -1.0, 1.0);
    let g = clamp(g, -MAX_ABS_G, MAX_ABS_G);
    let g2 = g * g;
    // `denom` is guarded away from zero so the `^1.5` term never divides by
    // (near) zero; `denom * sqrt(denom)` is the pure-`sqrt` form of `denom^1.5`.
    let denom = (1.0 + g2 - 2.0 * g * u).max(EPS);
    INV_FOUR_PI * (1.0 - g2) / (denom * denom.sqrt())
}

/// Evaluates a normalized `dual-lobe` `Henyey-Greenstein` phase.
///
/// The result is a convex blend of a forward lobe (`g_forward`) and a backward
/// lobe (`g_backward`): `lerp(hg(g_backward), hg(g_forward), blend)`. `blend`
/// is clamped to `[0, 1]` (`1` selects the forward lobe). Because a convex
/// combination of two normalized phases is itself normalized, this phase also
/// satisfies `integral p dOmega = 1`. `g_forward` should be positive and
/// `g_backward` negative to model the measured `Mie` forward peak plus the
/// softer backscatter wrap.
#[must_use]
pub fn dual_lobe_phase(cos_theta: f32, g_forward: f32, g_backward: f32, blend: f32) -> f32 {
    let forward = hg_phase(cos_theta, g_forward);
    let backward = hg_phase(cos_theta, g_backward);
    lerp(backward, forward, saturate(blend))
}

/// Evaluates the normalized `Draine` phase (`Jendersie` and `d'Eon` 2023).
///
/// `Draine` multiplies the `Henyey-Greenstein` shape by `1 + alpha*u^2`,
/// sharpening the forward `Mie` peak while a matching normalization factor
/// keeps `integral p dOmega = 1`. `alpha >= 0` is the shape parameter (clamped
/// to be non-negative); `alpha = 0` reduces exactly to [`hg_phase`]. `g` and
/// `cos_theta` are clamped as in [`hg_phase`].
#[must_use]
pub fn draine_phase(cos_theta: f32, g: f32, alpha: f32) -> f32 {
    let u = clamp(cos_theta, -1.0, 1.0);
    let g = clamp(g, -MAX_ABS_G, MAX_ABS_G);
    let alpha = alpha.max(0.0);
    let g2 = g * g;
    let denom = (1.0 + g2 - 2.0 * g * u).max(EPS);
    // The normalization uses the closed-form `HG` second moment
    // `<u^2> = (1 + 2*g^2) / 3`, so the extra `1 + alpha*u^2` weighting still
    // integrates to one over the sphere.
    let norm = 1.0 + alpha * (1.0 + 2.0 * g2) / 3.0;
    INV_FOUR_PI * (1.0 - g2) * (1.0 + alpha * u * u) / (norm * denom * denom.sqrt())
}

/// Evaluates a normalized `HG-Draine` blend phase.
///
/// The result is a convex blend `lerp(draine, hg, hg_weight)` of the
/// [`draine_phase`] (weight `1 - hg_weight`) and the [`hg_phase`] (weight
/// `hg_weight`), both evaluated at the same `cos_theta`. `hg_weight` is
/// clamped to `[0, 1]`. As a convex combination of two normalized phases the
/// blend integrates to one over the sphere. `g_hg` drives the `HG` lobe,
/// while `g_draine` and `alpha` drive the sharper `Draine` forward peak.
#[must_use]
pub fn hg_draine_phase(
    cos_theta: f32,
    g_hg: f32,
    g_draine: f32,
    alpha: f32,
    hg_weight: f32,
) -> f32 {
    let hg = hg_phase(cos_theta, g_hg);
    let draine = draine_phase(cos_theta, g_draine, alpha);
    lerp(draine, hg, saturate(hg_weight))
}

/// Evaluates the anisotropic `dual-lobe` cloud phase with a `Draine`-sharpened
/// forward lobe (design section 5, "anisotropic dual-lobe HG + Draine").
///
/// The backward lobe is a soft `Henyey-Greenstein` lobe (`g_backward`, meant to
/// be negative) that fills the ambient wrap-around light, while the forward
/// lobe is the [`hg_draine_phase`] `Mie` mixture: `g_forward` drives both its
/// `HG` and `Draine` terms, `alpha` sharpens the forward `Draine` peak, and
/// `draine_weight` in `[0, 1]` selects how much of the sharper `Draine` shape
/// to fold in (`1` = pure `Draine`, `0` = pure `HG`). That forward lobe
/// reproduces the silver-lining and glory response of measured water droplets.
/// The two lobes are mixed by `blend` (clamped to `[0, 1]`, `1` selecting the
/// forward lobe).
///
/// Each lobe is individually normalized and the outer mix is convex, so the
/// result still integrates to one over the sphere. With `alpha = 0` the forward
/// `Draine` term collapses to `HG` (independently of `draine_weight`), so the
/// phase reduces exactly to [`dual_lobe_phase`] and preserves the pure-`HG`
/// dual-lobe golden.
#[must_use]
pub fn dual_lobe_draine_phase(
    cos_theta: f32,
    g_forward: f32,
    g_backward: f32,
    alpha: f32,
    draine_weight: f32,
    blend: f32,
) -> f32 {
    let backward = hg_phase(cos_theta, g_backward);
    // `hg_weight = 1 - draine_weight`: the forward lobe leans on the sharp
    // pure-`Draine` peak as `draine_weight` -> 1 and on the softer `HG` lobe as
    // `draine_weight` -> 0. Both `g` arguments share `g_forward` so the lobe is
    // parameterized by a single forward eccentricity.
    let forward = hg_draine_phase(
        cos_theta,
        g_forward,
        g_forward,
        alpha,
        1.0 - saturate(draine_weight),
    );
    lerp(backward, forward, saturate(blend))
}

/// Evaluates the Nubis `powder` dark-edge term for a given view-ray optical
/// depth.
///
/// The `powder` effect fakes the multiple-scattering darkening seen near dense
/// cloud edges with the curve `1 - exp(-2 * density * strength)`. `density`
/// (optical depth accumulated along the view ray) and `strength` are clamped
/// to be non-negative, so the result rises monotonically from `0` at
/// `density = 0` toward `1` as density grows, and is always in `[0, 1]`.
#[must_use]
pub fn powder(density_along_view: f32, strength: f32) -> f32 {
    let density = density_along_view.max(0.0);
    let strength = strength.max(0.0);
    saturate(1.0 - exp_approx(-2.0 * density * strength))
}

/// Geometric-decay parameters for the Wrenninge-style `octave` scattering
/// approximation (design section 7).
///
/// Each successive scattering octave attenuates the scattering coefficient,
/// the extinction coefficient, and the phase eccentricity by a fixed
/// geometric factor, so the summed contribution forms a convergent series that
/// only ever loses energy. All factors are treated as living in `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OctaveParams {
    /// Per-octave decay of the extinction coefficient `sigma_t` in `[0, 1]`.
    pub attenuation: f32,
    /// Per-octave decay of the scattering coefficient `sigma_s` in `[0, 1]`.
    pub contribution: f32,
    /// Per-octave decay of the phase eccentricity `g` in `[0, 1]`, so higher
    /// octaves become progressively more isotropic.
    pub eccentricity_attenuation: f32,
    /// Number of octaves the approximation sums; indices at or beyond this are
    /// clamped to the final octave so queries never grow the energy.
    pub octave_count: u32,
}

impl OctaveParams {
    /// The default `octave` schedule: each octave halves the scattering,
    /// extinction, and eccentricity across four octaves (a balanced Nubis /
    /// Wrenninge preset).
    pub const DEFAULT: Self = Self {
        attenuation: 0.5,
        contribution: 0.5,
        eccentricity_attenuation: 0.5,
        octave_count: 4,
    };
}

impl Default for OctaveParams {
    /// Returns [`OctaveParams::DEFAULT`].
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Raises `base` to the non-negative integer power `exp` by repeated
/// multiplication.
///
/// This avoids [`super::math::pow_approx`] (and its `pow(0, 0)` ambiguity):
/// `pow_u(base, 0)` is exactly `1.0` and the loop is deterministic and
/// `panic`-free for any `base`. Used to form the per-octave geometric factor
/// `factor^octave_index`.
#[must_use]
fn pow_u(base: f32, exp: u32) -> f32 {
    let mut acc = 1.0;
    let mut i = 0;
    while i < exp {
        acc *= base;
        i += 1;
    }
    acc
}

/// Computes the scattering coefficient, extinction coefficient, and phase
/// eccentricity for a single Wrenninge `octave`.
///
/// Given the base `sigma_s`, base `sigma_t`, and base `g`, the octave index,
/// and the geometric [`OctaveParams`], this returns
/// `(sigma_s, sigma_t, g)` for that octave:
///
/// - `sigma_s = base_sigma_s * contribution^i`
/// - `sigma_t = base_sigma_t * attenuation^i`
/// - `g = base_g * eccentricity_attenuation^i`
///
/// where `i` is `octave_index` clamped to the last octave in
/// [`OctaveParams::octave_count`]. All three geometric factors are saturated
/// into `[0, 1]`, so every returned quantity is non-increasing in the octave
/// index (energy never grows) and the eccentricity converges toward zero
/// (higher octaves scatter more isotropically). Base coefficients are clamped
/// to be non-negative and the base eccentricity is clamped to the valid
/// anisotropy range; the result is deterministic and never `panic`s, even when
/// `octave_count` is zero.
#[must_use]
pub fn octave_scatter(
    base_sigma_s: f32,
    base_sigma_t: f32,
    base_g: f32,
    octave_index: u32,
    params: OctaveParams,
) -> (f32, f32, f32) {
    let sigma_s0 = base_sigma_s.max(0.0);
    let sigma_t0 = base_sigma_t.max(0.0);
    let g0 = clamp(base_g, -MAX_ABS_G, MAX_ABS_G);

    let attenuation = saturate(params.attenuation);
    let contribution = saturate(params.contribution);
    let eccentricity = saturate(params.eccentricity_attenuation);

    // Clamp the index to the last octave so out-of-range queries stay at the
    // lowest (never larger) energy; `octave_count == 0` collapses to octave 0.
    let last = params.octave_count.saturating_sub(1);
    let i = octave_index.min(last);

    let sigma_s = sigma_s0 * pow_u(contribution, i);
    let sigma_t = sigma_t0 * pow_u(attenuation, i);
    let g = g0 * pow_u(eccentricity, i);
    (sigma_s, sigma_t, g)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sample count for the midpoint solid-angle normalization integrals.
    const INTEGRATION_SAMPLES: u32 = 20_000;

    /// Fixed scattering cosines spanning full back scatter to full forward
    /// scatter, used across the property tests.
    const COSINES: [f32; 7] = [-1.0, -0.6, -0.2, 0.0, 0.35, 0.7, 1.0];

    /// `true` when `a` and `b` agree within `tol` absolute error.
    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// Numerically integrates a phase `f(cos_theta)` over the sphere using the
    /// midpoint rule: `integral over [-1, 1] of 2*PI*f d(cos_theta)`.
    fn integrate_phase(f: impl Fn(f32) -> f32) -> f32 {
        let two_pi = 2.0 * PI;
        let n = INTEGRATION_SAMPLES;
        let step = 2.0 / n as f32;
        let mut sum = 0.0;
        let mut i = 0;
        while i < n {
            let u = -1.0 + (i as f32 + 0.5) * step;
            sum += f(u) * step * two_pi;
            i += 1;
        }
        sum
    }

    #[test]
    fn hg_phase_is_normalized() {
        for g in [-0.7, -0.2, 0.0, 0.3, 0.6, 0.9] {
            let integral = integrate_phase(|u| hg_phase(u, g));
            assert!(
                close(integral, 1.0, 5e-3),
                "HG not normalized for g={g}: {integral}"
            );
        }
    }

    #[test]
    fn hg_phase_is_isotropic_at_zero_g() {
        for &u in &COSINES {
            assert!(
                close(hg_phase(u, 0.0), INV_FOUR_PI, 1e-6),
                "HG at g=0 should be isotropic"
            );
        }
        assert!(close(isotropic_phase(), INV_FOUR_PI, 1e-9));
    }

    #[test]
    fn hg_phase_is_forward_biased() {
        // A positive g must weight forward scatter above back scatter.
        assert!(hg_phase(1.0, 0.6) > hg_phase(-1.0, 0.6));
        // A negative g reverses the bias.
        assert!(hg_phase(-1.0, -0.6) > hg_phase(1.0, -0.6));
    }

    #[test]
    fn dual_lobe_phase_is_normalized() {
        let integral = integrate_phase(|u| dual_lobe_phase(u, 0.8, -0.3, 0.5));
        assert!(
            close(integral, 1.0, 1.5e-2),
            "dual-lobe not normalized: {integral}"
        );
    }

    #[test]
    fn dual_lobe_forward_lobe_dominates() {
        let forward = dual_lobe_phase(1.0, 0.8, -0.3, 0.5);
        let backward = dual_lobe_phase(-1.0, 0.8, -0.3, 0.5);
        assert!(
            forward > backward,
            "forward lobe must exceed backward: {forward} vs {backward}"
        );
    }

    #[test]
    fn draine_phase_is_normalized() {
        for (g, alpha) in [(0.5, 1.0), (0.3, 0.5), (0.7, 2.0), (0.0, 1.5)] {
            let integral = integrate_phase(|u| draine_phase(u, g, alpha));
            assert!(
                close(integral, 1.0, 1e-2),
                "Draine not normalized for g={g}, alpha={alpha}: {integral}"
            );
        }
    }

    #[test]
    fn draine_reduces_to_hg_at_zero_alpha() {
        for &u in &COSINES {
            assert!(
                close(draine_phase(u, 0.5, 0.0), hg_phase(u, 0.5), 1e-6),
                "Draine with alpha=0 must equal HG"
            );
        }
    }

    #[test]
    fn hg_draine_blend_is_normalized_and_forward_biased() {
        let integral = integrate_phase(|u| hg_draine_phase(u, 0.6, 0.7, 1.0, 0.5));
        assert!(
            close(integral, 1.0, 1.5e-2),
            "HG-Draine not normalized: {integral}"
        );
        let forward = hg_draine_phase(1.0, 0.6, 0.7, 1.0, 0.5);
        let backward = hg_draine_phase(-1.0, 0.6, 0.7, 1.0, 0.5);
        assert!(forward > backward);
    }

    #[test]
    fn dual_lobe_draine_phase_is_normalized() {
        for (alpha, draine_weight, blend) in [
            (1.0, 0.5, 0.6),
            (2.0, 1.0, 0.7),
            (0.5, 0.25, 0.4),
            (0.0, 0.8, 0.6),
        ] {
            let integral = integrate_phase(|u| {
                dual_lobe_draine_phase(u, 0.8, -0.3, alpha, draine_weight, blend)
            });
            assert!(
                close(integral, 1.0, 1.5e-2),
                "dual-lobe Draine not normalized for alpha={alpha},                  draine_weight={draine_weight}, blend={blend}: {integral}"
            );
        }
    }

    #[test]
    fn dual_lobe_draine_forward_lobe_dominates() {
        let forward = dual_lobe_draine_phase(1.0, 0.8, -0.3, 1.0, 1.0, 0.6);
        let backward = dual_lobe_draine_phase(-1.0, 0.8, -0.3, 1.0, 1.0, 0.6);
        assert!(
            forward > backward,
            "forward lobe must exceed backward: {forward} vs {backward}"
        );
    }

    #[test]
    fn dual_lobe_draine_sharper_peak_with_alpha() {
        // A positive Draine alpha must sharpen the forward peak relative to the
        // pure-HG dual lobe at the same forward/backward/blend configuration.
        let sharp = dual_lobe_draine_phase(1.0, 0.8, -0.3, 2.0, 1.0, 0.6);
        let base = dual_lobe_phase(1.0, 0.8, -0.3, 0.6);
        assert!(
            sharp > base,
            "Draine forward peak must exceed HG: {sharp} vs {base}"
        );
    }

    #[test]
    fn dual_lobe_draine_reduces_to_dual_lobe_at_zero_alpha() {
        // alpha = 0 collapses the Draine term to HG for any draine_weight, so
        // the phase must equal the pure-HG dual lobe bit-for-bit.
        for &draine_weight in &[0.0, 0.5, 1.0] {
            for &u in &COSINES {
                assert_eq!(
                    dual_lobe_draine_phase(u, 0.8, -0.3, 0.0, draine_weight, 0.6).to_bits(),
                    dual_lobe_phase(u, 0.8, -0.3, 0.6).to_bits(),
                    "alpha=0 must reduce to dual_lobe_phase"
                );
            }
        }
    }

    #[test]
    fn powder_stays_in_unit_range_and_starts_at_zero() {
        // density = 0 yields (approximately) zero, and never leaves [0, 1].
        assert!(powder(0.0, DEFAULT_POWDER_STRENGTH) <= 1e-5);
        assert!(powder(0.0, DEFAULT_POWDER_STRENGTH) >= 0.0);
        let mut d = 0.0;
        while d <= 10.0 {
            let v = powder(d, DEFAULT_POWDER_STRENGTH);
            assert!(
                (0.0..=1.0).contains(&v),
                "powder out of range at d={d}: {v}"
            );
            d += 0.25;
        }
    }

    #[test]
    fn powder_is_monotonic_in_density() {
        let mut prev = powder(0.0, 1.0);
        let mut d = 0.1;
        while d <= 8.0 {
            let cur = powder(d, 1.0);
            assert!(
                cur >= prev - 1e-6,
                "powder must be non-decreasing in density at d={d}"
            );
            prev = cur;
            d += 0.1;
        }
    }

    #[test]
    fn octave_energy_is_non_increasing_and_eccentricity_converges() {
        let params = OctaveParams::DEFAULT;
        let base_s = 0.8;
        let base_t = 1.2;
        let base_g = 0.85;
        let mut prev_s = f32::INFINITY;
        let mut prev_t = f32::INFINITY;
        let mut prev_abs_g = f32::INFINITY;
        for octave in 0u32..6 {
            let (sigma_s, sigma_t, g) = octave_scatter(base_s, base_t, base_g, octave, params);
            assert!(sigma_s <= prev_s + 1e-6, "sigma_s grew at octave {octave}");
            assert!(sigma_t <= prev_t + 1e-6, "sigma_t grew at octave {octave}");
            assert!(g.abs() <= prev_abs_g + 1e-6, "|g| grew at octave {octave}");
            assert!(sigma_s >= 0.0 && sigma_t >= 0.0);
            prev_s = sigma_s;
            prev_t = sigma_t;
            prev_abs_g = g.abs();
        }
        // The eccentricity converges toward isotropic as octaves accumulate.
        let (_, _, g_late) = octave_scatter(base_s, base_t, base_g, 3, params);
        assert!(g_late.abs() < base_g.abs());
    }

    #[test]
    fn octave_zero_index_returns_base() {
        let params = OctaveParams::DEFAULT;
        let (sigma_s, sigma_t, g) = octave_scatter(0.7, 1.1, 0.6, 0, params);
        assert!(close(sigma_s, 0.7, 1e-6));
        assert!(close(sigma_t, 1.1, 1e-6));
        assert!(close(g, 0.6, 1e-6));
    }

    #[test]
    fn octave_zero_count_does_not_panic() {
        let params = OctaveParams {
            attenuation: 0.5,
            contribution: 0.5,
            eccentricity_attenuation: 0.5,
            octave_count: 0,
        };
        // Index is clamped to octave 0, so the base values come back unchanged.
        let (sigma_s, sigma_t, g) = octave_scatter(0.9, 0.4, 0.5, 12, params);
        assert!(close(sigma_s, 0.9, 1e-6));
        assert!(close(sigma_t, 0.4, 1e-6));
        assert!(close(g, 0.5, 1e-6));
    }

    #[test]
    fn phases_are_deterministic() {
        for &u in &COSINES {
            assert_eq!(hg_phase(u, 0.4).to_bits(), hg_phase(u, 0.4).to_bits());
            assert_eq!(
                draine_phase(u, 0.4, 1.0).to_bits(),
                draine_phase(u, 0.4, 1.0).to_bits()
            );
            assert_eq!(
                dual_lobe_phase(u, 0.8, -0.3, 0.5).to_bits(),
                dual_lobe_phase(u, 0.8, -0.3, 0.5).to_bits()
            );
            assert_eq!(
                hg_draine_phase(u, 0.6, 0.7, 1.0, 0.5).to_bits(),
                hg_draine_phase(u, 0.6, 0.7, 1.0, 0.5).to_bits()
            );
            assert_eq!(
                powder(u.abs(), 1.0).to_bits(),
                powder(u.abs(), 1.0).to_bits()
            );
        }
        let params = OctaveParams::DEFAULT;
        for octave in 0u32..5 {
            let a = octave_scatter(0.8, 1.2, 0.85, octave, params);
            let b = octave_scatter(0.8, 1.2, 0.85, octave, params);
            assert_eq!(a.0.to_bits(), b.0.to_bits());
            assert_eq!(a.1.to_bits(), b.1.to_bits());
            assert_eq!(a.2.to_bits(), b.2.to_bits());
        }
    }
}
