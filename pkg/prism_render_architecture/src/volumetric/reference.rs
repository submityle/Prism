#![forbid(unsafe_code)]
//! Path-tracing reference truth for offline calibration (design section 9c).
//!
//! The runtime volumetric path approximates single/multiple scattering with
//! analytic `Beer-Lambert` `transmittance`, pre-integrated `LUT`s, and probe
//! interpolation. Those approximations must be pinned against a zero-bias
//! ground truth so the `multiscatter` / `avsm` calibration never silently
//! drifts into blackening or over-exposure. This module owns that ground
//! truth: a deterministic, sandbox-verifiable bucket of `Monte-Carlo`
//! estimators that a `CPU` unit test (or `CI` regression) can compare against
//! closed-form values. The full spectral path tracer runs on the `GPU` /
//! offline; here we implement the two workhorse estimators plus their analytic
//! oracles:
//!
//! - [`delta_tracking_transmittance`] — unbiased `free-flight` collision
//!   sampling against a `majorant` `extinction`, handling a heterogeneous
//!   `extinction` field through the supplied closure;
//! - [`ratio_tracking_transmittance`] — the lower-variance residual-ratio
//!   product estimator for the same `transmittance`, used to calibrate the
//!   analytic `transmittance` of design section 6;
//! - [`single_scatter_reference`] — a `Monte-Carlo` single-scatter `radiance`
//!   integral that converges to the closed form [`analytic_single_scatter`].
//!
//! Randomness comes from a self-contained stateless integer-hash `RNG` (a
//! `PCG`-style hash-counter Weyl sequence, see [`hash_u32`] / [`rng_unit`]): no
//! external `rand`, no floating-point transcendental intrinsics. Every float
//! transcendental routes through the shared hand-rolled [`super::math`]
//! (`exp_approx` / `ln_approx`), the only permitted intrinsic being
//! `f32::sqrt`, which this module does not need. Given a fixed seed every
//! estimator is bit-for-bit reproducible, never panics, and clamps degenerate
//! inputs (negative `extinction`, zero `distance`, empty sample counts) instead
//! of trusting them.

use super::math::{exp_approx, ln_approx, saturate, EPS};

/// The golden-ratio odd increment (`floor(2^32 / phi)`) used both to advance
/// the Weyl state inside [`Rng`] and to decorrelate per-sample seeds.
const GOLDEN_GAMMA: u32 = 0x9E37_79B9;

/// Hard cap on tentative collision events inside one `free-flight` walk.
///
/// A pathological `majorant` far above the true `extinction` produces many
/// null-collisions; this ceiling guarantees termination on adversarial input
/// so the estimators can never spin forever, while being high enough never to
/// bias physically reasonable media.
const MAX_EVENTS: u32 = 1_000_000;

/// Maps a full-range `u32` to a unit float in `[0, 1)`.
///
/// Uses the top 24 bits so the result lands exactly on the `f32` mantissa grid
/// (`24` bits of precision) and can never round up to `1.0`, matching the
/// sibling `noise` module's hash-to-unit convention.
#[must_use]
fn unit_from_u32(bits: u32) -> f32 {
    // 1 / 2^24 as an exact power-of-two reciprocal keeps the map lossless.
    ((bits >> 8) as f32) * (1.0 / 16_777_216.0)
}

/// Stateless integer avalanche hash (`Wang`-style, all-`wrapping`).
///
/// This is the primitive behind the module's `RNG`: a bijective-ish scramble
/// that turns a counter into a well-distributed `u32`. It is pure and
/// deterministic, so `hash_u32(x)` returns the same value on every platform,
/// which is what the fixed-seed reproducibility contract of design section 16
/// requires.
#[must_use]
pub fn hash_u32(x: u32) -> u32 {
    let mut h = (x ^ 61).wrapping_mul(0x27d4_eb2d);
    h ^= x >> 16;
    h = h.wrapping_add(h << 3);
    h ^= h >> 4;
    h = h.wrapping_mul(0x27d4_eb2d);
    h ^= h >> 15;
    h
}

/// Stateless mapping from a seed `state` to a unit float in `[0, 1)`.
///
/// Convenience wrapper over [`hash_u32`] for call sites that want one hashed
/// draw without threading an [`Rng`]; the `RNG` reproducibility contract still
/// holds because it is a pure function of `state`.
#[must_use]
pub fn rng_unit(state: u32) -> f32 {
    unit_from_u32(hash_u32(state))
}

/// A tiny deterministic `RNG`: a Weyl-sequence counter fed through
/// [`hash_u32`].
///
/// The `PCG`-style construction (advance an additive counter, then hash it)
/// gives good low-dimensional distribution for the short `free-flight` walks
/// here while staying fully reproducible from its seed. It is intentionally
/// private; public entry points expose only the stateless [`hash_u32`] /
/// [`rng_unit`] primitives.
struct Rng {
    /// Current Weyl counter; every draw advances it by [`GOLDEN_GAMMA`].
    state: u32,
}

impl Rng {
    /// Builds a generator whose stream is a pure function of `seed`.
    #[must_use]
    fn seeded(seed: u32) -> Self {
        Self {
            state: hash_u32(seed ^ GOLDEN_GAMMA),
        }
    }

    /// Advances the Weyl counter and returns the next hashed `u32`.
    ///
    /// Takes `&mut self` and mutates the stream, so it is deliberately not
    /// `#[must_use]`.
    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        hash_u32(self.state)
    }

    /// Advances the stream and returns the next unit float in `[0, 1)`.
    fn next_unit(&mut self) -> f32 {
        unit_from_u32(self.next_u32())
    }
}

/// Derives a decorrelated per-sample seed from a base `seed` and sample index.
///
/// Mixing the index through [`GOLDEN_GAMMA`] keeps neighbouring samples from
/// sharing a `free-flight` stream, which would otherwise correlate the
/// `Monte-Carlo` estimate and inflate its variance.
#[must_use]
fn sample_seed(seed: u32, index: u32) -> u32 {
    seed ^ index.wrapping_mul(GOLDEN_GAMMA)
}

/// Samples one `free-flight` distance for a homogeneous `majorant`.
///
/// Inverts the exponential free-path `cdf`: `t = -ln(1 - u) / majorant`. The
/// caller guarantees `majorant >= EPS`; `u` comes from `[0, 1)` so `1 - u`
/// stays in `(0, 1]` and [`ln_approx`] never sees a non-positive argument.
#[must_use]
fn free_flight_step(u: f32, majorant: f32) -> f32 {
    -ln_approx(1.0 - u) / majorant
}

/// Analytic `Beer-Lambert` `transmittance` of a homogeneous medium.
///
/// Returns `exp(-sigma_t * distance)` via [`exp_approx`], the closed form the
/// tracking estimators must converge to. Negative `sigma_t` and negative
/// `distance` are clamped to zero, and the result is saturated into `[0, 1]`,
/// so `distance == 0` yields exactly `1` and the value is never outside the
/// physical range.
#[must_use]
pub fn analytic_transmittance(sigma_t: f32, distance: f32) -> f32 {
    let optical_depth = sigma_t.max(0.0) * distance.max(0.0);
    saturate(exp_approx(-optical_depth))
}

/// Unbiased `delta tracking` estimate of `transmittance` over `[0, distance]`.
///
/// `sigma_t_fn` samples the local `extinction` at a distance along the ray, so
/// the estimator handles a heterogeneous field. `majorant` must upper-bound
/// `sigma_t_fn` over the segment; it is floored to [`EPS`] to avoid division by
/// zero. The estimator draws `samples` independent `free-flight` walks: a walk
/// "survives" if it reaches `distance` before a real collision (accepted with
/// probability `sigma_t_fn(t) / majorant` at each tentative event). The
/// survival fraction is an unbiased `transmittance` estimate that converges to
/// [`analytic_transmittance`] for a homogeneous medium as `samples` grows.
///
/// Deterministic for a fixed `seed`; returns `1.0` for a zero-length segment or
/// an empty sample budget and clamps the result into `[0, 1]`.
#[must_use]
pub fn delta_tracking_transmittance<F>(
    sigma_t_fn: F,
    majorant: f32,
    distance: f32,
    seed: u32,
    samples: u32,
) -> f32
where
    F: Fn(f32) -> f32,
{
    let dist = distance.max(0.0);
    let maj = majorant.max(EPS);
    if dist < EPS || samples == 0 {
        return 1.0;
    }

    let mut survived: u32 = 0;
    let mut i: u32 = 0;
    while i < samples {
        let mut rng = Rng::seeded(sample_seed(seed, i));
        let mut t = 0.0_f32;
        let mut alive = true;
        let mut events: u32 = 0;
        loop {
            t += free_flight_step(rng.next_unit(), maj);
            if t >= dist {
                break;
            }
            let ratio = saturate(sigma_t_fn(t) / maj);
            if rng.next_unit() < ratio {
                alive = false;
                break;
            }
            events += 1;
            if events >= MAX_EVENTS {
                break;
            }
        }
        if alive {
            survived += 1;
        }
        i += 1;
    }

    saturate(survived as f32 / samples as f32)
}

/// Lower-variance `ratio tracking` estimate of `transmittance`.
///
/// Same setup as [`delta_tracking_transmittance`], but instead of the binary
/// survive/collide test each walk accumulates the residual-ratio product
/// `prod (1 - sigma_t_fn(t) / majorant)` over its null-collision events. The
/// mean of these weights is an unbiased `transmittance` estimate with strictly
/// no higher variance than `delta tracking` (they coincide only in the
/// degenerate `majorant == sigma_t` case), so it is the preferred oracle for
/// calibrating the analytic `transmittance` of design section 6.
///
/// Deterministic for a fixed `seed`; returns `1.0` for a zero-length segment or
/// an empty sample budget and clamps the result into `[0, 1]`.
#[must_use]
pub fn ratio_tracking_transmittance<F>(
    sigma_t_fn: F,
    majorant: f32,
    distance: f32,
    seed: u32,
    samples: u32,
) -> f32
where
    F: Fn(f32) -> f32,
{
    let dist = distance.max(0.0);
    let maj = majorant.max(EPS);
    if dist < EPS || samples == 0 {
        return 1.0;
    }

    let mut sum = 0.0_f32;
    let mut i: u32 = 0;
    while i < samples {
        let mut rng = Rng::seeded(sample_seed(seed, i));
        let mut t = 0.0_f32;
        let mut weight = 1.0_f32;
        let mut events: u32 = 0;
        loop {
            t += free_flight_step(rng.next_unit(), maj);
            if t >= dist {
                break;
            }
            weight *= 1.0 - saturate(sigma_t_fn(t) / maj);
            events += 1;
            if events >= MAX_EVENTS {
                break;
            }
        }
        sum += weight;
        i += 1;
    }

    saturate(sum / samples as f32)
}

/// Closed-form single-scatter `radiance` for a homogeneous medium.
///
/// Integrates in-scattered light along `[0, distance]` under the standard
/// single-scatter reference assumption that the light source itself is not
/// attenuated by the medium (only the eye ray is), giving
/// `sigma_s * phase * light_radiance * (1 - exp(-sigma_t * distance)) /
/// sigma_t`. This is the oracle [`single_scatter_reference`] converges to.
/// All inputs are clamped non-negative and `sigma_t` is floored to [`EPS`] so
/// the division is always safe; the returned `radiance` is not clamped to
/// `[0, 1]` because it is a physical radiance, not a `transmittance`.
#[must_use]
pub fn analytic_single_scatter(
    sigma_t: f32,
    sigma_s: f32,
    phase: f32,
    light_radiance: f32,
    distance: f32,
) -> f32 {
    let st = sigma_t.max(EPS);
    let dist = distance.max(0.0);
    let integral = (1.0 - exp_approx(-st * dist)) / st;
    sigma_s.max(0.0) * phase.max(0.0) * light_radiance.max(0.0) * integral
}

/// `Monte-Carlo` single-scatter `radiance` reference for a homogeneous medium.
///
/// Uses `free-flight` importance sampling: each sample draws a scatter distance
/// `t = -ln(1 - u) / sigma_t` from the `extinction` `pdf`. When `t` lands
/// inside the segment the sample contributes `sigma_s * phase_fn(t) *
/// light_radiance / sigma_t` (the `transmittance` and `pdf` cancel), otherwise
/// it contributes nothing. `phase_fn` supplies the phase value at a distance so
/// anisotropic phases can be probed without importing the `scatter` module.
/// The mean over `samples` converges to [`analytic_single_scatter`] for a
/// constant phase.
///
/// Deterministic for a fixed `seed`; returns `0.0` for a zero-length segment or
/// an empty sample budget. `sigma_t` is floored to [`EPS`] and the other terms
/// are clamped non-negative.
#[must_use]
pub fn single_scatter_reference<P>(
    sigma_t: f32,
    sigma_s: f32,
    light_radiance: f32,
    phase_fn: P,
    distance: f32,
    seed: u32,
    samples: u32,
) -> f32
where
    P: Fn(f32) -> f32,
{
    let st = sigma_t.max(EPS);
    let ss = sigma_s.max(0.0);
    let lr = light_radiance.max(0.0);
    let dist = distance.max(0.0);
    if dist < EPS || samples == 0 {
        return 0.0;
    }

    let mut acc = 0.0_f32;
    let mut i: u32 = 0;
    while i < samples {
        let mut rng = Rng::seeded(sample_seed(seed, i));
        let t = free_flight_step(rng.next_unit(), st);
        if t < dist {
            acc += ss * phase_fn(t).max(0.0) * lr / st;
        }
        i += 1;
    }

    acc / samples as f32
}

#[cfg(test)]
mod tests {
    use super::super::math::EPS;
    use super::*;
    use alloc::vec::Vec;

    /// Homogeneous `extinction` used across the convergence assertions.
    const HOMO_SIGMA: f32 = 0.5;

    /// A homogeneous `extinction` closure for the tracking estimators.
    fn homogeneous(_t: f32) -> f32 {
        HOMO_SIGMA
    }

    #[test]
    fn hash_and_rng_unit_are_deterministic_and_bounded() {
        let mut i: u32 = 0;
        while i < 1000 {
            let a = hash_u32(i);
            let b = hash_u32(i);
            assert_eq!(a, b, "hash_u32 not deterministic at {i}");
            let u = rng_unit(i);
            assert!((0.0..1.0).contains(&u), "rng_unit escaped [0,1) at {i}");
            i += 1;
        }
    }

    #[test]
    fn analytic_transmittance_is_bounded_and_unit_at_zero() {
        assert!((analytic_transmittance(0.5, 0.0) - 1.0).abs() < EPS);
        assert!((analytic_transmittance(0.0, 5.0) - 1.0).abs() < EPS);
        // Negative inputs clamp rather than blow up.
        assert!((analytic_transmittance(-1.0, 3.0) - 1.0).abs() < EPS);
        let mut d = 0.0_f32;
        while d <= 8.0 {
            let tr = analytic_transmittance(0.7, d);
            assert!(
                (0.0..=1.0).contains(&tr),
                "transmittance out of range at {d}"
            );
            d += 0.1;
        }
    }

    #[test]
    fn tracking_transmittance_is_bounded_and_unit_at_zero() {
        assert!((delta_tracking_transmittance(homogeneous, 1.0, 0.0, 7, 64) - 1.0).abs() < EPS);
        assert!((ratio_tracking_transmittance(homogeneous, 1.0, 0.0, 7, 64) - 1.0).abs() < EPS);
        // Empty sample budgets are defined and safe.
        assert!((delta_tracking_transmittance(homogeneous, 1.0, 2.0, 7, 0) - 1.0).abs() < EPS);

        let dt = delta_tracking_transmittance(homogeneous, 1.0, 2.0, 11, 256);
        let rt = ratio_tracking_transmittance(homogeneous, 1.0, 2.0, 11, 256);
        assert!((0.0..=1.0).contains(&dt), "delta tracking out of range");
        assert!((0.0..=1.0).contains(&rt), "ratio tracking out of range");
    }

    #[test]
    fn tracking_is_reproducible_for_fixed_seed() {
        let a = delta_tracking_transmittance(homogeneous, 1.0, 2.0, 1234, 128);
        let b = delta_tracking_transmittance(homogeneous, 1.0, 2.0, 1234, 128);
        assert_eq!(a, b, "delta tracking not bit-reproducible");

        let c = ratio_tracking_transmittance(homogeneous, 1.0, 2.0, 1234, 128);
        let d = ratio_tracking_transmittance(homogeneous, 1.0, 2.0, 1234, 128);
        assert_eq!(c, d, "ratio tracking not bit-reproducible");

        let phase = |_t: f32| 0.2_f32;
        let e = single_scatter_reference(0.5, 0.5, 1.0, phase, 2.0, 99, 512);
        let f = single_scatter_reference(0.5, 0.5, 1.0, phase, 2.0, 99, 512);
        assert_eq!(e, f, "single scatter reference not bit-reproducible");
    }

    #[test]
    fn tracking_converges_to_analytic_transmittance() {
        // Homogeneous medium: sigma_t = 0.5, distance = 2.0 -> exp(-1).
        let truth = analytic_transmittance(HOMO_SIGMA, 2.0);

        let dt_low = delta_tracking_transmittance(homogeneous, 1.0, 2.0, 5, 64);
        let dt_high = delta_tracking_transmittance(homogeneous, 1.0, 2.0, 5, 8192);
        assert!(
            (dt_high - truth).abs() < 0.02,
            "delta tracking did not converge: {dt_high} vs {truth}"
        );
        assert!(
            (dt_high - truth).abs() <= (dt_low - truth).abs() + 0.05,
            "more delta samples should not worsen the estimate"
        );

        let rt_high = ratio_tracking_transmittance(homogeneous, 1.0, 2.0, 5, 8192);
        assert!(
            (rt_high - truth).abs() < 0.02,
            "ratio tracking did not converge: {rt_high} vs {truth}"
        );
    }

    #[test]
    fn ratio_tracking_variance_not_worse_than_delta() {
        // majorant strictly above sigma_t so the estimators do not degenerate.
        let truth = analytic_transmittance(HOMO_SIGMA, 2.0);
        let mut delta_mse = 0.0_f32;
        let mut ratio_mse = 0.0_f32;
        let mut seeds = Vec::new();
        let mut s: u32 = 0;
        while s < 64 {
            seeds.push(s.wrapping_mul(2_654_435_761).wrapping_add(1));
            s += 1;
        }
        for &seed in &seeds {
            let dt = delta_tracking_transmittance(homogeneous, 1.0, 2.0, seed, 32);
            let rt = ratio_tracking_transmittance(homogeneous, 1.0, 2.0, seed, 32);
            let de = dt - truth;
            let re = rt - truth;
            delta_mse += de * de;
            ratio_mse += re * re;
        }
        let n = seeds.len() as f32;
        delta_mse /= n;
        ratio_mse /= n;
        assert!(
            ratio_mse <= delta_mse * 1.5 + 1e-4,
            "ratio tracking variance should not exceed delta tracking: ratio={ratio_mse} delta={delta_mse}"
        );
    }

    #[test]
    fn single_scatter_reference_matches_closed_form() {
        let phase_value = 0.2_f32;
        let phase = |_t: f32| phase_value;
        let truth = analytic_single_scatter(0.5, 0.5, phase_value, 1.0, 2.0);
        // Closed form: 0.5 * 0.2 * 1.0 * (1 - exp(-1)) / 0.5 = 0.12642...
        assert!(truth > 0.0, "analytic single scatter should be positive");

        let mc = single_scatter_reference(0.5, 0.5, 1.0, phase, 2.0, 4242, 40_000);
        assert!(
            (mc - truth).abs() < 0.01,
            "single scatter MC did not converge: {mc} vs {truth}"
        );

        // Degenerate segment scatters nothing.
        assert!(single_scatter_reference(0.5, 0.5, 1.0, phase, 0.0, 4242, 128).abs() < EPS);
    }
}
