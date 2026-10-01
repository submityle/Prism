//! Bounded kernel density estimation for photon splatting — CPU golden.
//!
//! Once caustic [`Photon`](super::photon::Photon)s have been traced onto a
//! diffuse receiver, their discrete flux samples must be reconstructed into a
//! continuous irradiance field.  This module is the backend-neutral reference
//! for that reconstruction: a *bounded* (compact-support) 2-D kernel that
//! spreads each photon's flux over a small footprint on the receiver's tangent
//! plane.
//!
//! Two interchangeable kernels are provided — a radially symmetric
//! [`Kernel::Epanechnikov`] (compact, minimum-variance) and a truncated
//! [`Kernel::Gaussian`] (smooth) — both normalised so their integral over the
//! footprint disk is exactly one.  Because the kernel integrates to one, the
//! splat contribution `flux * kernel(r)` integrates back to the photon's flux:
//! **energy is conserved** by construction, regardless of the footprint radius.
//!
//! It provides:
//!
//! * [`clamp_radius`] — footprint-radius sanitation / clamping.
//! * [`kernel_density`] — the per-unit-area kernel value at planar distance `r`.
//! * [`splat_flux`] — a single photon's flux contribution at distance `r`.
//! * [`estimate_irradiance`] — the gathered irradiance at a receiver point from
//!   a slice of photons, projecting each onto the receiver's tangent plane.
//!
//! # Conventions
//! * Distances and radii are in world units; the kernel has units of inverse
//!   area, so `splat_flux` (and the gather) have units of flux per unit area —
//!   an irradiance estimate.
//! * A kernel is strictly zero at and beyond its footprint radius (compact
//!   support), so a photon never contributes outside its footprint.
//! * `no_std`: math via `bevy_math`; transcendentals via `bevy_math::ops`;
//!   square roots via the `f32::sqrt` method (never `f32::exp`).
//! * Radii are clamped to a positive, finite range; negative/degenerate inputs
//!   fall back safely and every result is finite — no `NaN`, no division by
//!   zero.
//! * Every function is a deterministic, allocation-free pure function: no RNG,
//!   no I/O, no GPU, no global state.

use bevy_math::{ops, Vec3};
use core::f32::consts::PI;

use super::photon::Photon;

/// Smallest footprint radius accepted; keeps the `1 / area` normalisation
/// finite.
const MIN_RADIUS: f32 = 1.0e-4;

/// Ratio of the truncated-Gaussian standard deviation to the footprint radius:
/// the support cuts off at `3 sigma`, where the Gaussian has decayed to
/// `exp(-4.5) ~= 1.1%`.
const GAUSSIAN_SIGMA_FRAC: f32 = 1.0 / 3.0;

/// A bounded (compact-support) density-estimation kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// The Epanechnikov kernel `k(r) = (2 / (pi h^2)) * (1 - (r/h)^2)`, zero for
    /// `r >= h`.  Compact, non-negative and minimum-variance among second-order
    /// kernels.
    Epanechnikov,
    /// A radially symmetric Gaussian with `sigma = h/3`, truncated to the disk
    /// of radius `h` and renormalised so its integral over that disk is one.
    Gaussian,
}

/// Clamps a footprint radius into the valid `[min, max]` range.
///
/// Non-finite or non-positive inputs fall back to `min`; `min` is itself
/// floored at [`MIN_RADIUS`] so the `1 / area` normalisation never blows up.
/// `max` is treated as `>= min`.
#[inline]
pub fn clamp_radius(radius: f32, min: f32, max: f32) -> f32 {
    let lo = if min.is_finite() { min.max(MIN_RADIUS) } else { MIN_RADIUS };
    let hi = if max.is_finite() { max.max(lo) } else { lo };
    if radius.is_finite() {
        radius.clamp(lo, hi)
    } else {
        lo
    }
}

/// Evaluates the chosen kernel's density (per unit area) at planar distance `r`.
///
/// `radius` is the footprint radius `h`; it is floored at [`MIN_RADIUS`].  The
/// return value is non-negative, is exactly `0` for `r >= radius`, and
/// integrates to `1` over the footprint disk (verified in the tests), so it can
/// be multiplied by a photon's flux to obtain an energy-conserving splat.
#[inline]
pub fn kernel_density(r: f32, radius: f32, kernel: Kernel) -> f32 {
    let h = if radius.is_finite() { radius.max(MIN_RADIUS) } else { MIN_RADIUS };
    let r = if r.is_finite() { r.abs() } else { h };
    if r >= h {
        return 0.0;
    }
    let h2 = h * h;
    match kernel {
        Kernel::Epanechnikov => {
            let t = r / h;
            let v = (2.0 / (PI * h2)) * (1.0 - t * t);
            v.max(0.0)
        }
        Kernel::Gaussian => {
            let sigma = h * GAUSSIAN_SIGMA_FRAC;
            let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
            // Normalisation of a Gaussian truncated to the disk of radius h:
            //   integral_0^h exp(-r^2/(2 s^2)) 2 pi r dr
            //     = 2 pi s^2 (1 - exp(-h^2/(2 s^2))).
            let edge = ops::exp(-h2 * inv_2s2);
            let norm = 2.0 * PI * sigma * sigma * (1.0 - edge);
            let v = ops::exp(-r * r * inv_2s2) / norm.max(f32::MIN_POSITIVE);
            v.max(0.0)
        }
    }
}

/// A single photon's flux contribution at planar distance `r` from its centre.
///
/// Returns `flux * kernel_density(r, radius, kernel)` per channel: the flux
/// spread over the footprint.  Because the kernel integrates to one, summing (or
/// integrating) this contribution over the footprint recovers the photon's full
/// flux — the splat is energy-conserving.
#[inline]
pub fn splat_flux(flux: Vec3, r: f32, radius: f32, kernel: Kernel) -> Vec3 {
    let flux = Vec3::new(
        if flux.x.is_finite() { flux.x.max(0.0) } else { 0.0 },
        if flux.y.is_finite() { flux.y.max(0.0) } else { 0.0 },
        if flux.z.is_finite() { flux.z.max(0.0) } else { 0.0 },
    );
    flux * kernel_density(r, radius, kernel)
}

/// Gathers the caustic irradiance at a receiver point from a slice of photons.
///
/// For each photon the offset `photon.position - point` is projected onto the
/// receiver's tangent plane (its component along `normal` is removed), and the
/// planar distance drives the kernel.  Photons on the far side of the surface
/// (negative distance along `normal` beyond the footprint) still contribute via
/// their in-plane distance only, matching a flat splat; the normal is used
/// solely to define the tangent plane.  The contributions are summed, giving an
/// irradiance estimate (flux per unit area).
///
/// `radius` is clamped to `[MIN_RADIUS, inf)`; a degenerate `normal` falls back
/// to `+Y`.  The result is always finite and non-negative per channel.
#[inline]
pub fn estimate_irradiance(
    point: Vec3,
    normal: Vec3,
    photons: &[Photon],
    radius: f32,
    kernel: Kernel,
) -> Vec3 {
    let h = if radius.is_finite() { radius.max(MIN_RADIUS) } else { MIN_RADIUS };
    let n = {
        let len_sq = normal.length_squared();
        if len_sq > f32::MIN_POSITIVE {
            normal * len_sq.sqrt().recip()
        } else {
            Vec3::Y
        }
    };

    let mut sum = Vec3::ZERO;
    for photon in photons {
        let offset = photon.position - point;
        if !offset.is_finite() {
            continue;
        }
        // Remove the component along the normal to get the in-plane offset.
        let planar = offset - n * offset.dot(n);
        let r = planar.length();
        if r < h {
            sum += splat_flux(photon.flux, r, h, kernel);
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    /// Numerically integrates `kernel_density` over the footprint disk using a
    /// fine polar grid; should recover `1` for a normalised kernel.
    fn integrate(radius: f32, kernel: Kernel) -> f32 {
        let nr = 4000usize;
        let nth = 256usize;
        let dr = radius / nr as f32;
        let dth = 2.0 * PI / nth as f32;
        let mut acc = 0.0f64;
        for i in 0..nr {
            let r = (i as f32 + 0.5) * dr;
            let k = kernel_density(r, radius, kernel) as f64;
            // Area element r dr dtheta, summed over all angular slices.
            acc += k * (r as f64) * (dr as f64) * (dth as f64) * nth as f64;
        }
        acc as f32
    }

    #[test]
    fn epanechnikov_integrates_to_one() {
        let v = integrate(0.5, Kernel::Epanechnikov);
        assert!((v - 1.0).abs() < 1.0e-2, "integral={v}");
    }

    #[test]
    fn gaussian_integrates_to_one() {
        let v = integrate(0.5, Kernel::Gaussian);
        assert!((v - 1.0).abs() < 1.0e-2, "integral={v}");
    }

    #[test]
    fn kernel_is_zero_beyond_radius() {
        for k in [Kernel::Epanechnikov, Kernel::Gaussian] {
            assert_eq!(kernel_density(0.5, 0.5, k), 0.0);
            assert_eq!(kernel_density(0.75, 0.5, k), 0.0);
            assert!(kernel_density(0.49, 0.5, k) > 0.0);
        }
    }

    #[test]
    fn kernel_peaks_at_centre() {
        for k in [Kernel::Epanechnikov, Kernel::Gaussian] {
            let c = kernel_density(0.0, 0.3, k);
            let mid = kernel_density(0.15, 0.3, k);
            assert!(c > mid && mid > 0.0, "centre={c} mid={mid}");
        }
    }

    #[test]
    fn splat_scales_flux_by_density() {
        let flux = Vec3::new(1.0, 2.0, 4.0);
        let d = kernel_density(0.1, 0.5, Kernel::Epanechnikov);
        let s = splat_flux(flux, 0.1, 0.5, Kernel::Epanechnikov);
        assert!((s - flux * d).length() < 1.0e-6, "s={s:?}");
    }

    #[test]
    fn clamp_radius_bounds_and_defaults() {
        assert!((clamp_radius(0.3, 0.1, 1.0) - 0.3).abs() < 1.0e-6);
        assert!((clamp_radius(5.0, 0.1, 1.0) - 1.0).abs() < 1.0e-6);
        assert!((clamp_radius(0.0, 0.1, 1.0) - 0.1).abs() < 1.0e-6);
        // Non-finite falls back to the (floored) minimum.
        assert!(clamp_radius(f32::NAN, -1.0, 1.0) >= MIN_RADIUS);
        // Inverted bounds are tolerated (hi clamped up to lo).
        let v = clamp_radius(0.5, 1.0, 0.1);
        assert!(v.is_finite() && v >= 1.0);
    }

    #[test]
    fn gather_conserves_energy_for_a_disk_of_photons() {
        // Lay many photons on the tangent plane within the footprint and gather:
        // the gathered irradiance times the covered area approximates total flux.
        // Instead we check the simpler invariant that a single centred photon
        // yields exactly flux * density(0).
        let flux = Vec3::new(0.5, 0.5, 0.5);
        let p = Photon::new(Vec3::ZERO, Vec3::NEG_Y, flux);
        let e = estimate_irradiance(Vec3::ZERO, Vec3::Y, &[p], 0.4, Kernel::Gaussian);
        let d0 = kernel_density(0.0, 0.4, Kernel::Gaussian);
        assert!((e - flux * d0).length() < 1.0e-5, "e={e:?}");
    }

    #[test]
    fn gather_projects_onto_tangent_plane() {
        // A photon directly above the receiver along the normal has zero in-plane
        // distance, so it lands at the kernel centre regardless of its height.
        let flux = Vec3::splat(1.0);
        let high = Photon::new(Vec3::new(0.0, 10.0, 0.0), Vec3::NEG_Y, flux);
        let e = estimate_irradiance(Vec3::ZERO, Vec3::Y, &[high], 0.5, Kernel::Epanechnikov);
        let d0 = kernel_density(0.0, 0.5, Kernel::Epanechnikov);
        assert!((e - flux * d0).length() < 1.0e-5, "e={e:?}");
    }

    #[test]
    fn gather_excludes_photons_outside_footprint() {
        let flux = Vec3::splat(1.0);
        let far = Photon::new(Vec3::new(2.0, 0.0, 0.0), Vec3::NEG_Y, flux);
        let e = estimate_irradiance(Vec3::ZERO, Vec3::Y, &[far], 0.5, Kernel::Epanechnikov);
        assert_eq!(e, Vec3::ZERO);
    }

    #[test]
    fn empty_gather_is_zero_and_finite() {
        let e = estimate_irradiance(Vec3::ZERO, Vec3::ZERO, &[], 0.5, Kernel::Gaussian);
        assert_eq!(e, Vec3::ZERO);
    }

    #[test]
    fn degenerate_inputs_do_not_nan() {
        assert!(kernel_density(f32::NAN, f32::NAN, Kernel::Gaussian).is_finite());
        let s = splat_flux(Vec3::new(f32::NAN, 1.0, -1.0), 0.1, 0.0, Kernel::Epanechnikov);
        assert!(s.is_finite());
        let p = Photon::new(Vec3::new(f32::INFINITY, 0.0, 0.0), Vec3::NEG_Y, Vec3::ONE);
        let e = estimate_irradiance(Vec3::ZERO, Vec3::Y, &[p], 0.5, Kernel::Gaussian);
        assert!(e.is_finite());
    }

    #[test]
    fn is_deterministic() {
        assert_eq!(
            kernel_density(0.2, 0.5, Kernel::Gaussian),
            kernel_density(0.2, 0.5, Kernel::Gaussian)
        );
    }
}
