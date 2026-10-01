//! BSDF-domain sample mappings — CPU golden.
//!
//! The samplers in [`super::sobol`] and [`super::blue_noise`] produce canonical
//! points in the unit square `[0, 1)^2`.  This module warps those points into
//! the geometric domains the GI integrator actually samples: a unit disk, a
//! cosine-weighted hemisphere (the importance distribution for Lambertian
//! diffuse bounces), and a uniform hemisphere (unbiased fallback / debugging).
//!
//! All results are expressed in a local tangent frame with the surface normal
//! along `+Z`; [`world_from_local`] rotates them onto an arbitrary world-space
//! normal using a branch-light orthonormal basis.
//!
//! # References
//! * Shirley & Chiu, *A Low Distortion Map Between Disk and Square* — the
//!   concentric disk mapping used by [`concentric_disk`].
//! * Malley's method: a cosine-weighted hemisphere direction is a concentric
//!   disk sample lifted onto the hemisphere, used by [`cosine_hemisphere`].
//! * Duff et al., *Building an Orthonormal Basis, Revisited* (JCGT 2017) — the
//!   branch-light frame used by [`orthonormal_basis`].
//!
//! Every function is deterministic and allocation-free so the GPU/WESL twin can
//! reproduce it bit-for-bit with the same `f32` arithmetic.

use core::f32::consts::{FRAC_1_PI, PI, TAU};

/// Maps a unit-square point `(u, v) ∈ [0, 1)^2` onto the unit disk with the
/// Shirley–Chiu concentric mapping, preserving stratification (adjacent square
/// samples stay adjacent on the disk) and area.
///
/// Returns `(x, y)` with `x^2 + y^2 <= 1`.
#[inline]
pub fn concentric_disk(u: f32, v: f32) -> (f32, f32) {
    // Remap to [-1, 1]^2.
    let a = 2.0 * u - 1.0;
    let b = 2.0 * v - 1.0;

    if a == 0.0 && b == 0.0 {
        return (0.0, 0.0);
    }

    // Choose the region (|a| > |b| vs otherwise) so that radius = max(|a|,|b|)
    // and the angle is measured off the dominant axis.
    let (r, theta) = if a * a > b * b {
        (a, (PI / 4.0) * (b / a))
    } else {
        (b, (PI / 2.0) - (PI / 4.0) * (a / b))
    };
    (r * theta.cos(), r * theta.sin())
}

/// Draws a cosine-weighted direction over the hemisphere about `+Z` from a
/// unit-square point (Malley's method: lift a concentric disk sample).
///
/// The returned direction is a unit vector with `z >= 0`; its pdf with respect
/// to solid angle is `cos(theta) / pi` (see [`cosine_hemisphere_pdf`]).  This is
/// the importance distribution for a Lambertian diffuse bounce, so the Monte
/// Carlo weight `f * cos / pdf` collapses to the constant albedo.
#[inline]
pub fn cosine_hemisphere(u: f32, v: f32) -> [f32; 3] {
    let (x, y) = concentric_disk(u, v);
    // Lift onto the hemisphere: z = sqrt(1 - r^2), clamped against round-off.
    let z = (1.0 - (x * x + y * y)).max(0.0).sqrt();
    [x, y, z]
}

/// Solid-angle pdf of [`cosine_hemisphere`] for a direction whose cosine with
/// the normal is `cos_theta` (`>= 0`): `cos_theta / pi`.
#[inline]
pub fn cosine_hemisphere_pdf(cos_theta: f32) -> f32 {
    cos_theta.max(0.0) * FRAC_1_PI
}

/// Draws a uniformly distributed direction over the hemisphere about `+Z`.
///
/// The returned direction is a unit vector with `z >= 0` and constant solid-angle
/// pdf `1 / (2*pi)` (see [`uniform_hemisphere_pdf`]).  Used as an unbiased
/// fallback / reference when cosine importance sampling is disabled.
#[inline]
pub fn uniform_hemisphere(u: f32, v: f32) -> [f32; 3] {
    let z = u; // cos(theta), uniform in [0, 1)
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = TAU * v;
    [r * phi.cos(), r * phi.sin(), z]
}

/// Solid-angle pdf of [`uniform_hemisphere`]: the constant `1 / (2*pi)`.
#[inline]
pub fn uniform_hemisphere_pdf() -> f32 {
    0.5 * FRAC_1_PI
}

/// Builds a right-handed orthonormal basis `(tangent, bitangent)` for a unit
/// normal `n` using Duff et al.'s branch-light construction.
///
/// The returned vectors are mutually orthogonal unit vectors, each orthogonal to
/// `n`, so `(tangent, bitangent, n)` forms a frame with `+Z == n`.
#[inline]
pub fn orthonormal_basis(n: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let sign = if n[2] >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + n[2]);
    let b = n[0] * n[1] * a;
    let tangent = [1.0 + sign * n[0] * n[0] * a, sign * b, -sign * n[0]];
    let bitangent = [b, sign + n[1] * n[1] * a, -n[1]];
    (tangent, bitangent)
}

/// Rotates a direction expressed in the local tangent frame (`+Z` = normal) into
/// world space for the given unit `normal`.
#[inline]
pub fn world_from_local(local: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    let (t, bt) = orthonormal_basis(normal);
    [
        t[0] * local[0] + bt[0] * local[1] + normal[0] * local[2],
        t[1] * local[0] + bt[1] * local[1] + normal[1] * local[2],
        t[2] * local[0] + bt[2] * local[1] + normal[2] * local[2],
    ]
}

/// Convenience: a world-space cosine-weighted direction about `normal`.
#[inline]
pub fn cosine_hemisphere_world(u: f32, v: f32, normal: [f32; 3]) -> [f32; 3] {
    world_from_local(cosine_hemisphere(u, v), normal)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn len(v: [f32; 3]) -> f32 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }
    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[test]
    fn disk_samples_stay_inside_unit_disk() {
        for i in 0..64u32 {
            for j in 0..64u32 {
                let (x, y) = concentric_disk(i as f32 / 64.0, j as f32 / 64.0);
                assert!(x * x + y * y <= 1.0 + 1e-5, "({x},{y}) outside disk");
            }
        }
    }

    #[test]
    fn disk_center_maps_to_center() {
        let (x, y) = concentric_disk(0.5, 0.5);
        assert!(x.abs() < 1e-6 && y.abs() < 1e-6);
    }

    #[test]
    fn cosine_directions_are_unit_upper_hemisphere() {
        for i in 0..64u32 {
            for j in 0..64u32 {
                let d = cosine_hemisphere(i as f32 / 64.0, j as f32 / 64.0);
                assert!((len(d) - 1.0).abs() < 1e-4, "not unit: {d:?}");
                assert!(d[2] >= -1e-6, "below hemisphere: {d:?}");
            }
        }
    }

    #[test]
    fn cosine_pdf_matches_definition() {
        assert!((cosine_hemisphere_pdf(1.0) - FRAC_1_PI).abs() < 1e-6);
        assert_eq!(cosine_hemisphere_pdf(-0.5), 0.0);
    }

    #[test]
    fn uniform_directions_are_unit_upper_hemisphere() {
        for i in 0..32u32 {
            for j in 0..32u32 {
                let d = uniform_hemisphere(i as f32 / 32.0, j as f32 / 32.0);
                assert!((len(d) - 1.0).abs() < 1e-4, "not unit: {d:?}");
                assert!(d[2] >= -1e-6);
            }
        }
    }

    #[test]
    fn basis_is_orthonormal() {
        for n in [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.577_350_26, 0.577_350_26, 0.577_350_26], [1.0, 0.0, 0.0]] {
            let (t, b) = orthonormal_basis(n);
            assert!((len(t) - 1.0).abs() < 1e-4, "t not unit for {n:?}");
            assert!((len(b) - 1.0).abs() < 1e-4, "b not unit for {n:?}");
            assert!(dot(t, b).abs() < 1e-4, "t.b != 0 for {n:?}");
            assert!(dot(t, n).abs() < 1e-4, "t.n != 0 for {n:?}");
            assert!(dot(b, n).abs() < 1e-4, "b.n != 0 for {n:?}");
        }
    }

    #[test]
    fn local_z_maps_to_normal() {
        let n = [0.267, 0.534, 0.802]; // ~unit
        let w = world_from_local([0.0, 0.0, 1.0], n);
        assert!((w[0] - n[0]).abs() < 1e-3);
        assert!((w[1] - n[1]).abs() < 1e-3);
        assert!((w[2] - n[2]).abs() < 1e-3);
    }

    #[test]
    fn cosine_world_samples_sit_in_normal_hemisphere() {
        let n = [0.0, 1.0, 0.0];
        for i in 0..32u32 {
            for j in 0..32u32 {
                let d = cosine_hemisphere_world(i as f32 / 32.0, j as f32 / 32.0, n);
                assert!(dot(d, n) >= -1e-4, "sample below surface: {d:?}");
            }
        }
    }

    #[test]
    fn cosine_mean_direction_tilts_toward_normal() {
        // Averaging cosine-weighted samples should give a vector pointing clearly
        // along +Z (the mean of a cosine lobe is 2/3), unlike a uniform
        // hemisphere whose mean z is 1/2.
        let n = 4096u32;
        let mut sz = 0.0f64;
        for i in 0..n {
            let (u, v) = super::super::sobol::sample_2d(i, 0x1234);
            sz += cosine_hemisphere(u, v)[2] as f64;
        }
        let mean_z = sz / n as f64;
        assert!(mean_z > 0.6 && mean_z < 0.72, "mean z = {mean_z}");
    }

    #[test]
    fn is_deterministic() {
        assert_eq!(concentric_disk(0.3, 0.7), concentric_disk(0.3, 0.7));
        assert_eq!(cosine_hemisphere(0.3, 0.7), cosine_hemisphere(0.3, 0.7));
    }
}
