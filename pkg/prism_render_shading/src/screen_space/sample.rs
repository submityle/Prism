//! Deterministic GGX importance sampling for multi-ray screen-space
//! reflections.
//!
//! A single mirror ray only matches a perfectly smooth surface.  To approximate
//! the GGX specular lobe on rougher surfaces the tracer shoots several rays per
//! pixel, each perturbed off the mirror direction by importance-sampling the
//! GGX normal-distribution function, marches every one, and averages the hits.
//!
//! Reproducibility matters as much as quality: the sample directions are driven
//! by a fixed low-discrepancy [`hammersley`] sequence (not a per-frame RNG) so
//! the CPU golden and the `ssr.wesl` twin pick byte-identical directions for a
//! given `(sample_index, sample_count, roughness, normal)`.  Every
//! transcendental routes through [`bevy_math::ops`] for cross-platform
//! determinism.

use bevy_math::{ops, Vec2, Vec3};
use core::f32::consts::PI;

/// Van der Corput radical inverse in base 2.
///
/// Reverses the bits of `bits` and scales into `[0, 1)`, the second dimension
/// of the [`hammersley`] set.  Mirrors the classic `bitfieldReverse` GPU idiom
/// so the WESL twin (which reconstructs the reversal with explicit shifts)
/// agrees bit-for-bit.
pub fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = bits.rotate_left(16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xaaaa_aaaa) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xcccc_cccc) >> 2);
    bits = ((bits & 0x0f0f_0f0f) << 4) | ((bits & 0xf0f0_f0f0) >> 4);
    bits = ((bits & 0x00ff_00ff) << 8) | ((bits & 0xff00_ff00) >> 8);
    (bits as f32) * 2.328_306_4e-10 // 1 / 2^32
}

/// The `i`-th point of the `n`-point Hammersley set on the unit square.
///
/// `x` is the stratified fraction `i / n` and `y` is the radical inverse, giving
/// a low-discrepancy 2D sequence with far less clumping than white noise.
/// `n == 0` degenerates to the origin.
pub fn hammersley(i: u32, n: u32) -> Vec2 {
    let x = if n == 0 { 0.0 } else { i as f32 / n as f32 };
    Vec2::new(x, radical_inverse_vdc(i))
}

/// Builds a right-handed orthonormal basis whose `z` axis is the unit `normal`.
///
/// Uses Duff et al. (2017) "Building an Orthonormal Basis, Revisited", which is
/// branch-light (a single sign copy) and numerically stable everywhere except
/// the exact `-z` pole, which the `sign`-based formulation still handles.
/// Returns `(tangent, bitangent)`.
pub fn orthonormal_basis(normal: Vec3) -> (Vec3, Vec3) {
    let sign = if normal.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + normal.z);
    let b = normal.x * normal.y * a;
    let tangent = Vec3::new(1.0 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    let bitangent = Vec3::new(b, sign + normal.y * normal.y * a, -normal.y);
    (tangent, bitangent)
}

/// Importance-samples a GGX half-vector around `normal` for `roughness`.
///
/// `xi` is a 2D sample in `[0, 1)^2` (typically from [`hammersley`]).  The
/// perceptual `roughness` is squared into the GGX `alpha` (the same
/// remap the BRDF uses), the polar angle is drawn from the GGX NDF and the
/// azimuth uniformly, then the tangent-space half-vector is rotated into view
/// space through [`orthonormal_basis`].  The returned vector is unit length and
/// lies in the hemisphere around `normal`.
pub fn importance_sample_ggx(xi: Vec2, roughness: f32, normal: Vec3) -> Vec3 {
    let alpha = (roughness * roughness).max(1.0e-4);
    let phi = 2.0 * PI * xi.x;
    // GGX NDF inverse-CDF for the half-vector polar angle.
    let cos_theta = ops::sqrt(((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y)).max(0.0));
    let sin_theta = ops::sqrt((1.0 - cos_theta * cos_theta).max(0.0));
    let h_tangent = Vec3::new(sin_theta * ops::cos(phi), sin_theta * ops::sin(phi), cos_theta);

    let n = normal.normalize_or_zero();
    if n == Vec3::ZERO {
        return Vec3::ZERO;
    }
    let (tangent, bitangent) = orthonormal_basis(n);
    (tangent * h_tangent.x + bitangent * h_tangent.y + n * h_tangent.z).normalize_or_zero()
}

/// The GGX NDF `D(h)` for perceptual `roughness` and half-angle cosine
/// `n_dot_h`.
///
/// Uses the Trowbridge-Reitz form with the perceptual-to-alpha remap; the
/// denominator is clamped away from zero so a grazing half-vector cannot
/// produce a non-finite density.
pub fn ggx_ndf(n_dot_h: f32, roughness: f32) -> f32 {
    let alpha = (roughness * roughness).max(1.0e-4);
    let a2 = alpha * alpha;
    let nh = n_dot_h.max(0.0);
    let d = nh * nh * (a2 - 1.0) + 1.0;
    a2 / (PI * d * d).max(1.0e-8)
}

/// Smith GGX height-correlated visibility term `V = G / (4 NoV NoL)`.
///
/// The BRDF weight applied to each importance sample so rougher lobes and
/// grazing rays are down-weighted exactly as the analytic specular integral
/// demands.  Clamps the cosines away from zero to keep the ratio finite.
pub fn smith_ggx_visibility(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let alpha = (roughness * roughness).max(1.0e-4);
    let a2 = alpha * alpha;
    let nv = n_dot_v.max(1.0e-4);
    let nl = n_dot_l.max(1.0e-4);
    let lambda_v = nl * ops::sqrt(nv * nv * (1.0 - a2) + a2);
    let lambda_l = nv * ops::sqrt(nl * nl * (1.0 - a2) + a2);
    0.5 / (lambda_v + lambda_l).max(1.0e-8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radical_inverse_matches_known_prefix() {
        // Classic van der Corput base-2 prefix: 0, 1/2, 1/4, 3/4, 1/8, ...
        assert!((radical_inverse_vdc(0) - 0.0).abs() < 1.0e-6);
        assert!((radical_inverse_vdc(1) - 0.5).abs() < 1.0e-6);
        assert!((radical_inverse_vdc(2) - 0.25).abs() < 1.0e-6);
        assert!((radical_inverse_vdc(3) - 0.75).abs() < 1.0e-6);
        assert!((radical_inverse_vdc(4) - 0.125).abs() < 1.0e-6);
    }

    #[test]
    fn hammersley_is_stratified_on_x_and_bounded() {
        let n = 16u32;
        for i in 0..n {
            let p = hammersley(i, n);
            assert!((p.x - i as f32 / n as f32).abs() < 1.0e-6);
            assert!((0.0..1.0).contains(&p.y));
        }
        assert_eq!(hammersley(3, 0), Vec2::new(0.0, radical_inverse_vdc(3)));
    }

    #[test]
    fn orthonormal_basis_is_orthonormal_for_varied_normals() {
        for n in [
            Vec3::Z,
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(1.0, 2.0, 3.0).normalize(),
            Vec3::new(-0.3, 0.7, -0.2).normalize(),
        ] {
            let (t, b) = orthonormal_basis(n);
            assert!((t.length() - 1.0).abs() < 1.0e-4, "tangent unit");
            assert!((b.length() - 1.0).abs() < 1.0e-4, "bitangent unit");
            assert!(t.dot(n).abs() < 1.0e-4, "t ⟂ n");
            assert!(b.dot(n).abs() < 1.0e-4, "b ⟂ n");
            assert!(t.dot(b).abs() < 1.0e-4, "t ⟂ b");
        }
    }

    #[test]
    fn ggx_sample_collapses_to_the_normal_as_roughness_vanishes() {
        // A near-mirror surface must return the normal itself for every sample.
        let n = Vec3::new(0.2, 0.3, 0.9).normalize();
        for i in 0..8u32 {
            let h = importance_sample_ggx(hammersley(i, 8), 1.0e-3, n);
            assert!((h - n).length() < 1.0e-2, "sample {i} should hug the normal");
        }
    }

    #[test]
    fn ggx_samples_stay_in_the_upper_hemisphere() {
        let n = Vec3::new(-0.1, 0.8, 0.5).normalize();
        for i in 0..32u32 {
            let h = importance_sample_ggx(hammersley(i, 32), 0.5, n);
            assert!(h.dot(n) >= -1.0e-4, "half-vector {i} left the hemisphere");
            assert!((h.length() - 1.0).abs() < 1.0e-3);
        }
    }

    #[test]
    fn ndf_and_visibility_are_finite_and_nonnegative() {
        for &r in &[0.05_f32, 0.3, 0.6, 1.0] {
            let d = ggx_ndf(0.7, r);
            let v = smith_ggx_visibility(0.6, 0.4, r);
            assert!(d.is_finite() && d >= 0.0);
            assert!(v.is_finite() && v >= 0.0);
        }
        // Grazing angles must not blow up.
        assert!(smith_ggx_visibility(0.0, 0.0, 0.5).is_finite());
    }
}
