//! Anisotropic `GGX` (Trowbridge-Reitz) microfacet reflection.
//!
//! The isotropic lobe in [`crate::reference_pt::microfacet`] assumes the
//! micro-geometry is rotationally symmetric about the normal, so its highlight
//! is always a round blob. Real brushed metal, hair, vinyl records, and
//! machined surfaces instead have a *grain*: parallel grooves that stretch the
//! highlight into a streak perpendicular to the brushing direction. The
//! anisotropic `GGX` distribution captures that by giving the micro-slope its
//! own width along two orthogonal tangent axes, `alpha_x` and `alpha_y`.
//!
//! Everything is expressed in the surface's local shading frame, whose `+z`
//! axis is the macroscopic normal and whose `x`/`y` axes are the tangent and
//! bitangent that orient the grain. Three pieces mirror the isotropic core:
//!
//! - the distribution `D(h)` concentrates micro-normals around `+z` with an
//!   elliptical cross-section scaled by `(alpha_x, alpha_y)`;
//! - the Smith `Lambda` auxiliary and the height-correlated masking `G2` use the
//!   same elliptical slope so masking is consistent with the lobe shape;
//! - visible-normal (`VNDF`) sampling draws a half vector proportional to
//!   `D(h) * G1(wo) * max(0, wo·h)`, generalising Heitz's stretch/unstretch warp
//!   to independent `x`/`y` widths.
//!
//! The whole routine uses only `sqrt` and a rejection-sampled unit-disk point,
//! honouring the crate's "no `sin`/`cos`" determinism policy. When
//! `alpha_x == alpha_y` every formula collapses exactly to the isotropic lobe.

use super::microfacet::MIN_ALPHA;
use super::sampler::uniform_disk;
use super::{Vec3, EPS_LEN_SQ, INV_PI};

/// An anisotropic `GGX` microfacet distribution with independent widths
/// `alpha_x` and `alpha_y` along the local tangent and bitangent axes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GgxAnisotropic {
    /// `GGX` width along the local tangent (`x`) axis, clamped to [`MIN_ALPHA`].
    pub alpha_x: f32,
    /// `GGX` width along the local bitangent (`y`) axis, clamped to [`MIN_ALPHA`].
    pub alpha_y: f32,
}

impl GgxAnisotropic {
    /// Builds the distribution from explicit `alpha_x`/`alpha_y`, each clamped to
    /// at least [`MIN_ALPHA`].
    #[must_use]
    pub fn new(alpha_x: f32, alpha_y: f32) -> Self {
        Self {
            alpha_x: alpha_x.max(MIN_ALPHA),
            alpha_y: alpha_y.max(MIN_ALPHA),
        }
    }

    /// Builds the distribution from a perceptual `roughness` in `[0, 1]` and an
    /// `anisotropy` in `[0, 1]`, using the Disney / `UE` (Burley) remap.
    ///
    /// `aspect = sqrt(1 - 0.9 * anisotropy)` stretches the base width
    /// `alpha = roughness^2` into `alpha_x = alpha / aspect` (the elongated
    /// axis) and `alpha_y = alpha * aspect`. `anisotropy = 0` yields the
    /// isotropic `alpha_x = alpha_y = roughness^2`.
    #[must_use]
    pub fn from_roughness_anisotropy(roughness: f32, anisotropy: f32) -> Self {
        let r = roughness.clamp(0.0, 1.0);
        let alpha = r * r;
        let aniso = anisotropy.clamp(0.0, 1.0);
        // `0.9` keeps `aspect` strictly positive even at full anisotropy.
        let aspect = (1.0 - 0.9 * aniso).max(1.0e-4).sqrt();
        Self::new(alpha / aspect, alpha * aspect)
    }

    /// The anisotropic `GGX` normal distribution `D(h)` for a half vector given
    /// in the local shading frame (`h.z` is its cosine to the normal).
    ///
    /// Returns zero for a back-facing half vector (`h.z <= 0`).
    #[must_use]
    pub fn distribution(&self, h_local: Vec3) -> f32 {
        if h_local.z <= 0.0 {
            return 0.0;
        }
        let hx = h_local.x / self.alpha_x;
        let hy = h_local.y / self.alpha_y;
        let hz = h_local.z;
        // Elliptical quadratic form; the squared denominator keeps it positive.
        let q = hx * hx + hy * hy + hz * hz;
        INV_PI / (self.alpha_x * self.alpha_y * q * q)
    }

    /// The Smith `Lambda` auxiliary for a direction given in the local frame.
    ///
    /// Grazing directions (`w.z -> 0`) drive `Lambda -> infinity`; perfectly
    /// normal incidence (`w.z = 1`, `w.x = w.y = 0`) returns zero.
    fn lambda(&self, w_local: Vec3) -> f32 {
        let cz = w_local.z.abs();
        if cz >= 1.0 {
            return 0.0;
        }
        let ax = self.alpha_x * w_local.x;
        let ay = self.alpha_y * w_local.y;
        let numer = ax * ax + ay * ay;
        if numer <= 0.0 {
            return 0.0;
        }
        let ratio = numer / (cz * cz);
        0.5 * ((1.0 + ratio).sqrt() - 1.0)
    }

    /// The Smith single-direction masking term `G1(w)` in `[0, 1]` for a local
    /// direction.
    #[must_use]
    pub fn g1(&self, w_local: Vec3) -> f32 {
        1.0 / (1.0 + self.lambda(w_local))
    }

    /// The height-correlated Smith masking-shadowing term `G2(wo, wi)` for two
    /// local directions.
    #[must_use]
    pub fn g2(&self, wo_local: Vec3, wi_local: Vec3) -> f32 {
        1.0 / (1.0 + self.lambda(wo_local) + self.lambda(wi_local))
    }

    /// The solid-angle probability density of the reflected direction produced
    /// by [`Self::sample_half_vector`], given the local view direction and the
    /// local half vector.
    ///
    /// This is the visible-normal density folded through the reflection
    /// Jacobian `1 / (4 wo·h)`; the `wo·h` factor cancels, leaving
    /// `G1(wo) * D(h) / (4 * wo.z)`.
    #[must_use]
    pub fn reflection_pdf(&self, wo_local: Vec3, h_local: Vec3) -> f32 {
        if wo_local.z <= 0.0 {
            return 0.0;
        }
        self.g1(wo_local) * self.distribution(h_local) / (4.0 * wo_local.z)
    }

    /// Importance-samples a local-frame microfacet half vector for the local
    /// view direction `wo_local`, using Heitz's anisotropic `VNDF` warp.
    ///
    /// Returns [`None`] when the view direction is below the surface or the warp
    /// degenerates, so the caller terminates rather than producing a `NaN`.
    #[must_use]
    pub fn sample_half_vector(
        &self,
        wo_local: Vec3,
        rng: &mut super::sampler::Rng,
    ) -> Option<Vec3> {
        if wo_local.z <= 0.0 {
            return None;
        }
        // 1. Stretch the view direction into the hemisphere configuration.
        let vh = Vec3::new(
            self.alpha_x * wo_local.x,
            self.alpha_y * wo_local.y,
            wo_local.z,
        )
        .normalize_or_zero();
        if vh.length_squared() <= EPS_LEN_SQ {
            return None;
        }
        // 2. Build an orthonormal basis around the stretched view direction.
        let lensq = vh.x * vh.x + vh.y * vh.y;
        let t1 = if lensq > EPS_LEN_SQ {
            Vec3::new(-vh.y, vh.x, 0.0).scale(1.0 / lensq.sqrt())
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };
        let t2 = vh.cross(t1);
        // 3. Sample a uniform disk point and warp it onto the projected cap.
        let (p1, p2_raw) = uniform_disk(rng);
        let s = 0.5 * (1.0 + vh.z);
        let p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * p2_raw;
        // 4. Lift to the hemisphere and un-stretch back to the GGX normal.
        let pz = (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt();
        let nh = t1.scale(p1).add(t2.scale(p2)).add(vh.scale(pz));
        let h_local =
            Vec3::new(self.alpha_x * nh.x, self.alpha_y * nh.y, nh.z.max(0.0)).normalize_or_zero();
        if h_local.length_squared() <= EPS_LEN_SQ {
            None
        } else {
            Some(h_local)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::microfacet::GgxIsotropic;
    use super::super::sampler::Rng;
    use super::super::PI;
    use super::*;

    /// A local direction from spherical-cap components, normalised.
    fn dir(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z).normalize_or_zero()
    }

    /// When `alpha_x == alpha_y` every anisotropic formula must match the
    /// isotropic lobe of the same width.
    #[test]
    fn reduces_to_isotropic_when_widths_match() {
        let alpha = 0.2;
        let aniso = GgxAnisotropic::new(alpha, alpha);
        let iso = GgxIsotropic { alpha };
        let h = dir(0.1, -0.2, 0.9);
        assert!((aniso.distribution(h) - iso.distribution(h.z)).abs() < 1e-4);
        let w = dir(0.3, 0.4, 0.8);
        assert!((aniso.g1(w) - iso.g1(w.z)).abs() < 1e-5);
        let wi = dir(-0.2, 0.1, 0.95);
        assert!((aniso.g2(w, wi) - iso.g2(w.z, wi.z)).abs() < 1e-5);
    }

    /// The anisotropic `NDF` integrates to one over the hemisphere (projected
    /// solid angle), the defining property of a normalised distribution.
    #[test]
    fn distribution_integrates_to_one() {
        let aniso = GgxAnisotropic::new(0.5, 0.15);
        let mut rng = Rng::seed(5);
        let count = 400_000u32;
        let mut sum = 0.0f64;
        for _ in 0..count {
            // Uniform hemisphere direction (pdf = 1/(2*pi)) by cube rejection.
            let h = loop {
                let v = Vec3::new(
                    2.0 * rng.next_f32() - 1.0,
                    2.0 * rng.next_f32() - 1.0,
                    2.0 * rng.next_f32() - 1.0,
                );
                let l2 = v.length_squared();
                if l2 > 1e-6 && l2 <= 1.0 {
                    let u = v.normalize_or_zero();
                    break if u.z < 0.0 { u.negate() } else { u };
                }
            };
            let d = f64::from(aniso.distribution(h));
            sum += d * f64::from(h.z) * (2.0 * f64::from(PI));
        }
        let integral = sum / f64::from(count);
        assert!(
            (integral - 1.0).abs() < 3e-2,
            "anisotropic NDF must integrate to one, got {integral}"
        );
    }

    /// A larger `alpha_x` than `alpha_y` widens the lobe along the tangent axis:
    /// an off-normal half vector tilted toward `x` is more probable than the
    /// same tilt toward `y`.
    #[test]
    fn wider_axis_has_a_broader_lobe() {
        let aniso = GgxAnisotropic::new(0.6, 0.1);
        let tilt_x = dir(0.4, 0.0, 0.9);
        let tilt_y = dir(0.0, 0.4, 0.9);
        assert!(
            aniso.distribution(tilt_x) > aniso.distribution(tilt_y),
            "tangent-axis tilt should be more probable for the wider axis"
        );
    }

    /// Sampled half vectors stay in the upper hemisphere and remain unit length.
    #[test]
    fn sampled_half_vectors_are_valid() {
        let aniso = GgxAnisotropic::new(0.5, 0.2);
        let mut rng = Rng::seed(99);
        let wo = dir(0.3, 0.1, 0.9);
        for _ in 0..20_000 {
            let h = aniso.sample_half_vector(wo, &mut rng).expect("valid half");
            assert!(h.z > 0.0, "half vector below surface");
            assert!((h.length() - 1.0).abs() < 1e-4, "half vector not unit");
        }
    }

    /// The analytic `reflection_pdf` is positive and finite for samples drawn by
    /// the matching sampler.
    #[test]
    fn sampling_density_is_positive_and_finite() {
        let aniso = GgxAnisotropic::new(0.4, 0.15);
        let mut rng = Rng::seed(321);
        let wo = dir(0.0, 0.0, 1.0);
        for _ in 0..20_000 {
            let h = aniso.sample_half_vector(wo, &mut rng).expect("valid half");
            let pdf = aniso.reflection_pdf(wo, h);
            assert!(pdf > 0.0 && pdf.is_finite(), "pdf must be positive finite");
        }
    }
}
