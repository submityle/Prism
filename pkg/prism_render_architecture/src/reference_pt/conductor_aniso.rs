//! Anisotropic rough conductor: brushed and machined metals.
//!
//! [`crate::reference_pt::conductor`] pairs the exact complex-index `Fresnel`
//! term with an *isotropic* `GGX` lobe, so its highlight is always a round
//! blob. Brushed aluminium, satin steel, and the grooves on a machined bezel
//! instead stretch that highlight into a streak aligned with the grain. This
//! module reuses the same spectral `Fresnel` reflectance but swaps the lobe for
//! the anisotropic `GGX` distribution in
//! [`crate::reference_pt::microfacet_aniso`], giving the offline reference
//! tracer a brushed-metal oracle the real-time Cook-Torrance path can be
//! measured against.
//!
//! The anisotropy axes are bound to the surface's geometric tangent basis
//! (built from the shading normal), so this oracle validates the *`BRDF`
//! mathematics itself* — the elliptical `D`, the matching Smith masking, and
//! `VNDF` sampling — independently of any mesh `UV` parameterisation. The
//! real-time twin is expected to feed its own `UV`-derived tangent frame so the
//! streak follows the artist's grain direction; here the frame is canonical and
//! deterministic.
//!
//! All directions follow the tracer convention: `wo`, `wi`, and `normal` are
//! unit vectors with `wo`/`wi` pointing away from the surface. The lobe is
//! glossy (non-delta), so it is importance-sampled and connected to lights by
//! next-event estimation like any rough surface.

use super::conductor::fresnel_conductor;
use super::microfacet_aniso::GgxAnisotropic;
use super::sampler::{orthonormal_basis, Rng};
use super::{Vec3, EPS_LEN_SQ};

/// The outcome of importance-sampling an [`AnisoConductor`] lobe.
#[derive(Clone, Copy, Debug)]
pub struct AnisoConductorSample {
    /// The sampled outgoing direction (away from the surface), unit length.
    pub direction: Vec3,
    /// The `BRDF` value `f_r(wo, wi)` for the sampled pair (no cosine applied).
    pub value: Vec3,
    /// The solid-angle probability density of `direction`.
    pub pdf: f32,
}

/// A rough anisotropic conductor `BRDF` driven by the metal's measured complex
/// index `eta + i*k` and an anisotropic `GGX` microfacet lobe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnisoConductor {
    /// Per-channel real index of refraction `eta` of the metal.
    eta: Vec3,
    /// Per-channel extinction coefficient `k` of the metal.
    k: Vec3,
    /// The anisotropic `GGX` microfacet distribution of the specular lobe.
    dist: GgxAnisotropic,
}

/// A local-frame triad `(tangent, bitangent, normal)` oriented into the view
/// hemisphere, together with the pre-projected local view direction.
struct LocalFrame {
    /// The local `x` axis (tangent, carrying `alpha_x`).
    tangent: Vec3,
    /// The local `y` axis (bitangent, carrying `alpha_y`).
    bitangent: Vec3,
    /// The local `z` axis (viewer-facing shading normal).
    normal: Vec3,
}

impl LocalFrame {
    /// Builds the shading frame around a viewer-facing `normal`.
    fn new(normal: Vec3) -> Self {
        let (tangent, bitangent) = orthonormal_basis(normal);
        Self {
            tangent,
            bitangent,
            normal,
        }
    }

    /// Projects a world-space vector into the local shading frame.
    fn to_local(&self, v: Vec3) -> Vec3 {
        Vec3::new(
            v.dot(self.tangent),
            v.dot(self.bitangent),
            v.dot(self.normal),
        )
    }

    /// Lifts a local-frame vector back into world space.
    fn to_world(&self, v: Vec3) -> Vec3 {
        self.tangent
            .scale(v.x)
            .add(self.bitangent.scale(v.y))
            .add(self.normal.scale(v.z))
    }
}

impl AnisoConductor {
    /// Builds an anisotropic conductor from its complex index `eta + i*k` and
    /// explicit `GGX` widths `alpha_x`/`alpha_y` (each clamped to the lobe's
    /// minimum width).
    #[must_use]
    pub fn new(eta: Vec3, k: Vec3, alpha_x: f32, alpha_y: f32) -> Self {
        Self {
            eta,
            k,
            dist: GgxAnisotropic::new(alpha_x, alpha_y),
        }
    }

    /// Builds an anisotropic conductor from its complex index and a perceptual
    /// `roughness`/`anisotropy` pair, using the Disney / `UE` (Burley) remap in
    /// [`GgxAnisotropic::from_roughness_anisotropy`].
    #[must_use]
    pub fn from_roughness_anisotropy(eta: Vec3, k: Vec3, roughness: f32, anisotropy: f32) -> Self {
        Self {
            eta,
            k,
            dist: GgxAnisotropic::from_roughness_anisotropy(roughness, anisotropy),
        }
    }

    /// Evaluates the anisotropic rough-conductor `BRDF`
    /// `f_r = F * D * G2 / (4 cos_o cos_i)`.
    ///
    /// Returns [`Vec3::ZERO`] when either direction is below the surface or the
    /// half vector degenerates.
    #[must_use]
    pub fn evaluate(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> Vec3 {
        let cos_o = normal.dot(wo);
        let cos_i = normal.dot(wi);
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return Vec3::ZERO;
        }
        let frame = LocalFrame::new(normal);
        let wo_local = frame.to_local(wo);
        let wi_local = frame.to_local(wi);
        let half_local = wo_local.add(wi_local).normalize_or_zero();
        if half_local.length_squared() <= EPS_LEN_SQ || half_local.z <= 0.0 {
            return Vec3::ZERO;
        }
        let d = self.dist.distribution(half_local);
        let g2 = self.dist.g2(wo_local, wi_local);
        let woh = wo_local.dot(half_local).max(0.0);
        let fresnel = fresnel_conductor(self.eta, self.k, woh);
        fresnel.scale(d * g2 / (4.0 * cos_o * cos_i))
    }

    /// The solid-angle density [`Self::sample`] assigns to `(wo, wi)`.
    ///
    /// Zero when either direction is below the surface or the half vector
    /// degenerates.
    #[must_use]
    pub fn pdf(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> f32 {
        let cos_o = normal.dot(wo);
        let cos_i = normal.dot(wi);
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return 0.0;
        }
        let frame = LocalFrame::new(normal);
        let wo_local = frame.to_local(wo);
        let wi_local = frame.to_local(wi);
        let half_local = wo_local.add(wi_local).normalize_or_zero();
        if half_local.length_squared() <= EPS_LEN_SQ {
            return 0.0;
        }
        self.dist.reflection_pdf(wo_local, half_local)
    }

    /// Importance-samples an outgoing direction via anisotropic visible-normal
    /// (`VNDF`) sampling of the `GGX` lobe.
    ///
    /// The returned `value`/`pdf` are the true `BRDF` and its solid-angle
    /// density, so the integrator's generic throughput update
    /// `value * cos_i / pdf` reduces to the clean microfacet weight
    /// `F * G2 / G1(wo)`. Returns `None` for a degenerate (grazing/zero-length)
    /// sample so the caller terminates the path rather than dividing by zero.
    #[must_use]
    pub fn sample(&self, wo: Vec3, normal: Vec3, rng: &mut Rng) -> Option<AnisoConductorSample> {
        // The integrator passes the raw geometric normal; orient it into the
        // view hemisphere so a back-facing hit still scatters.
        let frame = LocalFrame::new(normal.faced_toward(wo));
        let wo_local = frame.to_local(wo);
        let cos_o = wo_local.z;
        if cos_o <= 0.0 {
            return None;
        }
        let half_local = self.dist.sample_half_vector(wo_local, rng)?;
        let woh = wo_local.dot(half_local);
        if woh <= 0.0 {
            return None;
        }
        // Reflect the view direction about the sampled microfacet normal.
        let wi_local = wo_local.negate().reflect(half_local).normalize_or_zero();
        let cos_i = wi_local.z;
        if cos_i <= 0.0 {
            return None;
        }
        let pdf = self.dist.reflection_pdf(wo_local, half_local);
        if pdf <= 0.0 {
            return None;
        }
        let d = self.dist.distribution(half_local);
        let g2 = self.dist.g2(wo_local, wi_local);
        let fresnel = fresnel_conductor(self.eta, self.k, woh);
        let value = fresnel.scale(d * g2 / (4.0 * cos_o * cos_i));
        Some(AnisoConductorSample {
            direction: frame.to_world(wi_local),
            value,
            pdf,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference_pt::conductor::Conductor;

    /// Red/green/blue complex index of gold, a canonical warm metal.
    const GOLD_ETA: Vec3 = Vec3 {
        x: 0.143,
        y: 0.375,
        z: 1.442,
    };
    /// Extinction coefficient of gold.
    const GOLD_K: Vec3 = Vec3 {
        x: 3.983,
        y: 2.386,
        z: 1.603,
    };

    const N: Vec3 = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    fn unit(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z).normalize_or_zero()
    }

    /// A sampled direction stays in the view hemisphere and reports the same
    /// value and density [`AnisoConductor::evaluate`]/[`AnisoConductor::pdf`]
    /// recompute for the same pair.
    #[test]
    fn sample_is_consistent_with_evaluate_and_pdf() {
        let conductor = AnisoConductor::new(GOLD_ETA, GOLD_K, 0.4, 0.1);
        let wo = unit(0.4, 0.9, 0.2);
        let mut rng = Rng::seed(42);
        for _ in 0..512 {
            if let Some(s) = conductor.sample(wo, N, &mut rng) {
                assert!(N.dot(s.direction) > 0.0, "sample below surface");
                let eval = conductor.evaluate(wo, s.direction, N);
                let pdf = conductor.pdf(wo, s.direction, N);
                // The sharp `GGX` lobe amplifies tiny half-vector round-off in
                // the recomputed path, so compare relative to the peak value.
                let val_tol = 1e-3 * s.value.length().max(1.0);
                let pdf_tol = 1e-3 * pdf.max(1.0);
                assert!(s.value.sub(eval).length() <= val_tol, "value mismatch");
                assert!((s.pdf - pdf).abs() <= pdf_tol, "pdf mismatch");
            }
        }
    }

    /// With equal widths the anisotropic conductor must match the isotropic
    /// [`Conductor`] of the same roughness for a fixed direction pair.
    #[test]
    fn reduces_to_isotropic_when_widths_match() {
        let roughness = 0.3;
        let alpha = roughness * roughness;
        let aniso = AnisoConductor::new(GOLD_ETA, GOLD_K, alpha, alpha);
        let iso = Conductor::new(GOLD_ETA, GOLD_K, roughness);
        let wo = unit(0.3, 0.9, 0.1);
        let wi = unit(-0.2, 0.95, 0.1);
        let a = aniso.evaluate(wo, wi, N);
        let b = iso.evaluate(wo, wi, N);
        assert!(a.sub(b).length() < 1e-3, "aniso {a:?} vs iso {b:?}");
    }

    /// White-furnace energy check: the single-scatter directional albedo of the
    /// anisotropic conductor never exceeds one.
    #[test]
    fn directional_albedo_never_exceeds_one() {
        let conductor = AnisoConductor::new(GOLD_ETA, GOLD_K, 0.5, 0.15);
        let wo = unit(0.3, 0.9, 0.0);
        let mut rng = Rng::seed(7);
        let samples = 40_000u32;
        let mut sum = 0.0f32;
        for _ in 0..samples {
            if let Some(s) = conductor.sample(wo, N, &mut rng) {
                let cos_i = N.dot(s.direction).max(0.0);
                sum += s.value.x * cos_i / s.pdf;
            }
        }
        let albedo = sum / samples as f32;
        assert!(albedo <= 1.0 + 1e-2, "furnace gained energy: {albedo}");
    }

    /// The brushed streak is directional: a tangent-plane probe aligned with the
    /// narrow axis produces a different reflectance than one aligned with the
    /// wide axis, confirming the lobe is genuinely anisotropic.
    #[test]
    fn highlight_depends_on_azimuth() {
        let conductor = AnisoConductor::new(GOLD_ETA, GOLD_K, 0.6, 0.08);
        let wo = unit(0.0, 0.9, 0.0);
        // Two outgoing directions tilted the same polar amount but 90 degrees
        // apart in azimuth around the normal.
        let wi_x = unit(0.5, 0.86, 0.0);
        let wi_z = unit(0.0, 0.86, 0.5);
        let fx = conductor.evaluate(wo, wi_x, N).length();
        let fz = conductor.evaluate(wo, wi_z, N).length();
        assert!(
            (fx - fz).abs() > 1e-3 * fx.max(fz).max(1e-3),
            "anisotropic highlight should vary with azimuth: {fx} vs {fz}"
        );
    }
}
