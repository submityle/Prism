//! Offline `CPU` reference path tracer (ground-truth integrator).
//!
//! This module is Prism's *unbiased* reference renderer: a classical
//! `Monte Carlo` path integrator that reuses the software `BVH` and triangle
//! intersection in [`crate::ray_scene`] as its geometry back end. It exists so
//! every real-time global-illumination path (screen-space probes, `ReSTIR`
//! `DI`/`GI`, surface cache, froxel volumetrics) has a single, deterministic
//! ground truth to converge against, exactly as production renderers validate
//! their real-time `GI` against an offline reference (`PBRT`, `Mitsuba`, the
//! Arnold / Cycles reference modes). It is not a real-time path; it is the
//! correctness oracle the real-time paths are measured against.
//!
//! The integrator is deliberately dependency-free and uses only classical
//! numerical methods (no `AI`/`ML`). The only transcendental allowed by the
//! workspace determinism policy is `sqrt`, so every sampling routine is
//! expressed without `sin`/`cos`/`exp`: directions are drawn with rejection
//! sampling plus Malley's method and transformed through a branchless
//! orthonormal basis (Duff et al., "Building an Orthonormal Basis, Revisited").
//!
//! Submodules:
//! - [`sampler`] — the deterministic `PCG` random source, stratified sample
//!   generation, cosine-weighted hemisphere sampling, and the orthonormal
//!   basis used to orient local samples around a surface normal.
//! - [`bsdf`] — the surface scattering models ([`bsdf::Bsdf`]): a Lambertian
//!   diffuse lobe, a perfect specular mirror, a physically based GGX rough
//!   conductor, and a smooth dielectric (glass/water), each exposing
//!   evaluate / sample / `pdf` with matching conventions.
//! - [`microfacet`] — the shared isotropic GGX (Trowbridge-Reitz) core: the
//!   normal distribution, the height-correlated Smith masking term, Schlick
//!   Fresnel, and visible-normal (`VNDF`) importance sampling (Heitz 2018).
//! - [`dielectric`] — the smooth-dielectric building blocks: the unpolarized
//!   `Fresnel` reflectance (including total internal reflection) and Snell's-law
//!   refraction used by the [`bsdf::Bsdf::Dielectric`] lobe.
//! - [`fresnel_blend`] — the Ashikhmin-Shirley coupled diffuse-specular
//!   reflector behind the [`bsdf::Bsdf::Plastic`] lobe, an energy-conserving
//!   dielectric coat over a diffuse substrate.
//! - [`estimator`] — next-event estimation ([`estimator::Light`]) for direct
//!   lighting from point, directional, and quad area lights with visibility
//!   tested against the scene `BVH`.
//! - [`integrator`] — the path-tracing loop ([`integrator::PathIntegrator`]):
//!   `BSDF` importance sampling, direct-light `NEE`, Russian-roulette path
//!   termination, and the [`integrator::Scene`] it traces.
//! - [`camera`] — the pinhole camera ([`camera::PinholeCamera`]) that
//!   generates primary rays through a virtual image plane.
//! - [`film`] — the [`film::Film`] framebuffer and the [`film::render`]
//!   driver that averages jittered primary rays into a reference image.
//! - [`compare`] — image-difference metrics ([`compare::ErrorMetrics`])
//!   for validating a candidate render against this reference.
//!
//! All public math is `f32`; floating-point equality is never tested directly
//! (an epsilon or ordering comparison is used instead), matching the crate's
//! determinism conventions.

pub mod bsdf;
pub mod camera;
pub mod compare;
pub mod conductor;
pub mod conductor_aniso;
pub mod conductor_aniso_ms;
pub mod conductor_ms;
pub mod conductor_schlick_ms;
pub mod dielectric;
pub mod dielectric_energy;
pub mod dielectric_ms;
pub mod estimator;
pub mod film;
pub mod fresnel_blend;
pub mod ggx_energy;
pub mod integrator;
pub mod metal;
pub mod microfacet;
pub mod microfacet_aniso;
pub mod oren_nayar;
pub mod rough_dielectric;
pub mod sampler;

/// Mathematical constant pi, reused from `core` so no literal drifts.
pub const PI: f32 = core::f32::consts::PI;

/// Reciprocal of pi, precomputed for the Lambertian normalization `albedo/pi`.
pub const INV_PI: f32 = 1.0 / core::f32::consts::PI;

/// Squared-length threshold below which a vector is treated as the zero vector,
/// so normalization never divides by (near) zero and never yields `NaN`.
pub const EPS_LEN_SQ: f32 = 1e-12;

/// Geometric ray-offset epsilon used to push a bounce/shadow ray origin off the
/// surface along the shading normal, avoiding self-intersection ("shadow acne").
pub const RAY_EPS: f32 = 1e-4;

/// A hand-rolled three-component vector.
///
/// `prism_render_architecture` is a dependency-free contracts crate and
/// [`crate::ray_scene`] speaks in `[f32; 3]` arrays, so the path-tracer math is
/// spelled out here (mirroring the per-subsystem `Vec3` in `cloth`/`water`)
/// rather than pulling in a linear-algebra dependency. Only `sqrt` is used.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// The all-ones vector (useful as a unit throughput / unit albedo).
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };

    /// Builds a vector from components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Builds a vector with all three components set to `s`.
    #[must_use]
    pub const fn splat(s: f32) -> Self {
        Self { x: s, y: s, z: s }
    }

    /// Builds a vector from a `[f32; 3]` array (the [`crate::ray_scene`] layout).
    #[must_use]
    pub const fn from_array(a: [f32; 3]) -> Self {
        Self {
            x: a[0],
            y: a[1],
            z: a[2],
        }
    }

    /// Converts to a `[f32; 3]` array (the [`crate::ray_scene`] layout).
    #[must_use]
    pub const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// Component-wise sum.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The path-tracer math API is specified with named add/sub/mul methods for call-site uniformity (mirroring the cloth/water Vec3); operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: the specified API uses named sub for call-site uniformity, not operator traits."
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Component-wise (Hadamard) product, used to attenuate a radiance/throughput
    /// by a spectral albedo.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: the Hadamard product is exposed as a named mul for call-site uniformity, not the Mul operator trait (which would read as a scalar or matrix product)."
    )]
    pub fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }

    /// Dot (inner) product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Cross product `self × rhs`.
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Negation `-self`.
    #[must_use]
    pub fn negate(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }

    /// Squared Euclidean length.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Squared distance between two points.
    #[must_use]
    pub fn distance_squared(self, rhs: Self) -> f32 {
        self.sub(rhs).length_squared()
    }

    /// Returns the unit vector along `self`, or [`Vec3::ZERO`] when `self` is
    /// (numerically) the zero vector, so normalization never yields `NaN`.
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > EPS_LEN_SQ {
            self.scale(1.0 / len_sq.sqrt())
        } else {
            Self::ZERO
        }
    }

    /// The largest of the three components (used by Russian roulette).
    #[must_use]
    pub fn max_component(self) -> f32 {
        self.x.max(self.y).max(self.z)
    }

    /// `true` when every component is finite (no `NaN`/infinity).
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Reflects `self` about a unit `normal` (mirror direction).
    ///
    /// Uses the standard `v - 2 (v·n) n`; `self` is treated as the incident
    /// direction pointing *into* the surface for the usual mirror convention.
    #[must_use]
    pub fn reflect(self, normal: Self) -> Self {
        self.sub(normal.scale(2.0 * self.dot(normal)))
    }

    /// Returns `self` flipped, if needed, so it lies in the same hemisphere as
    /// `reference` (i.e. their dot product is non-negative). Used to orient a
    /// shading normal toward the viewer.
    #[must_use]
    pub fn faced_toward(self, reference: Self) -> Self {
        if self.dot(reference) < 0.0 {
            self.negate()
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_cross_basic() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert!((a.dot(b)).abs() < EPS_LEN_SQ);
        let c = a.cross(b);
        assert!(c.sub(Vec3::new(0.0, 0.0, 1.0)).length_squared() < EPS_LEN_SQ);
    }

    #[test]
    fn normalize_zero_is_zero() {
        assert_eq!(Vec3::ZERO.normalize_or_zero(), Vec3::ZERO);
        let n = Vec3::new(3.0, 0.0, 4.0).normalize_or_zero();
        assert!((n.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn reflect_about_up_flips_vertical() {
        // Incident going down-and-forward reflects to up-and-forward.
        let incident = Vec3::new(1.0, -1.0, 0.0);
        let r = incident.reflect(Vec3::new(0.0, 1.0, 0.0));
        assert!(r.sub(Vec3::new(1.0, 1.0, 0.0)).length_squared() < 1e-10);
    }

    #[test]
    fn faced_toward_flips_when_opposed() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.0, -1.0, 0.0);
        assert_eq!(n.faced_toward(wo), Vec3::new(0.0, -1.0, 0.0));
        let wo2 = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(n.faced_toward(wo2), n);
    }

    #[test]
    fn array_round_trip() {
        let v = Vec3::new(1.5, -2.0, 3.25);
        assert_eq!(Vec3::from_array(v.to_array()), v);
    }
}
