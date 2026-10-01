//! Spherical-Gaussian (SG) lobes for glossy GI reconstruction — CPU golden.
//!
//! A low-order L1 spherical-harmonic probe (see [`super::radiance_cache`])
//! captures diffuse irradiance well but blurs away all directional, glossy
//! detail: it simply cannot represent a sharp highlight.  Modern Lumen-/
//! Frostbite-style pipelines therefore keep a small *mixture of spherical
//! Gaussians* alongside the SH probe to reconstruct view-dependent, moderately
//! glossy indirect specular.  A single spherical Gaussian is the spherical
//! analogue of a 3-D Gaussian lobe:
//!
//! ```text
//! G(v) = amplitude * exp( sharpness * (dot(v, axis) - 1) )
//! ```
//!
//! where `axis` is a unit mean direction, `sharpness` (`lambda`) controls the
//! angular width (large = tight lobe), and `amplitude` is a per-RGB peak value.
//! The subtraction of one normalises the exponent so the peak value is exactly
//! `amplitude` at `v == axis`.
//!
//! This module is the backend-neutral, CPU-golden reference for the SG algebra
//! the GPU twin mirrors.  All closed forms follow Wang et al. 2009
//! ("All-Frequency Rendering of Dynamic, Spatially-Varying Reflectance") and
//! the SG tutorials by Matt Pettineo and Stephen Hill:
//!
//! * [`SphericalGaussian::evaluate`] — point evaluation.
//! * [`SphericalGaussian::integral`] — closed-form integral over the sphere.
//! * [`SphericalGaussian::inner_product`] — closed-form integral of the product
//!   of two SGs (the workhorse for lighting an SG by another SG).
//! * [`SphericalGaussian::product`] — the product of two SGs is itself an SG.
//! * [`cosine_lobe_sg`] / [`SphericalGaussian::irradiance`] — a clamped-cosine
//!   lobe fitted as an SG, so diffuse irradiance from an SG light is a single
//!   inner product.
//! * [`ggx_specular_sg`] — an isotropic GGX NDF approximated as an SG about the
//!   reflection vector, driving the glossy reconstruction.
//!
//! # Conventions
//! * Directions are unit vectors; constructors normalise the axis and fall back
//!   to `+Z` for a degenerate (zero) input so evaluation never produces NaNs.
//! * `amplitude` is linear RGB stored as [`Vec3`] to match the GPU twin; the
//!   three channels share one axis and sharpness.
//! * `sharpness` is clamped to be strictly positive; a non-positive sharpness
//!   would make the lobe constant/degenerate and break the closed forms.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU.
//!   The `#[cfg(test)]` suite cross-checks each closed form against a fixed,
//!   RNG-free Fibonacci-sphere quadrature.

use bevy_math::{ops, Vec3};

/// Smallest sharpness a lobe may hold; keeps the closed forms well-conditioned.
const MIN_SHARPNESS: f32 = 1.0e-4;

/// A single RGB spherical-Gaussian lobe `A * exp(lambda * (dot(v, axis) - 1))`.
///
/// The lobe shares one unit `axis` and scalar `sharpness` across all three
/// colour channels carried in `amplitude`.  Construct via [`new`](Self::new)
/// so the axis is normalised and the sharpness clamped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphericalGaussian {
    /// Unit mean direction of the lobe.
    pub axis: Vec3,
    /// Angular sharpness `lambda`; larger values give a tighter lobe. Always
    /// `>= MIN_SHARPNESS`.
    pub sharpness: f32,
    /// Per-RGB peak amplitude reached at `v == axis`.
    pub amplitude: Vec3,
}

impl SphericalGaussian {
    /// Builds a lobe, normalising `axis` and clamping `sharpness` positive.
    ///
    /// A zero-length axis falls back to `+Z`; a sharpness below
    /// [`MIN_SHARPNESS`] is raised to it so the closed-form integrals stay
    /// finite and well-conditioned.
    #[inline]
    pub fn new(axis: Vec3, sharpness: f32, amplitude: Vec3) -> Self {
        Self {
            axis: normalize_or_z(axis),
            sharpness: sharpness.max(MIN_SHARPNESS),
            amplitude,
        }
    }

    /// Evaluates the lobe in direction `v` (need not be normalised).
    ///
    /// Returns the per-RGB value `amplitude * exp(sharpness * (dot(v, axis) -
    /// 1))`.  With a unit `v` the exponent lies in `[-2*sharpness, 0]`, so the
    /// result is bounded by `amplitude` and never overflows.
    #[inline]
    pub fn evaluate(&self, v: Vec3) -> Vec3 {
        let cos = normalize_or_z(v).dot(self.axis);
        self.amplitude * libm_exp(self.sharpness * (cos - 1.0))
    }

    /// Closed-form integral of the lobe over the unit sphere.
    ///
    /// `∫_S2 G(v) dv = amplitude * 2*pi/lambda * (1 - exp(-2*lambda))`.  For a
    /// tight lobe the `exp(-2*lambda)` term vanishes and the integral tends to
    /// `amplitude * 2*pi/lambda`.
    #[inline]
    pub fn integral(&self) -> Vec3 {
        let l = self.sharpness;
        let scalar = core::f32::consts::TAU / l * (1.0 - libm_exp(-2.0 * l));
        self.amplitude * scalar
    }

    /// The product of two SGs is a third SG (unnormalised Gaussian algebra).
    ///
    /// With `d = lambda1*axis1 + lambda2*axis2`, the product lobe has
    /// `sharpness = |d|`, `axis = d/|d|`, and
    /// `amplitude = A1 * A2 * exp(|d| - lambda1 - lambda2)`.  A degenerate zero
    /// `d` (exactly opposed equal lobes) falls back to a minimal lobe about
    /// `axis1` carrying the correctly attenuated amplitude.
    #[inline]
    pub fn product(&self, other: &Self) -> Self {
        let d = self.axis * self.sharpness + other.axis * other.sharpness;
        let len = d.length();
        let amp_scale = libm_exp(len - self.sharpness - other.sharpness);
        let amplitude = self.amplitude * other.amplitude * amp_scale;
        if len > MIN_SHARPNESS {
            Self {
                axis: d / len,
                sharpness: len,
                amplitude,
            }
        } else {
            Self {
                axis: self.axis,
                sharpness: MIN_SHARPNESS,
                amplitude,
            }
        }
    }

    /// Closed-form integral over the sphere of the product of two SGs.
    ///
    /// This is the SG "dot product" used to light one SG lobe by another:
    /// `∫_S2 G1(v) G2(v) dv`.  Following Wang 2009, with
    /// `dm = lambda1*axis1 + lambda2*axis2`, `len = |dm|`:
    ///
    /// ```text
    /// expo  = exp(len - lambda1 - lambda2)
    /// other = 1 - exp(-2*len)
    /// result = 2*pi * A1 * A2 * expo * other / len
    /// ```
    ///
    /// A degenerate `len -> 0` uses the limit `other/len -> 2`, giving
    /// `4*pi * A1 * A2 * expo`.
    #[inline]
    pub fn inner_product(&self, other: &Self) -> Vec3 {
        let dm = self.axis * self.sharpness + other.axis * other.sharpness;
        let len = dm.length();
        let expo = libm_exp(len - self.sharpness - other.sharpness);
        let amp = self.amplitude * other.amplitude;
        if len > MIN_SHARPNESS {
            let other_term = 1.0 - libm_exp(-2.0 * len);
            amp * (core::f32::consts::TAU * expo * other_term / len)
        } else {
            // lim_{len->0} (1 - exp(-2 len)) / len = 2.
            amp * (2.0 * core::f32::consts::TAU * expo)
        }
    }

    /// Diffuse irradiance received at a surface with the given `normal` when
    /// this lobe is the incident radiance distribution.
    ///
    /// Computed as the inner product of the lobe with the clamped-cosine lobe
    /// fitted as an SG by [`cosine_lobe_sg`].  The result is the cosine-weighted
    /// integral `∫ G(v) max(dot(v, n), 0) dv`, clamped per channel to be
    /// non-negative (the SG cosine fit can dip slightly below zero on the far
    /// hemisphere).  Divide by `pi` externally for a Lambertian albedo of one.
    #[inline]
    pub fn irradiance(&self, normal: Vec3) -> Vec3 {
        let cosine = cosine_lobe_sg(normal);
        self.inner_product(&cosine).max(Vec3::ZERO)
    }
}

/// Fits the clamped-cosine lobe `max(dot(v, n), 0)` as a single SG.
///
/// The canonical fit (Pettineo, "SG Series") uses `sharpness = 2.133` and
/// `amplitude = 1.17` about the surface normal; it reproduces the cosine lobe's
/// shape and, crucially, its hemispherical integral `pi` to within a few
/// percent, which is accurate enough for indirect diffuse.  All three RGB
/// channels carry the same scalar amplitude.
#[inline]
pub fn cosine_lobe_sg(normal: Vec3) -> SphericalGaussian {
    SphericalGaussian::new(normal, 2.133, Vec3::splat(1.17))
}

/// Approximates the isotropic GGX NDF as an SG about the reflection vector.
///
/// For reconstructing moderately glossy indirect specular, the GGX lobe seen
/// around the mirror reflection direction `reflect(-view, normal)` is fitted as
/// an SG.  Following the Frostbite/Karis mapping, roughness maps to sharpness
/// as `lambda = 2 / max(alpha^2 * (1 + dot(n, v)), eps)` with `alpha =
/// roughness^2`; the amplitude is normalised so the lobe integrates to one
/// (`amplitude = lambda / (2*pi * (1 - exp(-2*lambda)))`).  `roughness` is
/// clamped to `[eps, 1]` and `n_dot_v` to `[eps, 1]`.
#[inline]
pub fn ggx_specular_sg(reflection: Vec3, roughness: f32, n_dot_v: f32) -> SphericalGaussian {
    let alpha = roughness.clamp(1.0e-3, 1.0) * roughness.clamp(1.0e-3, 1.0);
    let nov = n_dot_v.clamp(1.0e-3, 1.0);
    let sharpness = (2.0 / (alpha * alpha * (1.0 + nov)).max(1.0e-4)).max(MIN_SHARPNESS);
    // Normalise so the lobe integrates to 1 over the sphere.
    let norm = sharpness / (core::f32::consts::TAU * (1.0 - libm_exp(-2.0 * sharpness)));
    SphericalGaussian::new(reflection, sharpness, Vec3::splat(norm))
}

/// Normalises `v`, falling back to `+Z` for a degenerate (zero) input.
#[inline]
fn normalize_or_z(v: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq > f32::MIN_POSITIVE {
        v * len_sq.sqrt().recip()
    } else {
        Vec3::Z
    }
}

/// `exp` routed through [`bevy_math::ops`] for cross-platform determinism,
/// matching the convention of the sibling shading modules in this crate.
#[inline]
fn libm_exp(x: f32) -> f32 {
    ops::exp(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic, RNG-free Fibonacci-lattice quadrature of the unit
    /// sphere: `n` points each carrying solid angle `4*pi/n`.  Summing
    /// `f(dir) * weight` approximates `∫_S2 f`.
    fn fibonacci_sphere(n: usize) -> Vec<Vec3> {
        let mut out = Vec::with_capacity(n);
        let golden = core::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
        for i in 0..n {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let theta = golden * i as f32;
            out.push(Vec3::new(r * theta.cos(), r * theta.sin(), z));
        }
        out
    }

    fn integrate<F: Fn(Vec3) -> Vec3>(n: usize, f: F) -> Vec3 {
        let w = 2.0 * core::f32::consts::TAU / n as f32; // 4*pi / n
        let mut acc = Vec3::ZERO;
        for d in fibonacci_sphere(n) {
            acc += f(d) * w;
        }
        acc
    }

    #[test]
    fn evaluate_peaks_at_axis_and_decays() {
        let sg = SphericalGaussian::new(Vec3::Z, 4.0, Vec3::splat(2.0));
        let peak = sg.evaluate(Vec3::Z);
        assert!((peak.x - 2.0).abs() < 1e-5, "peak {peak:?}");
        // Off-axis is strictly smaller, opposite pole is tiny.
        assert!(sg.evaluate(Vec3::X).x < peak.x);
        assert!(sg.evaluate(Vec3::NEG_Z).x < sg.evaluate(Vec3::X).x);
    }

    #[test]
    fn integral_matches_quadrature() {
        for &lambda in &[1.0f32, 4.0, 12.0] {
            let sg = SphericalGaussian::new(Vec3::new(0.3, -0.6, 0.74), lambda, Vec3::splat(1.0));
            let closed = sg.integral().x;
            let numeric = integrate(4096, |d| sg.evaluate(d)).x;
            assert!(
                (closed - numeric).abs() < 2e-2 * closed.max(1.0),
                "lambda {lambda}: closed {closed} vs numeric {numeric}"
            );
        }
    }

    #[test]
    fn inner_product_matches_quadrature() {
        let a = SphericalGaussian::new(Vec3::new(0.0, 0.0, 1.0), 6.0, Vec3::new(1.0, 0.5, 0.25));
        let b = SphericalGaussian::new(Vec3::new(0.2, 0.1, 0.97), 3.0, Vec3::splat(2.0));
        let closed = a.inner_product(&b);
        let numeric = integrate(8192, |d| a.evaluate(d) * b.evaluate(d));
        for c in 0..3 {
            let (cl, nu) = (closed[c], numeric[c]);
            assert!(
                (cl - nu).abs() < 3e-2 * cl.max(1.0),
                "channel {c}: closed {cl} vs numeric {nu}"
            );
        }
    }

    #[test]
    fn product_lobe_matches_pointwise_product() {
        let a = SphericalGaussian::new(Vec3::Z, 5.0, Vec3::splat(1.3));
        let b = SphericalGaussian::new(Vec3::new(0.3, 0.2, 0.9), 7.0, Vec3::splat(0.8));
        let p = a.product(&b);
        // The product SG must equal the pointwise product at several directions.
        for d in [Vec3::Z, Vec3::X, Vec3::new(0.1, 0.4, 0.9), p.axis] {
            let lhs = p.evaluate(d);
            let rhs = a.evaluate(d) * b.evaluate(d);
            assert!(
                (lhs.x - rhs.x).abs() < 1e-4,
                "dir {d:?}: {lhs:?} vs {rhs:?}"
            );
        }
    }

    #[test]
    fn irradiance_is_nonnegative_and_tracks_normal() {
        let sg = SphericalGaussian::new(Vec3::Z, 4.0, Vec3::splat(3.0));
        let facing = sg.irradiance(Vec3::Z);
        let grazing = sg.irradiance(Vec3::X);
        let away = sg.irradiance(Vec3::NEG_Z);
        assert!(facing.min_element() >= 0.0);
        assert!(away.min_element() >= 0.0);
        // A normal aligned with the lobe gathers the most irradiance.
        assert!(facing.x > grazing.x, "{facing:?} !> {grazing:?}");
        assert!(grazing.x > away.x, "{grazing:?} !> {away:?}");
    }

    #[test]
    fn cosine_lobe_integral_is_near_pi() {
        // The clamped-cosine lobe integrates to exactly pi over the sphere; the
        // single-SG fit reproduces that integral to within a few percent (the
        // canonical 1.17 / 2.133 fit lands near 3.40).
        let n = Vec3::Z;
        let cos = cosine_lobe_sg(n);
        // The closed-form integral must agree with the quadrature of the lobe.
        let numeric = integrate(8192, |d| Vec3::splat(cos.evaluate(d).x));
        let closed = cos.integral().x;
        assert!(
            (numeric.x - closed).abs() < 2e-2 * closed,
            "integral parity: numeric {} vs closed {closed}",
            numeric.x
        );
        // And that integral sits close to the reference value pi.
        assert!(
            (closed - core::f32::consts::PI).abs() < 0.4,
            "cosine fit integral {closed} vs pi"
        );
    }

    #[test]
    fn ggx_sg_sharpens_with_smoothness() {
        let r = Vec3::Z;
        let rough = ggx_specular_sg(r, 0.6, 0.8);
        let smooth = ggx_specular_sg(r, 0.1, 0.8);
        assert!(
            smooth.sharpness > rough.sharpness,
            "smoother surface must give a tighter lobe: {} vs {}",
            smooth.sharpness,
            rough.sharpness
        );
        // Normalised lobe integrates to ~1.
        let numeric = integrate(16384, |d| Vec3::splat(smooth.evaluate(d).x)).x;
        assert!((numeric - 1.0).abs() < 0.1, "ggx sg integral {numeric}");
    }

    #[test]
    fn constructors_are_defensive() {
        let degenerate = SphericalGaussian::new(Vec3::ZERO, -5.0, Vec3::ONE);
        assert_eq!(degenerate.axis, Vec3::Z);
        assert!(degenerate.sharpness >= MIN_SHARPNESS);
        // Evaluation never yields NaN even for a zero query direction.
        assert!(degenerate.evaluate(Vec3::ZERO).x.is_finite());
    }

    #[test]
    fn results_are_deterministic() {
        let a = SphericalGaussian::new(Vec3::new(0.1, 0.2, 0.97), 5.0, Vec3::splat(1.0));
        let b = SphericalGaussian::new(Vec3::new(-0.3, 0.4, 0.86), 2.0, Vec3::splat(0.7));
        assert_eq!(a.inner_product(&b), a.inner_product(&b));
        assert_eq!(a.product(&b), a.product(&b));
        assert_eq!(a.irradiance(Vec3::Y), a.irradiance(Vec3::Y));
    }
}
