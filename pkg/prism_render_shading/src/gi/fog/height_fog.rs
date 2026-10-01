//! Closed-form exponential height fog optical depth (CPU golden reference).
//!
//! Exponential height fog models an atmosphere whose extinction density decays
//! exponentially with altitude:
//!
//! ```text
//! ρ(h) = density · exp(-falloff · (h - base_height))
//! ```
//!
//! where `h` is the altitude along the chosen world up axis, `density` is the
//! extinction at `base_height`, and `falloff` controls how quickly the fog
//! thins with height.  To shade a surface we need the *optical depth*
//! `τ = ∫ ρ ds` accumulated along the view ray from the camera to the surface,
//! which this module evaluates in closed form (no ray marching):
//!
//! ```text
//! τ = ∫_0^t ρ(h₀ + s·u) ds
//!   = ρ(h₀) · (1 - exp(-b·t)) / b,     b = falloff · u,  u ≠ 0
//!   = ρ(h₀) · t,                        b → 0 (horizontal ray / zero falloff)
//! ```
//!
//! Here `h₀` is the altitude of the ray start (the camera), `u` is the up-axis
//! component of the (unit) view direction, and `t` is the ray length.  The
//! `b → 0` limit recovers a constant-density slab, so a horizontal ray or a
//! zero `falloff` both reduce to `ρ · t` continuously.  The transmittance is
//! then `T = exp(-τ)` and the fog coverage is `1 - T`.
//!
//! The world up axis is the `+Z` axis: altitude is `position.z` and the ray's
//! up component is `direction.z`.  Scalar helpers accept the altitude and up
//! component directly for callers that use a different convention.
//!
//! # Conventions
//! * `density`, `falloff`, and `distance` are clamped non-negative; a
//!   degenerate (negative / non-finite) input falls back to the fog-free
//!   result (`τ = 0`, `T = 1`).
//! * The up component `u` may be any sign: `u > 0` climbs into thinner fog,
//!   `u < 0` descends into denser fog, `u ≈ 0` is the constant-density limit.
//! * Optical depth is clamped to `[0, MAX_OPTICAL_DEPTH]` and every exponent is
//!   bounded before [`bevy_math::ops::exp`], so transmittance stays in `(0, 1]`
//!   and no `NaN`/`inf` escapes.
//! * Transcendental maths goes through [`bevy_math::ops`]; `sqrt` (when needed)
//!   uses the value method.  Every function is a deterministic pure function:
//!   no RNG, no I/O, no GPU, no `unsafe`.

use bevy_math::{ops, Vec3};

/// Largest optical depth the integrator will report.
///
/// `exp(-80)` is already far below `f32` denormal relevance, so clamping here
/// keeps the transmittance meaningful while guaranteeing a finite exponent.
const MAX_OPTICAL_DEPTH: f32 = 80.0;

/// Threshold below which `|b| = |falloff · u|` is treated as the constant
/// density limit, selecting the `τ = ρ · t` branch to avoid a `0/0` form.
const B_EPSILON: f32 = 1.0e-6;

/// Result of integrating height fog along a finite view ray segment.
///
/// All fields are finite; `transmittance` and `fog_amount` are complementary
/// values in `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FogIntegral {
    /// Accumulated optical depth `τ = ∫ ρ ds`, clamped to
    /// `[0, MAX_OPTICAL_DEPTH]`.
    pub optical_depth: f32,
    /// Transmittance `T = exp(-τ)` in `(0, 1]`.
    pub transmittance: f32,
    /// Fog coverage `1 - T` in `[0, 1)`.
    pub fog_amount: f32,
}

/// Exponential height fog parameters mirroring the GPU twin layout.
///
/// Fields are stored as `f32` in declaration order to match the shader-side
/// uniform; see the module docs for the density model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightFog {
    /// Extinction density `ρ` at `base_height` (per unit length). Clamped
    /// non-negative on use.
    pub density: f32,
    /// Exponential altitude falloff `k` (per unit length). Clamped
    /// non-negative; `0` yields a constant-density slab.
    pub falloff: f32,
    /// Reference altitude at which `ρ = density`.
    pub base_height: f32,
}

impl Default for HeightFog {
    #[inline]
    fn default() -> Self {
        Self {
            density: 0.02,
            base_height: 0.0,
            falloff: 0.1,
        }
    }
}

impl HeightFog {
    /// Creates a height fog, sanitising `density` and `falloff` to be
    /// non-negative and finite.
    #[inline]
    pub fn new(density: f32, falloff: f32, base_height: f32) -> Self {
        Self {
            density: clamp_non_negative(density),
            falloff: clamp_non_negative(falloff),
            base_height: finite_or(base_height, 0.0),
        }
    }

    /// Returns the extinction density `ρ(h)` at altitude `h`.
    ///
    /// Equals `density · exp(-falloff · (h - base_height))`. The exponent is
    /// bounded so the result stays finite even far below `base_height`.
    #[inline]
    pub fn density_at(&self, height: f32) -> f32 {
        let density = clamp_non_negative(self.density);
        let falloff = clamp_non_negative(self.falloff);
        let height = finite_or(height, self.base_height);
        // ρ(h) = density · exp(-falloff·(h - base_height)); bound the exponent.
        let exponent = (-falloff * (height - self.base_height)).clamp(-MAX_OPTICAL_DEPTH, MAX_OPTICAL_DEPTH);
        let value = density * ops::exp(exponent);
        clamp_non_negative(value)
    }

    /// Optical depth along a ray described by its start altitude, up-axis
    /// direction component, and length.
    ///
    /// Evaluates the closed-form integral
    /// `τ = ρ(h₀) · (1 - exp(-b·t)) / b` with `b = falloff · up`, falling back
    /// to the constant-density `τ = ρ(h₀) · t` when `|b|` is negligible. The
    /// result is clamped to `[0, MAX_OPTICAL_DEPTH]`.
    #[inline]
    pub fn optical_depth_axis(&self, start_height: f32, up: f32, distance: f32) -> f32 {
        let distance = clamp_non_negative(distance);
        if distance <= 0.0 {
            return 0.0;
        }
        let falloff = clamp_non_negative(self.falloff);
        let up = finite_or(up, 0.0);
        let rho0 = self.density_at(start_height);
        if rho0 <= 0.0 {
            return 0.0;
        }

        let b = falloff * up;
        let integral = if b.abs() > B_EPSILON {
            // ∫_0^t exp(-b·s) ds = (1 - exp(-b·t)) / b.
            // Bound the exponent so a descending ray cannot overflow `exp`.
            let exponent = (-b * distance).clamp(-MAX_OPTICAL_DEPTH, MAX_OPTICAL_DEPTH);
            (1.0 - ops::exp(exponent)) / b
        } else {
            // b → 0: the integrand tends to 1, so the integral tends to `t`.
            distance
        };

        let tau = rho0 * integral;
        if tau.is_finite() {
            tau.clamp(0.0, MAX_OPTICAL_DEPTH)
        } else {
            MAX_OPTICAL_DEPTH
        }
    }

    /// Optical depth along a world-space ray with `+Z` as the up axis.
    ///
    /// Uses `camera.z` as the start altitude and `direction.z` as the up
    /// component. `direction` need not be unit length; its `z` component is
    /// used verbatim, so pass a normalised direction for a physical path
    /// length in `distance`.
    #[inline]
    pub fn optical_depth(&self, camera: Vec3, direction: Vec3, distance: f32) -> f32 {
        let camera_z = finite_or(camera.z, self.base_height);
        let dir_z = finite_or(direction.z, 0.0);
        self.optical_depth_axis(camera_z, dir_z, distance)
    }

    /// Integrates height fog along a `+Z`-up world ray, returning optical
    /// depth, transmittance, and fog coverage together.
    #[inline]
    pub fn integrate(&self, camera: Vec3, direction: Vec3, distance: f32) -> FogIntegral {
        let tau = self.optical_depth(camera, direction, distance);
        Self::integral_from_tau(tau)
    }

    /// Integrates height fog from a start altitude / up component, returning
    /// optical depth, transmittance, and fog coverage.
    #[inline]
    pub fn integrate_axis(&self, start_height: f32, up: f32, distance: f32) -> FogIntegral {
        let tau = self.optical_depth_axis(start_height, up, distance);
        Self::integral_from_tau(tau)
    }

    /// Builds a [`FogIntegral`] from a (already clamped) optical depth.
    #[inline]
    fn integral_from_tau(tau: f32) -> FogIntegral {
        let tau = if tau.is_finite() {
            tau.clamp(0.0, MAX_OPTICAL_DEPTH)
        } else {
            MAX_OPTICAL_DEPTH
        };
        let transmittance = ops::exp(-tau).clamp(0.0, 1.0);
        FogIntegral {
            optical_depth: tau,
            transmittance,
            fog_amount: (1.0 - transmittance).clamp(0.0, 1.0),
        }
    }
}

/// Clamps `value` to be non-negative and finite (non-finite → `0`).
#[inline]
fn clamp_non_negative(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

/// Returns `value` when finite, otherwise the supplied `fallback`.
#[inline]
fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trapezoidal numerical integration of `ρ(h₀ + s·u)` over `s ∈ [0, t]`.
    fn trapezoid_optical_depth(fog: &HeightFog, start_height: f32, up: f32, distance: f32) -> f32 {
        let steps = 20_000usize;
        let dt = distance / steps as f32;
        let mut acc = 0.0f64;
        for i in 0..=steps {
            let s = i as f32 * dt;
            let w = if i == 0 || i == steps { 0.5 } else { 1.0 };
            acc += (w * fog.density_at(start_height + s * up)) as f64;
        }
        (acc * dt as f64) as f32
    }

    #[test]
    fn horizontal_ray_is_constant_density_integration() {
        let fog = HeightFog::new(0.05, 0.2, 1.0);
        let h0 = 3.0;
        let dist = 42.0;
        // Up component zero → constant density ρ(h0) along the whole ray.
        let tau = fog.optical_depth_axis(h0, 0.0, dist);
        let expected = fog.density_at(h0) * dist;
        assert!((tau - expected).abs() < 1e-4, "tau={tau} expected={expected}");
    }

    #[test]
    fn vertical_ray_matches_closed_form_and_numeric() {
        let fog = HeightFog::new(0.08, 0.15, 0.0);
        let h0 = 0.5;
        let dist = 30.0;
        for up in [1.0f32, -1.0] {
            let tau = fog.optical_depth_axis(h0, up, dist);
            // Closed form reference computed independently.
            let b = fog.falloff * up;
            let closed = fog.density_at(h0) * (1.0 - ops::exp(-b * dist)) / b;
            assert!((tau - closed).abs() < 1e-3, "up={up} tau={tau} closed={closed}");
            let numeric = trapezoid_optical_depth(&fog, h0, up, dist);
            assert!(
                (tau - numeric).abs() < 2e-3,
                "up={up} tau={tau} numeric={numeric}"
            );
        }
    }

    #[test]
    fn slanted_ray_matches_trapezoid() {
        // Gentle parameters keep the optical depth well below the clamp so
        // the closed form and the trapezoid agree on both ascending and
        // descending rays.
        let fog = HeightFog::new(0.03, 0.05, 2.0);
        let h0 = 20.0;
        let dist = 30.0;
        for up in [-0.7f32, -0.3, 0.2, 0.6, 0.9] {
            let tau = fog.optical_depth_axis(h0, up, dist);
            let numeric = trapezoid_optical_depth(&fog, h0, up, dist);
            assert!(
                (tau - numeric).abs() < 3e-3,
                "up={up} tau={tau} numeric={numeric}"
            );
        }
    }

    #[test]
    fn zero_falloff_degenerates_to_constant_density() {
        let fog = HeightFog::new(0.07, 0.0, 10.0);
        let dist = 25.0;
        // With falloff = 0 the density is `density` everywhere, so τ = density·t
        // regardless of altitude or ray slope.
        for up in [-1.0f32, -0.4, 0.0, 0.5, 1.0] {
            for h0 in [-5.0f32, 0.0, 12.0] {
                let tau = fog.optical_depth_axis(h0, up, dist);
                let expected = fog.density * dist;
                assert!(
                    (tau - expected).abs() < 1e-4,
                    "up={up} h0={h0} tau={tau} expected={expected}"
                );
            }
        }
    }

    #[test]
    fn transmittance_complements_fog_amount() {
        let fog = HeightFog::new(0.1, 0.2, 0.0);
        let integral = fog.integrate_axis(1.0, 0.5, 40.0);
        assert!((0.0..=1.0).contains(&integral.transmittance));
        assert!((0.0..=1.0).contains(&integral.fog_amount));
        assert!((integral.transmittance + integral.fog_amount - 1.0).abs() < 1e-6);
        assert!((ops::exp(-integral.optical_depth) - integral.transmittance).abs() < 1e-6);
    }

    #[test]
    fn vec3_api_uses_z_as_up() {
        let fog = HeightFog::new(0.05, 0.1, 0.0);
        let camera = Vec3::new(100.0, -50.0, 2.0);
        let direction = Vec3::new(0.3, 0.4, 0.5);
        let via_vec = fog.optical_depth(camera, direction, 20.0);
        let via_axis = fog.optical_depth_axis(2.0, 0.5, 20.0);
        assert!((via_vec - via_axis).abs() < 1e-6);
    }

    #[test]
    fn descending_ray_saturates_without_overflow() {
        let fog = HeightFog::new(1.0, 2.0, 0.0);
        // Deep descent into exponentially denser fog must saturate, not blow up.
        let tau = fog.optical_depth_axis(0.0, -1.0, 1000.0);
        assert!(tau.is_finite());
        assert!(tau <= MAX_OPTICAL_DEPTH + 1e-3, "tau={tau}");
        let integral = fog.integrate_axis(0.0, -1.0, 1000.0);
        assert!(integral.transmittance >= 0.0 && integral.transmittance <= 1.0);
    }

    #[test]
    fn degenerate_inputs_fall_back_to_fog_free() {
        let fog = HeightFog::new(f32::NAN, -1.0, f32::INFINITY);
        // All density/falloff sanitised; distance guards zero / negative.
        assert_eq!(fog.optical_depth_axis(0.0, 1.0, 0.0), 0.0);
        assert_eq!(fog.optical_depth_axis(0.0, 1.0, -5.0), 0.0);
        let integral = fog.integrate_axis(f32::NAN, f32::INFINITY, f32::NAN);
        assert!(integral.optical_depth.is_finite());
        assert!(integral.transmittance.is_finite());
        assert!(integral.fog_amount.is_finite());
    }

    #[test]
    fn density_at_is_finite_everywhere() {
        let fog = HeightFog::new(0.5, 1.5, 0.0);
        for h in [-1000.0f32, -10.0, 0.0, 10.0, 1000.0, f32::NAN] {
            let d = fog.density_at(h);
            assert!(d.is_finite() && d >= 0.0, "h={h} d={d}");
        }
    }
}
