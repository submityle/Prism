//! Distance-based fog factors and colour compositing (CPU golden reference).
//!
//! Distance fog attenuates a shaded surface purely as a function of its
//! distance `d` from the camera, independent of altitude.  Three classic
//! factor curves are provided, each returning a *fog factor* `f ∈ [0, 1]`
//! (`0` = no fog, `1` = fully fogged):
//!
//! ```text
//! exponential:        f = 1 - exp(-d · density)
//! exponential-squared: f = 1 - exp(-(d · density)²)
//! linear:             f = clamp((d - start) / (end - start), 0, 1)
//! ```
//!
//! The exponential and exponential-squared curves derive from Beer-Lambert
//! extinction `T = exp(-τ)` with `f = 1 - T`; the squared variant ramps up more
//! sharply with distance (a denser near-field falloff popular for ground fog).
//! The linear curve is the artist-friendly `start`/`end` ramp.  The resulting
//! factor blends the scene colour toward the fog colour with a simple `lerp`:
//!
//! ```text
//! result = lerp(scene_color, fog_color, f)
//! ```
//!
//! # Conventions
//! * `distance`, `density`, `start`, and `end` are clamped non-negative and
//!   finite; a non-finite input falls back to the fog-free factor `0`.
//! * Every factor is clamped to `[0, 1]`, is monotonically non-decreasing in
//!   `distance`, and equals `0` at `distance = 0`.
//! * The linear ramp guards `end ≤ start`: a degenerate or inverted interval
//!   collapses to a hard step (fully fogged beyond `start`, clear before it).
//! * Exponents are bounded before [`bevy_math::ops::exp`] so no `NaN`/`inf`
//!   escapes; colour channels are sanitised to be finite and non-negative.
//! * Transcendental maths goes through [`bevy_math::ops`]. Every function is a
//!   deterministic pure function: no RNG, no I/O, no GPU, no `unsafe`.

use bevy_math::{ops, Vec3};

/// Largest optical-depth-like exponent fed to `exp`; `exp(-80)` underflows the
/// meaningful transmittance range, so clamping keeps the result finite.
const MAX_EXPONENT: f32 = 80.0;

/// Selects which distance-fog curve a [`DistanceFog`] evaluates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistanceFogMode {
    /// `f = 1 - exp(-d · density)`.
    Exponential,
    /// `f = 1 - exp(-(d · density)²)`.
    ExponentialSquared,
    /// `f = clamp((d - start) / (end - start), 0, 1)`.
    Linear,
}

impl Default for DistanceFogMode {
    #[inline]
    fn default() -> Self {
        Self::Exponential
    }
}

/// Distance-fog parameters mirroring the GPU twin layout.
///
/// `color` is linear-RGB; `density` drives the exponential curves while
/// `start`/`end` drive the linear ramp.  The active curve is chosen by `mode`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistanceFog {
    /// Linear-RGB fog colour the scene is blended toward.
    pub color: Vec3,
    /// Extinction density for the exponential curves (per unit length).
    pub density: f32,
    /// Near distance at which linear fog begins.
    pub start: f32,
    /// Far distance at which linear fog reaches full coverage.
    pub end: f32,
    /// Which fog curve [`DistanceFog::fog_factor`] evaluates.
    pub mode: DistanceFogMode,
}

impl Default for DistanceFog {
    #[inline]
    fn default() -> Self {
        Self {
            color: Vec3::splat(0.5),
            density: 0.02,
            start: 0.0,
            end: 100.0,
            mode: DistanceFogMode::Exponential,
        }
    }
}

impl DistanceFog {
    /// Evaluates the configured fog factor `f ∈ [0, 1]` at `distance`.
    #[inline]
    pub fn fog_factor(&self, distance: f32) -> f32 {
        match self.mode {
            DistanceFogMode::Exponential => exponential_fog_factor(distance, self.density),
            DistanceFogMode::ExponentialSquared => {
                exponential_squared_fog_factor(distance, self.density)
            }
            DistanceFogMode::Linear => linear_fog_factor(distance, self.start, self.end),
        }
    }

    /// Composites the configured fog over `scene_color` at `distance`.
    ///
    /// Equivalent to `lerp(scene_color, color, fog_factor(distance))`.
    #[inline]
    pub fn apply(&self, scene_color: Vec3, distance: f32) -> Vec3 {
        apply_fog_color(scene_color, self.color, self.fog_factor(distance))
    }
}

/// Exponential distance-fog factor `f = 1 - exp(-distance · density)`.
///
/// `distance` and `density` are clamped non-negative; the exponent is bounded
/// so the result is in `[0, 1]`, equals `0` at `distance = 0`, and increases
/// monotonically toward `1`.
#[inline]
pub fn exponential_fog_factor(distance: f32, density: f32) -> f32 {
    let distance = clamp_non_negative(distance);
    let density = clamp_non_negative(density);
    let tau = (distance * density).min(MAX_EXPONENT);
    let factor = 1.0 - ops::exp(-tau);
    sanitize_unit(factor)
}

/// Exponential-squared distance-fog factor `f = 1 - exp(-(distance · density)²)`.
///
/// Ramps up more sharply than [`exponential_fog_factor`]. Inputs are clamped
/// non-negative and the squared exponent is bounded; the result lies in
/// `[0, 1]`, is `0` at `distance = 0`, and is monotonically non-decreasing.
#[inline]
pub fn exponential_squared_fog_factor(distance: f32, density: f32) -> f32 {
    let distance = clamp_non_negative(distance);
    let density = clamp_non_negative(density);
    let d = distance * density;
    let tau = (d * d).min(MAX_EXPONENT);
    let factor = 1.0 - ops::exp(-tau);
    sanitize_unit(factor)
}

/// Linear distance-fog factor `f = clamp((distance - start) / (end - start), 0, 1)`.
///
/// When `end ≤ start` the interval is degenerate; the ramp collapses to a hard
/// step at `start` (fully fogged at or beyond `start`, clear before it). The
/// result is always in `[0, 1]`.
#[inline]
pub fn linear_fog_factor(distance: f32, start: f32, end: f32) -> f32 {
    let distance = clamp_non_negative(distance);
    let start = clamp_non_negative(start);
    let end = clamp_non_negative(end);
    let span = end - start;
    if span > 1.0e-6 {
        sanitize_unit((distance - start) / span)
    } else if distance >= start {
        1.0
    } else {
        0.0
    }
}

/// Blends `scene_color` toward `fog_color` by `fog_factor ∈ [0, 1]`.
///
/// Equal to `scene_color·(1 - f) + fog_color·f`. The factor is clamped to
/// `[0, 1]` and every output channel is sanitised to be finite and
/// non-negative.
#[inline]
pub fn apply_fog_color(scene_color: Vec3, fog_color: Vec3, fog_factor: f32) -> Vec3 {
    let f = sanitize_unit(fog_factor);
    let scene = sanitize_rgb(scene_color);
    let fog = sanitize_rgb(fog_color);
    sanitize_rgb(scene.lerp(fog, f))
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

/// Clamps `value` into `[0, 1]`, mapping non-finite inputs to `0`.
#[inline]
fn sanitize_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Replaces any non-finite channel with `0` and clamps every channel
/// non-negative.
#[inline]
fn sanitize_rgb(rgb: Vec3) -> Vec3 {
    Vec3::new(
        if rgb.x.is_finite() { rgb.x.max(0.0) } else { 0.0 },
        if rgb.y.is_finite() { rgb.y.max(0.0) } else { 0.0 },
        if rgb.z.is_finite() { rgb.z.max(0.0) } else { 0.0 },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_has_no_fog_at_zero_distance() {
        assert_eq!(exponential_fog_factor(0.0, 0.5), 0.0);
        assert_eq!(exponential_squared_fog_factor(0.0, 0.5), 0.0);
    }

    #[test]
    fn exponential_saturates_at_far_distance() {
        let near = exponential_fog_factor(1.0, 0.1);
        let far = exponential_fog_factor(10_000.0, 0.1);
        assert!(far > near);
        assert!(far > 0.999, "far={far}");
        assert!((0.0..=1.0).contains(&far));
    }

    #[test]
    fn exponential_squared_ramps_faster_than_exponential_mid_range() {
        // For d·density > 1 the squared exponent exceeds the linear one, so the
        // squared factor is the larger (denser) of the two.
        let d = 20.0;
        let density = 0.1; // d·density = 2 > 1.
        let lin = exponential_fog_factor(d, density);
        let sq = exponential_squared_fog_factor(d, density);
        assert!(sq > lin, "sq={sq} lin={lin}");
    }

    #[test]
    fn exponential_curves_are_monotonic() {
        let mut prev_e = -1.0;
        let mut prev_s = -1.0;
        for i in 0..200 {
            let d = i as f32 * 0.5;
            let e = exponential_fog_factor(d, 0.05);
            let s = exponential_squared_fog_factor(d, 0.05);
            assert!(e >= prev_e - 1e-6, "exp not monotonic at d={d}");
            assert!(s >= prev_s - 1e-6, "exp2 not monotonic at d={d}");
            assert!((0.0..=1.0).contains(&e));
            assert!((0.0..=1.0).contains(&s));
            prev_e = e;
            prev_s = s;
        }
    }

    #[test]
    fn linear_ramp_boundaries() {
        let start = 10.0;
        let end = 30.0;
        assert_eq!(linear_fog_factor(0.0, start, end), 0.0);
        assert_eq!(linear_fog_factor(10.0, start, end), 0.0);
        assert!((linear_fog_factor(20.0, start, end) - 0.5).abs() < 1e-6);
        assert_eq!(linear_fog_factor(30.0, start, end), 1.0);
        assert_eq!(linear_fog_factor(100.0, start, end), 1.0);
    }

    #[test]
    fn linear_ramp_is_monotonic() {
        let mut prev = -1.0;
        for i in 0..100 {
            let d = i as f32;
            let f = linear_fog_factor(d, 20.0, 60.0);
            assert!(f >= prev - 1e-6, "not monotonic at d={d}");
            assert!((0.0..=1.0).contains(&f));
            prev = f;
        }
    }

    #[test]
    fn linear_degenerate_interval_is_a_step() {
        // end == start → hard step at start.
        assert_eq!(linear_fog_factor(4.0, 5.0, 5.0), 0.0);
        assert_eq!(linear_fog_factor(5.0, 5.0, 5.0), 1.0);
        assert_eq!(linear_fog_factor(6.0, 5.0, 5.0), 1.0);
        // Inverted interval (end < start) collapses the same way.
        assert_eq!(linear_fog_factor(1.0, 10.0, 2.0), 0.0);
        assert_eq!(linear_fog_factor(10.0, 10.0, 2.0), 1.0);
    }

    #[test]
    fn apply_fog_color_endpoints() {
        let scene = Vec3::new(0.2, 0.4, 0.8);
        let fog = Vec3::new(0.9, 0.9, 0.9);
        assert_eq!(apply_fog_color(scene, fog, 0.0), scene);
        assert_eq!(apply_fog_color(scene, fog, 1.0), fog);
        let mid = apply_fog_color(scene, fog, 0.5);
        assert!((mid - (scene + fog) * 0.5).length() < 1e-6);
    }

    #[test]
    fn apply_saturates_scene_to_fog_color() {
        let fog = DistanceFog {
            color: Vec3::new(0.7, 0.75, 0.8),
            density: 0.2,
            start: 0.0,
            end: 100.0,
            mode: DistanceFogMode::Exponential,
        };
        let scene = Vec3::new(0.1, 0.1, 0.1);
        let near = fog.apply(scene, 0.0);
        assert!((near - scene).length() < 1e-6);
        let far = fog.apply(scene, 100_000.0);
        assert!((far - fog.color).length() < 1e-3, "far={far:?}");
    }

    #[test]
    fn distance_fog_mode_dispatch() {
        let base = DistanceFog {
            color: Vec3::splat(1.0),
            density: 0.1,
            start: 10.0,
            end: 50.0,
            mode: DistanceFogMode::Exponential,
        };
        let d = 30.0;
        let exp = DistanceFog {
            mode: DistanceFogMode::Exponential,
            ..base
        };
        let exp2 = DistanceFog {
            mode: DistanceFogMode::ExponentialSquared,
            ..base
        };
        let lin = DistanceFog {
            mode: DistanceFogMode::Linear,
            ..base
        };
        assert_eq!(exp.fog_factor(d), exponential_fog_factor(d, 0.1));
        assert_eq!(exp2.fog_factor(d), exponential_squared_fog_factor(d, 0.1));
        assert_eq!(lin.fog_factor(d), linear_fog_factor(d, 10.0, 50.0));
    }

    #[test]
    fn factors_stay_bounded_on_extreme_inputs() {
        for d in [-10.0f32, 0.0, 1e9, f32::NAN, f32::INFINITY] {
            for density in [-1.0f32, 0.0, 1e9, f32::NAN, f32::INFINITY] {
                let e = exponential_fog_factor(d, density);
                let s = exponential_squared_fog_factor(d, density);
                assert!(e.is_finite() && (0.0..=1.0).contains(&e), "d={d} dens={density} e={e}");
                assert!(s.is_finite() && (0.0..=1.0).contains(&s), "d={d} dens={density} s={s}");
            }
        }
        let out = apply_fog_color(
            Vec3::splat(f32::NAN),
            Vec3::new(f32::INFINITY, 0.5, -1.0),
            f32::NAN,
        );
        assert!(out.is_finite());
    }
}
