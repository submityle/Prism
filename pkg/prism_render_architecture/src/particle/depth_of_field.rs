//! Circle-of-confusion (`CoC`) evaluation for the particle depth-of-field
//! (`DoF`) post pass (design §16, §21).
//!
//! A thin-lens camera focuses a single `view-space` plane sharply; every other
//! depth projects to a blurred disc whose diameter is the *circle of confusion*.
//! Production `VFX` and post stacks (Unreal's diaphragm `DoF`, `Frostbite`'s
//! gather `DoF`, Unity `HDRP`'s physical camera) drive their bokeh gather from
//! exactly this quantity. This module owns the `CPU`-verifiable *maths* of that
//! contract: given a `view-space` depth and the lens parameters, it returns the
//! signed `CoC`, the clamped gather radius, and the normalized bokeh-kernel
//! scale a `GPU` blur pass reads, plus the `std430` parameter block the pass
//! binds.
//!
//! # Thin-lens model
//!
//! For focal length `f`, aperture diameter `A` (`= f / N` for `f`-number `N`),
//! and focus distance `focus`, the `CoC` *diameter* at a positive `view-space`
//! depth `z` is
//!
//! ```text
//! coc = | A * f * (z - focus) / (z * (focus - f)) |.
//! ```
//!
//! The bracketed value (before the absolute value) is *signed*: it is negative
//! for near defocus (`z < focus`, in front of the focus plane) and positive for
//! far defocus (`z > focus`, behind it). This is the sign convention every
//! method here follows.
//!
//! # Determinism
//!
//! The `CoC` is pure algebra and division, so it is bit-reproducible against a
//! future `GPU` evaluation. The optional blur-fade is a `smoothstep`
//! (a cubic polynomial) and the per-sample energy term is a rational
//! polynomial; no transcendental function (`sin`/`cos`/`exp`/`ln`/`pow`) and no
//! `f32::round`/`f32::ceil` is ever called, matching the determinism contract
//! of the sibling [`super::simulation`] module.

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Denominators (and divisors) with magnitude below this are treated as zero so
/// evaluation falls back to a defined result instead of dividing by (near) zero
/// or propagating `NaN`.
const MIN_DENOM: f32 = 1e-6;

/// Evaluates the `smoothstep` interpolation of `x` across `[edge0, edge1]`,
/// returning `0.0` at or below `edge0`, `1.0` at or above `edge1`, and the
/// cubic `3t^2 - 2t^3` in between. A degenerate (near-zero-width) edge interval
/// falls back to a hard step at `edge0`.
#[must_use]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if span.abs() < MIN_DENOM {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / span).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The result of evaluating [`DofParams::evaluate`] at one `view-space` depth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DofSample {
    /// Signed `CoC` *diameter*: negative for near defocus (`z < focus`),
    /// positive for far defocus (`z > focus`), zero on the focus plane.
    pub signed_coc: f32,
    /// Gather *radius* in `CoC` units, always non-negative and clamped to
    /// [`DofParams::max_coc_radius`].
    pub radius: f32,
    /// Whether the depth lies in front of the focus plane (near defocus).
    pub is_near: bool,
}

/// The thin-lens parameters the depth-of-field pass evaluates against.
///
/// Distances are in the same `view-space` unit as the sampled depth; the
/// aperture diameter is stored directly (use [`DofParams::from_f_number`] to
/// derive it from an `f`-number).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DofParams {
    /// `view-space` distance of the sharply focused plane (must exceed
    /// [`DofParams::focal_length`] for a valid thin lens).
    pub focus_distance: f32,
    /// Lens focal length `f`.
    pub focal_length: f32,
    /// Aperture diameter `A` (`= f / N` for `f`-number `N`).
    pub aperture_diameter: f32,
    /// Upper bound on the gather radius, in `CoC` units, so the blur kernel
    /// stays within a bounded footprint.
    pub max_coc_radius: f32,
}

impl DofParams {
    /// The `std430` stride of the packed parameter block: a single `vec4`.
    pub const STD430_STRIDE: usize = VEC4_STRIDE;

    /// Builds parameters from an explicit aperture diameter.
    #[must_use]
    pub const fn new(
        focus_distance: f32,
        focal_length: f32,
        aperture_diameter: f32,
        max_coc_radius: f32,
    ) -> Self {
        Self {
            focus_distance,
            focal_length,
            aperture_diameter,
            max_coc_radius,
        }
    }

    /// Builds parameters from an `f`-number `N`, deriving the aperture diameter
    /// as `A = f / N`. A near-zero `f`-number falls back to a closed aperture
    /// (`A = 0`), which yields a `CoC` of zero everywhere.
    #[must_use]
    pub fn from_f_number(
        focus_distance: f32,
        focal_length: f32,
        f_number: f32,
        max_coc_radius: f32,
    ) -> Self {
        let aperture_diameter = if f_number.abs() < MIN_DENOM {
            0.0
        } else {
            focal_length / f_number
        };
        Self::new(
            focus_distance,
            focal_length,
            aperture_diameter,
            max_coc_radius,
        )
    }

    /// The signed `CoC` diameter at `view-space` depth `depth`.
    ///
    /// Negative for near defocus (`depth < focus`), positive for far defocus
    /// (`depth > focus`), and exactly zero on the focus plane. Guards a
    /// non-positive depth, a focus distance not exceeding the focal length, and
    /// a near-zero denominator by returning `0.0`.
    #[must_use]
    pub fn coc_signed(&self, depth: f32) -> f32 {
        let f = self.focal_length;
        let focus = self.focus_distance;
        if depth <= 0.0 || focus <= f {
            return 0.0;
        }
        let denom = depth * (focus - f);
        if denom.abs() < MIN_DENOM {
            return 0.0;
        }
        self.aperture_diameter * f * (depth - focus) / denom
    }

    /// The unsigned `CoC` diameter at `depth` (the magnitude of
    /// [`DofParams::coc_signed`]).
    #[must_use]
    pub fn coc_diameter(&self, depth: f32) -> f32 {
        self.coc_signed(depth).abs()
    }

    /// The gather radius at `depth`: half the `CoC` diameter, clamped to
    /// `[0, max_coc_radius]`.
    #[must_use]
    pub fn coc_radius_clamped(&self, depth: f32) -> f32 {
        let radius = self.coc_diameter(depth) * 0.5;
        let max_radius = self.max_coc_radius.max(0.0);
        radius.min(max_radius)
    }

    /// Maps a gather radius linearly onto the normalized bokeh-kernel scale in
    /// `[0, 1]`, where [`DofParams::max_coc_radius`] maps to `1.0`. A near-zero
    /// maximum radius returns `0.0` (no blur).
    #[must_use]
    pub fn bokeh_scale(&self, coc_radius: f32) -> f32 {
        let max_radius = self.max_coc_radius;
        if max_radius.abs() < MIN_DENOM {
            return 0.0;
        }
        (coc_radius / max_radius).clamp(0.0, 1.0)
    }

    /// A `smoothstep` blur-fade weight in `[0, 1]` for the gather radius at
    /// `depth`: `0.0` on the focus plane (fully sharp) rising smoothly to `1.0`
    /// once the radius reaches [`DofParams::max_coc_radius`] (fully blurred).
    /// Monotonically non-decreasing in the radius, so it never re-sharpens as
    /// defocus grows.
    #[must_use]
    pub fn blur_fade(&self, depth: f32) -> f32 {
        let max_radius = self.max_coc_radius.max(0.0);
        smoothstep(0.0, max_radius, self.coc_radius_clamped(depth))
    }

    /// A per-sample energy attenuation in `(0, 1]` as a rational polynomial of
    /// the normalized gather radius `r = coc_radius / max_coc_radius`:
    /// `1 / (1 + r^2)`. A wider `CoC` spreads a sample's energy over a larger
    /// disc, so its per-tap weight falls off; the term is `1.0` at zero radius
    /// and decreases monotonically. A near-zero maximum radius returns `1.0`.
    #[must_use]
    pub fn energy_attenuation(&self, coc_radius: f32) -> f32 {
        let max_radius = self.max_coc_radius;
        if max_radius.abs() < MIN_DENOM {
            return 1.0;
        }
        let r = coc_radius / max_radius;
        1.0 / (1.0 + r * r)
    }

    /// Evaluates the signed `CoC`, clamped gather radius, and near/far side at a
    /// single `view-space` depth.
    #[must_use]
    pub fn evaluate(&self, depth: f32) -> DofSample {
        DofSample {
            signed_coc: self.coc_signed(depth),
            radius: self.coc_radius_clamped(depth),
            is_near: depth > 0.0 && depth < self.focus_distance,
        }
    }

    /// Packs the parameters into a `std430` `vec4`
    /// `(focus_distance, focal_length, aperture_diameter, max_coc_radius)`.
    #[must_use]
    pub fn to_std430(&self) -> [f32; 4] {
        [
            self.focus_distance,
            self.focal_length,
            self.aperture_diameter,
            self.max_coc_radius,
        ]
    }

    /// Total `std430` byte size of a storage buffer holding `count` packed
    /// [`DofParams`] blocks, clamped up to a single element per the shared
    /// [`storage_bytes`] rule.
    #[must_use]
    pub fn gpu_storage_bytes(count: usize) -> usize {
        storage_bytes(VEC4_STRIDE, count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for `f32` equality decisions in tests.
    const CMP_EPS: f32 = 1e-6;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    /// A representative lens: focus at 10, focal length 1, aperture 2, with a
    /// tight radius clamp so far defocus saturates.
    fn params() -> DofParams {
        DofParams::new(10.0, 1.0, 2.0, 0.05)
    }

    #[test]
    fn focus_plane_has_zero_coc() {
        let p = params();
        assert!(approx(p.coc_signed(p.focus_distance), 0.0));
        assert!(approx(p.coc_diameter(p.focus_distance), 0.0));
        assert!(approx(p.coc_radius_clamped(p.focus_distance), 0.0));
        let s = p.evaluate(p.focus_distance);
        assert!(approx(s.signed_coc, 0.0));
        assert!(approx(s.radius, 0.0));
        assert!(!s.is_near);
    }

    #[test]
    fn near_defocus_is_negative_and_marked_near() {
        let p = params();
        let depth = 5.0;
        let signed = p.coc_signed(depth);
        assert!(signed < 0.0, "near defocus must be negative, got {signed}");
        assert!(p.coc_radius_clamped(depth) > 0.0);
        let s = p.evaluate(depth);
        assert!(s.is_near);
        assert!(s.signed_coc < 0.0);
        assert!(s.radius > 0.0);
    }

    #[test]
    fn far_defocus_is_positive_and_clamped() {
        let p = params();
        // Unclamped far radius grows past max_coc_radius, so it saturates.
        let near_far = p.coc_radius_clamped(11.0);
        let deep = p.coc_radius_clamped(20.0);
        let very_deep = p.coc_radius_clamped(1.0e6);
        assert!(p.coc_signed(20.0) > 0.0);
        assert!(near_far < p.max_coc_radius);
        assert!(approx(deep, p.max_coc_radius));
        assert!(approx(very_deep, p.max_coc_radius));
        assert!(!p.evaluate(20.0).is_near);
    }

    #[test]
    fn far_coc_is_monotonic_beyond_focus() {
        // Generous clamp so the raw thin-lens growth is observable.
        let p = DofParams::new(10.0, 1.0, 2.0, 100.0);
        let mut previous = 0.0;
        let depths = [11.0_f32, 15.0, 20.0, 40.0, 100.0, 1000.0];
        for depth in depths {
            let radius = p.coc_radius_clamped(depth);
            assert!(
                radius >= previous - CMP_EPS,
                "far CoC must be non-decreasing: {radius} < {previous} at depth {depth}"
            );
            previous = radius;
        }
    }

    #[test]
    fn denominator_and_range_guards_never_panic() {
        // focus == focal_length: (focus - f) is zero, so CoC is guarded to 0.
        let degenerate = DofParams::new(1.0, 1.0, 2.0, 0.05);
        assert!(approx(degenerate.coc_signed(5.0), 0.0));
        assert!(approx(degenerate.coc_diameter(5.0), 0.0));
        // Non-positive depths are guarded.
        let p = params();
        assert!(approx(p.coc_signed(0.0), 0.0));
        assert!(approx(p.coc_signed(-3.0), 0.0));
        assert!(!p.evaluate(0.0).is_near);
        // A closed aperture from a near-zero f-number yields zero CoC.
        let closed = DofParams::from_f_number(10.0, 1.0, 0.0, 0.05);
        assert!(approx(closed.aperture_diameter, 0.0));
        assert!(approx(closed.coc_signed(20.0), 0.0));
    }

    #[test]
    fn from_f_number_derives_aperture() {
        let p = DofParams::from_f_number(10.0, 2.0, 4.0, 0.05);
        assert!(approx(p.aperture_diameter, 0.5));
    }

    #[test]
    fn bokeh_scale_is_linear_and_clamped() {
        let p = params();
        assert!(approx(p.bokeh_scale(0.0), 0.0));
        assert!(approx(p.bokeh_scale(p.max_coc_radius * 0.5), 0.5));
        assert!(approx(p.bokeh_scale(p.max_coc_radius), 1.0));
        assert!(approx(p.bokeh_scale(p.max_coc_radius * 2.0), 1.0));
        assert!(approx(p.bokeh_scale(-1.0), 0.0));
    }

    #[test]
    fn blur_fade_is_monotonic_with_defocus() {
        let p = params();
        assert!(approx(p.blur_fade(p.focus_distance), 0.0));
        let shallow = p.blur_fade(11.0);
        let deep = p.blur_fade(20.0);
        assert!((0.0..=1.0).contains(&shallow));
        assert!(deep >= shallow);
        assert!(approx(deep, 1.0));
    }

    #[test]
    fn energy_attenuation_falls_off_monotonically() {
        let p = params();
        assert!(approx(p.energy_attenuation(0.0), 1.0));
        assert!(approx(
            p.energy_attenuation(p.max_coc_radius * 0.5),
            1.0 / 1.25
        ));
        assert!(approx(p.energy_attenuation(p.max_coc_radius), 0.5));
        let a = p.energy_attenuation(0.01);
        let b = p.energy_attenuation(0.02);
        let c = p.energy_attenuation(0.03);
        assert!(a > b && b > c);
    }

    #[test]
    fn std430_layout_matches_shared_stride() {
        let p = params();
        let packed = p.to_std430();
        assert_eq!(packed.len() * 4, VEC4_STRIDE);
        assert!(approx(packed[0], 10.0));
        assert!(approx(packed[3], 0.05));
        assert_eq!(DofParams::STD430_STRIDE, VEC4_STRIDE);
        assert_eq!(DofParams::STD430_STRIDE % VEC4_STRIDE, 0);
        assert_eq!(DofParams::gpu_storage_bytes(1), VEC4_STRIDE);
        assert_eq!(DofParams::gpu_storage_bytes(4), 4 * VEC4_STRIDE);
        // An empty buffer still reserves one element.
        assert_eq!(DofParams::gpu_storage_bytes(0), VEC4_STRIDE);
    }

    #[test]
    fn evaluation_is_deterministic() {
        let p = params();
        let depths = [2.0_f32, 5.0, 10.0, 11.0, 50.0, 500.0];
        for depth in depths {
            let first = p.evaluate(depth);
            let second = p.evaluate(depth);
            assert_eq!(first, second);
        }
    }
}
