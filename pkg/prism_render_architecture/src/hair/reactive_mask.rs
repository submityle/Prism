//! Fibre-aware temporal reactive mask and blue-noise dither `alpha` for hair
//! anti-aliasing (design doc §8.5 item12).
//!
//! Thin hair fibres are the worst case for temporal anti-aliasing (`TAA`) and
//! temporal upscalers: a single strand usually covers only a fraction of a
//! pixel, so its sub-pixel coverage flickers on and off between frames and the
//! history buffer smears those flickers into ghosts and trails. The film-grade
//! and real-time fix (as popularised by `UE5`'s `TAA`/`TSR` reactive mask and
//! visible in motion-heavy titles such as `Alan Wake 2`) is to emit, per pixel,
//! a `reactivity` value in `[0, 1]`: `0` trusts the temporal history fully,
//! while `1` leans on the current frame and rejects stale history. Hair pushes
//! `reactivity` up where it is most prone to ghosting -- low sub-pixel
//! `coverage`, high screen-space `velocity`, and large `depth_delta` (edges and
//! disocclusions) -- so the accumulator stops dragging a stale strand behind a
//! moving silhouette.
//!
//! This module is the deterministic, panic-free mapping kernel for that policy.
//! It is a pure function of scalar / array inputs to scalar / array outputs with
//! no device state, exactly like the analytic coverage kernel in
//! [`crate::hair::line_coverage`]: array in, array out, golden-comparable.
//!
//! Reactivity model (closed form, no transcendental math). Each contribution is
//! a monotone ramp: the `coverage` term is `1 - coverage` (less coverage -> more
//! reactive), and the `velocity` / `depth_delta` terms are rational saturation
//! ramps of the form `v / (v + k)` (faster / deeper change -> more reactive,
//! asymptotically approaching `1`). The three terms are combined as a weighted
//! sum and clamped into `[0, max_reactivity]`. Only `+`, `-`, `*`, `/`, and
//! `clamp` are used, so the kernel never touches `exp`, `pow`, or trigonometry.
//!
//! Blue-noise dither path. The analytic `line_coverage` ramp is the preferred,
//! flicker-free way to resolve a sub-pixel strand, but it cannot model every
//! case (dense overlapping fibres, order-independent-transparency budget
//! exhaustion). For those the module provides a stochastic fallback: a purely
//! integer `blue-noise`-style hash produces a per-pixel threshold in `[0, 1)`
//! (rotated over `frame` for temporal decorrelation), and [`dither_alpha`] turns
//! a sub-pixel `coverage` into a hard `0` / `1` draw decision by comparing it to
//! that threshold. Analytic first, dither as the safety net.

use alloc::vec::Vec;

/// Reference epsilon for degeneracy / equality-free comparisons and for test
/// closeness checks, kept module-level so the kernel and its tests agree.
pub const EPS: f32 = 1.0e-6;

/// Saturation constant (in pixels per frame) for the screen-`velocity` ramp
/// `v / (v + k)`: at `v == VELOCITY_SATURATION` the velocity term reaches `0.5`.
pub const VELOCITY_SATURATION: f32 = 2.0;

/// Saturation constant for the `depth_delta` ramp `d / (d + k)`: at
/// `d == DEPTH_SATURATION` the depth term reaches `0.5`. Relative depth changes
/// are small, so this constant is correspondingly small.
pub const DEPTH_SATURATION: f32 = 0.1;

/// Weighting / clamping parameters for [`pixel_reactivity`]. Each weight scales
/// one monotone reactivity term; `max_reactivity` is the upper clamp applied to
/// the weighted sum. All fields are sanitised by [`ReactiveParams::sanitized`]
/// before use, so non-finite or negative values never panic or escape range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReactiveParams {
    /// Weight on the `1 - coverage` term (low sub-pixel `coverage` is reactive).
    pub coverage_weight: f32,
    /// Weight on the screen-`velocity` saturation term (fast motion is reactive).
    pub velocity_weight: f32,
    /// Weight on the `depth_delta` saturation term (edges / disocclusions).
    pub depth_weight: f32,
    /// Upper clamp on the final `reactivity`, itself clamped to `[0, 1]`.
    pub max_reactivity: f32,
}

impl ReactiveParams {
    /// Explicit parameters without sanitisation; call [`Self::sanitized`] (as
    /// [`pixel_reactivity`] does internally) before relying on the values.
    #[must_use]
    pub const fn new(
        coverage_weight: f32,
        velocity_weight: f32,
        depth_weight: f32,
        max_reactivity: f32,
    ) -> Self {
        Self {
            coverage_weight,
            velocity_weight,
            depth_weight,
            max_reactivity,
        }
    }

    /// A sanitised copy: each weight becomes `0` if it is non-finite or negative,
    /// and `max_reactivity` is clamped to `[0, 1]` (non-finite -> `1`). The result
    /// always yields a finite `reactivity` in range.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            coverage_weight: sanitize_weight(self.coverage_weight),
            velocity_weight: sanitize_weight(self.velocity_weight),
            depth_weight: sanitize_weight(self.depth_weight),
            max_reactivity: if self.max_reactivity.is_finite() {
                self.max_reactivity.clamp(0.0, 1.0)
            } else {
                1.0
            },
        }
    }
}

impl Default for ReactiveParams {
    /// Balanced defaults: `velocity` is weighted most (motion is the dominant
    /// ghosting driver), `coverage` next, `depth_delta` least, with the weights
    /// summing to `1` and a full `max_reactivity` of `1`.
    fn default() -> Self {
        Self::new(0.3, 0.45, 0.25, 1.0)
    }
}

/// Per-pixel inputs to [`pixel_reactivity`]. `coverage` is sub-pixel coverage in
/// `[0, 1]`, `screen_velocity` is motion in pixels per frame, and `depth_delta`
/// is a (possibly signed) relative depth change. All are sanitised before use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixelReactiveInput {
    /// Sub-pixel coverage in `[0, 1]`; non-finite is treated as fully covered.
    pub coverage: f32,
    /// Screen-space motion in pixels per frame; the magnitude is used.
    pub screen_velocity: f32,
    /// Relative depth change; the magnitude is used.
    pub depth_delta: f32,
}

impl PixelReactiveInput {
    /// Explicit per-pixel inputs.
    #[must_use]
    pub const fn new(coverage: f32, screen_velocity: f32, depth_delta: f32) -> Self {
        Self {
            coverage,
            screen_velocity,
            depth_delta,
        }
    }
}

/// A finite, non-negative weight: non-finite or negative inputs collapse to `0`
/// without any floating-point equality test.
#[must_use]
fn sanitize_weight(w: f32) -> f32 {
    if w.is_finite() && w > 0.0 {
        w
    } else {
        0.0
    }
}

/// Sub-pixel `coverage` clamped to `[0, 1]`; non-finite is treated as fully
/// covered (`1`) so corrupt data does not spuriously inflate `reactivity`.
#[must_use]
fn sanitize_coverage(coverage: f32) -> f32 {
    if coverage.is_finite() {
        coverage.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// A finite, non-negative magnitude: non-finite -> `0`, otherwise the absolute
/// value (so signed velocities / depth deltas contribute by magnitude).
#[must_use]
fn sanitize_magnitude(v: f32) -> f32 {
    if v.is_finite() {
        v.abs()
    } else {
        0.0
    }
}

/// A rational saturation ramp `v / (v + k)` for a non-negative magnitude `v` and
/// a positive constant `k`: `0` at `v == 0`, `0.5` at `v == k`, and
/// asymptotically `1` as `v` grows. Uses no transcendental math and never
/// divides by zero because `k > 0` and `v >= 0`.
#[must_use]
fn saturation_ramp(v: f32, k: f32) -> f32 {
    v / (v + k)
}

/// The reactive-mask value for a single pixel: a weighted sum of the three
/// monotone reactivity terms, clamped to `[0, max_reactivity]`. Inputs and
/// parameters are sanitised first, so the result is always finite and in range
/// and the function never panics. `reactivity` is monotonically non-increasing
/// in `coverage` and non-decreasing in both `screen_velocity` magnitude and
/// `depth_delta` magnitude.
#[must_use]
pub fn pixel_reactivity(params: ReactiveParams, input: PixelReactiveInput) -> f32 {
    let p = params.sanitized();
    let coverage = sanitize_coverage(input.coverage);
    let velocity = sanitize_magnitude(input.screen_velocity);
    let depth = sanitize_magnitude(input.depth_delta);

    let coverage_term = 1.0 - coverage;
    let velocity_term = saturation_ramp(velocity, VELOCITY_SATURATION);
    let depth_term = saturation_ramp(depth, DEPTH_SATURATION);

    let raw = p.coverage_weight * coverage_term
        + p.velocity_weight * velocity_term
        + p.depth_weight * depth_term;

    raw.clamp(0.0, p.max_reactivity)
}

/// Per-pixel `reactivity` for a whole span of inputs: maps each through
/// [`pixel_reactivity`] against the same `params`, preserving input order. An
/// empty slice returns an empty [`Vec`] (no panic).
#[must_use]
pub fn reactivity_map(params: ReactiveParams, inputs: &[PixelReactiveInput]) -> Vec<f32> {
    let mut out = Vec::with_capacity(inputs.len());
    for &input in inputs {
        out.push(pixel_reactivity(params, input));
    }
    out
}

/// A deterministic `blue-noise`-style dither threshold in `[0, 1)` for pixel
/// `(x, y)` on `frame`. The three coordinates are mixed with large odd constants
/// and passed through an integer xorshift / multiply finaliser, then normalised
/// by `2^32` so the result is strictly below `1`. `frame` rotates the sequence
/// in time to decorrelate the dither pattern across frames; the mapping is pure
/// integer arithmetic, so it is bit-exact and never panics.
#[must_use]
pub fn dither_threshold(x: u32, y: u32, frame: u32) -> f32 {
    let mut h = x.wrapping_mul(0x9E37_79B1);
    h ^= y.wrapping_mul(0x85EB_CA77);
    h ^= frame.wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    // Divide by 2^32 (not u32::MAX) so the result lands in [0, 1), never 1.
    (h as f32) / 4_294_967_296.0
}

/// A hard sub-pixel draw decision: returns `1.0` when the sanitised `coverage`
/// (clamped to `[0, 1]`, non-finite -> `0`) is at least `threshold`, else `0.0`.
/// `threshold` is sanitised to `[0, 1]` (non-finite -> `1`) so a corrupt
/// threshold conservatively suppresses the fibre rather than panicking. This is
/// the stochastic fallback used for the thinnest fibres where the analytic
/// `line_coverage` ramp is not applied.
#[must_use]
pub fn dither_alpha(coverage: f32, threshold: f32) -> f32 {
    let coverage = if coverage.is_finite() {
        coverage.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let threshold = if threshold.is_finite() {
        threshold.clamp(0.0, 1.0)
    } else {
        1.0
    };
    if coverage >= threshold {
        1.0
    } else {
        0.0
    }
}

/// Per-pixel dither `alpha` for a span of `coverages`, preserving input order.
/// Pixel `i` uses the threshold at `(base_x + i, base_y)` on `frame`, so a tile
/// is dithered with a spatially varying, temporally rotated pattern. An empty
/// slice returns an empty [`Vec`] (no panic); the pixel index wraps rather than
/// overflowing.
#[must_use]
pub fn dither_alpha_map(coverages: &[f32], base_x: u32, base_y: u32, frame: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity(coverages.len());
    for (i, &coverage) in coverages.iter().enumerate() {
        let x = base_x.wrapping_add(i as u32);
        let threshold = dither_threshold(x, base_y, frame);
        out.push(dither_alpha(coverage, threshold));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn reactivity_is_monotone_decreasing_in_coverage() {
        let params = ReactiveParams::default();
        let low = pixel_reactivity(params, PixelReactiveInput::new(0.1, 0.0, 0.0));
        let mid = pixel_reactivity(params, PixelReactiveInput::new(0.5, 0.0, 0.0));
        let high = pixel_reactivity(params, PixelReactiveInput::new(0.9, 0.0, 0.0));
        assert!(low >= mid);
        assert!(mid >= high);
    }

    #[test]
    fn reactivity_is_monotone_increasing_in_velocity() {
        let params = ReactiveParams::default();
        let slow = pixel_reactivity(params, PixelReactiveInput::new(0.5, 1.0, 0.0));
        let fast = pixel_reactivity(params, PixelReactiveInput::new(0.5, 10.0, 0.0));
        let faster = pixel_reactivity(params, PixelReactiveInput::new(0.5, 100.0, 0.0));
        assert!(fast >= slow);
        assert!(faster >= fast);
    }

    #[test]
    fn reactivity_is_monotone_increasing_in_depth_delta() {
        let params = ReactiveParams::default();
        let shallow = pixel_reactivity(params, PixelReactiveInput::new(0.5, 0.0, 0.02));
        let deep = pixel_reactivity(params, PixelReactiveInput::new(0.5, 0.0, 0.5));
        assert!(deep >= shallow);
    }

    #[test]
    fn reactivity_stays_within_zero_and_max() {
        let params = ReactiveParams::default();
        let coverages = [-1.0, 0.0, 0.3, 0.7, 1.0, 2.0];
        let velocities = [0.0, 1.0, 25.0, 1000.0];
        let depths = [-0.5, 0.0, 0.1, 5.0];
        for &c in &coverages {
            for &v in &velocities {
                for &d in &depths {
                    let r = pixel_reactivity(params, PixelReactiveInput::new(c, v, d));
                    assert!(r.is_finite());
                    assert!(r >= 0.0);
                    assert!(r <= params.max_reactivity);
                }
            }
        }
    }

    #[test]
    fn all_zero_input_is_low_reactivity() {
        let params = ReactiveParams::default();
        let zero = pixel_reactivity(params, PixelReactiveInput::new(0.0, 0.0, 0.0));
        let moving = pixel_reactivity(params, PixelReactiveInput::new(0.0, 100.0, 1.0));
        assert!(zero < 0.5);
        assert!(zero < moving);
    }

    #[test]
    fn bad_input_does_not_panic_and_is_finite() {
        let params = ReactiveParams::default();
        let inputs = [
            PixelReactiveInput::new(f32::NAN, 1.0, 0.1),
            PixelReactiveInput::new(0.5, f32::INFINITY, 0.1),
            PixelReactiveInput::new(0.5, 1.0, f32::NAN),
            PixelReactiveInput::new(f32::NEG_INFINITY, f32::NAN, f32::INFINITY),
        ];
        for &input in &inputs {
            let r = pixel_reactivity(params, input);
            assert!(r.is_finite());
            assert!(r >= 0.0);
            assert!(r <= params.max_reactivity);
        }
    }

    #[test]
    fn sanitize_repairs_bad_params() {
        let bad = ReactiveParams::new(-1.0, f32::NAN, f32::INFINITY, 5.0);
        let fixed = bad.sanitized();
        assert!(close(fixed.coverage_weight, 0.0));
        assert!(close(fixed.velocity_weight, 0.0));
        assert!(close(fixed.depth_weight, 0.0));
        assert!(close(fixed.max_reactivity, 1.0));

        let bad_max = ReactiveParams::new(0.2, 0.2, 0.2, f32::NAN).sanitized();
        assert!(close(bad_max.max_reactivity, 1.0));

        // All-zero weights force reactivity to zero regardless of input.
        let zeroed = ReactiveParams::new(0.0, 0.0, 0.0, 1.0);
        let r = pixel_reactivity(zeroed, PixelReactiveInput::new(0.0, 100.0, 1.0));
        assert!(close(r, 0.0));
    }

    #[test]
    fn dither_threshold_in_unit_range_and_deterministic() {
        for y in 0..16u32 {
            for x in 0..16u32 {
                let t = dither_threshold(x, y, 7);
                assert!(t.is_finite());
                assert!(t >= 0.0);
                assert!(t < 1.0);
                // Deterministic: same inputs -> identical bits.
                assert!(close(t, dither_threshold(x, y, 7)));
            }
        }
    }

    #[test]
    fn dither_alpha_threshold_behaviour() {
        // Coverage at or above threshold draws; below does not.
        assert!(close(dither_alpha(0.6, 0.5), 1.0));
        assert!(close(dither_alpha(0.5, 0.5), 1.0));
        assert!(close(dither_alpha(0.4, 0.5), 0.0));
        // Non-finite coverage -> treated as 0 -> no draw (threshold > 0).
        assert!(close(dither_alpha(f32::NAN, 0.5), 0.0));
        // Non-finite threshold -> treated as 1 -> only full coverage draws.
        assert!(close(dither_alpha(1.0, f32::INFINITY), 1.0));
        assert!(close(dither_alpha(0.9, f32::INFINITY), 0.0));
    }

    #[test]
    fn empty_maps_are_empty_without_panic() {
        let params = ReactiveParams::default();
        assert!(reactivity_map(params, &[]).is_empty());
        assert!(dither_alpha_map(&[], 0, 0, 0).is_empty());
    }

    #[test]
    fn reactivity_map_matches_scalar_and_preserves_order() {
        let params = ReactiveParams::default();
        let inputs = [
            PixelReactiveInput::new(0.0, 0.0, 0.0),
            PixelReactiveInput::new(0.3, 5.0, 0.1),
            PixelReactiveInput::new(1.0, 50.0, 0.4),
        ];
        let mapped = reactivity_map(params, &inputs);
        assert_eq!(mapped.len(), inputs.len());
        for (input, got) in inputs.iter().zip(mapped.iter()) {
            assert!(close(*got, pixel_reactivity(params, *input)));
        }
    }

    #[test]
    fn dither_alpha_map_matches_scalar_and_preserves_order() {
        let coverages = [0.1, 0.5, 0.9, 0.0];
        let mapped = dither_alpha_map(&coverages, 100, 200, 3);
        assert_eq!(mapped.len(), coverages.len());
        for (i, (&coverage, &got)) in coverages.iter().zip(mapped.iter()).enumerate() {
            let threshold = dither_threshold(100u32.wrapping_add(i as u32), 200, 3);
            assert!(close(got, dither_alpha(coverage, threshold)));
        }
    }

    #[test]
    fn frame_rotation_changes_threshold() {
        // Over a small tile there must exist a pixel whose threshold differs
        // between two frames (temporal decorrelation).
        let mut differs = false;
        for y in 0..8u32 {
            for x in 0..8u32 {
                if !close(dither_threshold(x, y, 0), dither_threshold(x, y, 1)) {
                    differs = true;
                }
            }
        }
        assert!(differs);
    }
}
