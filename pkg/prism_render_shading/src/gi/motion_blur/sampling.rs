//! Deterministic along-velocity sample placement for the reconstruction filter.
//!
//! McGuire's reconstruction filter gathers a small, fixed number of taps along
//! the *dominant* (tile NeighborMax) velocity vector that passes through a
//! pixel, then weights them with the primitives in
//! [`crate::gi::motion_blur::weights`].  This module is the backend-neutral CPU
//! golden reference for *where* those taps live.
//!
//! Given a center pixel, a dominant velocity (in pixels, from
//! [`crate::gi::motion`]'s tile stage), and a sample count `S`, it produces `S`
//! pixel-space offsets spread **symmetrically** across the segment
//! `[-half_velocity, +half_velocity]`.  Placement is fully analytic and
//! deterministic — no RNG — using a per-pixel interleaved-gradient-noise jitter
//! so neighbouring pixels decorrelate their tap positions and the blur reads as
//! continuous rather than banded.
//!
//! # Conventions
//! * Offsets are **relative** to the pixel center, in pixels; add them to the
//!   center coordinate to obtain absolute sample positions.
//! * The dominant velocity defines the sampling *line*; taps are parameterised
//!   by `t in [-1, 1]` and placed at `t * half_velocity`.  With zero jitter the
//!   set is exactly symmetric (its offsets sum to the zero vector).
//! * A (near) zero velocity is a degenerate line, so every offset collapses to
//!   `Vec2::ZERO` — the documented "not moving" fallback.
//! * `jitter` is a value in `[0, 1)`; callers may obtain a deterministic one
//!   from [`interleaved_gradient_jitter`].  Out-of-range or non-finite jitter is
//!   wrapped/sanitized into `[0, 1)`.
//! * Deterministic pure functions only: no RNG/IO/GPU/unsafe.  The returned
//!   `Vec` is the single allocation; every component is finite.

use alloc::vec::Vec;
use bevy_math::{ops, Vec2};

use super::weights::{half_velocity, velocity_magnitude};

/// Smallest sample count the generator will emit.
const MIN_SAMPLES: u32 = 1;

/// Velocity magnitudes below this (in pixels) are treated as "not moving", so
/// the sampler returns all-zero offsets.
pub const DEFAULT_MIN_VELOCITY_PX: f32 = 0.5;

/// Returns the fractional part `x - floor(x)` via [`ops::floor`].
#[inline]
fn fract(x: f32) -> f32 {
    x - ops::floor(x)
}

/// Sanitizes a jitter value into `[0, 1)`.
///
/// Non-finite jitter becomes `0.0`; finite values are reduced modulo `1.0` so
/// any real input maps into the half-open unit interval.
#[inline]
fn sanitize_jitter(jitter: f32) -> f32 {
    if !jitter.is_finite() {
        return 0.0;
    }
    let f = fract(jitter);
    if f < 0.0 { f + 1.0 } else { f }
}

/// Deterministic per-pixel jitter in `[0, 1)` from interleaved gradient noise.
///
/// This is the classic Jimenez IGN hash, which gives a visually pleasing,
/// low-correlation dither across neighbouring pixels without any RNG state.  It
/// is a pure function of the integer-ish pixel coordinate, so the whole
/// reconstruction remains deterministic and reproducible.
///
/// Non-finite coordinates are sanitized to `0.0` before hashing, so the result
/// is always finite and in `[0, 1)`.
#[inline]
pub fn interleaved_gradient_jitter(pixel: Vec2) -> f32 {
    let px = if pixel.x.is_finite() { pixel.x } else { 0.0 };
    let py = if pixel.y.is_finite() { pixel.y } else { 0.0 };
    // Jimenez 2014, "Next Generation Post Processing in Call of Duty".
    let magic = 52.982_918_6_f32;
    let dotted = px * 0.067_110_56 + py * 0.005_837_15;
    sanitize_jitter(magic * fract(dotted))
}

/// Maps a sample index to its parameter `t in [-1, 1]` along the velocity line.
///
/// Uses the McGuire parameterisation `t = mix(-1, 1, (i + 1 + j) / (S + 1))`
/// where `j in [0, 1)` is the jitter.  With `j == 0` the indices `0..S` produce
/// parameters symmetric about `0`, so the overall tap set is balanced around the
/// pixel center; a non-zero `j` slides the whole set along the line to
/// decorrelate neighbouring pixels.  The result is clamped to `[-1, 1]`.
#[inline]
fn sample_parameter(index: u32, count: u32, jitter: f32) -> f32 {
    let denom = (count + 1) as f32;
    let numer = (index + 1) as f32 + jitter;
    let unit = (numer / denom).clamp(0.0, 1.0);
    (2.0 * unit - 1.0).clamp(-1.0, 1.0)
}

/// Generates `count` pixel-space sample **offsets** spread symmetrically along
/// the dominant `velocity` (in pixels), using `jitter in [0, 1)`.
///
/// The taps lie on the segment `[-half_velocity, +half_velocity]`.  When
/// `velocity` is below `DEFAULT_MIN_VELOCITY_PX` the segment is degenerate and
/// every offset is `Vec2::ZERO` (the "not moving" fallback).  `count` is floored
/// to `1`; `jitter` is sanitized into `[0, 1)`.
///
/// The returned vector has exactly `max(count, 1)` finite entries.
#[inline]
pub fn along_velocity_offsets(velocity: Vec2, count: u32, jitter: f32) -> Vec<Vec2> {
    along_velocity_offsets_thresholded(velocity, count, jitter, DEFAULT_MIN_VELOCITY_PX)
}

/// Like [`along_velocity_offsets`] but with a caller-supplied minimum velocity
/// (in pixels) below which the fallback to all-zero offsets kicks in.
///
/// `min_velocity_px` is clamped non-negative; `count` is floored to `1`.
pub fn along_velocity_offsets_thresholded(
    velocity: Vec2,
    count: u32,
    jitter: f32,
    min_velocity_px: f32,
) -> Vec<Vec2> {
    let count = count.max(MIN_SAMPLES);
    let mut offsets = Vec::with_capacity(count as usize);

    let min_velocity_px = if min_velocity_px.is_finite() {
        min_velocity_px.max(0.0)
    } else {
        0.0
    };

    // Degenerate / sub-threshold velocity: no displacement.
    if velocity_magnitude(velocity) < min_velocity_px {
        for _ in 0..count {
            offsets.push(Vec2::ZERO);
        }
        return offsets;
    }

    let half = half_velocity(velocity);
    // Raw jitter in [0, 1): zero yields a symmetric set, non-zero slides it.
    let j = sanitize_jitter(jitter);

    for i in 0..count {
        let t = sample_parameter(i, count, j);
        offsets.push(half * t);
    }
    offsets
}

/// Generates absolute sample **positions** by offsetting `center` with the
/// along-velocity offsets from [`along_velocity_offsets`].
///
/// Convenience wrapper for callers that gather by absolute pixel coordinate.
/// Non-finite `center` components are sanitized to `0.0`.
pub fn along_velocity_positions(
    center: Vec2,
    velocity: Vec2,
    count: u32,
    jitter: f32,
) -> Vec<Vec2> {
    let cx = if center.x.is_finite() { center.x } else { 0.0 };
    let cy = if center.y.is_finite() { center.y } else { 0.0 };
    let center = Vec2::new(cx, cy);
    let mut positions = along_velocity_offsets(velocity, count, jitter);
    for p in positions.iter_mut() {
        *p += center;
    }
    positions
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    #[test]
    fn zero_velocity_falls_back_to_no_offset() {
        let offsets = along_velocity_offsets(Vec2::ZERO, 8, 0.3);
        assert_eq!(offsets.len(), 8);
        for o in offsets {
            assert_eq!(o, Vec2::ZERO);
        }
    }

    #[test]
    fn sub_threshold_velocity_falls_back() {
        // 0.4 px < 0.5 px default threshold.
        let offsets = along_velocity_offsets(Vec2::new(0.4, 0.0), 5, 0.0);
        for o in offsets {
            assert_eq!(o, Vec2::ZERO);
        }
    }

    #[test]
    fn offsets_lie_on_the_velocity_line() {
        let velocity = Vec2::new(10.0, 4.0);
        let offsets = along_velocity_offsets(velocity, 7, 0.0);
        let dir = velocity.normalize();
        for o in offsets {
            // The offset must be colinear with the velocity direction: its
            // perpendicular component is ~0.
            let perp = o - dir * o.dot(dir);
            assert!(perp.length() < EPS, "offset off the line: {o:?}");
            // And it must stay within the +/- half-velocity extent.
            assert!(o.length() <= velocity.length() * 0.5 + EPS);
        }
    }

    #[test]
    fn zero_jitter_is_symmetric() {
        let velocity = Vec2::new(6.0, -8.0);
        let offsets = along_velocity_offsets(velocity, 8, 0.0);
        let sum: Vec2 = offsets.iter().copied().fold(Vec2::ZERO, |a, b| a + b);
        assert!(sum.length() < EPS, "offsets not balanced: sum={sum:?}");
    }

    #[test]
    fn deterministic_across_calls() {
        let velocity = Vec2::new(3.0, 9.0);
        let a = along_velocity_offsets(velocity, 11, 0.37);
        let b = along_velocity_offsets(velocity, 11, 0.37);
        assert_eq!(a, b);
    }

    #[test]
    fn positions_offset_from_center() {
        let center = Vec2::new(100.0, 50.0);
        let velocity = Vec2::new(8.0, 0.0);
        let offsets = along_velocity_offsets(velocity, 5, 0.0);
        let positions = along_velocity_positions(center, velocity, 5, 0.0);
        assert_eq!(offsets.len(), positions.len());
        for (o, p) in offsets.iter().zip(positions.iter()) {
            assert!((*p - (center + *o)).length() < EPS);
        }
    }

    #[test]
    fn count_is_floored_to_one() {
        let offsets = along_velocity_offsets(Vec2::new(5.0, 0.0), 0, 0.0);
        assert_eq!(offsets.len(), 1);
    }

    #[test]
    fn jitter_hash_is_deterministic_and_in_range() {
        for (x, y) in [(0.0, 0.0), (12.0, 7.0), (123.0, 456.0)] {
            let a = interleaved_gradient_jitter(Vec2::new(x, y));
            let b = interleaved_gradient_jitter(Vec2::new(x, y));
            assert_eq!(a, b);
            assert!((0.0..1.0).contains(&a), "jitter out of range: {a}");
        }
        // Different pixels generally differ.
        let p0 = interleaved_gradient_jitter(Vec2::new(1.0, 1.0));
        let p1 = interleaved_gradient_jitter(Vec2::new(2.0, 1.0));
        assert!((p0 - p1).abs() > EPS);
    }

    #[test]
    fn non_finite_inputs_are_sanitized() {
        let offsets = along_velocity_offsets(Vec2::new(f32::NAN, 10.0), 4, f32::INFINITY);
        assert_eq!(offsets.len(), 4);
        for o in offsets {
            assert!(o.x.is_finite() && o.y.is_finite());
        }
        assert!(interleaved_gradient_jitter(Vec2::new(f32::NAN, f32::INFINITY)).is_finite());
    }
}
