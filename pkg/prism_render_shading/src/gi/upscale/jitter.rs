//! Sub-pixel camera jitter sequences for temporal super-resolution.
//!
//! Temporal upsampling (TAA/TSR style) relies on nudging the camera's
//! projection by a *sub-pixel* offset every frame so that, over a short window,
//! the low-resolution raster samples cover distinct interior positions of each
//! high-resolution pixel.  Accumulating those offset samples reconstructs detail
//! far beyond the native raster resolution — a purely classical signal-recovery
//! trick, with no learned priors anywhere in the pipeline.
//!
//! The offsets come from the **Halton (2, 3)** low-discrepancy sequence, the
//! de-facto standard jitter generator in real-time rendering.  The 1-D radical
//! inverse in base 2 and base 3 produces a 2-D point set that is maximally
//! spread for *any* prefix length, so the first `N` frames of a session already
//! tile the pixel evenly rather than clustering (as white noise would).
//!
//! Offsets are reported in *pixel units* in `[-0.5, 0.5]` so they can be added
//! directly to the normalized-device-coordinate projection as
//! `offset_ndc = jitter_px * 2 / resolution`.
//!
//! # Conventions
//! * `halton_2_3(index)` returns the raw sequence point in `[0, 1)^2`; index `0`
//!   is the sequence origin `(0, 0)`.
//! * Jitter offsets skip the origin (they start at Halton index `1`) so no frame
//!   lands exactly on the pixel's lower-left corner, and are clamped to the
//!   half-open-centred range `[-0.5, 0.5]`.
//! * A jitter *sequence length* is the period after which the offset pattern
//!   repeats; it is clamped to `>= 1` and typically a small power-of-two-ish
//!   count ([`DEFAULT_SEQUENCE_LENGTH`]).
//! * No transcendental maths is required here; all arithmetic is exact-ish f32
//!   division matching the GPU twin.  Every function is a deterministic pure
//!   function with no RNG, I/O, GPU, or `unsafe`, and never returns `NaN`.

use bevy_math::Vec2;

/// Default number of distinct jitter phases before the pattern repeats.
///
/// Eight phases is the common TAA/TSR sweet spot: long enough that the Halton
/// set tiles the pixel well, short enough that disocclusion recovery stays fast.
pub const DEFAULT_SEQUENCE_LENGTH: u32 = 8;

/// Clamps a requested jitter sequence length to the valid non-zero range.
///
/// A length of `0` would make the phase modulo undefined, so it is promoted to
/// `1` (a static, un-jittered camera).  The value is otherwise returned as-is.
#[inline]
pub fn sequence_length_clamped(length: u32) -> u32 {
    length.max(1)
}

/// Van der Corput / radical-inverse of `index` in the given integer `base`.
///
/// Returns the digit-reversed fraction in `[0, 1)`.  Degenerate bases (`< 2`)
/// cannot form a sequence and fall back to `0.0`.
#[inline]
pub fn radical_inverse(base: u32, mut index: u32) -> f32 {
    if base < 2 {
        return 0.0;
    }
    let inv_base = 1.0 / base as f32;
    let mut inv_bi = inv_base;
    let mut result = 0.0f32;
    while index > 0 {
        let digit = index % base;
        result += digit as f32 * inv_bi;
        index /= base;
        inv_bi *= inv_base;
    }
    // The construction keeps `result` in `[0, 1)`; clamp defends against any
    // f32 rounding that would nudge it to exactly `1.0`.
    if result.is_finite() {
        result.clamp(0.0, 0.999_999_94)
    } else {
        0.0
    }
}

/// The raw Halton (2, 3) sequence point for `index`, `(x, y) ∈ [0, 1)^2`.
///
/// Index `0` is the origin `(0, 0)`; successive indices spread the point set
/// with low discrepancy.
#[inline]
pub fn halton_2_3(index: u32) -> Vec2 {
    Vec2::new(radical_inverse(2, index), radical_inverse(3, index))
}

/// The sub-pixel jitter offset for sequence position `index`, in pixel units.
///
/// Uses Halton index `index + 1` so position `0` is a genuine interior offset
/// rather than the pixel corner, then centres the `[0, 1)` point about zero.
/// The result lies in `[-0.5, 0.5]` on both axes and is always finite.
#[inline]
pub fn jitter_offset(index: u32) -> Vec2 {
    // `+1` skips the Halton origin so no phase is the degenerate corner offset.
    let p = halton_2_3(index.wrapping_add(1));
    let offset = p - Vec2::splat(0.5);
    sanitize_offset(offset)
}

/// The current jitter *phase* for an absolute `frame` index given a repeating
/// sequence of `length` phases.
///
/// `length` is clamped via [`sequence_length_clamped`]; the result is
/// `frame % length` and therefore in `[0, length)`.
#[inline]
pub fn jitter_phase(frame: u32, length: u32) -> u32 {
    frame % sequence_length_clamped(length)
}

/// The sub-pixel jitter offset for an absolute `frame` within a repeating
/// sequence of `length` phases, in pixel units `[-0.5, 0.5]`.
#[inline]
pub fn jitter_offset_phased(frame: u32, length: u32) -> Vec2 {
    jitter_offset(jitter_phase(frame, length))
}

/// Converts a pixel-space jitter offset to a normalized-device-coordinate (NDC)
/// projection shift for a framebuffer of the given dimensions.
///
/// NDC spans `[-1, 1]` across `resolution` pixels, so one pixel is `2 /
/// resolution` of NDC.  Zero or negative dimensions are degenerate and yield a
/// zero shift (no jitter) rather than a division by zero.
#[inline]
pub fn jitter_to_ndc(jitter_px: Vec2, width: u32, height: u32) -> Vec2 {
    let sx = if width == 0 { 0.0 } else { 2.0 / width as f32 };
    let sy = if height == 0 { 0.0 } else { 2.0 / height as f32 };
    sanitize_offset(Vec2::new(jitter_px.x * sx, jitter_px.y * sy))
}

/// The mean jitter offset over one full `length`-phase sequence.
///
/// A well-balanced jitter sequence integrates to approximately `(0, 0)` so the
/// time-averaged camera matches the un-jittered camera (no persistent bias).
/// Useful for validating a chosen sequence length.  `length` is clamped to
/// `>= 1`.
#[inline]
pub fn sequence_mean(length: u32) -> Vec2 {
    let n = sequence_length_clamped(length);
    let mut acc = Vec2::ZERO;
    for i in 0..n {
        acc += jitter_offset(i);
    }
    acc / n as f32
}

/// Replaces any non-finite component with `0.0` and clamps to `[-0.5, 0.5]`.
#[inline]
fn sanitize_offset(v: Vec2) -> Vec2 {
    let x = if v.x.is_finite() { v.x.clamp(-0.5, 0.5) } else { 0.0 };
    let y = if v.y.is_finite() { v.y.clamp(-0.5, 0.5) } else { 0.0 };
    Vec2::new(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radical_inverse_base2_known_values() {
        // Classic base-2 van der Corput prefix: 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8.
        let expected = [0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875];
        for (i, &e) in expected.iter().enumerate() {
            let got = radical_inverse(2, (i + 1) as u32);
            assert!((got - e).abs() < 1e-6, "idx {} => {got} want {e}", i + 1);
        }
    }

    #[test]
    fn radical_inverse_base3_known_values() {
        // Base-3: 1/3, 2/3, 1/9, 4/9, 7/9.
        let expected = [1.0 / 3.0, 2.0 / 3.0, 1.0 / 9.0, 4.0 / 9.0, 7.0 / 9.0];
        for (i, &e) in expected.iter().enumerate() {
            let got = radical_inverse(3, (i + 1) as u32);
            assert!((got - e).abs() < 1e-6, "idx {} => {got} want {e}", i + 1);
        }
    }

    #[test]
    fn degenerate_base_falls_back_to_zero() {
        assert_eq!(radical_inverse(0, 7), 0.0);
        assert_eq!(radical_inverse(1, 7), 0.0);
    }

    #[test]
    fn halton_point_in_unit_square() {
        for i in 0..4096u32 {
            let p = halton_2_3(i);
            assert!((0.0..1.0).contains(&p.x), "x={}", p.x);
            assert!((0.0..1.0).contains(&p.y), "y={}", p.y);
        }
    }

    #[test]
    fn jitter_offsets_in_half_pixel_range() {
        for i in 0..4096u32 {
            let o = jitter_offset(i);
            assert!((-0.5..=0.5).contains(&o.x), "x={}", o.x);
            assert!((-0.5..=0.5).contains(&o.y), "y={}", o.y);
            assert!(o.x.is_finite() && o.y.is_finite());
        }
    }

    #[test]
    fn jitter_never_lands_on_corner() {
        // The `+1` origin skip guarantees phase 0 is not the (-0.5, -0.5) corner.
        let o0 = jitter_offset(0);
        assert!(o0.length() > 1e-4, "phase 0 degenerate: {o0:?}");
    }

    #[test]
    fn phase_wraps_within_length() {
        for frame in 0..100u32 {
            let p = jitter_phase(frame, 8);
            assert!(p < 8, "phase {p} out of range");
            assert_eq!(p, frame % 8);
        }
    }

    #[test]
    fn phase_handles_zero_length() {
        // Zero length is promoted to 1: a single static phase.
        assert_eq!(jitter_phase(5, 0), 0);
        assert_eq!(sequence_length_clamped(0), 1);
    }

    #[test]
    fn phased_offset_matches_direct_offset() {
        for frame in 0..64u32 {
            assert_eq!(jitter_offset_phased(frame, 8), jitter_offset(frame % 8));
        }
    }

    #[test]
    fn sequence_mean_is_near_zero() {
        // A balanced jitter window must integrate to ~zero (no camera bias).
        // Short Halton prefixes carry a mild low-index bias; longer windows
        // integrate increasingly close to the pixel centre.
        for &(len, bound) in &[(4u32, 0.2f32), (8, 0.12), (16, 0.08), (32, 0.05)] {
            let m = sequence_mean(len);
            assert!(m.length() < bound, "len {len} mean {m:?} too biased");
        }
    }

    #[test]
    fn low_discrepancy_quadrant_coverage() {
        // The first 8 phases must touch all four pixel quadrants on each axis.
        let mut hit_x = [false; 2];
        let mut hit_y = [false; 2];
        for i in 0..8u32 {
            let o = jitter_offset(i);
            hit_x[(o.x >= 0.0) as usize] = true;
            hit_y[(o.y >= 0.0) as usize] = true;
        }
        assert!(hit_x.iter().all(|&b| b), "x half-coverage incomplete");
        assert!(hit_y.iter().all(|&b| b), "y half-coverage incomplete");
    }

    #[test]
    fn ndc_scale_is_two_over_resolution() {
        let shift = jitter_to_ndc(Vec2::new(0.5, -0.5), 1920, 1080);
        assert!((shift.x - 0.5 * 2.0 / 1920.0).abs() < 1e-9);
        assert!((shift.y + 0.5 * 2.0 / 1080.0).abs() < 1e-9);
    }

    #[test]
    fn ndc_zero_resolution_is_safe() {
        let shift = jitter_to_ndc(Vec2::new(0.3, 0.4), 0, 0);
        assert_eq!(shift, Vec2::ZERO);
    }

    #[test]
    fn is_deterministic() {
        assert_eq!(jitter_offset(13), jitter_offset(13));
        assert_eq!(jitter_offset_phased(27, 8), jitter_offset_phased(27, 8));
    }
}
