//! Sub-pixel camera jitter for **temporal upscaling** (FSR2 / UE-TSR class).
//!
//! Temporal upscaling reconstructs a high-resolution image from a stream of
//! low-resolution render targets. Exactly like TAA it relies on a per-frame
//! sub-pixel camera jitter so that, integrated over the sequence, each *output*
//! (display-resolution) pixel accumulates coverage from many sub-pixel sample
//! positions — but because the render target is *smaller* than the display, one
//! render pixel must resolve `1 / render_scale²` display pixels, so the jitter
//! sequence has to be correspondingly longer to cover them all.
//!
//! This module reuses the low-discrepancy **Halton(2, 3)** generator from
//! [`crate::halton`] (the TAA jitter source) and layers the FSR2 phase-count
//! rule on top: the number of distinct sub-pixel positions grows as the square
//! of the upscale ratio, so a 50% render scale (4x pixels) cycles through 4x as
//! many phases as native. The offsets are expressed in **render-resolution
//! pixels**, centred on zero (`[-0.5, 0.5)` per axis), which is the space the
//! projection shear is applied in before the low-resolution frame is drawn.
//!
//! The renderer removes the same jitter before writing the motion-vector
//! G-buffer (the temporal contract shared with [`crate::taa`]), so history
//! reprojection tracks true surface motion rather than the artificial wobble.

use bevy_math::Vec2;

use crate::halton;

/// The native (unscaled) Halton phase count, matching the TAA default of eight
/// sub-pixel positions. At `render_scale == 1.0` temporal upscaling degrades to
/// plain TAA and reuses exactly this many phases.
pub const BASE_UPSCALE_JITTER_LEN: u32 = 8;

/// An upper bound on the jitter phase count so an extreme (near-zero) render
/// scale cannot ask for an unbounded sequence the history could never
/// integrate. `128` covers render scales down to `25%` at the base length.
pub const MAX_UPSCALE_JITTER_LEN: u32 = 128;

/// The number of distinct Halton phases to cycle for a given `render_scale`
/// (render-resolution / display-resolution, in `(0, 1]`).
///
/// Follows the FSR2 rule `ceil(base / render_scale²)`: halving the render scale
/// quadruples the pixel count each render pixel must resolve, so it quadruples
/// the phase count. The result is clamped to `[1, MAX_UPSCALE_JITTER_LEN]` and
/// a non-positive or non-finite scale falls back to the base length.
#[must_use]
pub fn upscale_phase_count(render_scale: f32) -> u32 {
    if render_scale <= 0.0 || !render_scale.is_finite() {
        return BASE_UPSCALE_JITTER_LEN;
    }
    let scale = render_scale.min(1.0);
    let count = (BASE_UPSCALE_JITTER_LEN as f32 / (scale * scale)).ceil();
    (count as u32).clamp(1, MAX_UPSCALE_JITTER_LEN)
}

/// The sub-pixel jitter offset (in **render-resolution pixels**, centred on
/// zero) for frame `frame` at the given `render_scale`.
///
/// The phase count comes from [`upscale_phase_count`]; the frame index wraps
/// modulo that count and is shifted to one-based so the cycle skips Halton's
/// zero origin, and the raw `[0, 1)` Halton pair is re-centred to `[-0.5, 0.5)`
/// so the mean offset over a full cycle sits on the pixel centre (no net image
/// shift). Identical in spirit to [`crate::taa_jitter`], only with the
/// upscale-aware sequence length.
#[must_use]
pub fn upscale_jitter(frame: u64, render_scale: f32) -> Vec2 {
    let len = upscale_phase_count(render_scale);
    let index = (frame % len as u64) as u32 + 1;
    Vec2::new(halton(index, 2) - 0.5, halton(index, 3) - 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_scale_uses_the_base_phase_count() {
        assert_eq!(upscale_phase_count(1.0), BASE_UPSCALE_JITTER_LEN);
    }

    #[test]
    fn phase_count_scales_with_the_inverse_square_of_render_scale() {
        // 50% render scale => 4x pixels => 4x phases.
        assert_eq!(upscale_phase_count(0.5), BASE_UPSCALE_JITTER_LEN * 4);
        // 1/3 render scale => 9x pixels => 9x phases (72).
        assert_eq!(upscale_phase_count(1.0 / 3.0), BASE_UPSCALE_JITTER_LEN * 9);
    }

    #[test]
    fn phase_count_is_monotonic_in_lower_scales() {
        let a = upscale_phase_count(0.75);
        let b = upscale_phase_count(0.5);
        assert!(
            b >= a,
            "lower render scale must not need fewer phases: {a} vs {b}"
        );
    }

    #[test]
    fn phase_count_is_clamped_and_guards_degenerate_scales() {
        assert_eq!(upscale_phase_count(0.0), BASE_UPSCALE_JITTER_LEN);
        assert_eq!(upscale_phase_count(-1.0), BASE_UPSCALE_JITTER_LEN);
        assert_eq!(upscale_phase_count(f32::NAN), BASE_UPSCALE_JITTER_LEN);
        // A tiny scale saturates at the maximum length rather than exploding.
        assert_eq!(upscale_phase_count(0.01), MAX_UPSCALE_JITTER_LEN);
        // Scales above one are treated as native (never fewer than the base).
        assert_eq!(upscale_phase_count(2.0), BASE_UPSCALE_JITTER_LEN);
    }

    #[test]
    fn jitter_stays_within_the_half_pixel_box() {
        for frame in 0..64u64 {
            let j = upscale_jitter(frame, 0.5);
            assert!(
                j.x >= -0.5 && j.x < 0.5 && j.y >= -0.5 && j.y < 0.5,
                "jitter must stay in [-0.5, 0.5): {j:?}"
            );
        }
    }

    #[test]
    fn jitter_cycle_averages_near_the_pixel_centre() {
        let len = upscale_phase_count(0.5);
        let mut mean = Vec2::ZERO;
        for frame in 0..len as u64 {
            mean += upscale_jitter(frame, 0.5);
        }
        mean /= len as f32;
        assert!(
            mean.length() < 0.1,
            "the jitter cycle must average near the pixel centre: {mean:?}"
        );
    }

    #[test]
    fn jitter_wraps_on_the_phase_count() {
        let len = upscale_phase_count(0.5) as u64;
        assert_eq!(upscale_jitter(0, 0.5), upscale_jitter(len, 0.5));
        assert_eq!(upscale_jitter(3, 0.5), upscale_jitter(len + 3, 0.5));
    }

    #[test]
    fn native_scale_matches_the_taa_jitter_sequence() {
        // At render_scale == 1 the phase count equals the TAA base length, so
        // the two jitter sequences agree frame for frame.
        for frame in 0..BASE_UPSCALE_JITTER_LEN as u64 {
            let up = upscale_jitter(frame, 1.0);
            let taa = crate::taa_jitter(frame, BASE_UPSCALE_JITTER_LEN);
            assert!(
                (up - taa).length() < 1.0e-6,
                "frame {frame}: {up:?} vs {taa:?}"
            );
        }
    }
}
