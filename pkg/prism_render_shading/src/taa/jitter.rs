//! Sub-pixel camera jitter for temporal anti-aliasing.
//!
//! TAA turns the temporal history into a supersampler: every frame the camera's
//! projection is nudged by a fraction of a pixel so that, integrated over the
//! sequence, each output pixel accumulates coverage from many sub-pixel sample
//! positions. The classic AAA choice is a low-discrepancy **Halton(2, 3)**
//! sequence — its 2D points cover the pixel far more evenly than a random or a
//! regular grid at any prefix length, so even a handful of frames already
//! resolve edges cleanly and the pattern never clumps.
//!
//! The offsets here are expressed in **pixels**, centred on zero
//! (`[-0.5, 0.5]` per axis). The renderer converts them to a clip-space shear
//! (`2 * offset / viewport`) added to the projection's `z` column, and — this
//! is the crucial TAA contract — **removes the same jitter before writing the
//! motion-vector G-buffer**, so history reprojection tracks true surface motion
//! rather than the artificial per-frame wobble.
//!
//! Mirrored bit-for-bit by the GPU jitter upload; the sequence is pure integer
//! math so CPU and GPU agree exactly.

use bevy_math::Vec2;

/// The default Halton sequence length. Eight sub-pixel positions is the AAA
/// sweet spot: long enough to resolve edges smoothly, short enough that the
/// history never has to remember a stale sample for long (which would show as
/// lag on motion).
pub const DEFAULT_TAA_JITTER_LEN: u32 = 8;

/// The radical-inverse **Halton** value for a one-based `index` in `base`.
///
/// `base` must be a prime (`2` and `3` for the two screen axes). Index `0`
/// returns `0`; callers use one-based indices so the sequence never starts at
/// the pixel centre `(0, 0)` (which would waste a frame on a zero offset).
pub fn halton(mut index: u32, base: u32) -> f32 {
    debug_assert!(base >= 2, "Halton base must be at least 2");
    let mut fraction = 1.0_f32;
    let mut result = 0.0_f32;
    while index > 0 {
        fraction /= base as f32;
        result += fraction * (index % base) as f32;
        index /= base;
    }
    result
}

/// The sub-pixel jitter offset (in pixels, centred on zero) for frame `frame`
/// of a `sequence_len`-long Halton(2, 3) cycle.
///
/// The frame index wraps modulo `sequence_len` and is shifted to one-based so
/// the cycle skips Halton's zero start; the raw `[0, 1)` Halton pair is then
/// re-centred to `[-0.5, 0.5)` so the mean offset over the cycle sits on the
/// pixel centre (no net image shift).
pub fn taa_jitter(frame: u64, sequence_len: u32) -> Vec2 {
    let len = sequence_len.max(1);
    let index = (frame % len as u64) as u32 + 1;
    Vec2::new(halton(index, 2) - 0.5, halton(index, 3) - 0.5)
}
