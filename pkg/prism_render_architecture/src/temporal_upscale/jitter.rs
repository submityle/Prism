//! Deterministic sub-pixel jitter sequence for temporal reconstruction.
//!
//! Temporal upsamplers (`DLSS` / `FSR` / `XeSS`-class reconstructors) sample the
//! scene at a fractional sub-pixel offset that changes every frame. Integrated
//! over the reconstruction window those offsets tile the display pixel, so the
//! accumulated history carries more spatial detail than any single low-render
//! frame. The camera jitter that produces those offsets must be *deterministic*
//! and *low-discrepancy*: deterministic so the `CPU` reference and a future
//! `GPU` kernel agree bit-for-bit, low-discrepancy so the offsets spread evenly
//! instead of clustering.
//!
//! The classic choice is the Halton sequence with bases `(2, 3)`. This module
//! generates it with the **integer radical inverse**: the base-`b` digits of an
//! index are read most-significant-first after reflection about the radix
//! point. That is pure integer arithmetic plus a reciprocal multiply — no
//! transcendental functions — which keeps the sequence reproducible under the
//! crate's determinism policy.
//!
//! The `GPU`-side constant-buffer upload of the active jitter offset is out of
//! scope and pending the `GPU` backend; this module fixes the exact offsets
//! that upload must carry.

/// The Halton base used for the horizontal jitter axis.
pub const HALTON_BASE_X: u32 = 2;

/// The Halton base used for the vertical jitter axis.
pub const HALTON_BASE_Y: u32 = 3;

/// Default number of frames after which the jitter phase repeats.
///
/// Eight phases is the common `TAA` / upsampler default: long enough to tile a
/// pixel well, short enough that a paused camera converges quickly.
pub const DEFAULT_JITTER_PERIOD: u32 = 8;

/// Computes the base-`base` radical inverse of `index` in `[0, 1)`.
///
/// The radical inverse reflects the base-`base` digits of `index` about the
/// radix point: digit `d0 d1 d2 ...` becomes `0.d0 d1 d2 ...`. It is evaluated
/// with integer division/modulo and a running reciprocal-power multiply, so the
/// only floating-point operations are `+`, `*`, and the initial reciprocal —
/// no transcendental calls, keeping the result deterministic.
///
/// A `base` below `2` has no meaningful positional expansion, so it returns
/// `0.0` rather than looping forever.
#[must_use]
pub fn radical_inverse(base: u32, mut index: u32) -> f32 {
    if base < 2 {
        return 0.0;
    }
    // Reciprocal of the base; each successive digit is weighted by an
    // additional factor of this reciprocal (i.e. base^-1, base^-2, ...).
    let inv_base = 1.0 / base as f32;
    let mut inv_weight = inv_base;
    let mut result = 0.0f32;
    while index > 0 {
        let digit = index % base;
        result += digit as f32 * inv_weight;
        index /= base;
        inv_weight *= inv_base;
    }
    result
}

/// A repeating, low-discrepancy camera-jitter schedule.
///
/// The sequence maps an absolute frame number to a sub-pixel offset in
/// `[-0.5, 0.5)` on each axis, using Halton `(2, 3)`. The offsets repeat with
/// [`Self::period`] frames so a static view converges to a fixed accumulation
/// pattern instead of drifting forever.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct JitterSequence {
    period: u32,
    base_x: u32,
    base_y: u32,
}

impl JitterSequence {
    /// Builds a sequence with an explicit period, clamped to at least one frame.
    ///
    /// A zero period would make the phase mapping divide by zero, so it is
    /// promoted to `1` (a constant, un-jittered phase).
    #[must_use]
    pub const fn with_period(period: u32) -> Self {
        let period = if period == 0 { 1 } else { period };
        Self {
            period,
            base_x: HALTON_BASE_X,
            base_y: HALTON_BASE_Y,
        }
    }

    /// The number of distinct phases before the offsets repeat.
    #[must_use]
    pub const fn period(self) -> u32 {
        self.period
    }

    /// Maps an absolute frame number to a phase in `[0, period)`.
    ///
    /// The phase is the frame number modulo the period, so consecutive frames
    /// walk the whole sequence and then wrap.
    #[must_use]
    pub const fn phase_for_frame(self, frame: u64) -> u32 {
        (frame % self.period as u64) as u32
    }

    /// Returns the sub-pixel offset for a phase, each axis in `[-0.5, 0.5)`.
    ///
    /// The Halton index is `phase + 1`: index `0` yields `(0, 0)`, which would
    /// waste a phase on the un-jittered center, so the schedule skips it. The
    /// raw radical inverse lies in `[0, 1)`; subtracting `0.5` recenters it on
    /// the pixel so positive and negative offsets are balanced.
    #[must_use]
    pub fn offset_for_phase(self, phase: u32) -> [f32; 2] {
        let index = phase.wrapping_add(1);
        let x = radical_inverse(self.base_x, index) - 0.5;
        let y = radical_inverse(self.base_y, index) - 0.5;
        [x, y]
    }

    /// Convenience: the sub-pixel offset for an absolute frame number.
    #[must_use]
    pub fn offset_for_frame(self, frame: u64) -> [f32; 2] {
        self.offset_for_phase(self.phase_for_frame(frame))
    }
}

impl Default for JitterSequence {
    fn default() -> Self {
        Self::with_period(DEFAULT_JITTER_PERIOD)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-6
    }

    #[test]
    fn radical_inverse_known_values() {
        // Base 2: index 1 -> 0.1b = 0.5, index 2 -> 0.01b = 0.25,
        // index 3 -> 0.11b = 0.75.
        assert!(approx(radical_inverse(2, 1), 0.5));
        assert!(approx(radical_inverse(2, 2), 0.25));
        assert!(approx(radical_inverse(2, 3), 0.75));
        // Base 3: index 1 -> 0.1(base3) = 1/3.
        assert!(approx(radical_inverse(3, 1), 1.0 / 3.0));
        // Index 0 is always the origin.
        assert!(approx(radical_inverse(2, 0), 0.0));
    }

    #[test]
    fn degenerate_base_returns_zero() {
        assert!(approx(radical_inverse(0, 7), 0.0));
        assert!(approx(radical_inverse(1, 7), 0.0));
    }

    #[test]
    fn offsets_lie_in_half_open_pixel() {
        let seq = JitterSequence::default();
        for frame in 0u64..4096 {
            let [x, y] = seq.offset_for_frame(frame);
            assert!((-0.5..0.5).contains(&x), "x out of range: {x}");
            assert!((-0.5..0.5).contains(&y), "y out of range: {y}");
            assert!(!x.is_nan() && !y.is_nan());
        }
    }

    #[test]
    fn sequence_is_deterministic_and_periodic() {
        let seq = JitterSequence::with_period(8);
        // Same frame -> same offset, and frame N matches frame N + period.
        for frame in 0u64..64 {
            let a = seq.offset_for_frame(frame);
            let b = seq.offset_for_frame(frame);
            assert_eq!(a, b);
            let wrapped = seq.offset_for_frame(frame + seq.period() as u64);
            assert_eq!(a, wrapped);
        }
    }

    #[test]
    fn zero_period_is_promoted_to_one() {
        let seq = JitterSequence::with_period(0);
        assert_eq!(seq.period(), 1);
        // Every frame collapses to phase 0 with a single fixed offset.
        assert_eq!(seq.offset_for_frame(0), seq.offset_for_frame(999));
    }

    #[test]
    fn phase_mapping_wraps_at_period() {
        let seq = JitterSequence::with_period(4);
        assert_eq!(seq.phase_for_frame(0), 0);
        assert_eq!(seq.phase_for_frame(3), 3);
        assert_eq!(seq.phase_for_frame(4), 0);
        assert_eq!(seq.phase_for_frame(5), 1);
    }
}
