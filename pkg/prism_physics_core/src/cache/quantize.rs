//! Fixed-step position quantization for offline physics caches.
//!
//! Baked trajectories store particle positions as integers on a uniform grid
//! rather than raw [`f32`] components. Quantizing to a fixed spatial [`step`] (in
//! metres) makes the on-disk representation compact and, crucially, *exactly*
//! reproducible: encoding then decoding is a pure integer round-trip, so two
//! bakes of the same simulation yield byte-identical caches. This determinism
//! is what lets the golden-replay test compare a stored digest against a fresh
//! bake without floating-point tolerance drift.
//!
//! The grid step is a lossy but bounded approximation: every stored coordinate
//! is within half a [`step`] of the true value, so a [`step`] of `1e-4` m caps the
//! per-axis error at 50 micrometres, far below what soft-body rendering
//! resolves.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Uniform
//! scalar quantization (round-to-nearest on a fixed grid) is a textbook
//! signal-processing primitive.

use glam::Vec3;

use crate::math::scalar::Real;

/// The default quantization grid step, in metres.
///
/// At `1e-4` m the maximum per-axis round-trip error is 50 micrometres, which
/// is imperceptible for cloth, rope, and soft-body playback while keeping the
/// integer coordinates small.
pub const DEFAULT_STEP: Real = 1.0e-4;

/// Maps continuous [`Vec3`] positions onto a uniform integer grid and back.
///
/// The quantizer is defined by a single positive [`step`]: a coordinate `c` encodes
/// to `round(c / step)` and decodes back to `q * step`. Because both directions
/// are deterministic integer/float arithmetic, re-encoding a decoded value is
/// idempotent, which is the property the golden digest relies on.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PositionQuantizer {
    /// The grid step in metres. Always strictly positive.
    step: Real,
}

impl PositionQuantizer {
    /// Creates a quantizer with the given grid `step` in metres.
    ///
    /// A non-finite or non-positive `step` is rejected in favour of
    /// [`DEFAULT_STEP`] so the quantizer can never divide by zero or invert the
    /// grid.
    #[must_use]
    pub fn new(step: Real) -> PositionQuantizer {
        let step = if step.is_finite() && step > 0.0 {
            step
        } else {
            DEFAULT_STEP
        };
        PositionQuantizer { step }
    }

    /// Returns the grid step in metres.
    #[must_use]
    pub fn step(&self) -> Real {
        self.step
    }

    /// Encodes a single scalar coordinate to its nearest grid index.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "round() yields an integral Real; the grid indices we bake stay well within i32 range for physically sized scenes."
    )]
    fn encode_scalar(&self, c: Real) -> i32 {
        (c / self.step).round() as i32
    }

    /// Encodes a position to three grid indices (round-to-nearest per axis).
    #[must_use]
    pub fn encode(&self, position: Vec3) -> [i32; 3] {
        [
            self.encode_scalar(position.x),
            self.encode_scalar(position.y),
            self.encode_scalar(position.z),
        ]
    }

    /// Decodes three grid indices back to a continuous position.
    #[must_use]
    pub fn decode(&self, quantized: [i32; 3]) -> Vec3 {
        Vec3::new(
            quantized[0] as Real * self.step,
            quantized[1] as Real * self.step,
            quantized[2] as Real * self.step,
        )
    }
}

impl Default for PositionQuantizer {
    fn default() -> Self {
        PositionQuantizer::new(DEFAULT_STEP)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_positive_step_falls_back_to_default() {
        assert_eq!(PositionQuantizer::new(0.0).step(), DEFAULT_STEP);
        assert_eq!(PositionQuantizer::new(-1.0).step(), DEFAULT_STEP);
        assert_eq!(PositionQuantizer::new(Real::NAN).step(), DEFAULT_STEP);
        assert_eq!(PositionQuantizer::new(Real::INFINITY).step(), DEFAULT_STEP);
    }

    #[test]
    fn encode_decode_round_trips_within_half_step() {
        let q = PositionQuantizer::new(1.0e-3);
        let p = Vec3::new(0.123_45, -1.987_65, 42.0);
        let decoded = q.decode(q.encode(p));
        let err = (decoded - p).abs();
        let half = q.step() * 0.5 + 1e-6;
        assert!(
            err.x <= half && err.y <= half && err.z <= half,
            "err={err:?}"
        );
    }

    #[test]
    fn re_encoding_a_decoded_value_is_idempotent() {
        let q = PositionQuantizer::new(2.5e-4);
        let p = Vec3::new(-3.3, 0.75, 9.001);
        let once = q.encode(p);
        let twice = q.encode(q.decode(once));
        assert_eq!(once, twice);
    }

    #[test]
    fn origin_encodes_to_zero() {
        let q = PositionQuantizer::default();
        assert_eq!(q.encode(Vec3::ZERO), [0, 0, 0]);
        assert_eq!(q.decode([0, 0, 0]), Vec3::ZERO);
    }
}
