//! Fixed-point quantisation shared by the `GPU` scatter kernels and the `CPU`
//! golden twin.
//!
//! The particle-to-grid (`P2G`) transfer is a *scatter*: many particles add
//! their trilinearly weighted momentum and weight to the same `MAC` face. `WGSL`
//! atomics operate only on 32-bit integers, so the accumulators are kept as
//! fixed-point `i32` values: each floating contribution is multiplied by a
//! scale, rounded to the nearest integer (ties to even, matching `WGSL`
//! `round`), and added with an integer atomic. Integer addition is exact and
//! order-independent, so the massively parallel device sum is deterministic and
//! reproducible on the `CPU`.
//!
//! The `CPU` golden performs the identical quantisation and integer
//! accumulation, so the only float divergence between the two engines is the
//! final `momentum / weight` division after de-quantisation, which the parity
//! test bounds with a tight tolerance.
//!
//! # Overflow budget
//!
//! Accumulators are `i32`, so the sum of quantised contributions per face must
//! stay within `+/- 2^31`. With [`MOMENTUM_SCALE`] and [`WEIGHT_SCALE`] at
//! `2^16` this comfortably covers thousands of overlapping contributions at
//! metre-per-second velocities; larger scenes should lower the scale.
//!
//! # Provenance
//!
//! Fixed-point atomic scatter is a standard `GPU` transfer technique; this
//! module contains no Unreal Engine source or derived code.

/// Fixed-point scale for accumulated face momentum (velocity times weight).
pub const MOMENTUM_SCALE: f32 = 65536.0;

/// Fixed-point scale for accumulated trilinear weights.
pub const WEIGHT_SCALE: f32 = 65536.0;

/// Quantises `value * scale` to the nearest integer using round-half-to-even,
/// matching the `WGSL` `round` builtin.
///
/// The Rust `as i32` cast saturates on overflow and truncates the already
/// integer-valued float, which agrees with the in-range `WGSL` `i32(f32)`
/// conversion.
#[must_use]
pub fn quantise(value: f32, scale: f32) -> i32 {
    (value * scale).round_ties_even() as i32
}

/// De-quantises a fixed-point integer back to a float by dividing by `scale`.
#[must_use]
pub fn dequantise(value: i32, scale: f32) -> f32 {
    value as f32 / scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_is_close() {
        let v = 3.5_f32;
        let q = quantise(v, MOMENTUM_SCALE);
        let back = dequantise(q, MOMENTUM_SCALE);
        assert!((back - v).abs() < 1.0 / MOMENTUM_SCALE);
    }

    #[test]
    fn ties_round_to_even() {
        // 0.5 rounds to 0, 1.5 rounds to 2 (round-half-to-even).
        assert_eq!(quantise(0.5, 1.0), 0);
        assert_eq!(quantise(1.5, 1.0), 2);
        assert_eq!(quantise(-0.5, 1.0), 0);
    }

    #[test]
    fn negative_values_round_symmetrically() {
        assert_eq!(quantise(-2.5, 1.0), -2);
        assert_eq!(quantise(-3.5, 1.0), -4);
    }
}
