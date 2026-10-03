//! Colour-endpoint quantisation for the ASTC LDR encoder.
//!
//! This is the encode-side inverse of [`super::super::color_unquant`]: given a
//! target 8-bit colour channel and a colour quant level, it returns the
//! "scrambled packed quant" integer whose unquantization lands closest to the
//! target. Because the decode table already folds *unscramble* and
//! *unquantize* into one indexed read, the chosen packed value can be fed
//! straight into the colour Bounded Integer Sequence (`low | (digit << bits)`)
//! and will decode back through `unquant_color` to the value we selected here.
//!
//! The search is an exhaustive nearest-value lookup over the (at most 256)
//! table entries -- exact, deterministic, pure integer arithmetic (no AI/ML
//! path).

use super::super::color_unquant::{color_quant_num_levels, unquant_color};

/// Quantize an 8-bit colour channel `target` to the packed integer at colour
/// quant level `level_index` (`0..=16`, `QUANT_6 ..= QUANT_256`).
///
/// Returns the packed value whose [`unquant_color`] reconstruction minimises
/// `|unquant - target|`; ties keep the lower packed value. For `QUANT_256`
/// (`level_index >= 16`) the table is the identity, so the result equals
/// `target`.
pub(super) fn quantize_color_channel(level_index: usize, target: u8) -> u8 {
    let levels = color_quant_num_levels(level_index);
    let mut best = 0u8;
    let mut best_err = i32::MAX;
    for packed in 0..levels {
        let got = i32::from(unquant_color(level_index, packed as u8));
        let err = (got - i32::from(target)).abs();
        if err < best_err {
            best_err = err;
            best = packed as u8;
            if err == 0 {
                break;
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::super::super::color_unquant::unquant_color;
    use super::*;

    /// The quantizer is a true nearest-neighbour inverse: no packed value in
    /// the table reconstructs closer to the target than the one it returns.
    #[test]
    fn quantizer_is_nearest_over_all_targets_and_levels() {
        for level in 0..=16usize {
            let levels = super::color_quant_num_levels(level);
            for target in 0..=255u8 {
                let chosen = quantize_color_channel(level, target);
                let chosen_err =
                    (i32::from(unquant_color(level, chosen)) - i32::from(target)).abs();
                for packed in 0..levels {
                    let err =
                        (i32::from(unquant_color(level, packed as u8)) - i32::from(target)).abs();
                    assert!(
                        err >= chosen_err,
                        "level {level} target {target}: packed {packed} (err {err}) beats \
                         chosen {chosen} (err {chosen_err})"
                    );
                }
            }
        }
    }

    /// QUANT_256 (index 16) is the identity: every target maps to itself.
    #[test]
    fn quant256_round_trips_exactly() {
        for target in 0..=255u8 {
            let packed = quantize_color_channel(16, target);
            assert_eq!(packed, target);
            assert_eq!(unquant_color(16, packed), target);
        }
    }

    /// QUANT_192 (index 15) reconstructs every target within the table's
    /// worst-case step (<= 2 LSB on an evenly spaced 192-level ramp).
    #[test]
    fn quant192_is_within_two_lsb() {
        for target in 0..=255u8 {
            let packed = quantize_color_channel(15, target);
            let got = unquant_color(15, packed);
            assert!(
                (i32::from(got) - i32::from(target)).abs() <= 2,
                "QUANT_192 target {target} -> {got} exceeds 2 LSB"
            );
        }
    }
}
