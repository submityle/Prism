//! ASTC weight *unquantization* for trit- and quint-form weight ranges.
//!
//! The bit-only (power-of-two level count) weight ranges unquantize by the
//! Khronos bit-replication rule (see [`super::weights::unquant_weight_bits`]).
//! The trit- and quint-form ranges instead use the small fixed lookup tables
//! transcribed verbatim here from the ARM `astcenc` reference encoder
//! (`astcenc_weight_quant_xfer_tables.cpp`, Apache-2.0), specifically the
//! `unscramble_and_unquant_map` sub-array of each `quant_and_xfer_tables`
//! entry. That map is indexed by the raw value produced by the BISE decoder
//! (`low_bits | (digit << bits)`, exactly what [`super::trit_quint`] emits) and
//! yields the final weight on the `0..=64` interpolation scale.
//!
//! Only the ranges a *weight* grid can use appear here: a weight is at most a
//! five-level trit (QUANT_24) or three-level quint (QUANT_20); QUANT_32 and
//! smaller power-of-two ranges are bit-only. The maps therefore need at most
//! 24 entries each.
//!
//! Decode is pure integer table lookup -- no AI/ML path.

use super::bise::{IseKind, IseRange};
use super::weights::unquant_weight_bits;

/// QUANT_3 (trit, 0 low bits): 3 levels.
const TRIT_B0: [u8; 3] = [0, 32, 64];
/// QUANT_6 (trit, 1 low bit): 6 levels.
const TRIT_B1: [u8; 6] = [0, 64, 12, 52, 25, 39];
/// QUANT_12 (trit, 2 low bits): 12 levels.
const TRIT_B2: [u8; 12] = [0, 64, 17, 47, 5, 59, 23, 41, 11, 53, 28, 36];
/// QUANT_24 (trit, 3 low bits): 24 levels.
const TRIT_B3: [u8; 24] = [
    0, 64, 8, 56, 16, 48, 24, 40, 2, 62, 11, 53, 19, 45, 27, 37, 5, 59, 13, 51, 22, 42, 30, 34,
];

/// QUANT_5 (quint, 0 low bits): 5 levels.
const QUINT_B0: [u8; 5] = [0, 16, 32, 48, 64];
/// QUANT_10 (quint, 1 low bit): 10 levels.
const QUINT_B1: [u8; 10] = [0, 64, 7, 57, 14, 50, 21, 43, 28, 36];
/// QUANT_20 (quint, 2 low bits): 20 levels.
const QUINT_B2: [u8; 20] = [
    0, 64, 16, 48, 3, 61, 19, 45, 6, 58, 23, 41, 9, 55, 26, 38, 13, 51, 29, 35,
];

/// Unquantize a raw weight `value` drawn from `range` onto the `0..=64`
/// interpolation scale.
///
/// Bit-only ranges delegate to the bit-replication rule. Trit/quint ranges
/// index the `astcenc` `unscramble_and_unquant_map` for the matching level
/// count; the index is the BISE decoder output `low | (digit << bits)`.
///
/// # Panics (debug only)
/// Panics in debug builds if `value` is out of range for `range`, or if a
/// trit/quint range carries more low bits than any valid weight range.
#[must_use]
pub(super) fn unquant_weight(value: u32, range: IseRange) -> u8 {
    match range.kind {
        IseKind::Bits => unquant_weight_bits(value, range.bits),
        IseKind::Trit => {
            let map: &[u8] = match range.bits {
                0 => &TRIT_B0,
                1 => &TRIT_B1,
                2 => &TRIT_B2,
                3 => &TRIT_B3,
                _ => {
                    debug_assert!(false, "trit weight range has at most 3 low bits");
                    &TRIT_B3
                }
            };
            debug_assert!(
                (value as usize) < map.len(),
                "trit weight value out of range"
            );
            map[value as usize]
        }
        IseKind::Quint => {
            let map: &[u8] = match range.bits {
                0 => &QUINT_B0,
                1 => &QUINT_B1,
                2 => &QUINT_B2,
                _ => {
                    debug_assert!(false, "quint weight range has at most 2 low bits");
                    &QUINT_B2
                }
            };
            debug_assert!(
                (value as usize) < map.len(),
                "quint weight value out of range"
            );
            map[value as usize]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(kind: IseKind, bits: u32) -> IseRange {
        IseRange { kind, bits }
    }

    #[test]
    fn endpoints_map_to_the_full_scale() {
        // Index 0 is always weight 0; index 1 is always weight 64 for the
        // trit/quint forms (the astcenc maps place the two extreme weights
        // first), while bit-only maxima are at the top index.
        for b in 0u32..=3 {
            assert_eq!(
                unquant_weight(0, range(IseKind::Trit, b)),
                0,
                "trit b{b} zero"
            );
        }
        for b in 0u32..=2 {
            assert_eq!(
                unquant_weight(0, range(IseKind::Quint, b)),
                0,
                "quint b{b} zero"
            );
        }
        // For every range with at least one low bit the astcenc map places the
        // maximum weight (64) at index 1.
        for b in 1u32..=3 {
            assert_eq!(
                unquant_weight(1, range(IseKind::Trit, b)),
                64,
                "trit b{b} max"
            );
        }
        for b in 1u32..=2 {
            assert_eq!(
                unquant_weight(1, range(IseKind::Quint, b)),
                64,
                "quint b{b} max"
            );
        }
    }

    #[test]
    fn trit_quint_tables_match_astcenc() {
        assert_eq!(TRIT_B0, [0, 32, 64]);
        assert_eq!(TRIT_B1, [0, 64, 12, 52, 25, 39]);
        assert_eq!(QUINT_B0, [0, 16, 32, 48, 64]);
        assert_eq!(QUINT_B1, [0, 64, 7, 57, 14, 50, 21, 43, 28, 36]);
        // Spot-check the largest tables' extremes.
        assert_eq!(TRIT_B3[0], 0);
        assert_eq!(TRIT_B3[1], 64);
        assert_eq!(QUINT_B2[0], 0);
        assert_eq!(QUINT_B2[1], 64);
    }

    #[test]
    fn every_table_value_is_within_scale() {
        for b in 0u32..=3 {
            for v in 0..(3u32 << b) {
                assert!(unquant_weight(v, range(IseKind::Trit, b)) <= 64);
            }
        }
        for b in 0u32..=2 {
            for v in 0..(5u32 << b) {
                assert!(unquant_weight(v, range(IseKind::Quint, b)) <= 64);
            }
        }
    }

    #[test]
    fn bit_only_delegates_to_replication() {
        // 2-bit (QUANT_4) weight unquantization: 0, 21, 43, 64.
        let r = range(IseKind::Bits, 2);
        assert_eq!(unquant_weight(0, r), 0);
        assert_eq!(unquant_weight(1, r), 21);
        assert_eq!(unquant_weight(2, r), 43);
        assert_eq!(unquant_weight(3, r), 64);
    }
}
