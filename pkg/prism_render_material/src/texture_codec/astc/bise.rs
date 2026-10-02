//! ASTC Bounded Integer Sequence Encoding (BISE).
//!
//! ASTC stores both weight and endpoint values as *integer sequences*. Each
//! value is drawn from one of a fixed set of quantisation ranges whose level
//! count `L` is always of one of three forms, and the form selects how the
//! sequence is packed (Khronos Data Format Specification 1.3, "Integer
//! Sequence Encoding"):
//!
//! * `L = 2^b`      -- pure binary: every value is just `b` bits.
//! * `L = 3 * 2^b`  -- a *trit* (base-3 digit, 0..2) plus `b` low bits.
//! * `L = 5 * 2^b`  -- a *quint* (base-5 digit, 0..4) plus `b` low bits.
//!
//! For the trit/quint forms the trits (resp. quints) of five (resp. three)
//! consecutive values are grouped and packed together into an 8-bit (resp.
//! 7-bit) block, interleaved with each value's low bits, so a value cannot be
//! read in isolation -- the whole block must be assembled first.
//!
//! # Milestones
//! The pure-binary path carries no trit/quint tables, so it is provably
//! correct by inspection and is exercised exhaustively by the unit tests
//! below. The trit/quint block-unpack tables (8-bit -> 5 trits, 7-bit ->
//! 3 quints) are error-prone to reproduce from memory and can only be trusted
//! once a full weighted block is proven bit-for-bit against the GPU hardware
//! decoder; until then [`decode_ise`] rejects those ranges with
//! [`AstcError::UnsupportedIse`] rather than emitting unverified values.

use super::block_reader::read_bits;
use super::AstcError;

/// Which of the three BISE packing forms a quantisation range uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum IseKind {
    /// `L = 2^bits`: each value is `bits` raw binary bits.
    Bits,
    /// `L = 3 * 2^bits`: a base-3 trit plus `bits` low bits per value.
    Trit,
    /// `L = 5 * 2^bits`: a base-5 quint plus `bits` low bits per value.
    Quint,
}

/// A decoded BISE quantisation range: its packing form plus the number of
/// per-value low bits `b`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct IseRange {
    pub(super) kind: IseKind,
    pub(super) bits: u32,
}

impl IseRange {
    /// Classify a range from its *number of levels* `levels` (the count of
    /// distinct values the range can represent, e.g. `levels == 2` is a single
    /// bit, `levels == 3` is one trit).
    ///
    /// Returns `None` for a level count that is not one of the three BISE
    /// forms, which cannot occur for a valid ASTC quantisation range.
    #[must_use]
    pub(super) fn from_num_levels(levels: u32) -> Option<Self> {
        if levels == 0 {
            return None;
        }
        if levels.is_power_of_two() {
            return Some(Self {
                kind: IseKind::Bits,
                bits: levels.trailing_zeros(),
            });
        }
        if levels % 3 == 0 && (levels / 3).is_power_of_two() {
            return Some(Self {
                kind: IseKind::Trit,
                bits: (levels / 3).trailing_zeros(),
            });
        }
        if levels % 5 == 0 && (levels / 5).is_power_of_two() {
            return Some(Self {
                kind: IseKind::Quint,
                bits: (levels / 5).trailing_zeros(),
            });
        }
        None
    }
}

/// The number of bits a BISE sequence of `count` values occupies when packed
/// with the given `range`.
///
/// The trit/quint block counts use the standard ceiling forms so that a
/// partially filled final block (fewer than five trits / three quints) is
/// charged only for the bits it actually consumes.
#[must_use]
pub(super) fn ise_sequence_bits(count: u32, range: IseRange) -> u32 {
    let low = count * range.bits;
    match range.kind {
        IseKind::Bits => low,
        // ceil(count * 8 / 5): five trits share one 8-bit block.
        IseKind::Trit => low + (count * 8 + 4) / 5,
        // ceil(count * 7 / 3): three quints share one 7-bit block.
        IseKind::Quint => low + (count * 7 + 2) / 3,
    }
}

/// Decode a BISE integer sequence of `count` values beginning at bit
/// `start_bit` of `block` into `out`.
///
/// Only the pure-binary form is decoded today; see the module docs for why the
/// trit/quint forms are deferred until GPU-proven.
///
/// # Errors
/// * [`AstcError::UnsupportedIse`] if `range` uses trits or quints, which are
///   not yet GPU-validated.
///
/// # Panics (debug only)
/// Panics in debug builds if `out.len() != count` or the sequence runs past
/// bit 128.
pub(super) fn decode_ise(
    block: &[u8; 16],
    start_bit: u32,
    range: IseRange,
    count: u32,
    out: &mut [u8],
) -> Result<(), AstcError> {
    debug_assert_eq!(out.len() as u32, count, "ISE output slice length mismatch");
    match range.kind {
        IseKind::Bits => {
            for i in 0..count {
                let v = read_bits(block, start_bit + i * range.bits, range.bits);
                out[i as usize] = v as u8;
            }
            Ok(())
        }
        IseKind::Trit | IseKind::Quint => Err(AstcError::UnsupportedIse),
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_ise, ise_sequence_bits, IseKind, IseRange};
    use crate::texture_codec::astc::AstcError;

    #[test]
    fn classifies_every_standard_quant_range() {
        // (levels, kind, bits) for the full ASTC quantisation ladder.
        let cases = [
            (2, IseKind::Bits, 1),
            (3, IseKind::Trit, 0),
            (4, IseKind::Bits, 2),
            (5, IseKind::Quint, 0),
            (6, IseKind::Trit, 1),
            (8, IseKind::Bits, 3),
            (10, IseKind::Quint, 1),
            (12, IseKind::Trit, 2),
            (16, IseKind::Bits, 4),
            (20, IseKind::Quint, 2),
            (24, IseKind::Trit, 3),
            (32, IseKind::Bits, 5),
            (40, IseKind::Quint, 3),
            (48, IseKind::Trit, 4),
            (64, IseKind::Bits, 6),
            (80, IseKind::Quint, 4),
            (96, IseKind::Trit, 5),
            (128, IseKind::Bits, 7),
            (160, IseKind::Quint, 5),
            (192, IseKind::Trit, 6),
            (256, IseKind::Bits, 8),
        ];
        for (levels, kind, bits) in cases {
            let r = IseRange::from_num_levels(levels)
                .unwrap_or_else(|| panic!("level {levels} should classify"));
            assert_eq!(r.kind, kind, "kind for {levels} levels");
            assert_eq!(r.bits, bits, "bits for {levels} levels");
        }
    }

    #[test]
    fn rejects_non_bise_level_counts() {
        for levels in [7, 9, 11, 13, 14, 15, 17] {
            assert!(
                IseRange::from_num_levels(levels).is_none(),
                "{levels} is not a BISE form"
            );
        }
        assert!(IseRange::from_num_levels(0).is_none());
    }

    #[test]
    fn sequence_bits_match_spec_forms() {
        // Pure binary: exactly count * bits.
        let b = IseRange {
            kind: IseKind::Bits,
            bits: 5,
        };
        assert_eq!(ise_sequence_bits(0, b), 0);
        assert_eq!(ise_sequence_bits(4, b), 20);

        // Trits: ceil(count*8/5) + count*bits.
        let t = IseRange {
            kind: IseKind::Trit,
            bits: 0,
        };
        assert_eq!(ise_sequence_bits(1, t), 2);
        assert_eq!(ise_sequence_bits(2, t), 4);
        assert_eq!(ise_sequence_bits(3, t), 5);
        assert_eq!(ise_sequence_bits(4, t), 7);
        assert_eq!(ise_sequence_bits(5, t), 8);
        let t2 = IseRange {
            kind: IseKind::Trit,
            bits: 2,
        };
        assert_eq!(ise_sequence_bits(5, t2), 8 + 10);

        // Quints: ceil(count*7/3) + count*bits.
        let q = IseRange {
            kind: IseKind::Quint,
            bits: 0,
        };
        assert_eq!(ise_sequence_bits(1, q), 3);
        assert_eq!(ise_sequence_bits(2, q), 5);
        assert_eq!(ise_sequence_bits(3, q), 7);
        let q1 = IseRange {
            kind: IseKind::Quint,
            bits: 1,
        };
        assert_eq!(ise_sequence_bits(3, q1), 7 + 3);
    }

    #[test]
    fn binary_sequence_round_trips_for_every_width() {
        // For each width 1..=8, pack an ascending sequence LSB-first and verify
        // decode_ise reproduces it exactly.
        for bits in 1u32..=8 {
            let range = IseRange {
                kind: IseKind::Bits,
                bits,
            };
            let mask = if bits == 32 {
                u32::MAX
            } else {
                (1u32 << bits) - 1
            };
            // How many values fit in 128 bits.
            let count = (128 / bits).min(12);
            let mut raw = [0u8; 16];
            let mut expected = [0u8; 12];
            for i in 0..count {
                let v = (i * 7 + 1) & mask; // arbitrary but deterministic
                expected[i as usize] = v as u8;
                let lo = i * bits;
                for bit in 0..bits {
                    if (v >> bit) & 1 == 1 {
                        let pos = lo + bit;
                        raw[(pos >> 3) as usize] |= 1 << (pos & 7);
                    }
                }
            }
            let mut out = alloc::vec![0u8; count as usize];
            decode_ise(&raw, 0, range, count, &mut out).expect("binary ise decodes");
            assert_eq!(&out[..], &expected[..count as usize], "width {bits}");
        }
    }

    #[test]
    fn binary_sequence_honours_start_offset() {
        // Three 4-bit values starting at bit 10: 0xA, 0x5, 0xF.
        let range = IseRange {
            kind: IseKind::Bits,
            bits: 4,
        };
        let vals = [0xAu32, 0x5, 0xF];
        let mut raw = [0u8; 16];
        for (i, v) in vals.iter().enumerate() {
            let lo = 10 + i as u32 * 4;
            for bit in 0..4 {
                if (v >> bit) & 1 == 1 {
                    let pos = lo + bit;
                    raw[(pos >> 3) as usize] |= 1 << (pos & 7);
                }
            }
        }
        let mut out = [0u8; 3];
        decode_ise(&raw, 10, range, 3, &mut out).expect("offset binary ise decodes");
        assert_eq!(out, [0xA, 0x5, 0xF]);
    }

    #[test]
    fn trit_and_quint_ranges_are_rejected_until_gpu_proven() {
        let t = IseRange {
            kind: IseKind::Trit,
            bits: 1,
        };
        let q = IseRange {
            kind: IseKind::Quint,
            bits: 1,
        };
        let block = [0u8; 16];
        let mut out = [0u8; 2];
        assert_eq!(
            decode_ise(&block, 0, t, 2, &mut out),
            Err(AstcError::UnsupportedIse)
        );
        assert_eq!(
            decode_ise(&block, 0, q, 2, &mut out),
            Err(AstcError::UnsupportedIse)
        );
    }
}
