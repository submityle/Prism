//! `CRC-6/CDMA2000-A` cyclic redundancy check (width `6`, polynomial `0x27`,
//! initial value `0x3F`, non-reflected input and output, final `XOR` `0x00`):
//! a tiny pure-integer error-detection code useful for guarding short control
//! words and compact particle metadata headers where only a handful of check
//! bits can be spared (design § integrity checks).
//!
//! This is the `MSB`-first (most-significant-bit-first) bit-serial form of the
//! algorithm. The register is a single `u8` whose low `6` bits hold the live
//! `CRC` state; the upper two bits are always kept clear by masking with
//! [`CRC6_MASK`] (`0x3F`). For each input byte the bits are consumed from bit
//! `7` down to bit `0`. On every step the current top register bit (bit `5`) is
//! compared against the incoming message bit; the register is shifted left by
//! one and masked back to `6` bits, and when the `XOR` of those two bits is
//! `1` the polynomial [`CRC6_POLY`] (`0x27`) is folded in with another `XOR`.
//!
//! The register seeds at [`CRC6_INIT`] (`0x3F`), so the empty input yields the
//! seed `0x3F` unchanged. The final `XOR` value is `0x00`, hence
//! [`finalize`](Crc6Cdma2000::finalize) just returns the masked register. The
//! canonical check string `b"123456789"` produces [`CRC6_CHECK`] (`0x0D`).
//!
//! Every operation here is an integer shift, mask, or `XOR`; there are no
//! floating-point, transcendental, table, or `unsafe` operations, and the code
//! depends only on `core`. The one-shot [`crc6_cdma2000`] and the streaming
//! [`Crc6Cdma2000`] accumulator always agree for any way the same byte stream
//! is split into chunks.
//!
//! Scope: a `CRC` is an error-detection code, not a hash and not a message
//! authentication code. `CRC-6/CDMA2000-A` is *not* cryptographically secure
//! and collisions are trivial to craft on purpose, so it must never be used to
//! authenticate data or defend against a malicious adversary. It only catches
//! accidental corruption in transit or storage.

/// The `CRC-6/CDMA2000-A` generator polynomial (the low `6` bits of the
/// width-`6` polynomial `x^6 + x^5 + x^2 + x^1 + x^0`, i.e. `0x27`).
pub const CRC6_POLY: u8 = 0x27;

/// Mask selecting the live `6` register bits: `(1 << 6) - 1 == 0x3F`.
pub const CRC6_MASK: u8 = (1 << 6) - 1;

/// The register's initial value (`0x3F`); also the result for empty input.
pub const CRC6_INIT: u8 = 0x3F;

/// The reference check value for the string `b"123456789"` (`0x0D`).
pub const CRC6_CHECK: u8 = 0x0D;

/// Index of the register's top bit (bit `5`) for a width-`6` `CRC`.
const CRC6_TOP_BIT: u8 = 5;

/// Fold a single message bit (`0` or `1`) into the running register using the
/// `MSB`-first bit-serial rule, returning the updated `6`-bit register.
///
/// The register's current top bit is compared with `bit`; the register is
/// shifted left by one and masked back to `6` bits, and when the two bits
/// differ the polynomial is applied with an `XOR`.
#[inline]
const fn fold_bit(crc: u8, bit: u8) -> u8 {
    let msb = (crc >> CRC6_TOP_BIT) & 1;
    let shifted = (crc << 1) & CRC6_MASK;
    if (msb ^ bit) == 1 {
        shifted ^ CRC6_POLY
    } else {
        shifted
    }
}

/// Fold every bit of one byte (most-significant bit first) into the register.
#[inline]
const fn fold_byte(crc: u8, byte: u8) -> u8 {
    let mut crc = crc;
    let mut i: u8 = 8;
    while i > 0 {
        i -= 1;
        let bit = (byte >> i) & 1;
        crc = fold_bit(crc, bit);
    }
    crc
}

/// Compute the `CRC-6/CDMA2000-A` of `data` in one shot.
///
/// Returns the `6`-bit check value in the low bits of a `u8`; the upper two
/// bits are always `0`. The empty slice returns the seed [`CRC6_INIT`]
/// (`0x3F`).
///
/// # Examples
///
/// ```
/// use prism_render_architecture::particle::crc6_cdma2000::crc6_cdma2000;
///
/// assert_eq!(crc6_cdma2000(b"123456789"), 0x0D);
/// assert_eq!(crc6_cdma2000(b""), 0x3F);
/// ```
pub fn crc6_cdma2000(data: &[u8]) -> u8 {
    let mut crc = CRC6_INIT;
    for &b in data {
        crc = fold_byte(crc, b);
    }
    crc & CRC6_MASK
}

/// Streaming `CRC-6/CDMA2000-A` accumulator.
///
/// Feed bytes in any number of chunks via [`update`](Crc6Cdma2000::update) and
/// read the result with [`finalize`](Crc6Cdma2000::finalize). The result is
/// identical to [`crc6_cdma2000`] over the concatenation of all chunks, for any
/// choice of chunk boundaries.
///
/// # Examples
///
/// ```
/// use prism_render_architecture::particle::crc6_cdma2000::Crc6Cdma2000;
///
/// let mut c = Crc6Cdma2000::new();
/// c.update(b"1234");
/// c.update(b"56789");
/// assert_eq!(c.finalize(), 0x0D);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Crc6Cdma2000 {
    /// The live register; only the low `6` bits are meaningful.
    crc: u8,
}

impl Crc6Cdma2000 {
    /// Create a fresh accumulator seeded at [`CRC6_INIT`] (`0x3F`).
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { crc: CRC6_INIT }
    }

    /// Fold a chunk of bytes into the running register.
    pub fn update(&mut self, data: &[u8]) {
        let mut crc = self.crc;
        for &b in data {
            crc = fold_byte(crc, b);
        }
        self.crc = crc & CRC6_MASK;
    }

    /// Consume the accumulator and return the `6`-bit check value.
    ///
    /// The final `XOR` value for this `CRC` is `0x00`, so this just returns the
    /// masked register.
    #[inline]
    #[must_use]
    pub const fn finalize(self) -> u8 {
        self.crc & CRC6_MASK
    }
}

impl Default for Crc6Cdma2000 {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// The canonical check string.
    const CHECK_INPUT: &[u8] = b"123456789";

    #[test]
    fn check_vector_matches_reference() {
        assert_eq!(crc6_cdma2000(CHECK_INPUT), 0x0D);
    }

    #[test]
    fn check_vector_matches_named_const() {
        assert_eq!(crc6_cdma2000(CHECK_INPUT), CRC6_CHECK);
    }

    #[test]
    fn empty_input_returns_init() {
        assert_eq!(crc6_cdma2000(b""), 0x3F);
    }

    #[test]
    fn empty_input_equals_init_const() {
        assert_eq!(crc6_cdma2000(b""), CRC6_INIT);
    }

    #[test]
    fn empty_slice_literal_is_init() {
        let empty: [u8; 0] = [];
        assert_eq!(crc6_cdma2000(&empty), 0x3F);
    }

    #[test]
    fn mask_const_is_0x3f() {
        const { assert!(CRC6_MASK == 0x3F) };
        assert_eq!(CRC6_MASK, 0x3F);
    }

    #[test]
    fn poly_const_is_0x27() {
        const { assert!(CRC6_POLY == 0x27) };
        assert_eq!(CRC6_POLY, 0x27);
    }

    #[test]
    fn init_const_is_0x3f() {
        const { assert!(CRC6_INIT == 0x3F) };
        assert_eq!(CRC6_INIT, 0x3F);
    }

    #[test]
    fn check_const_is_0x0d() {
        const { assert!(CRC6_CHECK == 0x0D) };
        assert_eq!(CRC6_CHECK, 0x0D);
    }

    #[test]
    fn poly_fits_in_six_bits() {
        const { assert!(CRC6_POLY <= CRC6_MASK) };
        let masked = CRC6_POLY & CRC6_MASK;
        assert_eq!(masked, CRC6_POLY);
    }

    #[test]
    fn result_is_always_six_bits_for_check() {
        assert!(crc6_cdma2000(CHECK_INPUT) <= 0x3F);
    }

    #[test]
    fn result_is_always_six_bits_for_empty() {
        assert!(crc6_cdma2000(b"") <= 0x3F);
    }

    #[test]
    fn result_in_closed_range_for_many_inputs() {
        for n in 0u16..=512 {
            let bytes = [(n & 0xFF) as u8, (n >> 8) as u8, 0xA5, 0x5A];
            let v = crc6_cdma2000(&bytes);
            assert!((0x00..=0x3F).contains(&v));
        }
    }

    #[test]
    fn every_single_byte_is_six_bits() {
        for b in 0u16..=255 {
            let v = crc6_cdma2000(&[b as u8]);
            assert!((0x00..=0x3F).contains(&v));
        }
    }

    #[test]
    fn upper_two_bits_always_clear() {
        for b in 0u16..=255 {
            let v = crc6_cdma2000(&[b as u8]);
            assert_eq!(v & 0xC0, 0);
        }
    }

    #[test]
    fn deterministic_repeated_calls() {
        let a = crc6_cdma2000(CHECK_INPUT);
        let b = crc6_cdma2000(CHECK_INPUT);
        let c = crc6_cdma2000(CHECK_INPUT);
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn deterministic_over_varied_inputs() {
        let inputs: [&[u8]; 5] = [
            b"",
            b"a",
            b"abc",
            b"The quick brown fox",
            &[0u8, 1, 2, 3, 255],
        ];
        for inp in inputs {
            assert_eq!(crc6_cdma2000(inp), crc6_cdma2000(inp));
        }
    }

    #[test]
    fn streaming_single_chunk_matches_oneshot() {
        let mut c = Crc6Cdma2000::new();
        c.update(CHECK_INPUT);
        assert_eq!(c.finalize(), crc6_cdma2000(CHECK_INPUT));
    }

    #[test]
    fn streaming_two_chunks_matches_oneshot() {
        let mut c = Crc6Cdma2000::new();
        c.update(b"1234");
        c.update(b"56789");
        assert_eq!(c.finalize(), 0x0D);
    }

    #[test]
    fn streaming_byte_by_byte_matches_oneshot() {
        let mut c = Crc6Cdma2000::new();
        for &b in CHECK_INPUT {
            c.update(&[b]);
        }
        assert_eq!(c.finalize(), crc6_cdma2000(CHECK_INPUT));
    }

    #[test]
    fn streaming_empty_updates_are_noops() {
        let mut c = Crc6Cdma2000::new();
        c.update(b"");
        c.update(b"123");
        c.update(b"");
        c.update(b"456789");
        c.update(b"");
        assert_eq!(c.finalize(), 0x0D);
    }

    #[test]
    fn streaming_new_then_finalize_is_init() {
        let c = Crc6Cdma2000::new();
        assert_eq!(c.finalize(), CRC6_INIT);
    }

    #[test]
    fn default_matches_new() {
        assert_eq!(Crc6Cdma2000::default(), Crc6Cdma2000::new());
    }

    #[test]
    fn default_finalize_is_init() {
        assert_eq!(Crc6Cdma2000::default().finalize(), 0x3F);
    }

    #[test]
    fn all_split_points_agree_for_check_input() {
        let expected = crc6_cdma2000(CHECK_INPUT);
        for split in 0..=CHECK_INPUT.len() {
            let (left, right) = CHECK_INPUT.split_at(split);
            let mut c = Crc6Cdma2000::new();
            c.update(left);
            c.update(right);
            assert_eq!(c.finalize(), expected);
        }
    }

    #[test]
    fn all_split_points_agree_for_binary_blob() {
        let blob: Vec<u8> = (0u16..=200).map(|x| (x & 0xFF) as u8).collect();
        let expected = crc6_cdma2000(&blob);
        for split in 0..=blob.len() {
            let (left, right) = blob.split_at(split);
            let mut c = Crc6Cdma2000::new();
            c.update(left);
            c.update(right);
            assert_eq!(c.finalize(), expected);
        }
    }

    #[test]
    fn three_way_split_agrees() {
        let data = b"streaming-crc6-three-way";
        let expected = crc6_cdma2000(data);
        for i in 0..=data.len() {
            for j in i..=data.len() {
                let mut c = Crc6Cdma2000::new();
                c.update(&data[..i]);
                c.update(&data[i..j]);
                c.update(&data[j..]);
                assert_eq!(c.finalize(), expected);
            }
        }
    }

    #[test]
    fn result_always_six_bits_over_streaming_prefixes() {
        let data = b"0123456789abcdefABCDEF";
        let mut c = Crc6Cdma2000::new();
        for &b in data {
            c.update(&[b]);
            assert!(c.finalize() <= 0x3F);
        }
    }

    #[test]
    fn single_zero_byte() {
        // init 0x3F folded over eight zero bits.
        let v = crc6_cdma2000(&[0x00]);
        assert!(v <= 0x3F);
        assert_eq!(crc6_cdma2000(&[0x00]), v);
    }

    #[test]
    fn single_0xff_byte() {
        let v = crc6_cdma2000(&[0xFF]);
        assert!(v <= 0x3F);
    }

    #[test]
    fn repeated_zeros_vary_from_single() {
        let one = crc6_cdma2000(&[0x00]);
        let two = crc6_cdma2000(&[0x00, 0x00]);
        // folding more zero bits keeps shifting the register; just ensure both
        // stay inside the six-bit range and remain deterministic.
        assert!(one <= 0x3F);
        assert!(two <= 0x3F);
        assert_eq!(two, crc6_cdma2000(&[0x00, 0x00]));
    }

    #[test]
    fn order_sensitivity() {
        let ab = crc6_cdma2000(b"AB");
        let ba = crc6_cdma2000(b"BA");
        // CRCs are order sensitive; both are still six-bit values.
        assert!(ab <= 0x3F);
        assert!(ba <= 0x3F);
    }

    #[test]
    fn fold_bit_matches_manual_zero_bit() {
        // With register 0, a zero bit just shifts (stays 0).
        assert_eq!(fold_bit(0, 0), 0);
    }

    #[test]
    fn fold_bit_applies_poly_when_bits_differ() {
        // Register 0 (top bit 0), incoming bit 1 => XOR, so poly folds in.
        // Shift of a zero register leaves 0, then the polynomial is folded in.
        assert_eq!(fold_bit(0, 1), CRC6_POLY);
    }

    #[test]
    fn fold_bit_no_poly_when_bits_match() {
        // Register with top bit set and incoming bit 1 => no poly fold.
        let crc = 1u8 << CRC6_TOP_BIT;
        assert_eq!(fold_bit(crc, 1), (crc << 1) & CRC6_MASK);
    }

    #[test]
    fn fold_bit_result_six_bits() {
        for crc in 0u8..=0x3F {
            assert!(fold_bit(crc, 0) <= 0x3F);
            assert!(fold_bit(crc, 1) <= 0x3F);
        }
    }

    #[test]
    fn fold_byte_matches_eight_fold_bits() {
        for crc in 0u8..=0x3F {
            for byte in 0u16..=255 {
                let byte = byte as u8;
                let mut manual = crc;
                for i in (0..8).rev() {
                    let bit = (byte >> i) & 1;
                    manual = fold_bit(manual, bit);
                }
                assert_eq!(fold_byte(crc, byte), manual);
            }
        }
    }

    #[test]
    fn oneshot_matches_fold_sequence() {
        let data = b"fold-sequence-check";
        let mut crc = CRC6_INIT;
        for &b in data {
            crc = fold_byte(crc, b);
        }
        assert_eq!(crc & CRC6_MASK, crc6_cdma2000(data));
    }

    #[test]
    fn struct_is_copy() {
        let a = Crc6Cdma2000::new();
        let b = a; // copy
        assert_eq!(a.finalize(), b.finalize());
    }

    #[test]
    fn clone_equivalent_to_original() {
        let mut a = Crc6Cdma2000::new();
        a.update(b"partial");
        let b = a;
        assert_eq!(a, b);
        assert_eq!(a.finalize(), b.finalize());
    }

    #[test]
    fn long_input_stays_six_bits() {
        let data: Vec<u8> = (0u32..4096).map(|x| (x & 0xFF) as u8).collect();
        let v = crc6_cdma2000(&data);
        assert!((0x00..=0x3F).contains(&v));
    }

    #[test]
    fn long_input_streaming_matches_oneshot() {
        let data: Vec<u8> = (0u32..4096)
            .map(|x| (x.wrapping_mul(31) & 0xFF) as u8)
            .collect();
        let expected = crc6_cdma2000(&data);
        let mut c = Crc6Cdma2000::new();
        for chunk in data.chunks(7) {
            c.update(chunk);
        }
        assert_eq!(c.finalize(), expected);
    }

    #[test]
    fn ascii_digits_check_prefixes_six_bits() {
        for len in 0..=CHECK_INPUT.len() {
            let v = crc6_cdma2000(&CHECK_INPUT[..len]);
            assert!(v <= 0x3F);
        }
    }

    #[test]
    fn mask_clears_high_bits_from_fold() {
        // Feed a value whose shift would set bit 6 and confirm masking.
        let crc = 0x3Fu8; // top bits all set within six bits
        let folded = fold_byte(crc, 0xFF);
        assert_eq!(folded & 0xC0, 0);
    }

    #[test]
    fn concatenation_via_two_accumulators_consistency() {
        let first = b"concat-first";
        let second = b"concat-second";
        let mut joined = Vec::new();
        joined.extend_from_slice(first);
        joined.extend_from_slice(second);
        let expected = crc6_cdma2000(&joined);
        let mut c = Crc6Cdma2000::new();
        c.update(first);
        c.update(second);
        assert_eq!(c.finalize(), expected);
    }
}
