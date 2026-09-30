//! Fixed-width `u32` bit packing: the tiny, device-free integer contract that
//! densely packs an array of small unsigned integers into a stream of 32-bit
//! words and unpacks them back, with elements free to straddle word
//! boundaries (design §27 attribute compression, GPU vertex-attribute and
//! index-stream packing).
//!
//! Every value contributes exactly `bits` bits (`1..=32`), laid down
//! back-to-back with no separators, no length prefixes, and no continuation
//! flags. This is the *fixed-width, dense* member of the integer-coding family
//! and is deliberately distinct from its neighbors:
//!
//! * [`super::varint_leb128`] is *variable length*: it emits a whole byte per
//!   7 payload bits with a high continuation bit, so small values are cheap and
//!   large ones grow. This module has no continuation bits and no variable
//!   width — the width is a fixed parameter shared by the whole array.
//! * [`super::morton_code`] *interleaves* the bits of several coordinates into
//!   one sortable key for spatial locality. This module never interleaves; it
//!   concatenates independent values in order.
//! * [`super::zigzag_delta_encode`] is a *signed-to-unsigned value transform*
//!   (delta then `ZigZag`) that reshapes the numbers themselves. This module
//!   does not transform values; it only relocates their bits.
//! * [`super::compression`] is the higher-level, policy-driven codec layer.
//!   This module is a primitive it can build on.
//!
//! # Bit layout (least-significant-bit first)
//!
//! Packing is **`LSB`-first**. The lowest-significant bit of each value is
//! written into the lowest currently-free bit of the current word, and the
//! value's remaining bits climb toward the most-significant bit (`MSB`). When a
//! value would run past bit 31 of the current word, the overflow high bits spill
//! into the low bits of the next word. Reconstruction reverses this with a pair
//! of shifts (a low-word right shift joined with a high-word left shift) and a
//! low-`bits` mask.
//!
//! Concretely, for `bits == 1` the values form a bitmap where value `i` lives in
//! bit `i % 32` of word `i / 32`; for `bits == 4` two values `[0xA, 0xB]` pack
//! into a single word `0x0000_00BA`.
//!
//! # Truncation
//!
//! When `bits < 32`, [`pack`] masks each input to its low `bits` bits before
//! writing, so out-of-range inputs are silently truncated rather than
//! panicking. This keeps [`pack`] a total, pure function: the round-trip
//! identity `unpack(pack(v, bits), bits, v.len()) == v` holds whenever every
//! value already fits in `bits` bits. `bits == 32` degenerates to a plain copy.

use alloc::vec::Vec;

/// Returns a mask with the low `bits` bits set (`bits` in `1..=32`).
///
/// For `bits == 32` this is `u32::MAX`; the shift is only evaluated for
/// `bits < 32`, so `1u32 << bits` never uses a shift amount of 32.
const fn low_mask(bits: u32) -> u32 {
    if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    }
}

/// Number of 32-bit words needed to hold `count` values of `bits` bits each.
///
/// Equal to `(count * bits).div_ceil(32)`.
#[must_use]
pub fn packed_len_words(count: usize, bits: u32) -> usize {
    (count * (bits as usize)).div_ceil(32)
}

/// Packs the low `bits` bits of each value into a dense `u32` word stream.
///
/// `bits` must be in `1..=32`. When `bits < 32`, each value is masked to its
/// low `bits` bits, so inputs outside that range are truncated. `bits == 32`
/// copies the values verbatim.
#[must_use]
pub fn pack(values: &[u32], bits: u32) -> Vec<u32> {
    debug_assert!((1..=32).contains(&bits), "bits must be in 1..=32");
    let word_count = packed_len_words(values.len(), bits);
    let mut out = alloc::vec![0u32; word_count];
    if bits == 32 {
        out.copy_from_slice(values);
        return out;
    }
    let mask = low_mask(bits);
    let mut bit_pos: usize = 0;
    for &raw in values {
        let value = raw & mask;
        let word = bit_pos >> 5;
        let offset = (bit_pos & 31) as u32;
        // The low part lands in `word`; `offset < 32` keeps this shift valid.
        out[word] |= value << offset;
        if offset + bits > 32 {
            // Spill the high bits into the next word. `offset + bits > 32`
            // with `bits < 32` forces `offset >= 1`, so `32 - offset` stays in
            // `1..=31` and never becomes a 32-bit shift.
            out[word + 1] |= value >> (32 - offset);
        }
        bit_pos += bits as usize;
    }
    out
}

/// Unpacks `count` values of `bits` bits each from a dense `u32` word stream.
///
/// `bits` must be in `1..=32` and `packed` must hold at least
/// `packed_len_words(count, bits)` words. `bits == 32` copies the words
/// verbatim.
#[must_use]
pub fn unpack(packed: &[u32], bits: u32, count: usize) -> Vec<u32> {
    debug_assert!((1..=32).contains(&bits), "bits must be in 1..=32");
    let mut out = Vec::with_capacity(count);
    if bits == 32 {
        out.extend_from_slice(&packed[..count]);
        return out;
    }
    let mask = low_mask(bits);
    let mut bit_pos: usize = 0;
    for _ in 0..count {
        let word = bit_pos >> 5;
        let offset = (bit_pos & 31) as u32;
        let mut value = packed[word] >> offset;
        if offset + bits > 32 {
            // Join the high bits from the next word. As in `pack`, this branch
            // only runs when `offset >= 1`, so `32 - offset` is a valid shift.
            value |= packed[word + 1] << (32 - offset);
        }
        out.push(value & mask);
        bit_pos += bits as usize;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Small linear-congruential generator for deterministic random samples.
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u32(&mut self) -> u32 {
            // Numerical Recipes constants.
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.state >> 32) as u32
        }
    }

    #[test]
    fn pack_empty_returns_empty() {
        assert_eq!(pack(&[], 7), Vec::<u32>::new());
    }

    #[test]
    fn unpack_empty_returns_empty() {
        assert_eq!(unpack(&[], 7, 0), Vec::<u32>::new());
    }

    #[test]
    fn packed_len_words_zero_count_is_zero() {
        assert_eq!(packed_len_words(0, 13), 0);
    }

    #[test]
    fn packed_len_words_rounds_up() {
        assert_eq!(packed_len_words(10, 3), 1); // 30 bits -> 1 word
        assert_eq!(packed_len_words(11, 3), 2); // 33 bits -> 2 words
    }

    #[test]
    fn packed_len_words_bits32_is_count() {
        assert_eq!(packed_len_words(5, 32), 5);
    }

    #[test]
    fn packed_len_words_exact_word_boundary() {
        assert_eq!(packed_len_words(32, 1), 1);
        assert_eq!(packed_len_words(33, 1), 2);
        assert_eq!(packed_len_words(4, 8), 1);
    }

    #[test]
    fn bits1_bitmap_layout_is_lsb_first() {
        let values = [1, 0, 1, 1];
        let packed = pack(&values, 1);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0b1101);
    }

    #[test]
    fn bits1_bitmap_roundtrip() {
        let values: Vec<u32> = (0..70).map(|i| (i % 3 == 0) as u32).collect();
        let packed = pack(&values, 1);
        assert_eq!(packed.len(), 3); // 70 bits -> 3 words
        assert_eq!(unpack(&packed, 1, values.len()), values);
    }

    #[test]
    fn bits4_two_values_one_word_layout() {
        let packed = pack(&[0xA, 0xB], 4);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0x0000_00BA);
    }

    #[test]
    fn bits32_is_verbatim_copy() {
        let values = [0xDEAD_BEEF, 0x0000_0000, 0xFFFF_FFFF, 0x1234_5678];
        let packed = pack(&values, 32);
        assert_eq!(packed, values);
    }

    #[test]
    fn bits32_roundtrip() {
        let values = [0xDEAD_BEEF, 0x0000_0000, 0xFFFF_FFFF, 0x1234_5678];
        let packed = pack(&values, 32);
        assert_eq!(unpack(&packed, 32, values.len()), values);
    }

    #[test]
    fn single_value_roundtrip() {
        for bits in 1..=32u32 {
            let mask = low_mask(bits);
            let value = 0x9E37_79B9 & mask;
            let packed = pack(&[value], bits);
            assert_eq!(unpack(&packed, bits, 1), alloc::vec![value]);
        }
    }

    #[test]
    fn bits3_cross_word_roundtrip() {
        let values: Vec<u32> = (0..40).map(|i| i as u32 & 0x7).collect();
        let packed = pack(&values, 3);
        assert_eq!(packed.len(), packed_len_words(values.len(), 3));
        assert_eq!(unpack(&packed, 3, values.len()), values);
    }

    #[test]
    fn bits7_roundtrip() {
        let values: Vec<u32> = (0..50).map(|i| (i as u32 * 5) & 0x7F).collect();
        let packed = pack(&values, 7);
        assert_eq!(unpack(&packed, 7, values.len()), values);
    }

    #[test]
    fn bits17_cross_word_roundtrip() {
        let values: Vec<u32> = (0..30).map(|i| (i as u32 * 1103) & 0x1_FFFF).collect();
        let packed = pack(&values, 17);
        assert_eq!(packed.len(), packed_len_words(values.len(), 17));
        assert_eq!(unpack(&packed, 17, values.len()), values);
    }

    #[test]
    fn bits31_cross_word_roundtrip() {
        let values: Vec<u32> = (0..20)
            .map(|i| (i as u32 * 0x0410_1101) & 0x7FFF_FFFF)
            .collect();
        let packed = pack(&values, 31);
        assert_eq!(packed.len(), packed_len_words(values.len(), 31));
        assert_eq!(unpack(&packed, 31, values.len()), values);
    }

    #[test]
    fn value_spanning_word_boundary_explicit_layout() {
        // 7 values of 5 bits: the 7th starts at bit 30 and spills 3 bits.
        let values = [0, 0, 0, 0, 0, 0, 0x1F];
        let packed = pack(&values, 5);
        assert_eq!(packed.len(), 2); // 35 bits -> 2 words
        assert_eq!(packed[0], 0xC000_0000); // bits 30,31 set
        assert_eq!(packed[1], 0b111); // top 3 bits of 0x1F
        assert_eq!(unpack(&packed, 5, values.len()), values);
    }

    #[test]
    fn value_exactly_fills_word_boundary() {
        // Two 16-bit values fill exactly one 32-bit word.
        let values = [0xABCD, 0x1234];
        let packed = pack(&values, 16);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0x1234_ABCD);
        assert_eq!(unpack(&packed, 16, values.len()), values);
    }

    #[test]
    fn max_value_per_bit_width_roundtrips() {
        for bits in 1..=32u32 {
            let max = low_mask(bits);
            let values = alloc::vec![max; 9];
            let packed = pack(&values, bits);
            assert_eq!(unpack(&packed, bits, values.len()), values);
        }
    }

    #[test]
    fn mask_truncates_out_of_range_values() {
        // 3-bit width: 0xFF masks to 0x7, 0x08 masks to 0x0.
        let packed = pack(&[0xFF, 0x08, 0x05], 3);
        assert_eq!(unpack(&packed, 3, 3), alloc::vec![0x7, 0x0, 0x5]);
    }

    #[test]
    fn bits2_roundtrip() {
        let values: Vec<u32> = (0..37).map(|i| i as u32 & 0x3).collect();
        let packed = pack(&values, 2);
        assert_eq!(unpack(&packed, 2, values.len()), values);
    }

    #[test]
    fn bits5_roundtrip() {
        let values: Vec<u32> = (0..41).map(|i| (i as u32 * 3) & 0x1F).collect();
        let packed = pack(&values, 5);
        assert_eq!(unpack(&packed, 5, values.len()), values);
    }

    #[test]
    fn bits12_cross_word_roundtrip() {
        let values: Vec<u32> = (0..25).map(|i| (i as u32 * 167) & 0xFFF).collect();
        let packed = pack(&values, 12);
        assert_eq!(packed.len(), packed_len_words(values.len(), 12));
        assert_eq!(unpack(&packed, 12, values.len()), values);
    }

    #[test]
    fn every_bit_width_roundtrips() {
        for bits in 1..=32u32 {
            let mask = low_mask(bits);
            let values: Vec<u32> = (0..37)
                .map(|i| (i as u32).wrapping_mul(0x9E37_79B1) & mask)
                .collect();
            let packed = pack(&values, bits);
            assert_eq!(packed.len(), packed_len_words(values.len(), bits));
            assert_eq!(unpack(&packed, bits, values.len()), values, "bits = {bits}");
        }
    }

    #[test]
    fn pack_output_length_matches_packed_len_words() {
        for bits in 1..=32u32 {
            for count in [0usize, 1, 2, 7, 31, 32, 33, 100] {
                let values = alloc::vec![1u32; count];
                let packed = pack(&values, bits);
                assert_eq!(packed.len(), packed_len_words(count, bits));
            }
        }
    }

    #[test]
    fn unpack_fewer_than_packed_capacity() {
        let values: Vec<u32> = (0..20).map(|i| i as u32 & 0xF).collect();
        let packed = pack(&values, 4);
        // Only decode the first 5 values.
        assert_eq!(unpack(&packed, 4, 5), values[..5].to_vec());
    }

    #[test]
    fn lcg_random_roundtrip_bits11() {
        let mut rng = Lcg::new(0x1234_5678_9ABC_DEF0);
        let bits = 11u32;
        let mask = low_mask(bits);
        let values: Vec<u32> = (0..500).map(|_| rng.next_u32() & mask).collect();
        let packed = pack(&values, bits);
        assert_eq!(packed.len(), packed_len_words(values.len(), bits));
        assert_eq!(unpack(&packed, bits, values.len()), values);
    }

    #[test]
    fn lcg_random_roundtrip_all_bit_widths() {
        let mut rng = Lcg::new(0xDEAD_BEEF_CAFE_F00D);
        for bits in 1..=32u32 {
            let mask = low_mask(bits);
            let values: Vec<u32> = (0..200).map(|_| rng.next_u32() & mask).collect();
            let packed = pack(&values, bits);
            assert_eq!(
                unpack(&packed, bits, values.len()),
                values,
                "random roundtrip failed at bits = {bits}"
            );
        }
    }

    #[test]
    fn low_mask_edges() {
        assert_eq!(low_mask(1), 0x1);
        assert_eq!(low_mask(8), 0xFF);
        assert_eq!(low_mask(31), 0x7FFF_FFFF);
        assert_eq!(low_mask(32), 0xFFFF_FFFF);
    }
}
