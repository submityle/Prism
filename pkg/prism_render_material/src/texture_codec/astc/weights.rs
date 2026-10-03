//! ASTC weight-grid decode for single-plane LDR blocks.
//!
//! After the block-mode and endpoint fields, an ASTC block stores one weight
//! per texel (two per texel for dual-plane modes). The weights live in the
//! *high* end of the 128-bit block and are stored **bit-reversed**: if you
//! mirror the whole 128-bit block end-for-end, the weights become an ordinary
//! LSB-first Integer-Sequence-Encoded stream starting at bit 0. Equivalently,
//! texel `t`'s weight value bit `b` (bit 0 = least significant) sits at block
//! bit `127 - (bits_per_weight * t + b)`.
//!
//! Each raw weight is then *unquantized* onto the fixed `0..=64` interpolation
//! scale. For the bit-only (power-of-two level count) ranges this is the
//! Khronos Data Format Specification 1.3 bit-replication rule: replicate the
//! `b`-bit value to fill six bits, then add one when the result exceeds 32 so
//! the midpoint lands exactly on 32 and the maximum on 64.
//!
//! # GPU ground truth
//! The `bits == 4` (QUANT_16) single-plane 4x4 path is proven bit-for-bit
//! against the Apple M2 Metal ASTC hardware decoder (block mode 578, black and
//! white endpoints): the sixteen unquantized levels decode to
//! `0,4,8,12,17,21,25,29,35,39,43,47,52,56,60,64`, which the hardware renders
//! as the gray ramp `0,16,32,...,255`. Other weight widths use the identical
//! algorithm but await their own block-mode milestone before being claimed as
//! hardware-proven.
//!
//! Decode is pure integer arithmetic -- no AI/ML path.

/// Unquantize a raw weight `value` drawn from a bit-only range of `bits` bits
/// (level count `2^bits`) onto the `0..=64` interpolation scale.
///
/// Replicates the `bits`-bit value to six bits (MSB-aligned, wrapping the
/// pattern to fill the low bits) and adds one past the midpoint so the range
/// is the full closed interval `[0, 64]`.
///
/// # Panics (debug only)
/// Panics in debug builds if `bits` is `0` or greater than `6`, neither of
/// which is a valid bit-only weight range.
#[must_use]
pub(super) fn unquant_weight_bits(value: u32, bits: u32) -> u8 {
    debug_assert!((1..=6).contains(&bits), "weight range bits out of 1..=6");
    // Replicate the `bits`-bit pattern to fill six bits, MSB-aligned.
    let mut acc = 0u32;
    let mut shift = 6i32 - bits as i32;
    while shift > 0 {
        acc |= value << shift;
        shift -= bits as i32;
    }
    // The final (possibly negative) shift folds the high bits into the low end.
    acc |= value >> (-shift) as u32;
    let mut w = acc & 0x3f;
    if w > 32 {
        w += 1;
    }
    w as u8
}

/// Read texel `t`'s raw (still quantized) weight from a single-plane block
/// whose weights are `bits` bits wide.
#[inline]
fn read_weight_raw(block: &[u8; 16], t: u32, bits: u32) -> u32 {
    let mut v = 0u32;
    for b in 0..bits {
        let pos = 127 - (bits * t + b);
        let byte = (pos >> 3) as usize;
        if (block[byte] >> (pos & 7)) & 1 == 1 {
            v |= 1 << b;
        }
    }
    v
}

/// Decode the sixteen unquantized weights (each `0..=64`) of a single-plane
/// 4x4 LDR block whose weight range is the bit-only range of `weight_bits`
/// bits.
///
/// This is a GPU-validatable building block for the general single-partition
/// decoder: it performs only the weight-grid read and unquantization, with no
/// endpoint or interpolation work, so it can be proven against the hardware
/// decoder in isolation (black/white endpoints turn each unquantized weight
/// straight into a gray level).
///
/// # Panics (debug only)
/// Panics in debug builds if `weight_bits` is `0` or greater than `6`.
#[must_use]
pub fn decode_astc_4x4_weights(block: &[u8; 16], weight_bits: u32) -> [u8; 16] {
    debug_assert!(
        (1..=6).contains(&weight_bits),
        "weight range bits out of 1..=6"
    );
    let mut out = [0u8; 16];
    for (t, w) in out.iter_mut().enumerate() {
        let raw = read_weight_raw(block, t as u32, weight_bits);
        *w = unquant_weight_bits(raw, weight_bits);
    }
    out
}

/// Mirror the 128-bit `block` end-for-end: output bit `p` is input bit
/// `127 - p`. ASTC stores the weight ISE stream bit-reversed at the top of the
/// block, so reversing the whole block turns it into an ordinary LSB-first
/// Integer-Sequence-Encoded stream beginning at bit 0.
#[must_use]
fn reverse_block_bits(block: &[u8; 16]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for p in 0..128u32 {
        let src = 127 - p;
        if (block[(src >> 3) as usize] >> (src & 7)) & 1 == 1 {
            out[(p >> 3) as usize] |= 1 << (p & 7);
        }
    }
    out
}

/// Decode the sixteen unquantized weights (each `0..=64`) of a single-plane
/// 4x4 LDR block whose weight range has `levels` distinct levels (e.g. `6` for
/// a QUANT_6 trit range, `5` for a QUANT_5 quint range). Power-of-two level
/// counts take the bit-only path; trit/quint level counts use the
/// `astcenc`-derived unquantization tables.
///
/// The weight stream is read by mirroring the block (see
/// [`reverse_block_bits`]) and decoding `16` BISE values from bit 0, then
/// unquantizing each via [`super::weight_unquant::unquant_weight`]. As with
/// [`decode_astc_4x4_weights`], no endpoint or interpolation work is done, so
/// the result can be proven against the hardware decoder in isolation using
/// black/white endpoints.
///
/// Returns `None` if `levels` is not one of the three BISE range forms, which
/// cannot occur for a valid ASTC weight range.
#[must_use]
pub fn decode_astc_4x4_weights_ise(block: &[u8; 16], levels: u32) -> Option<[u8; 16]> {
    let range = super::bise::IseRange::from_num_levels(levels)?;
    let reversed = reverse_block_bits(block);
    let mut raw = [0u8; 16];
    // Infallible for every well-formed BISE range; the 4x4 single-plane grid is
    // always exactly sixteen weights.
    let _ = super::bise::decode_ise(&reversed, 0, range, 16, &mut raw);
    let mut out = [0u8; 16];
    for (o, &r) in out.iter_mut().zip(raw.iter()) {
        *o = super::weight_unquant::unquant_weight(r as u32, range);
    }
    Some(out)
}

/// Decode `weight_count` single-plane weights (quantised to `levels` BISE
/// levels) from `block` into `out`, each unquantized onto the `0..=64` scale.
///
/// Generalizes [`decode_astc_4x4_weights_ise`] to arbitrary single-plane grid
/// sizes (`weight_count == weights_x * weights_y`) for the non-4x4 bilinear
/// infill path. The weights share the same bit-reversed top-of-block layout as
/// the 4x4 grid; only the count differs.
///
/// `out.len()` must equal `weight_count`, and `weight_count` must not exceed
/// the sixty-four-weight single-plane budget.
///
/// Returns `None` if `levels` is not a valid BISE level count.
#[must_use]
pub(super) fn decode_grid_weights_ise(
    block: &[u8; 16],
    weight_count: u32,
    levels: u32,
    out: &mut [u8],
) -> Option<()> {
    debug_assert_eq!(
        out.len() as u32,
        weight_count,
        "weight output slice length mismatch"
    );
    if weight_count as usize > 64 {
        return None;
    }
    let range = super::bise::IseRange::from_num_levels(levels)?;
    let reversed = reverse_block_bits(block);
    let mut raw = [0u8; 64];
    let n = weight_count as usize;
    // Infallible for every well-formed BISE range.
    let _ = super::bise::decode_ise(&reversed, 0, range, weight_count, &mut raw[..n]);
    for (o, &r) in out.iter_mut().zip(raw[..n].iter()) {
        *o = super::weight_unquant::unquant_weight(r as u32, range);
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::{
        decode_astc_4x4_weights, decode_astc_4x4_weights_ise, reverse_block_bits,
        unquant_weight_bits,
    };

    #[test]
    fn quant16_levels_match_gpu_ground_truth() {
        // The sixteen 4-bit (QUANT_16) unquantized levels, proven against the
        // Metal ASTC hardware decoder.
        let expected = [
            0u8, 4, 8, 12, 17, 21, 25, 29, 35, 39, 43, 47, 52, 56, 60, 64,
        ];
        for (v, &e) in expected.iter().enumerate() {
            assert_eq!(unquant_weight_bits(v as u32, 4), e, "4-bit weight {v}");
        }
    }

    #[test]
    fn bit_only_ranges_span_the_full_scale() {
        // Every bit-only range must map 0 -> 0 and its maximum level -> 64.
        for bits in 1u32..=6 {
            assert_eq!(unquant_weight_bits(0, bits), 0, "{bits}-bit zero");
            let max = (1u32 << bits) - 1;
            assert_eq!(unquant_weight_bits(max, bits), 64, "{bits}-bit max");
        }
    }

    #[test]
    fn one_bit_range_is_endpoints_only() {
        assert_eq!(unquant_weight_bits(0, 1), 0);
        assert_eq!(unquant_weight_bits(1, 1), 64);
    }

    #[test]
    fn two_bit_range_matches_spec_table() {
        // RANGE_4: 0, 21, 43, 64 (Khronos DFS 1.3 weight unquantization).
        let expected = [0u8, 21, 43, 64];
        for (v, &e) in expected.iter().enumerate() {
            assert_eq!(unquant_weight_bits(v as u32, 2), e, "2-bit weight {v}");
        }
    }

    #[test]
    fn weights_are_read_bit_reversed_from_the_top() {
        // Lay texel t's 4-bit weight at block bits [127-4t .. 124-4t], value
        // bit b at bit 127-(4t+b), and confirm the reader recovers it, then
        // unquantizes via the proven QUANT_16 table.
        let raw = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let mut block = [0u8; 16];
        for (t, &w) in raw.iter().enumerate() {
            for b in 0..4u32 {
                if (w >> b) & 1 == 1 {
                    let pos = 127 - (4 * t as u32 + b);
                    block[(pos >> 3) as usize] |= 1 << (pos & 7);
                }
            }
        }
        let expected = [
            0u8, 4, 8, 12, 17, 21, 25, 29, 35, 39, 43, 47, 52, 56, 60, 64,
        ];
        assert_eq!(decode_astc_4x4_weights(&block, 4), expected);
    }

    #[test]
    fn reverse_block_bits_is_an_involution() {
        // Mirroring the block twice must return the original bytes.
        let mut block = [0u8; 16];
        for (i, b) in block.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(37).wrapping_add(1);
        }
        assert_eq!(reverse_block_bits(&reverse_block_bits(&block)), block);
    }

    #[test]
    fn ise_path_matches_bit_only_reader_for_quant16() {
        // The generic trit/quint-capable ISE weight reader must agree bit-for-
        // bit with the GPU-proven bit-only reader on a power-of-two range. This
        // ties the new reverse-and-decode wiring to the hardware-validated
        // QUANT_16 path. Build the same bit-reversed 4-bit weight grid used in
        // `weights_are_read_bit_reversed_from_the_top`.
        let raw = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let mut block = [0u8; 16];
        for (t, &w) in raw.iter().enumerate() {
            for b in 0..4u32 {
                if (w >> b) & 1 == 1 {
                    let pos = 127 - (4 * t as u32 + b);
                    block[(pos >> 3) as usize] |= 1 << (pos & 7);
                }
            }
        }
        let via_bits = decode_astc_4x4_weights(&block, 4);
        let via_ise = decode_astc_4x4_weights_ise(&block, 16).expect("16 is a valid range");
        assert_eq!(via_ise, via_bits);
    }

    #[test]
    fn ise_rejects_non_bise_level_counts() {
        assert!(decode_astc_4x4_weights_ise(&[0u8; 16], 7).is_none());
        assert!(decode_astc_4x4_weights_ise(&[0u8; 16], 0).is_none());
    }

    #[test]
    fn ise_all_zero_block_is_all_zero_weights() {
        // Index 0 unquantizes to 0 in every range form, so an empty block maps
        // to the minimum weight everywhere regardless of the chosen range.
        for levels in [6u32, 5, 12, 10, 24, 20, 3] {
            let w = decode_astc_4x4_weights_ise(&[0u8; 16], levels).expect("valid range");
            assert_eq!(w, [0u8; 16], "levels {levels}");
        }
    }
}
