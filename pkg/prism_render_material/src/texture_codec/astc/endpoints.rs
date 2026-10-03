//! ASTC colour-endpoint decode for single-partition LDR blocks.
//!
//! After the block mode and partition/CEM fields, an ASTC block stores the
//! colour endpoints as a Bounded Integer Sequence (BISE) whose quantisation
//! range is derived from the bits left over once the weights are accounted
//! for. The decoded integers are *unquantized* to 8-bit colour components and
//! grouped into the two endpoint colours according to the Colour Endpoint Mode
//! (CEM). This module implements the CEM 8 path (LDR direct RGB), the common
//! opaque case; further CEMs land in later milestones.
//!
//! The quantisation-level and blue-contraction logic is transcribed from the
//! ARM `astcenc` reference decoder (`astcenc_symbolic_physical.cpp` and
//! `astcenc_color_unquantize.cpp`, Apache-2.0).
//!
//! Decode is pure integer arithmetic -- no AI/ML path.

use super::bise::{decode_ise, IseRange};
use super::AstcError;

/// A pair of unquantized 8-bit LDR endpoint colours, RGBA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Endpoints {
    pub(super) e0: [u8; 4],
    pub(super) e1: [u8; 4],
}

/// `astcenc` quant_mode index for QUANT_256 (identity 8-bit colour unquant).
const QUANT_256: i8 = 20;

/// `quant_mode_table[integer_count / 2][color_bits]` from the reference
/// decoder: given the number of colour integers (here fixed at six, so row
/// index three) and the colour bit budget, yields the `astcenc` colour
/// quantisation level (an index into the QUANT_* enum, or `-1` if the budget
/// is too small to encode the endpoints).
///
/// Row three (six colour integers, i.e. CEM 8 RGB) is transcribed verbatim.
const QUANT_MODE_TABLE_ROW3: [i8; 128] = [
    -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, //
    4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, //
    12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, //
];

/// Blue-uncontraction (reference `uncontract_color`): recover the full-range
/// R and G from a blue-contracted endpoint. B and A pass through unchanged.
#[inline]
fn uncontract(c: [u8; 4]) -> [u8; 4] {
    let r = ((i32::from(c[0]) + i32::from(c[2])) >> 1) as u8;
    let g = ((i32::from(c[1]) + i32::from(c[2])) >> 1) as u8;
    [r, g, c[2], c[3]]
}

/// Sum of the R, G and B components (reference `hadd_rgb_s`).
#[inline]
fn hadd_rgb(c: [u8; 4]) -> i32 {
    i32::from(c[0]) + i32::from(c[1]) + i32::from(c[2])
}

/// Decode the two LDR RGB endpoint colours of a **single-partition CEM 8**
/// block from a 4x4 LDR ASTC `block`.
///
/// `weight_bits` is the number of bits the weight stream occupies (from the
/// block mode); the colour bit budget is what remains of the single-partition
/// single-plane layout. The endpoint integer sequence begins at bit 17 (after
/// the 11-bit mode, 2-bit partition field and 4-bit CEM).
///
/// # Errors
/// Returns [`AstcError::UnsupportedIse`] when the derived colour quantisation
/// level is not QUANT_256; the non-identity colour unquant tables land in a
/// later milestone. Returns [`AstcError::Reserved`] when the colour budget is
/// too small to hold the endpoints (reference "error block").
pub(super) fn decode_cem8_endpoints(
    block: &[u8; 16],
    weight_bits: u32,
) -> Result<Endpoints, AstcError> {
    // Single partition, single plane: color_bits = 111 - weight_bits
    // (color_bits_arr[1] == 115 - 4). Reference clamps negatives to zero.
    let color_bits = 111i32 - weight_bits as i32;
    if color_bits < 0 {
        return Err(AstcError::Reserved);
    }
    let color_bits = color_bits as usize;

    // CEM 8 -> six colour integers -> quant_mode_table row index 3.
    let level = *QUANT_MODE_TABLE_ROW3.get(color_bits).unwrap_or(&QUANT_256);
    if level < 0 {
        return Err(AstcError::Reserved);
    }
    if level != QUANT_256 {
        // Only the identity (8-bit) colour unquant is implemented so far.
        return Err(AstcError::UnsupportedIse);
    }

    // QUANT_256 is the pure-binary 8-bit range: decode six raw bytes.
    let range = IseRange::from_num_levels(256).ok_or(AstcError::Reserved)?;
    let mut vals = [0u8; 6];
    decode_ise(block, 17, range, 6, &mut vals)?;

    // CEM 8 LDR direct RGB: (v0,v2,v4) = endpoint0 RGB, (v1,v3,v5) = endpoint1.
    // QUANT_256 unquant is the identity, so the values are already 8-bit.
    let mut e0 = [vals[0], vals[2], vals[4], 255];
    let mut e1 = [vals[1], vals[3], vals[5], 255];

    // Blue uncontraction + endpoint swap (reference `rgba_unpack`).
    if hadd_rgb(e0) > hadd_rgb(e1) {
        e0 = uncontract(e0);
        e1 = uncontract(e1);
        core::mem::swap(&mut e0, &mut e1);
    }

    Ok(Endpoints { e0, e1 })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write six 8-bit endpoint values LSB-first starting at bit 17.
    fn pack_endpoints(vals: [u8; 6]) -> [u8; 16] {
        let mut block = [0u8; 16];
        for (i, v) in vals.iter().enumerate() {
            let base = 17 + i as u32 * 8;
            for b in 0..8u32 {
                if (v >> b) & 1 == 1 {
                    let pos = base + b;
                    block[(pos >> 3) as usize] |= 1 << (pos & 7);
                }
            }
        }
        block
    }

    #[test]
    fn quant256_identity_direct_rgb() {
        // weight_bits = 42 (mode 67) -> color_bits = 69 -> QUANT_256.
        // Endpoints with hadd(e0) <= hadd(e1) pass through unchanged.
        let vals = [10u8, 200, 20, 210, 30, 220];
        let block = pack_endpoints(vals);
        let ep = decode_cem8_endpoints(&block, 42).expect("QUANT_256 decodes");
        assert_eq!(ep.e0, [10, 20, 30, 255]);
        assert_eq!(ep.e1, [200, 210, 220, 255]);
    }

    #[test]
    fn blue_contraction_swaps_and_uncontracts() {
        // hadd(e0) > hadd(e1) triggers uncontract + swap.
        let vals = [200u8, 10, 210, 20, 220, 30];
        let block = pack_endpoints(vals);
        // raw e0 = (200,210,220), e1 = (10,20,30); hadd 630 > 60.
        let u0 = uncontract([200, 210, 220, 255]);
        let u1 = uncontract([10, 20, 30, 255]);
        let ep = decode_cem8_endpoints(&block, 42).expect("decodes");
        // After swap, output e0 is uncontracted e1.
        assert_eq!(ep.e0, u1);
        assert_eq!(ep.e1, u0);
    }

    #[test]
    fn non_quant256_budget_is_unsupported_for_now() {
        // weight_bits = 96 -> color_bits = 15 -> QUANT_mode_table_row3[15] = 3,
        // a non-identity range not yet implemented.
        assert_eq!(QUANT_MODE_TABLE_ROW3[15], 3);
        let block = [0u8; 16];
        assert_eq!(
            decode_cem8_endpoints(&block, 96),
            Err(AstcError::UnsupportedIse)
        );
    }

    #[test]
    fn uncontract_matches_reference_formula() {
        assert_eq!(uncontract([100, 150, 50, 255]), [75, 100, 50, 255]);
    }
}
