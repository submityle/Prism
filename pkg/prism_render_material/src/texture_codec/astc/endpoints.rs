//! ASTC colour-endpoint decode for single-partition LDR blocks.
//!
//! After the block mode and partition/CEM fields, an ASTC block stores the
//! colour endpoints as a Bounded Integer Sequence (BISE) whose quantisation
//! range is derived from the bits left over once the weights are accounted
//! for. The decoded integers are *unquantized* to 8-bit colour components and
//! grouped into the two endpoint colours according to the Colour Endpoint Mode
//! (CEM).
//!
//! This module orchestrates that pipeline for the ten LDR CEMs: it derives the
//! colour quant level ([`super::quant_mode`]), classifies and decodes the
//! integer sequence ([`super::bise`]), unquantizes each integer
//! ([`super::color_unquant`]) and hands the result to the per-CEM endpoint
//! assembler ([`super::cem`]). The six HDR CEMs are a later milestone and are
//! rejected here.
//!
//! The quantisation-level selection and per-CEM maths are transcribed from the
//! ARM `astcenc` reference decoder (`astcenc_symbolic_physical.cpp`,
//! `astcenc_quantization.cpp` and `astcenc_color_unquantize.cpp`, Apache-2.0).
//!
//! Decode is pure integer arithmetic -- no AI/ML path.

use super::bise::{decode_ise, IseRange};
use super::cem::{cem_integer_count, cem_is_ldr, unpack_endpoints};
use super::color_unquant::{color_quant_num_levels, unquant_color};
use super::quant_mode::{color_quant_level, QUANT_6};
use super::AstcError;

/// A pair of unquantized 8-bit LDR endpoint colours, RGBA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Endpoints {
    pub(super) e0: [u8; 4],
    pub(super) e1: [u8; 4],
}

/// Decode the two LDR endpoint colours of a **single-partition** block from a
/// 4x4 LDR ASTC `block`, for Colour Endpoint Mode `cem`.
///
/// `weight_bits` is the number of bits the weight stream occupies (from the
/// block mode); the colour bit budget is what remains of the single-partition
/// layout, minus two further bits when `dual_plane` is set (the colour
/// component selector). The endpoint integer sequence begins at bit 17 (after
/// the 11-bit mode, 2-bit partition field and 4-bit CEM).
///
/// # Errors
/// * [`AstcError::UnsupportedHdr`] for the six HDR CEMs (2, 3, 7, 11, 14, 15),
///   which have no LDR decode here yet.
/// * [`AstcError::Reserved`] when the derived colour quantisation level is
///   below QUANT_6 -- either the colour budget is too small to hold the
///   endpoints, or the reference decoder would flag an "error block".
pub(super) fn decode_cem_endpoints(
    block: &[u8; 16],
    weight_bits: u32,
    cem: u32,
    dual_plane: bool,
) -> Result<Endpoints, AstcError> {
    if !cem_is_ldr(cem) {
        return Err(AstcError::UnsupportedHdr);
    }
    let (vals, integer_count) = decode_cem_color_vals(block, weight_bits, cem, dual_plane)?;
    Ok(unpack_endpoints(cem, &vals[..integer_count as usize]))
}

/// Decode and unquantize the colour integer sequence of a **single-partition**
/// block, returning the 8-bit unquantized integers plus their count.
///
/// This is the profile-agnostic half of the single-partition endpoint
/// pipeline: it performs the colour quant-level selection, Integer Sequence
/// Encoding decode and per-level unquantization, but does *not* assemble the
/// integers into endpoint colours. The LDR path (`decode_cem_endpoints`) and
/// the HDR path (`super::hdr_endpoints`) share this stage and then diverge on
/// `unpack_endpoints` vs `unpack_hdr_endpoints`. HDR and LDR CEMs use the
/// identical quant/unquant machinery, so no profile branch is needed here.
///
/// # Errors
/// [`AstcError::Reserved`] when the colour budget cannot hold the endpoints
/// (quant level below QUANT_6), or an [`AstcError`] propagated from the ISE
/// decode.
pub(super) fn decode_cem_color_vals(
    block: &[u8; 16],
    weight_bits: u32,
    cem: u32,
    dual_plane: bool,
) -> Result<([u8; 8], u32), AstcError> {
    let integer_count = cem_integer_count(cem);

    // Single partition: color_bits = 111 - weight_bits (color_bits_arr[1] ==
    // 115 - 4). A dual-plane block additionally spends 2 bits on the colour
    // component selector (CCS) that live below the weights, so the colour
    // budget drops by 2 (astcenc: `if is_dual_plane { color_bits -= 2 }`).
    // Reference rejects a negative budget.
    let color_bits = (if dual_plane { 109 } else { 111 }) - weight_bits as i32;
    if color_bits < 0 {
        return Err(AstcError::Reserved);
    }
    let color_bits = color_bits as usize;

    // quant_mode_table row is picked by the colour integer count. Any level
    // below QUANT_6 (including the `-1` "budget too small" sentinel) is an
    // error block in the reference.
    let level = color_quant_level(integer_count, color_bits);
    if level < QUANT_6 {
        return Err(AstcError::Reserved);
    }

    // Classify the colour integer sequence from its level count, decode the
    // "scrambled packed quant" integers, then unquantize each to 8-bit through
    // the per-level `color_scrambled_pquant_to_uquant` table (identity for
    // QUANT_256). The table index is `level - QUANT_6`, matching the reference.
    let level_index = (level - QUANT_6) as usize;
    let num_levels = color_quant_num_levels(level_index);
    let range = IseRange::from_num_levels(num_levels).ok_or(AstcError::Reserved)?;

    let mut packed = [0u8; 8];
    decode_ise(
        block,
        17,
        range,
        integer_count,
        &mut packed[..integer_count as usize],
    )?;

    let mut vals = [0u8; 8];
    for (v, p) in vals.iter_mut().zip(packed.iter()) {
        *v = unquant_color(level_index, *p);
    }

    Ok((vals, integer_count))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write `n` 8-bit endpoint values LSB-first starting at bit 17.
    fn pack_endpoints(vals: &[u8]) -> [u8; 16] {
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
    fn cem8_quant256_identity_direct_rgb() {
        // weight_bits = 42 (mode 67) -> color_bits = 69 -> QUANT_256.
        // Endpoints with hadd(e0) <= hadd(e1) pass through unchanged.
        let block = pack_endpoints(&[10, 200, 20, 210, 30, 220]);
        let ep = decode_cem_endpoints(&block, 42, 8, false).expect("QUANT_256 decodes");
        assert_eq!(ep.e0, [10, 20, 30, 255]);
        assert_eq!(ep.e1, [200, 210, 220, 255]);
    }

    #[test]
    fn cem8_blue_contraction_swaps_and_uncontracts() {
        // hadd(e0) > hadd(e1) triggers uncontract + swap. Raw e0 = (200,210,220),
        // e1 = (10,20,30); hadd 630 > 60.
        let block = pack_endpoints(&[200, 10, 210, 20, 220, 30]);
        let ep = decode_cem_endpoints(&block, 42, 8, false).expect("decodes");
        // uncontract(R,G) = ((c0+c2)>>1,(c1+c2)>>1,c2); then swap.
        // uncontract(R,G) = ((c0+c2)>>1,(c1+c2)>>1,c2); then swap endpoints.
        let u0 = [210u8, 215, 220, 255];
        let u1 = [20u8, 25, 30, 255];
        assert_eq!(ep.e0, u1);
        assert_eq!(ep.e1, u0);
    }

    #[test]
    fn cem12_rgba_direct_carries_alpha() {
        // CEM 12 needs 8 integers; mode 67 (color_bits 69) still gives
        // QUANT_256 for the 8-integer row (row4[69] == 20).
        let block = pack_endpoints(&[10, 200, 20, 210, 30, 220, 40, 230]);
        let ep = decode_cem_endpoints(&block, 42, 12, false).expect("QUANT_256 decodes");
        assert_eq!(ep.e0, [10, 20, 30, 40]);
        assert_eq!(ep.e1, [200, 210, 220, 230]);
    }

    #[test]
    fn cem0_luminance_direct() {
        let block = pack_endpoints(&[64, 192]);
        let ep = decode_cem_endpoints(&block, 42, 0, false).expect("decodes");
        assert_eq!(ep.e0, [64, 64, 64, 255]);
        assert_eq!(ep.e1, [192, 192, 192, 255]);
    }

    #[test]
    fn hdr_cems_are_unsupported() {
        for cem in [2u32, 3, 7, 11, 14, 15] {
            assert_eq!(
                decode_cem_endpoints(&[0u8; 16], 42, cem, false),
                Err(AstcError::UnsupportedHdr)
            );
        }
    }

    #[test]
    fn cem8_budget_below_quant6_is_an_error_block() {
        // weight_bits = 96 -> color_bits = 15 -> QUANT_5, below QUANT_6.
        assert_eq!(
            decode_cem_endpoints(&[0u8; 16], 96, 8, false),
            Err(AstcError::Reserved)
        );
        // color_bits = 5 (weight_bits = 106) -> -1 sentinel.
        assert_eq!(
            decode_cem_endpoints(&[0u8; 16], 106, 8, false),
            Err(AstcError::Reserved)
        );
    }
}
