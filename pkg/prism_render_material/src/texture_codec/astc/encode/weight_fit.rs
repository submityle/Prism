//! Weight quantisation for the single-partition LDR encoder.
//!
//! For each texel we brute-force every raw weight level in a bit-only range of
//! `bits` bits, reconstruct the RGB colour with the **exact** decode-side
//! unquantisation (`weights::unquant_weight_bits`) and interpolation
//! (`single_partition::lerp_component`), and keep the level with the smallest
//! squared RGB error. On a 4x4 footprint the 4x4 weight grid is the identity
//! (no infill resampling), so the chosen levels decode back bit-for-bit.
//!
//! Pure integer arithmetic -- no AI/ML path.

use super::super::bise::IseRange;
use super::super::single_partition::lerp_component;
use super::super::weight_unquant::unquant_weight;
use super::super::weights::unquant_weight_bits;

/// Choose the best raw weight level per texel for a bit-only range of `bits`
/// bits, given fitted RGB endpoints `e0`/`e1`.
///
/// Returns the sixteen raw levels in row-major texel order, ready to pack with
/// `bits::BlockWriter::write_weights_reversed`.
pub(super) fn quantize_weights_bits(
    texels: &[[u8; 4]; 16],
    e0: [u8; 3],
    e1: [u8; 3],
    bits: u32,
) -> [u8; 16] {
    let levels = 1u32 << bits;
    core::array::from_fn(|t| {
        let texel = texels[t];
        let mut best_raw = 0u8;
        let mut best_err = u32::MAX;
        for v in 0..levels {
            let w = u32::from(unquant_weight_bits(v, bits));
            let mut err = 0u32;
            for c in 0..3 {
                let got = i32::from(lerp_component(e0[c], e1[c], w));
                let want = i32::from(texel[c]);
                let d = got - want;
                err += (d * d) as u32;
            }
            if err < best_err {
                best_err = err;
                best_raw = v as u8;
                if err == 0 {
                    break;
                }
            }
        }
        best_raw
    })
}

/// Choose the best raw weight level per texel for a bit-only range of `bits`
/// bits, given fitted **RGBA** endpoints `e0`/`e1` (CEM 12, single plane).
///
/// Identical in spirit to [`quantize_weights_bits`] but the squared error is
/// summed over all four channels, so the alpha carried by the CEM-12 endpoints
/// participates in the fit instead of being ignored. On a 4x4 footprint the
/// weight grid is the identity, so the chosen levels decode back exactly.
pub(super) fn quantize_weights_bits_rgba(
    texels: &[[u8; 4]; 16],
    e0: [u8; 4],
    e1: [u8; 4],
    bits: u32,
) -> [u8; 16] {
    let levels = 1u32 << bits;
    core::array::from_fn(|t| {
        let texel = texels[t];
        let mut best_raw = 0u8;
        let mut best_err = u32::MAX;
        for v in 0..levels {
            let w = u32::from(unquant_weight_bits(v, bits));
            let mut err = 0u32;
            for c in 0..4 {
                let got = i32::from(lerp_component(e0[c], e1[c], w));
                let want = i32::from(texel[c]);
                let d = got - want;
                err += (d * d) as u32;
            }
            if err < best_err {
                best_err = err;
                best_raw = v as u8;
                if err == 0 {
                    break;
                }
            }
        }
        best_raw
    })
}

/// Choose the best raw weight level per texel for a general BISE range of
/// `num_levels` distinct levels (trit/quint as well as bit-only), given fitted
/// **RGBA** endpoints `e0`/`e1` (CEM 12, single plane).
///
/// Unlike [`quantize_weights_bits_rgba`], which is limited to power-of-two
/// bit-only ranges, this uses the full decode-side unquantisation
/// (`weight_unquant::unquant_weight`) so trit/quint weight ranges such as
/// QUANT_6 (six interpolation levels from one trit + one low bit) are
/// expressible. Each candidate `v` in `0..num_levels` is the raw BISE value as
/// read by the decoder (`low | (digit << bits)`), so the chosen levels feed
/// straight into `trit_quint::encode_trit_sequence` / `encode_quint_sequence`
/// and `bits::BlockWriter::mirror_weight_stream`.
///
/// The squared error is summed over all four channels, so the alpha carried by
/// the CEM-12 endpoints participates in the fit. On a 4x4 footprint the weight
/// grid is the identity, so the chosen levels decode back exactly.
pub(super) fn quantize_weights_ise_rgba(
    texels: &[[u8; 4]; 16],
    e0: [u8; 4],
    e1: [u8; 4],
    num_levels: u32,
) -> [u8; 16] {
    let range = IseRange::from_num_levels(num_levels)
        .expect("weight level count must be a valid BISE range");
    core::array::from_fn(|t| {
        let texel = texels[t];
        let mut best_raw = 0u8;
        let mut best_err = u32::MAX;
        for v in 0..num_levels {
            let w = u32::from(unquant_weight(v, range));
            let mut err = 0u32;
            for c in 0..4 {
                let got = i32::from(lerp_component(e0[c], e1[c], w));
                let want = i32::from(texel[c]);
                let d = got - want;
                err += (d * d) as u32;
            }
            if err < best_err {
                best_err = err;
                best_raw = v as u8;
                if err == 0 {
                    break;
                }
            }
        }
        best_raw
    })
}
