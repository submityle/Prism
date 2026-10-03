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

use super::super::single_partition::lerp_component;
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
