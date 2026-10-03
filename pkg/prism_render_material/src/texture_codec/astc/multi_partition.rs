//! Full multi-partition 4x4 LDR ASTC decode (single weight plane).
//!
//! A multi-partition block splits the 4x4 footprint into 2, 3 or 4 regions,
//! each with its own endpoint pair, and assigns every texel to a region with
//! the procedural partition hash ([`super::partition::select_partition`]). The
//! single weight stream is shared across all partitions; each texel
//! interpolates between its own partition's endpoints.
//!
//! Layout (little-endian 128-bit block), transcribed from astcenc
//! `astcenc_symbolic_physical.cpp` (`physical_to_symbolic`, Apache-2.0):
//!
//! * bits `[0, 11)`    block mode,
//! * bits `[11, 13)`   partition count minus one,
//! * bits `[13, 23)`   10-bit partition seed (`PARTITION_INDEX_BITS == 10`),
//! * bits `[23, 29)`   low 6 bits of the colour-endpoint-mode field,
//! * from bit `29`     the concatenated endpoint integer sequence,
//! * below the weights  the high part of the CEM field
//!                      (`3 * partition_count - 4` bits) when the per-partition
//!                      form is used,
//! * top of the block   the shared weight integer sequence (bit-reversed).
//!
//! Only the single-plane LDR subset is handled here; multi-partition dual-plane
//! and any HDR colour format return an [`AstcError`] so no unsupported block is
//! decoded to approximate pixels. Those are later milestones.
//!
//! Decode is pure integer arithmetic -- no AI/ML path.

use super::bise::{decode_ise, IseRange};
use super::block_mode::decode_block_mode_2d;
use super::block_reader::read_bits;
use super::cem::{cem_integer_count, cem_is_ldr, unpack_endpoints};
use super::color_unquant::{color_quant_num_levels, unquant_color};
use super::endpoints::Endpoints;
use super::infill::infill_weights_4x4;
use super::quant_mode::{color_quant_level, QUANT_6};
use super::single_partition::lerp_component;
use super::AstcError;

/// `PARTITION_INDEX_BITS` from the ASTC spec: the partition seed is 10 bits.
const PARTITION_INDEX_BITS: u32 = 10;

/// Colour bit budget per partition count (`color_bits_arr` in astcenc).
/// Index 0 is unused; index 1 is single-partition (`115 - 4`); indices 2..=4
/// are `113 - 4 - PARTITION_INDEX_BITS`.
const COLOR_BITS_ARR: [i32; 5] = [-1, 111, 99, 99, 99];

/// Decode a multi-partition (2/3/4) single-plane 4x4 LDR ASTC `block` to
/// sixteen RGBA8 texels in row-major order (`texel = y * 4 + x`).
///
/// # Errors
/// Returns an [`AstcError`] for any block outside the supported subset: a
/// single-partition block (handled elsewhere), a dual-plane block, an
/// oversized weight grid, an HDR colour format, or any encoding whose derived
/// colour quant level is below QUANT_6. No unsupported block is decoded to
/// approximate pixels.
pub(super) fn decode_multi_partition_4x4_ldr(block: &[u8; 16]) -> Result<[[u8; 4]; 16], AstcError> {
    let mode = (u16::from(block[1]) << 8 | u16::from(block[0])) & 0x07FF;
    let bm = decode_block_mode_2d(mode).ok_or(AstcError::UnsupportedBlockMode)?;

    // The weight grid must fit inside the 4x4 texel footprint (see the
    // single-partition path for the legality rule).
    if bm.weights_x > 4 || bm.weights_y > 4 {
        return Err(AstcError::UnsupportedBlockMode);
    }

    // Multi-partition dual-plane is a later milestone; reject it rather than
    // mis-decode. (astcenc also forbids dual-plane with four partitions.)
    if bm.dual_plane {
        return Err(AstcError::UnsupportedBlockMode);
    }

    let partition_count = (read_bits(block, 11, 2) + 1) as i32;
    if !(2..=4).contains(&partition_count) {
        return Err(AstcError::UnsupportedBlockMode);
    }

    let seed = read_bits(block, 13, PARTITION_INDEX_BITS) as i32;

    // --- Colour endpoint mode field (per-partition encoding) ---------------
    // The 6 low bits sit at [23, 29); the high part (3*pc - 4 bits) lives just
    // below the weight stream. If the "all partitions share one class" form is
    // used (base class 0), the high part is not consumed and the colour budget
    // is not reduced.
    let bits_for_weights = bm.weight_bits;
    let highpart_size = (3 * partition_count - 4) as u32;
    let below_weights_pos = 128 - bits_for_weights - highpart_size;

    let encoded_type =
        read_bits(block, 23, 6) | (read_bits(block, below_weights_pos, highpart_size) << 6);
    let baseclass = encoded_type & 0x3;

    let mut color_formats = [0u32; 4];
    let pc = partition_count as usize;
    let effective_highpart_size;
    if baseclass == 0 {
        let fmt = (encoded_type >> 2) & 0xF;
        for f in color_formats.iter_mut().take(pc) {
            *f = fmt;
        }
        // Shared-class form does not spend the high part of the field.
        effective_highpart_size = 0;
    } else {
        let base = baseclass - 1;
        let mut bitpos = 2u32;
        for f in color_formats.iter_mut().take(pc) {
            *f = (((encoded_type >> bitpos) & 1) + base) << 2;
            bitpos += 1;
        }
        for f in color_formats.iter_mut().take(pc) {
            *f |= (encoded_type >> bitpos) & 0x3;
            bitpos += 2;
        }
        effective_highpart_size = highpart_size;
    }

    // Every partition must use an LDR colour format for this milestone.
    for &fmt in color_formats.iter().take(pc) {
        if !cem_is_ldr(fmt) {
            return Err(AstcError::UnsupportedHdr);
        }
    }

    // --- Colour quantisation level -----------------------------------------
    let mut color_integer_count = 0u32;
    for &fmt in color_formats.iter().take(pc) {
        color_integer_count += cem_integer_count(fmt);
    }
    if color_integer_count > 18 {
        return Err(AstcError::Reserved);
    }

    let color_bits = COLOR_BITS_ARR[pc] - bits_for_weights as i32 - effective_highpart_size as i32;
    if color_bits < 0 {
        return Err(AstcError::Reserved);
    }
    let level = color_quant_level(color_integer_count, color_bits as usize);
    if level < QUANT_6 {
        return Err(AstcError::Reserved);
    }
    let level_index = (level - QUANT_6) as usize;
    let num_levels = color_quant_num_levels(level_index);
    let range = IseRange::from_num_levels(num_levels).ok_or(AstcError::Reserved)?;

    // --- Decode and unquantize the concatenated endpoint sequence ----------
    // Multi-partition endpoints start at bit 19 + PARTITION_INDEX_BITS == 29.
    let start_bit = 19 + PARTITION_INDEX_BITS;
    let mut packed = [0u8; 18];
    let n = color_integer_count as usize;
    decode_ise(
        block,
        start_bit,
        range,
        color_integer_count,
        &mut packed[..n],
    )?;

    let mut vals = [0u8; 18];
    for (v, p) in vals[..n].iter_mut().zip(packed[..n].iter()) {
        *v = unquant_color(level_index, *p);
    }

    // Split the integer run per partition and assemble each endpoint pair.
    let mut endpoints = [Endpoints {
        e0: [0; 4],
        e1: [0; 4],
    }; 4];
    let mut off = 0usize;
    for (i, &fmt) in color_formats.iter().take(pc).enumerate() {
        let count = cem_integer_count(fmt) as usize;
        endpoints[i] = unpack_endpoints(fmt, &vals[off..off + count]);
        off += count;
    }

    // --- Per-texel partition dispatch + interpolation ----------------------
    // 4x4 has 16 texels (< 32) so the small-block coordinate bias is active.
    let weights = infill_weights_4x4(block, bm.weights_x, bm.weights_y, bm.weight_levels)?;
    let mut out = [[0u8; 4]; 16];
    for y in 0..4i32 {
        for x in 0..4i32 {
            let texel = (y * 4 + x) as usize;
            let part =
                super::partition::select_partition(seed, x, y, 0, partition_count, true) as usize;
            let ep = &endpoints[part];
            let w = u32::from(weights[texel]);
            out[texel] = [
                lerp_component(ep.e0[0], ep.e1[0], w),
                lerp_component(ep.e0[1], ep.e1[1], w),
                lerp_component(ep.e0[2], ep.e1[2], w),
                lerp_component(ep.e0[3], ep.e1[3], w),
            ];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_partition_block_is_rejected() {
        // Partition-count field 0 => single partition; this path only handles
        // 2..=4 and must refuse it (the dispatcher routes single partition
        // elsewhere).
        let mut block = [0u8; 16];
        let mode = 578u16;
        block[0] = mode as u8;
        block[1] = (mode >> 8) as u8;
        assert_eq!(
            decode_multi_partition_4x4_ldr(&block),
            Err(AstcError::UnsupportedBlockMode)
        );
    }

    #[test]
    fn dual_plane_multi_partition_is_rejected() {
        // Mode 583 is a dual-plane 4x4 mode; with two partitions this is out of
        // scope for the single-plane milestone and must error rather than
        // mis-decode.
        let mut block = [0u8; 16];
        let mode = 583u16;
        block[0] = mode as u8;
        block[1] = (mode >> 8) as u8;
        block[1] |= 1 << 3; // partition count field => 2 partitions
        let r = decode_multi_partition_4x4_ldr(&block);
        assert!(matches!(r, Err(AstcError::UnsupportedBlockMode)), "{r:?}");
    }
}
