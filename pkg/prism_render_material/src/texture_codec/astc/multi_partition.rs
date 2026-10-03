//! Full multi-partition 4x4 ASTC decode (shared header parse + LDR decode).
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
//! The CEM-field parse, colour quantisation and endpoint integer-sequence
//! decode are identical for LDR and HDR colour formats, so they live in the
//! shared [`parse_multi_partition_color`] helper. The LDR decoder below adds
//! the LDR endpoint expansion and integer interpolation; the HDR decoder (see
//! [`super::multi_partition_hdr`]) adds the HDR endpoint unpack and the
//! logarithmic FP16 interpolation.
//!
//! Both single-plane and dual-plane blocks are handled (dual-plane is legal
//! only for two and three partitions -- the ASTC spec forbids four-way
//! dual-plane, which astcenc also rejects).
//!
//! Decode is pure integer arithmetic -- no AI/ML path.

use super::bise::{decode_ise, IseRange};
use super::block_mode::{decode_block_mode_2d, BlockMode2d};
use super::block_reader::read_bits;
use super::cem::{cem_integer_count, cem_is_ldr, unpack_endpoints};
use super::color_unquant::{color_quant_num_levels, unquant_color};
use super::endpoints::Endpoints;
use super::infill::{infill_dual_plane_4x4, infill_weights_4x4};
use super::quant_mode::{color_quant_level, QUANT_6};
use super::single_partition::lerp_component;
use super::AstcError;

/// `PARTITION_INDEX_BITS` from the ASTC spec: the partition seed is 10 bits.
const PARTITION_INDEX_BITS: u32 = 10;

/// Colour bit budget per partition count (`color_bits_arr` in astcenc).
/// Index 0 is unused; index 1 is single-partition (`115 - 4`); indices 2..=4
/// are `113 - 4 - PARTITION_INDEX_BITS`.
const COLOR_BITS_ARR: [i32; 5] = [-1, 111, 99, 99, 99];

/// The profile-agnostic result of parsing a multi-partition block header: the
/// block mode, partition geometry and the unquantized colour endpoint integers
/// (one run per partition, concatenated in `vals`). The caller expands `vals`
/// into endpoint pairs with the LDR or HDR unpack appropriate to each
/// partition's colour format.
pub(super) struct MultiPartitionColor {
    /// The decoded weight-grid block mode (shared across partitions).
    pub(super) bm: BlockMode2d,
    /// Partition count, always `2..=4` here.
    pub(super) partition_count: i32,
    /// 10-bit partition hash seed.
    pub(super) seed: i32,
    /// Per-partition colour endpoint mode (CEM); only the first
    /// `partition_count` entries are meaningful.
    pub(super) color_formats: [u32; 4],
    /// Size in bits of the CEM high part actually consumed below the weights
    /// (`0` for the shared-class form). The dual-plane CCS sits two bits below
    /// `128 - weight_bits - effective_highpart_size`.
    pub(super) effective_highpart_size: u32,
    /// Unquantized colour endpoint integers, concatenated per partition.
    pub(super) vals: [u8; 18],
}

/// Parse a multi-partition (2/3/4) 4x4 block header and decode its colour
/// endpoint integer sequence, independent of LDR/HDR colour profile.
///
/// # Errors
/// Returns an [`AstcError`] for any block outside the supported subset: a
/// single-partition block (handled elsewhere), a four-partition dual-plane
/// block (forbidden by the spec), an oversized weight grid, or any encoding
/// whose derived colour quant level is below QUANT_6 or whose integer count
/// exceeds 18. No colour-profile (LDR vs HDR) check is applied here; the caller
/// enforces it per partition.
pub(super) fn parse_multi_partition_color(
    block: &[u8; 16],
) -> Result<MultiPartitionColor, AstcError> {
    let mode = (u16::from(block[1]) << 8 | u16::from(block[0])) & 0x07FF;
    let bm = decode_block_mode_2d(mode).ok_or(AstcError::UnsupportedBlockMode)?;

    // The weight grid must fit inside the 4x4 texel footprint (see the
    // single-partition path for the legality rule).
    if bm.weights_x > 4 || bm.weights_y > 4 {
        return Err(AstcError::UnsupportedBlockMode);
    }

    let partition_count = (read_bits(block, 11, 2) + 1) as i32;
    if !(2..=4).contains(&partition_count) {
        return Err(AstcError::UnsupportedBlockMode);
    }

    // Dual-plane weights with four partitions are forbidden by the ASTC spec
    // (astcenc returns an error block for this combination). Two- and
    // three-partition dual-plane are valid and handled below.
    if bm.dual_plane && partition_count == 4 {
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

    // --- Colour quantisation level -----------------------------------------
    let mut color_integer_count = 0u32;
    for &fmt in color_formats.iter().take(pc) {
        color_integer_count += cem_integer_count(fmt);
    }
    if color_integer_count > 18 {
        return Err(AstcError::Reserved);
    }

    let mut color_bits =
        COLOR_BITS_ARR[pc] - bits_for_weights as i32 - effective_highpart_size as i32;
    if bm.dual_plane {
        // The dual-plane colour component selector steals two bits from the
        // colour budget (astcenc `color_bits -= 2`).
        color_bits -= 2;
    }
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

    Ok(MultiPartitionColor {
        bm,
        partition_count,
        seed,
        color_formats,
        effective_highpart_size,
        vals,
    })
}

/// Decode a multi-partition (2/3/4) 4x4 **LDR** ASTC `block` to sixteen RGBA8
/// texels in row-major order (`texel = y * 4 + x`). Both single- and dual-plane
/// weights are handled.
///
/// # Errors
/// Returns an [`AstcError`] for any block outside the supported subset: a
/// single-partition block (handled elsewhere), a four-partition dual-plane
/// block (forbidden by the spec), an oversized weight grid, an HDR colour
/// format in any partition, or any encoding whose derived colour quant level is
/// below QUANT_6. No unsupported block is decoded to approximate pixels.
pub(super) fn decode_multi_partition_4x4_ldr(block: &[u8; 16]) -> Result<[[u8; 4]; 16], AstcError> {
    let parsed = parse_multi_partition_color(block)?;
    let MultiPartitionColor {
        bm,
        partition_count,
        seed,
        color_formats,
        effective_highpart_size,
        vals,
    } = parsed;
    let pc = partition_count as usize;

    // Every partition must use an LDR colour format on this (LDR) path.
    for &fmt in color_formats.iter().take(pc) {
        if !cem_is_ldr(fmt) {
            return Err(AstcError::UnsupportedHdr);
        }
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
    let mut out = [[0u8; 4]; 16];
    if bm.dual_plane {
        // Two interleaved weight planes plus a 2-bit colour component selector
        // (CCS). The CCS sits immediately below the weight region and, when the
        // per-partition CEM form is used, below its high part as well -- this is
        // astcenc's final `below_weights_pos - 2`
        // (`128 - bits_for_weights - effective_highpart_size - 2`). The selected
        // channel interpolates with plane 1; the other three use plane 0.
        let ccs = read_bits(block, 128 - bm.weight_bits - effective_highpart_size - 2, 2);
        let (plane0, plane1) =
            infill_dual_plane_4x4(block, bm.weights_x, bm.weights_y, bm.weight_levels)?;
        for y in 0..4i32 {
            for x in 0..4i32 {
                let texel = (y * 4 + x) as usize;
                let part = super::partition::select_partition(seed, x, y, 0, partition_count, true)
                    as usize;
                let ep = &endpoints[part];
                let mut px = [0u8; 4];
                for (c, p) in px.iter_mut().enumerate() {
                    let w = u32::from(if c as u32 == ccs {
                        plane1[texel]
                    } else {
                        plane0[texel]
                    });
                    *p = lerp_component(ep.e0[c], ep.e1[c], w);
                }
                out[texel] = px;
            }
        }
    } else {
        let weights = infill_weights_4x4(block, bm.weights_x, bm.weights_y, bm.weight_levels)?;
        for y in 0..4i32 {
            for x in 0..4i32 {
                let texel = (y * 4 + x) as usize;
                let part = super::partition::select_partition(seed, x, y, 0, partition_count, true)
                    as usize;
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
    fn dual_plane_four_partition_is_rejected() {
        // Mode 1089 is a legal dual-plane 4x4 grid. Two- and three-partition
        // dual-plane are supported, but four-partition dual-plane is forbidden
        // by the ASTC spec (astcenc returns an error block), so it must be
        // rejected rather than mis-decoded.
        let mut block = [0u8; 16];
        let mode = 1089u16;
        block[0] = mode as u8;
        block[1] = (mode >> 8) as u8;
        // Partition-count field (bits 11-12) = 0b11 => four partitions.
        block[1] |= 0b11 << 3;
        let r = decode_multi_partition_4x4_ldr(&block);
        assert_eq!(r, Err(AstcError::UnsupportedBlockMode), "{r:?}");
    }
}
