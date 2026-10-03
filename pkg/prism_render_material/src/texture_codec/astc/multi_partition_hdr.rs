//! Full multi-partition 4x4 ASTC **HDR** decode (all-HDR-partition subset).
//!
//! A multi-partition HDR block reuses the exact header, partition-seed and
//! colour-endpoint integer-sequence parse of the LDR multi-partition path
//! (shared in [`super::multi_partition::parse_multi_partition_color`]); only
//! the endpoint expansion and the per-texel interpolation differ. Each
//! partition's integer run is expanded with the HDR endpoint unpack
//! ([`super::hdr_endpoints::unpack_hdr_endpoints`]) and every lane is
//! interpolated in the logarithmic FP16 domain
//! ([`super::hdr_endpoints::lerp_hdr_lane`]), mirroring the single-partition
//! HDR decoder.
//!
//! # Honest subset
//! This milestone decodes blocks where **every** partition uses one of the six
//! HDR Colour Endpoint Modes (2, 3, 7, 11, 14, 15). A block that mixes LDR and
//! HDR partitions is a distinct, later milestone: the ASTC spec expands the LDR
//! partitions into the HDR interpolation domain (`lns = false`, UNORM lanes
//! scaled x257), which this decoder does not yet implement. Rather than
//! approximate those pixels, a mixed or all-LDR multi-partition block routed
//! here returns [`AstcError::UnsupportedBlockMode`].
//!
//! Decode is pure integer / `f32` arithmetic -- no AI/ML path.

use super::block_reader::read_bits;
use super::cem::{cem_integer_count, cem_is_ldr};
use super::hdr_endpoints::{lerp_hdr_lane, unpack_hdr_endpoints, HdrEndpoints};
use super::infill::{infill_dual_plane_4x4, infill_weights_4x4};
use super::multi_partition::{parse_multi_partition_color, MultiPartitionColor};
use super::partition::select_partition;
use super::AstcError;

/// Decode a multi-partition (2/3/4) 4x4 **HDR** ASTC `block` to sixteen RGBA
/// texels in `f32`, row-major (`texel = y * 4 + x`). Both single- and
/// dual-plane weights are handled.
///
/// # Errors
/// Returns an [`AstcError`] for any block outside the supported subset: a
/// single-partition block (handled elsewhere), a four-partition dual-plane
/// block (forbidden by the spec), an oversized weight grid, any encoding whose
/// derived colour quant level is below QUANT_6, or a block with **any** LDR
/// partition (the mixed-domain case is a later milestone). No unsupported block
/// is decoded to approximate pixels.
pub(super) fn decode_multi_partition_4x4_hdr(
    block: &[u8; 16],
) -> Result<[[f32; 4]; 16], AstcError> {
    let MultiPartitionColor {
        bm,
        partition_count,
        seed,
        color_formats,
        effective_highpart_size,
        vals,
    } = parse_multi_partition_color(block)?;
    let pc = partition_count as usize;

    // Every partition must use an HDR colour format on this (HDR) path. A mixed
    // LDR/HDR block needs the LDR->HDR-domain expansion and is a later
    // milestone; refuse it rather than approximate pixels.
    for &fmt in color_formats.iter().take(pc) {
        if cem_is_ldr(fmt) {
            return Err(AstcError::UnsupportedBlockMode);
        }
    }

    // Split the integer run per partition and expand each HDR endpoint pair.
    let mut endpoints = [HdrEndpoints {
        e0: [0; 4],
        e1: [0; 4],
        lns: [false; 4],
    }; 4];
    let mut off = 0usize;
    for (i, &fmt) in color_formats.iter().take(pc).enumerate() {
        let count = cem_integer_count(fmt) as usize;
        endpoints[i] = unpack_hdr_endpoints(fmt, &vals[off..off + count]);
        off += count;
    }

    // --- Per-texel partition dispatch + logarithmic interpolation ----------
    // 4x4 has 16 texels (< 32) so the small-block partition coordinate bias is
    // active (`select_partition(.., small_block = true)`).
    let mut out = [[0.0f32; 4]; 16];
    if bm.dual_plane {
        // Dual-plane colour component selector: two bits immediately below the
        // weight region and, when the per-partition CEM form is used, below its
        // high part as well (astcenc `below_weights_pos - 2`). The selected
        // channel interpolates with plane 1; the other three use plane 0.
        let ccs = read_bits(block, 128 - bm.weight_bits - effective_highpart_size - 2, 2);
        let (plane0, plane1) =
            infill_dual_plane_4x4(block, bm.weights_x, bm.weights_y, bm.weight_levels)?;
        for y in 0..4i32 {
            for x in 0..4i32 {
                let texel = (y * 4 + x) as usize;
                let part = select_partition(seed, x, y, 0, partition_count, true) as usize;
                let ep = &endpoints[part];
                for c in 0..4usize {
                    let w = u32::from(if c as u32 == ccs {
                        plane1[texel]
                    } else {
                        plane0[texel]
                    });
                    out[texel][c] = lerp_hdr_lane(ep.e0[c], ep.e1[c], w, ep.lns[c]);
                }
            }
        }
    } else {
        let weights = infill_weights_4x4(block, bm.weights_x, bm.weights_y, bm.weight_levels)?;
        for y in 0..4i32 {
            for x in 0..4i32 {
                let texel = (y * 4 + x) as usize;
                let part = select_partition(seed, x, y, 0, partition_count, true) as usize;
                let ep = &endpoints[part];
                let w = u32::from(weights[texel]);
                for c in 0..4usize {
                    out[texel][c] = lerp_hdr_lane(ep.e0[c], ep.e1[c], w, ep.lns[c]);
                }
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
        // Partition-count field 0 => single partition; this multi-partition
        // path must refuse it (the dispatcher routes single partition
        // elsewhere).
        let mut block = [0u8; 16];
        let mode = 578u16;
        block[0] = mode as u8;
        block[1] = (mode >> 8) as u8;
        assert_eq!(
            decode_multi_partition_4x4_hdr(&block),
            Err(AstcError::UnsupportedBlockMode)
        );
    }

    #[test]
    fn all_ldr_multi_partition_is_rejected() {
        // A legal two-partition single-plane block whose shared CEM is an LDR
        // mode (0 => luminance direct) must be refused on the HDR path rather
        // than mis-decoded: there are no HDR partitions to interpolate.
        let mut block = [0u8; 16];
        let mode = 578u16;
        block[0] = mode as u8;
        block[1] = (mode >> 8) as u8;
        // Partition-count field (bits 11-12) = 0b01 => two partitions.
        block[1] |= 0b01 << 3;
        // CEM low field at [23, 29) left zero => shared-class form, CEM 0 (LDR
        // luminance direct). The HDR path must reject it.
        let r = decode_multi_partition_4x4_hdr(&block);
        assert_eq!(r, Err(AstcError::UnsupportedBlockMode), "{r:?}");
    }
}
