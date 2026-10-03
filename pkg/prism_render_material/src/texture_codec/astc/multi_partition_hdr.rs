//! Full multi-partition 4x4 ASTC **HDR** decode (HDR, LDR and mixed partitions).
//!
//! A multi-partition HDR block reuses the exact header, partition-seed and
//! colour-endpoint integer-sequence parse of the LDR multi-partition path
//! (shared in [`super::multi_partition::parse_multi_partition_color`]); only
//! the endpoint expansion and the per-texel interpolation differ. Each
//! partition's integer run is expanded per its own Colour Endpoint Mode: an
//! HDR partition through the HDR unpack
//! ([`super::hdr_endpoints::unpack_hdr_endpoints`], logarithmic lanes), an LDR
//! partition through the LDR unpack ([`super::cem::unpack_endpoints`]) widened
//! into the 16-bit linear HDR domain
//! ([`super::hdr_endpoints::expand_ldr_endpoints_to_hdr`], `x257`,
//! `lns = false`). Every lane is then interpolated by
//! [`super::hdr_endpoints::lerp_hdr_lane`], which routes each lane through the
//! logarithmic or linear FP16 conversion according to its per-channel `lns`
//! bit -- so LDR and HDR partitions mix within one block exactly as a
//! hardware HDR-profile decode does.
//!
//! # Partition coverage
//! This path decodes multi-partition (2/3/4) blocks under the HDR profile with
//! **any** mix of the six HDR Colour Endpoint Modes (2, 3, 7, 11, 14, 15) and
//! the ten LDR modes across partitions: all-HDR, all-LDR, and mixed LDR/HDR.
//! The LDR->HDR expansion is the reference `unpack_color_endpoints` behaviour
//! for `ASTCENC_PRF_HDR` (`output_scale = select(257, 1, hdr_lanes)`), so no
//! pixel is approximated. (A single-partition block routes to the dedicated
//! single-partition decoders, not here.)
//!
//! Decode is pure integer / `f32` arithmetic -- no AI/ML path.

use super::block_reader::read_bits;
use super::cem::{cem_integer_count, cem_is_ldr, unpack_endpoints};
use super::hdr_endpoints::{
    expand_ldr_endpoints_to_hdr, lerp_hdr_lane, unpack_hdr_endpoints, HdrEndpoints,
};
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
/// block (forbidden by the spec), an oversized weight grid, or any encoding
/// whose derived colour quant level is below QUANT_6. LDR, HDR and mixed
/// LDR/HDR partition combinations all decode. No unsupported block is decoded
/// to approximate pixels.
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

    // Split the integer run per partition and expand each endpoint pair into
    // the HDR interpolation domain (HDR CEM: native logarithmic lanes; LDR CEM:
    // linear lanes widened x257).
    let mut endpoints = [HdrEndpoints {
        e0: [0; 4],
        e1: [0; 4],
        lns: [false; 4],
    }; 4];
    let mut off = 0usize;
    for (i, &fmt) in color_formats.iter().take(pc).enumerate() {
        let count = cem_integer_count(fmt) as usize;
        let run = &vals[off..off + count];
        endpoints[i] = if cem_is_ldr(fmt) {
            // LDR partition in the HDR profile: unpack the 8-bit LDR endpoints
            // and widen each lane into the 16-bit linear HDR domain (x257,
            // `lns = false`). The per-texel loop below mixes these linear lanes
            // with the logarithmic lanes of any HDR partition transparently via
            // the per-channel `lns` mask.
            expand_ldr_endpoints_to_hdr(&unpack_endpoints(fmt, run))
        } else {
            unpack_hdr_endpoints(fmt, run)
        };
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
    fn all_ldr_multi_partition_decodes_in_hdr_profile() {
        // A legal two-partition single-plane block whose shared CEM is an LDR
        // mode (0 => luminance direct) now decodes under the HDR profile: both
        // partitions are expanded with the x257 linear widening, so every lane
        // is finite and the decode succeeds rather than being refused. (This is
        // the degenerate all-LDR case of the mixed LDR/HDR path.)
        let mut block = [0u8; 16];
        let mode = 578u16;
        block[0] = mode as u8;
        block[1] = (mode >> 8) as u8;
        // Partition-count field (bits 11-12) = 0b01 => two partitions.
        block[1] |= 0b01 << 3;
        // CEM low field at [23, 29) left zero => shared-class form, CEM 0 (LDR
        // luminance direct).
        let texels = decode_multi_partition_4x4_hdr(&block).expect("all-LDR HDR decode");
        for t in &texels {
            for &c in t {
                assert!(c.is_finite(), "lane must be finite, got {c}");
            }
        }
    }
}
