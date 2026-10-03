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
use super::infill::{infill_dual_plane, infill_weights, MAX_TEXELS};
use super::multi_partition::{parse_multi_partition_color, MultiPartitionColor};
use super::partition::select_partition;
use super::AstcError;

/// Decode a multi-partition (2/3/4) 4x4 **HDR** ASTC `block` to sixteen RGBA
/// texels in `f32`, row-major (`texel = y * 4 + x`). Thin wrapper over
/// [`decode_multi_partition_hdr`] on the 4x4 footprint.
///
/// # Errors
/// Propagates [`AstcError`] from [`decode_multi_partition_hdr`].
pub(super) fn decode_multi_partition_4x4_hdr(
    block: &[u8; 16],
) -> Result<[[f32; 4]; 16], AstcError> {
    let mut out = [[0.0f32; 4]; 16];
    decode_multi_partition_hdr(block, 4, 4, &mut out)?;
    Ok(out)
}

/// Decode a multi-partition (2/3/4) **HDR** ASTC `block` for an arbitrary 2D
/// footprint `bx` x `by` (4..=12 per axis), writing `bx * by` RGBA `f32` texels
/// into `out[..bx*by]` in row-major order (`texel = y * bx + x`). Both single-
/// and dual-plane weights are handled; LDR, HDR and mixed LDR/HDR partition
/// combinations all decode.
///
/// The procedural partition assignment uses the small-block coordinate bias
/// when the footprint has fewer than 32 texels (astcenc `small_block`).
///
/// # Errors
/// Returns [`AstcError::Reserved`] for an out-of-range footprint or short `out`
/// slice, and otherwise an [`AstcError`] for any block outside the supported
/// subset: a single-partition block (handled elsewhere), a four-partition
/// dual-plane block (forbidden by the spec), an oversized weight grid, or any
/// encoding whose derived colour quant level is below QUANT_6. No unsupported
/// block is decoded to approximate pixels.
pub(super) fn decode_multi_partition_hdr(
    block: &[u8; 16],
    bx: u32,
    by: u32,
    out: &mut [[f32; 4]],
) -> Result<(), AstcError> {
    let texels = (bx as usize) * (by as usize);
    if texels == 0 || texels > MAX_TEXELS || out.len() < texels {
        return Err(AstcError::Reserved);
    }

    let MultiPartitionColor {
        bm,
        partition_count,
        seed,
        color_formats,
        effective_highpart_size,
        vals,
    } = parse_multi_partition_color(block, bx, by)?;
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
            expand_ldr_endpoints_to_hdr(&unpack_endpoints(fmt, run))
        } else {
            unpack_hdr_endpoints(fmt, run)
        };
        off += count;
    }

    // --- Per-texel partition dispatch + logarithmic interpolation ----------
    // The small-block coordinate bias is active when the footprint has < 32
    // texels (astcenc `small_block`).
    let small_block = texels < 32;
    let bxi = bx as i32;
    let byi = by as i32;
    if bm.dual_plane {
        let ccs = read_bits(block, 128 - bm.weight_bits - effective_highpart_size - 2, 2);
        let mut plane0 = [0u8; MAX_TEXELS];
        let mut plane1 = [0u8; MAX_TEXELS];
        infill_dual_plane(
            block,
            bm.weights_x,
            bm.weights_y,
            bm.weight_levels,
            bx,
            by,
            &mut plane0[..texels],
            &mut plane1[..texels],
        )?;
        for y in 0..byi {
            for x in 0..bxi {
                let texel = (y * bxi + x) as usize;
                let part = select_partition(seed, x, y, 0, partition_count, small_block) as usize;
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
        let mut weights = [0u8; MAX_TEXELS];
        infill_weights(
            block,
            bm.weights_x,
            bm.weights_y,
            bm.weight_levels,
            bx,
            by,
            &mut weights[..texels],
        )?;
        for y in 0..byi {
            for x in 0..bxi {
                let texel = (y * bxi + x) as usize;
                let part = select_partition(seed, x, y, 0, partition_count, small_block) as usize;
                let ep = &endpoints[part];
                let w = u32::from(weights[texel]);
                for c in 0..4usize {
                    out[texel][c] = lerp_hdr_lane(ep.e0[c], ep.e1[c], w, ep.lns[c]);
                }
            }
        }
    }
    Ok(())
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

    struct Rng(u32);
    impl Rng {
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            x
        }
    }

    #[test]
    fn generic_4x4_matches_fixed_wrapper() {
        let mut rng = Rng(0xFEED_BEEF);
        for _ in 0..512 {
            let mut block = [0u8; 16];
            for b in &mut block {
                *b = (rng.next_u32() & 0xFF) as u8;
            }
            block[1] = (block[1] & !(0b11 << 3)) | (((rng.next_u32() % 3 + 1) as u8) << 3);
            let fixed = decode_multi_partition_4x4_hdr(&block);
            let mut generic = [[0.0f32; 4]; 16];
            let gres = decode_multi_partition_hdr(&block, 4, 4, &mut generic);
            match (fixed, gres) {
                (Ok(f), Ok(())) => {
                    for (fx, gx) in f.iter().zip(generic.iter()) {
                        assert_eq!(fx.map(f32::to_bits), gx.map(f32::to_bits));
                    }
                }
                (Err(a), Err(b)) => assert_eq!(a, b),
                (a, b) => panic!("wrapper/generic disagree: {a:?} vs {b:?}"),
            }
        }
    }

    #[test]
    fn generic_rejects_bad_footprint_and_short_slice() {
        let block = [0u8; 16];
        let mut out = [[0.0f32; 4]; MAX_TEXELS];
        assert_eq!(
            decode_multi_partition_hdr(&block, 13, 12, &mut out),
            Err(AstcError::Reserved)
        );
        let mut tiny = [[0.0f32; 4]; 16];
        assert_eq!(
            decode_multi_partition_hdr(&block, 8, 8, &mut tiny),
            Err(AstcError::Reserved)
        );
    }
}
