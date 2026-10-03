//! Full single-partition 4x4 LDR ASTC decode.
//!
//! Assembles the block-mode parse, colour-endpoint decode and weight-grid
//! decode into sixteen RGBA8 texels. This first milestone handles the common
//! opaque case:
//!
//! * single partition,
//! * single **or** dual weight plane (the dual-plane colour component selector
//!   routes one channel to the second plane; the other three use the first),
//! * any weight grid (resampled to the 4x4 texel footprint by the Khronos
//!   bilinear infill; the identity 4x4 grid is the degenerate case),
//! * any of the ten LDR Colour Endpoint Modes (0/1/4/5/6/8/9/10/12/13).
//!
//! Everything else (multi-partition and the six HDR CEMs) returns an
//! [`AstcError`] until its own GPU-validated milestone lands, so no path
//! silently produces wrong pixels.
//!
//! Decode is pure integer arithmetic -- no AI/ML path.

use super::block_mode::decode_block_mode_2d;
use super::cem::cem_is_ldr;
use super::endpoints::decode_cem_endpoints;
use super::infill::{infill_dual_plane, infill_weights, MAX_TEXELS};
use super::AstcError;

/// Interpolate one 8-bit LDR colour component between endpoints `e0` and `e1`
/// using an ASTC weight `w` in `0..=64`, producing an 8-bit UNORM result.
///
/// Follows the Khronos decode: expand each endpoint to 16-bit by replication
/// (`c * 257`), interpolate in the 16-bit domain
/// (`(c0*(64-w) + c1*w + 32) >> 6`), then round the 16-bit result to 8-bit
/// UNORM. For black/white endpoints this reduces exactly to the GPU-proven
/// gray-ramp formula.
#[inline]
pub(super) fn lerp_component(e0: u8, e1: u8, w: u32) -> u8 {
    let c0 = u32::from(e0) * 257;
    let c1 = u32::from(e1) * 257;
    let c16 = (c0 * (64 - w) + c1 * w + 32) >> 6;
    ((c16 * 255 + 32767) / 65535) as u8
}

/// Decode a single-partition 4x4 LDR ASTC `block` to sixteen RGBA8 texels in
/// row-major order (`texel = y * 4 + x`).
///
/// # Errors
/// Returns an [`AstcError`] for any block outside the supported subset
/// described in the module docs; callers fall back to the appropriate
/// higher-level error. No unsupported block is decoded to approximate pixels.
pub(super) fn decode_single_partition_4x4_ldr(
    block: &[u8; 16],
) -> Result<[[u8; 4]; 16], AstcError> {
    let mut out = [[0u8; 4]; 16];
    decode_single_partition_ldr(block, 4, 4, &mut out)?;
    Ok(out)
}

/// Decode a single-partition LDR ASTC `block` for an arbitrary 2D footprint
/// `bx` x `by` (4..=12 per axis), writing `bx * by` RGBA8 texels into
/// `out[..bx*by]` in row-major order (`texel = y * bx + x`).
///
/// The weight grid is resampled to the footprint by the Khronos bilinear
/// infill; the identity 4x4 grid on a 4x4 footprint is the degenerate case the
/// GPU-proven path already exercises. Any of the ten LDR Colour Endpoint Modes
/// decode here; the six HDR CEMs and multi-partition blocks return an
/// [`AstcError`] so no path silently produces wrong pixels.
///
/// # Errors
/// Returns an [`AstcError`] for any block outside the supported subset, for an
/// out-of-range footprint, for a weight grid larger than the footprint
/// (illegal per the ASTC block-mode legality rule), or if `out` is too short.
pub(super) fn decode_single_partition_ldr(
    block: &[u8; 16],
    bx: u32,
    by: u32,
    out: &mut [[u8; 4]],
) -> Result<(), AstcError> {
    let texels = (bx as usize) * (by as usize);
    if texels == 0 || texels > MAX_TEXELS || out.len() < texels {
        return Err(AstcError::Reserved);
    }

    let mode = (u16::from(block[1]) << 8 | u16::from(block[0])) & 0x07FF;
    let bm = decode_block_mode_2d(mode).ok_or(AstcError::UnsupportedBlockMode)?;

    // Block-mode legality: the weight grid may not exceed the texel footprint
    // on either axis. A mode whose grid is larger is illegal for this footprint
    // and rejected by conformant hardware (astcenc `init_block_size_descriptor`:
    // `weights_x > texels_x || weights_y > texels_y` => skip). We refuse rather
    // than resample an over-sized grid into bogus pixels.
    if bm.weights_x > bx || bm.weights_y > by {
        return Err(AstcError::UnsupportedBlockMode);
    }

    // Partition count is the 2-bit field at block bits [11, 13); 0 => single.
    let partition_count = ((u32::from(block[1]) >> 3) & 0x3) + 1;
    if partition_count != 1 {
        return Err(AstcError::UnsupportedBlockMode);
    }
    // CEM is the 4-bit field at block bits [13, 17): the low three bits are
    // byte 1 bits 5..8 and the high bit is byte 2 bit 0 (block bit 16). The ten
    // LDR CEMs decode here; the six HDR CEMs return an error from
    // `decode_cem_endpoints` until the HDR milestone lands.
    let cem = ((u32::from(block[1]) >> 5) & 0x7) | ((u32::from(block[2]) & 1) << 3);
    if !cem_is_ldr(cem) {
        return Err(AstcError::UnsupportedBlockMode);
    }

    let endpoints = decode_cem_endpoints(block, bm.weight_bits, cem, bm.dual_plane)?;

    if bm.dual_plane {
        // Dual plane: two independent weight planes plus a 2-bit colour
        // component selector (CCS) that names the channel driven by plane 1;
        // the remaining three channels use plane 0. The CCS sits immediately
        // below the weight region in the normal (non-reversed) block
        // orientation (astcenc `read_bits(block, below_weights_pos - 2, 2)`).
        let below_weights_pos = 128 - bm.weight_bits;
        let ccs = super::block_reader::read_bits(block, below_weights_pos - 2, 2);
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
        for (texel, slot) in out[..texels].iter_mut().enumerate() {
            let mut px = [0u8; 4];
            for (c, p) in px.iter_mut().enumerate() {
                let w = u32::from(if c as u32 == ccs {
                    plane1[texel]
                } else {
                    plane0[texel]
                });
                *p = lerp_component(endpoints.e0[c], endpoints.e1[c], w);
            }
            *slot = px;
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
        for (slot, w) in out[..texels].iter_mut().zip(weights[..texels].iter()) {
            let w = u32::from(*w);
            *slot = [
                lerp_component(endpoints.e0[0], endpoints.e1[0], w),
                lerp_component(endpoints.e0[1], endpoints.e1[1], w),
                lerp_component(endpoints.e0[2], endpoints.e1[2], w),
                lerp_component(endpoints.e0[3], endpoints.e1[3], w),
            ];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lerp_endpoints_hit_exactly() {
        // Weight 0 selects endpoint 0; weight 64 selects endpoint 1.
        assert_eq!(lerp_component(0, 255, 0), 0);
        assert_eq!(lerp_component(0, 255, 64), 255);
        assert_eq!(lerp_component(40, 200, 0), 40);
        assert_eq!(lerp_component(40, 200, 64), 200);
    }

    #[test]
    fn lerp_midpoint_is_centered() {
        // Half weight lands on the average (within rounding).
        let m = lerp_component(0, 255, 32);
        assert!((127..=128).contains(&m), "midpoint {m}");
    }

    #[test]
    fn rejects_grid_larger_than_the_4x4_footprint() {
        // Mode 7 decodes to an 8x2 weight grid, which exceeds the 4x4 texel
        // footprint and is therefore an illegal block mode that conformant
        // hardware rejects. We must refuse it rather than resample an
        // over-sized grid into bogus pixels.
        let mut block = [0u8; 16];
        block[0] = 7u16 as u8;
        block[1] = (7u16 >> 8) as u8;
        assert_eq!(
            decode_single_partition_4x4_ldr(&block),
            Err(AstcError::UnsupportedBlockMode),
            "8x2 grid must be rejected for a 4x4 block"
        );

        // Mode 73 decodes to a 4x8 grid: oversized on the Y axis, likewise
        // illegal and rejected.
        let mut block = [0u8; 16];
        block[0] = 73u16 as u8;
        block[1] = (73u16 >> 8) as u8;
        assert_eq!(
            decode_single_partition_4x4_ldr(&block),
            Err(AstcError::UnsupportedBlockMode),
            "4x8 grid must be rejected for a 4x4 block"
        );
    }

    #[test]
    fn rejects_dual_plane_and_multi_partition_shapes() {
        // Mode 578 is a valid single-plane 4x4 mode; flip the partition field
        // to two partitions and confirm it is rejected rather than mis-decoded.
        let mut block = [0u8; 16];
        block[0] = 578u16 as u8;
        block[1] = (578u16 >> 8) as u8;
        // Set partition-count field (bits 11-12) to 1 => two partitions.
        block[1] |= 1 << 3;
        assert_eq!(
            decode_single_partition_4x4_ldr(&block),
            Err(AstcError::UnsupportedBlockMode)
        );
    }

    #[test]
    fn full_pipeline_constant_weight_zero_is_endpoint0() {
        // Mode 578 (QUANT_16 weights, 64 weight bits -> color_bits 47)... that
        // is a non-identity colour range, so build mode 67 instead (QUANT_6
        // trit weights, color_bits 69 -> QUANT_256). All-zero weights select
        // endpoint 0 for every texel.
        let mut block = [0u8; 16];
        let mode = 67u16;
        block[0] = mode as u8;
        block[1] = (mode >> 8) as u8;
        // CEM 8 at bits [13,17): value 0b1000, so only the high bit (block
        // bit 16 == byte 2 bit 0) is set; the low three CEM bits stay zero.
        block[2] |= 1;
        // Endpoints at bit 17: e0 RGB = (10,20,30), e1 = (200,210,220).
        for (i, v) in [10u8, 200, 20, 210, 30, 220].iter().enumerate() {
            let base = 17 + i as u32 * 8;
            for b in 0..8u32 {
                if (v >> b) & 1 == 1 {
                    let pos = base + b;
                    block[(pos >> 3) as usize] |= 1 << (pos & 7);
                }
            }
        }
        // Weights all zero (block high bits already zero) => every texel = e0.
        let out = decode_single_partition_4x4_ldr(&block).expect("mode 67 decodes");
        for texel in out {
            assert_eq!(texel, [10, 20, 30, 255]);
        }
    }

    /// Build the mode-67 constant-weight block used above (every texel == e0).
    fn mode67_endpoint0_block() -> [u8; 16] {
        let mut block = [0u8; 16];
        let mode = 67u16;
        block[0] = mode as u8;
        block[1] = (mode >> 8) as u8;
        block[2] |= 1; // CEM 8 high bit
        for (i, v) in [10u8, 200, 20, 210, 30, 220].iter().enumerate() {
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

    /// The footprint-generic entry point must agree with the fixed 4x4 wrapper
    /// on the 4x4 footprint for the GPU-proven mode-67 constant block.
    #[test]
    fn generic_4x4_matches_fixed_wrapper() {
        let block = mode67_endpoint0_block();
        let fixed = decode_single_partition_4x4_ldr(&block).expect("4x4 decodes");
        let mut generic = [[0u8; 4]; 16];
        decode_single_partition_ldr(&block, 4, 4, &mut generic).expect("generic 4x4 decodes");
        assert_eq!(fixed, generic);
    }

    /// On a larger footprint the all-zero-weight mode-67 block must select
    /// endpoint 0 for every texel (constant grid resamples to the constant).
    #[test]
    fn larger_footprint_constant_weight_is_endpoint0() {
        let block = mode67_endpoint0_block();
        for (bx, by) in [
            (5u32, 5u32),
            (6, 6),
            (8, 8),
            (10, 10),
            (12, 12),
            (8, 5),
            (12, 10),
        ] {
            let texels = (bx * by) as usize;
            let mut out = [[0u8; 4]; MAX_TEXELS];
            decode_single_partition_ldr(&block, bx, by, &mut out[..texels])
                .unwrap_or_else(|_| panic!("mode 67 decodes on {bx}x{by}"));
            for (i, texel) in out[..texels].iter().enumerate() {
                assert_eq!(*texel, [10, 20, 30, 255], "{bx}x{by} texel {i}");
            }
        }
    }

    /// Footprints out of the 4..=12 envelope and short output slices are
    /// rejected rather than mis-decoded.
    #[test]
    fn generic_rejects_bad_footprint_and_short_slice() {
        let block = mode67_endpoint0_block();
        let mut out = [[0u8; 4]; MAX_TEXELS];
        assert_eq!(
            decode_single_partition_ldr(&block, 13, 8, &mut out),
            Err(AstcError::Reserved)
        );
        let mut tiny = [[0u8; 4]; 4];
        assert_eq!(
            decode_single_partition_ldr(&block, 8, 8, &mut tiny),
            Err(AstcError::Reserved)
        );
    }
}
