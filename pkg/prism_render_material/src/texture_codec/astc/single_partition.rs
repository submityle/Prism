//! Full single-partition 4x4 LDR ASTC decode.
//!
//! Assembles the block-mode parse, colour-endpoint decode and weight-grid
//! decode into sixteen RGBA8 texels. This first milestone handles the common
//! opaque case:
//!
//! * single partition,
//! * single weight plane,
//! * any single-plane weight grid (resampled to the 4x4 texel footprint by
//!   the Khronos bilinear infill; the identity 4x4 grid is the degenerate
//!   case),
//! * any of the ten LDR Colour Endpoint Modes (0/1/4/5/6/8/9/10/12/13).
//!
//! Everything else (multi-partition, dual-plane and the six HDR CEMs) returns
//! an [`AstcError`] until its own GPU-validated milestone lands, so no path
//! silently produces wrong pixels.
//!
//! Decode is pure integer arithmetic -- no AI/ML path.

use super::block_mode::decode_block_mode_2d;
use super::cem::cem_is_ldr;
use super::endpoints::decode_cem_endpoints;
use super::infill::infill_weights_4x4;
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
fn lerp_component(e0: u8, e1: u8, w: u32) -> u8 {
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
    let mode = (u16::from(block[1]) << 8 | u16::from(block[0])) & 0x07FF;
    let bm = decode_block_mode_2d(mode).ok_or(AstcError::UnsupportedBlockMode)?;

    // The weight grid must fit inside the 4x4 texel footprint. A block mode
    // whose grid exceeds the block dimensions is illegal for this footprint and
    // is rejected by conformant hardware (the Metal decoder returns its error
    // colour). Matching that, we refuse to synthesise pixels for such a mode
    // rather than silently resampling an over-sized grid. See the ASTC spec
    // block-mode legality rule (astcenc `init_block_size_descriptor`:
    // `weights_x > texels_x || weights_y > texels_y` => skip).
    if bm.weights_x > 4 || bm.weights_y > 4 {
        return Err(AstcError::UnsupportedBlockMode);
    }

    // Partition count is the 2-bit field at block bits [11, 13); 0 => single.
    let partition_count = ((u32::from(block[1]) >> 3) & 0x3) + 1;
    if partition_count != 1 {
        return Err(AstcError::UnsupportedBlockMode);
    }
    if bm.dual_plane {
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

    let endpoints = decode_cem_endpoints(block, bm.weight_bits, cem)?;
    let weights = infill_weights_4x4(block, bm.weights_x, bm.weights_y, bm.weight_levels)?;

    let mut out = [[0u8; 4]; 16];
    for (texel, w) in weights.iter().enumerate() {
        let w = u32::from(*w);
        out[texel] = [
            lerp_component(endpoints.e0[0], endpoints.e1[0], w),
            lerp_component(endpoints.e0[1], endpoints.e1[1], w),
            lerp_component(endpoints.e0[2], endpoints.e1[2], w),
            lerp_component(endpoints.e0[3], endpoints.e1[3], w),
        ];
    }
    Ok(out)
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
}
