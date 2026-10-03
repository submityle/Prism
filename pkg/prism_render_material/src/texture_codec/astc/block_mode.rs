//! ASTC 2D block-mode decode.
//!
//! The first eleven bits of every non-void-extent ASTC block encode the
//! *block mode*: the weight-grid dimensions, the weight quantisation range,
//! and whether the block is dual-plane. The packing is a dense bit-field with
//! several sub-encodings (Khronos Data Format Specification 1.3, "Block Mode"),
//! transcribed here from the ARM `astcenc` reference decoder
//! (`decode_block_mode_2d`, Apache-2.0) which every conformant unit implements.
//!
//! Decode is pure integer arithmetic -- no AI/ML path.

use super::bise::{ise_sequence_bits, IseRange};

/// A decoded 2D block mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BlockMode2d {
    /// Weight-grid width (number of weights along X), always `2..=12`.
    pub(super) weights_x: u32,
    /// Weight-grid height (number of weights along Y), always `2..=12`.
    pub(super) weights_y: u32,
    /// Number of distinct levels in the weight quantisation range (the BISE
    /// level count, e.g. `6` for a QUANT_6 trit range).
    pub(super) weight_levels: u32,
    /// Whether the block carries two weight planes.
    pub(super) dual_plane: bool,
    /// Number of bits the packed weight integer sequence occupies.
    pub(super) weight_bits: u32,
}

/// Map an `astcenc` weight `quant_mode` index (0 == QUANT_2 .. 11 == QUANT_32)
/// to its BISE level count. Weight ranges never exceed QUANT_32, so indices
/// above 11 are invalid.
#[must_use]
fn weight_quant_levels(quant_mode: u32) -> Option<u32> {
    // astcenc quant_method enum values QUANT_2..QUANT_32.
    const LEVELS: [u32; 12] = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32];
    LEVELS.get(quant_mode as usize).copied()
}

/// Decode an 11-bit `block_mode` field into its weight-grid geometry and
/// quantisation.
///
/// Returns `None` for void-extent, reserved, or otherwise undecodable modes
/// (including ones whose weight count or bit budget falls outside the ASTC
/// limits), mirroring the reference decoder's validity test.
#[must_use]
pub(super) fn decode_block_mode_2d(block_mode: u16) -> Option<BlockMode2d> {
    let block_mode = u32::from(block_mode);

    let mut base_quant_mode = (block_mode >> 4) & 1;
    let mut h = (block_mode >> 9) & 1;
    let mut d = (block_mode >> 10) & 1;
    let a = (block_mode >> 5) & 0x3;

    let weights_x;
    let weights_y;

    if (block_mode & 3) != 0 {
        base_quant_mode |= (block_mode & 3) << 1;
        let b = (block_mode >> 7) & 3;
        match (block_mode >> 2) & 3 {
            0 => {
                weights_x = b + 4;
                weights_y = a + 2;
            }
            1 => {
                weights_x = b + 8;
                weights_y = a + 2;
            }
            2 => {
                weights_x = a + 2;
                weights_y = b + 8;
            }
            _ => {
                let b = b & 1;
                if block_mode & 0x100 != 0 {
                    weights_x = b + 2;
                    weights_y = a + 2;
                } else {
                    weights_x = a + 2;
                    weights_y = b + 6;
                }
            }
        }
    } else {
        base_quant_mode |= ((block_mode >> 2) & 3) << 1;
        if ((block_mode >> 2) & 3) == 0 {
            return None;
        }
        let b = (block_mode >> 9) & 3;
        match (block_mode >> 7) & 3 {
            0 => {
                weights_x = 12;
                weights_y = a + 2;
            }
            1 => {
                weights_x = a + 2;
                weights_y = 12;
            }
            2 => {
                weights_x = a + 6;
                weights_y = b + 6;
                d = 0;
                h = 0;
            }
            _ => match (block_mode >> 5) & 3 {
                0 => {
                    weights_x = 6;
                    weights_y = 10;
                }
                1 => {
                    weights_x = 10;
                    weights_y = 6;
                }
                _ => return None,
            },
        }
    }

    let weight_count = weights_x * weights_y * (d + 1);
    // quant_mode = (base_quant_mode - 2) + 6 * H; base_quant_mode < 2 would
    // underflow, which the reference avoids because the low mode bits guarantee
    // base_quant_mode >= 2 for every reachable path.
    let quant_mode = base_quant_mode.checked_sub(2)? + 6 * h;
    let dual_plane = d != 0;

    let weight_levels = weight_quant_levels(quant_mode)?;
    let range = IseRange::from_num_levels(weight_levels)?;
    let weight_bits = ise_sequence_bits(weight_count, range);

    // ASTC validity limits (Khronos DFS 1.3): a block carries at most 64
    // weights and the weight bitstream must occupy 24..=96 bits.
    const BLOCK_MAX_WEIGHTS: u32 = 64;
    const BLOCK_MIN_WEIGHT_BITS: u32 = 24;
    const BLOCK_MAX_WEIGHT_BITS: u32 = 96;
    if weight_count > BLOCK_MAX_WEIGHTS
        || weight_bits < BLOCK_MIN_WEIGHT_BITS
        || weight_bits > BLOCK_MAX_WEIGHT_BITS
    {
        return None;
    }

    Some(BlockMode2d {
        weights_x,
        weights_y,
        weight_levels,
        dual_plane,
        weight_bits,
    })
}

#[cfg(test)]
mod tests {
    use super::decode_block_mode_2d;

    #[test]
    fn mode_67_is_4x4_quant6_trit() {
        // Block mode 67: 4x4 weight grid, QUANT_6 (trit, b1), single plane,
        // 42 weight bits. Cross-checked against the reference decoder.
        let bm = decode_block_mode_2d(67).expect("mode 67 is valid");
        assert_eq!(bm.weights_x, 4);
        assert_eq!(bm.weights_y, 4);
        assert_eq!(bm.weight_levels, 6);
        assert!(!bm.dual_plane);
        assert_eq!(bm.weight_bits, 42);
    }

    #[test]
    fn mode_577_is_4x4_quant10_quint() {
        // Block mode 577: 4x4 grid, QUANT_10 (quint, b1), 54 weight bits.
        let bm = decode_block_mode_2d(577).expect("mode 577 is valid");
        assert_eq!((bm.weights_x, bm.weights_y), (4, 4));
        assert_eq!(bm.weight_levels, 10);
        assert!(!bm.dual_plane);
        assert_eq!(bm.weight_bits, 54);
    }

    #[test]
    fn mode_578_is_4x4_quant16_bits() {
        // Block mode 578: 4x4 grid, QUANT_16 (4 raw bits), 64 weight bits.
        let bm = decode_block_mode_2d(578).expect("mode 578 is valid");
        assert_eq!((bm.weights_x, bm.weights_y), (4, 4));
        assert_eq!(bm.weight_levels, 16);
        assert!(!bm.dual_plane);
        assert_eq!(bm.weight_bits, 64);
    }

    #[test]
    fn low_two_bits_zero_and_sub_field_zero_is_rejected() {
        // The (block_mode & 3) == 0 path with ((block_mode >> 2) & 3) == 0 is a
        // reserved encoding.
        assert!(decode_block_mode_2d(0).is_none());
    }

    #[test]
    fn weight_levels_cover_the_valid_quant_range() {
        // Exercise the quant-mode -> level mapping indirectly: every decodable
        // mode must report a BISE-valid level count (2..=32).
        for mode in 0u16..0x800 {
            if let Some(bm) = decode_block_mode_2d(mode) {
                assert!(
                    matches!(
                        bm.weight_levels,
                        2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 32
                    ),
                    "mode {mode} produced non-weight level count {}",
                    bm.weight_levels
                );
                assert!((2..=12).contains(&bm.weights_x));
                assert!((2..=12).contains(&bm.weights_y));
            }
        }
    }
}
