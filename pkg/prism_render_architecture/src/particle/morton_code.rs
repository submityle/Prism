//! `Morton` (`Z-order`) space-filling-curve bit interleaving: the tiny,
//! device-free integer contract that maps 2D/3D grid coordinates to a single
//! sortable key and back (design §12 sort-key quantization, §7
//! `PerNeighborCell`).
//!
//! A `Morton` code interleaves the bits of each coordinate so that points close
//! together in space stay close together along a one-dimensional ordering. That
//! locality is what makes it valuable to the particle engine: sorting particles
//! (or grid cells) by their `Morton` key groups spatial neighbors into
//! contiguous runs, which improves cache behavior on the `CPU` and coalesced
//! memory access on the `GPU`, and lets a bounding box be summarized by the two
//! codes of its corner points.
//!
//! The interleave itself is the classic *magic-bits* trick: instead of shifting
//! one bit at a time, a short cascade of shift-or-and steps spreads every input
//! bit into its interleaved slot in `O(log bits)` operations. Two coordinate
//! layouts are provided:
//!
//! * **2D.** [`part1by1`] spreads a 16-bit value so each bit is separated by one
//!   empty bit (`interval 1`), and [`compact1by1`] gathers it back. Two spread
//!   values, one shifted up by one, pack into a 32-bit key via
//!   [`morton_encode_2d`] / [`morton_decode_2d`].
//! * **3D.** [`part1by2`] spreads a 10-bit value with two empty bits between
//!   each (`interval 2`), and [`compact1by2`] reverses it. Three spread values
//!   pack into the low 30 bits of a 32-bit key via [`morton_encode_3d`] /
//!   [`morton_decode_3d`].
//!
//! [`morton_aabb_range_2d`] encodes the min and max corners of an axis-aligned
//! bounding box (`AABB`). Because the code is nondecreasing in each coordinate
//! independently, the corner codes bound the codes of every point inside the
//! box, so the pair is a valid `[min, max]` key range.
//!
//! Deliberately out of scope, and never imported here:
//! [`super::spatial_hash`] owns the uniform-grid *cell hashing* pipeline
//! (`hash_cell`, per-cell counting, prefix-sum offsets and stable scatter). This
//! module performs none of that — no hash buckets, no prefix sums, no scatter,
//! only the reversible `Morton` bit interleave and the corner range. Everything
//! is pure `u16`/`u32` integer arithmetic with no floating point and no
//! transcendental functions, so the result is deterministic and platform
//! independent.

/// Spreads the 16 low bits of `x` so each occupies an even bit position with a
/// single empty (odd) bit between them (`interval 1`), the 2D `Morton` spread.
///
/// Input `abcd efgh ijkl mnop` becomes `0a0b 0c0d 0e0f 0g0h 0i0j 0k0l 0m0n 0o0p`.
#[must_use]
pub fn part1by1(x: u16) -> u32 {
    let mut v = u32::from(x);
    v = (v | (v << 8)) & 0x00FF_00FF;
    v = (v | (v << 4)) & 0x0F0F_0F0F;
    v = (v | (v << 2)) & 0x3333_3333;
    v = (v | (v << 1)) & 0x5555_5555;
    v
}

/// Inverse of [`part1by1`]: gathers the even bits of `x` back into a dense
/// 16-bit value, discarding the interleaved odd bits.
#[must_use]
pub fn compact1by1(x: u32) -> u16 {
    let mut v = x & 0x5555_5555;
    v = (v | (v >> 1)) & 0x3333_3333;
    v = (v | (v >> 2)) & 0x0F0F_0F0F;
    v = (v | (v >> 4)) & 0x00FF_00FF;
    v = (v | (v >> 8)) & 0x0000_FFFF;
    u16::try_from(v).unwrap_or(0)
}

/// Spreads the 10 low bits of `x` so each occupies every third bit position
/// with two empty bits between them (`interval 2`), the 3D `Morton` spread.
/// Bits above bit 9 of the input are ignored.
#[must_use]
pub fn part1by2(x: u32) -> u32 {
    let mut v = x & 0x0000_03FF;
    v = (v | (v << 16)) & 0xFF00_00FF;
    v = (v | (v << 8)) & 0x0300_F00F;
    v = (v | (v << 4)) & 0x030C_30C3;
    v = (v | (v << 2)) & 0x0924_9249;
    v
}

/// Inverse of [`part1by2`]: gathers every third bit of `x` back into a dense
/// 10-bit value, discarding the two interleaved bits between each.
#[must_use]
pub fn compact1by2(x: u32) -> u32 {
    let mut v = x & 0x0924_9249;
    v = (v | (v >> 2)) & 0x030C_30C3;
    v = (v | (v >> 4)) & 0x0300_F00F;
    v = (v | (v >> 8)) & 0xFF00_00FF;
    v = (v | (v >> 16)) & 0x0000_03FF;
    v
}

/// Interleaves two 16-bit coordinates into a 32-bit 2D `Morton` code, with `x`
/// on the even bits and `y` on the odd bits.
#[must_use]
pub fn morton_encode_2d(x: u16, y: u16) -> u32 {
    part1by1(x) | (part1by1(y) << 1)
}

/// Inverse of [`morton_encode_2d`]: recovers the `(x, y)` coordinate pair from a
/// 2D `Morton` code.
#[must_use]
pub fn morton_decode_2d(code: u32) -> (u16, u16) {
    (compact1by1(code), compact1by1(code >> 1))
}

/// Interleaves three 10-bit coordinates into the low 30 bits of a 32-bit 3D
/// `Morton` code (`x` on bit 0, `y` on bit 1, `z` on bit 2 of each triple).
/// Bits above bit 9 of any input are ignored.
#[must_use]
pub fn morton_encode_3d(x: u32, y: u32, z: u32) -> u32 {
    part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2)
}

/// Inverse of [`morton_encode_3d`]: recovers the `(x, y, z)` coordinate triple
/// (each a 10-bit value) from a 3D `Morton` code.
#[must_use]
pub fn morton_decode_3d(code: u32) -> (u32, u32, u32) {
    (
        compact1by2(code),
        compact1by2(code >> 1),
        compact1by2(code >> 2),
    )
}

/// Encodes the min and max corners of a 2D axis-aligned bounding box (`AABB`)
/// into their `Morton` codes, returned as `(min_code, max_code)`.
///
/// The corners are normalized so the returned `min_code` is never greater than
/// `max_code`. Because a `Morton` code is nondecreasing in each coordinate
/// independently, these two corner codes bound the `Morton` code of every point
/// contained in the box.
#[must_use]
pub fn morton_aabb_range_2d(min: (u16, u16), max: (u16, u16)) -> (u32, u32) {
    let lo_x = if min.0 <= max.0 { min.0 } else { max.0 };
    let lo_y = if min.1 <= max.1 { min.1 } else { max.1 };
    let hi_x = if min.0 >= max.0 { min.0 } else { max.0 };
    let hi_y = if min.1 >= max.1 { min.1 } else { max.1 };
    (morton_encode_2d(lo_x, lo_y), morton_encode_2d(hi_x, hi_y))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_2d_known_small_values_are_exact() {
        assert_eq!(morton_encode_2d(0, 0), 0);
        assert_eq!(morton_encode_2d(1, 0), 1);
        assert_eq!(morton_encode_2d(0, 1), 2);
        assert_eq!(morton_encode_2d(1, 1), 3);
        assert_eq!(morton_encode_2d(2, 0), 4);
        assert_eq!(morton_encode_2d(0, 2), 8);
        assert_eq!(morton_encode_2d(3, 3), 15);
    }

    #[test]
    fn part1by1_known_values() {
        assert_eq!(part1by1(0), 0);
        assert_eq!(part1by1(1), 1);
        assert_eq!(part1by1(2), 4);
        assert_eq!(part1by1(0xFFFF), 0x5555_5555);
    }

    #[test]
    fn compact1by1_known_values() {
        assert_eq!(compact1by1(0), 0);
        assert_eq!(compact1by1(0x5555_5555), 0xFFFF);
        assert_eq!(compact1by1(0xAAAA_AAAA), 0);
    }

    #[test]
    fn part1by1_and_compact1by1_are_inverse() {
        for x in [0u16, 1, 2, 7, 255, 256, 0x0F0F, 0x1234, 0xABCD, 0xFFFF] {
            assert_eq!(compact1by1(part1by1(x)), x);
        }
    }

    #[test]
    fn compact1by1_ignores_odd_bits() {
        // The odd bits carry the `y` channel and must not leak into `x`.
        let x = part1by1(0x1234);
        let contaminated = x | 0xAAAA_AAAA;
        assert_eq!(compact1by1(contaminated), 0x1234);
    }

    #[test]
    fn encode_decode_2d_roundtrips_many_points() {
        let coords = [0u16, 1, 5, 42, 511, 512, 4095, 4096, 0x8000, 0xFFFF];
        for &x in &coords {
            for &y in &coords {
                let code = morton_encode_2d(x, y);
                assert_eq!(morton_decode_2d(code), (x, y));
            }
        }
    }

    #[test]
    fn encode_2d_max_fills_all_bits() {
        assert_eq!(morton_encode_2d(0xFFFF, 0xFFFF), 0xFFFF_FFFF);
    }

    #[test]
    fn encode_2d_monotonic_along_x_axis() {
        let y = 0x1357;
        let mut prev = morton_encode_2d(0, y);
        for x in 1u16..=1024 {
            let cur = morton_encode_2d(x, y);
            assert!(cur > prev);
            prev = cur;
        }
    }

    #[test]
    fn encode_2d_monotonic_along_y_axis() {
        let x = 0x2468;
        let mut prev = morton_encode_2d(x, 0);
        for y in 1u16..=1024 {
            let cur = morton_encode_2d(x, y);
            assert!(cur > prev);
            prev = cur;
        }
    }

    #[test]
    fn encode_2d_axes_occupy_disjoint_bits() {
        // `x` lands on even bits, `y` on odd bits: OR-ing them equals the code.
        let x = 0x0F31;
        let y = 0x7A2C;
        assert_eq!(morton_encode_2d(x, y), part1by1(x) | (part1by1(y) << 1));
        assert_eq!(part1by1(x) & (part1by1(y) << 1), 0);
    }

    #[test]
    fn encode_3d_known_small_values_are_exact() {
        assert_eq!(morton_encode_3d(0, 0, 0), 0);
        assert_eq!(morton_encode_3d(1, 0, 0), 1);
        assert_eq!(morton_encode_3d(0, 1, 0), 2);
        assert_eq!(morton_encode_3d(0, 0, 1), 4);
        assert_eq!(morton_encode_3d(1, 1, 1), 7);
        assert_eq!(morton_encode_3d(2, 0, 0), 8);
    }

    #[test]
    fn part1by2_known_values() {
        assert_eq!(part1by2(0), 0);
        assert_eq!(part1by2(1), 1);
        assert_eq!(part1by2(2), 8);
        assert_eq!(part1by2(0x3FF), 0x0924_9249);
    }

    #[test]
    fn part1by2_ignores_input_above_ten_bits() {
        assert_eq!(part1by2(0xFFFF_FC00), 0);
        assert_eq!(part1by2(0x0000_0400 | 1), 1);
    }

    #[test]
    fn compact1by2_known_values() {
        assert_eq!(compact1by2(0), 0);
        assert_eq!(compact1by2(0x0924_9249), 0x3FF);
    }

    #[test]
    fn part1by2_and_compact1by2_are_inverse() {
        for x in [0u32, 1, 2, 7, 63, 256, 511, 512, 0x155, 0x2AA, 0x3FF] {
            assert_eq!(compact1by2(part1by2(x)), x);
        }
    }

    #[test]
    fn encode_decode_3d_roundtrips_many_points() {
        let coords = [0u32, 1, 3, 17, 100, 255, 256, 511, 512, 1023];
        for &x in &coords {
            for &y in &coords {
                for &z in &coords {
                    let code = morton_encode_3d(x, y, z);
                    assert_eq!(morton_decode_3d(code), (x, y, z));
                }
            }
        }
    }

    #[test]
    fn encode_3d_axes_are_independent() {
        // Each axis decodes independently of the others.
        assert_eq!(
            morton_decode_3d(morton_encode_3d(0x3FF, 0, 0)),
            (0x3FF, 0, 0)
        );
        assert_eq!(
            morton_decode_3d(morton_encode_3d(0, 0x3FF, 0)),
            (0, 0x3FF, 0)
        );
        assert_eq!(
            morton_decode_3d(morton_encode_3d(0, 0, 0x3FF)),
            (0, 0, 0x3FF)
        );
    }

    #[test]
    fn encode_3d_max_fills_low_thirty_bits() {
        assert_eq!(morton_encode_3d(0x3FF, 0x3FF, 0x3FF), 0x3FFF_FFFF);
    }

    #[test]
    fn encode_3d_monotonic_along_z_axis() {
        let (x, y) = (0x123, 0x2AB);
        let mut prev = morton_encode_3d(x, y, 0);
        for z in 1u32..=512 {
            let cur = morton_encode_3d(x, y, z);
            assert!(cur > prev);
            prev = cur;
        }
    }

    #[test]
    fn aabb_range_orders_min_and_max() {
        let (lo, hi) = morton_aabb_range_2d((10, 3), (2, 40));
        assert!(lo <= hi);
        // Corners are normalized: min = (2, 3), max = (10, 40).
        assert_eq!(lo, morton_encode_2d(2, 3));
        assert_eq!(hi, morton_encode_2d(10, 40));
    }

    #[test]
    fn aabb_range_already_ordered_corners() {
        let (lo, hi) = morton_aabb_range_2d((5, 6), (9, 20));
        assert_eq!(lo, morton_encode_2d(5, 6));
        assert_eq!(hi, morton_encode_2d(9, 20));
        assert!(lo <= hi);
    }

    #[test]
    fn aabb_range_bounds_interior_points() {
        let min = (4u16, 7u16);
        let max = (12u16, 19u16);
        let (lo, hi) = morton_aabb_range_2d(min, max);
        for x in min.0..=max.0 {
            for y in min.1..=max.1 {
                let code = morton_encode_2d(x, y);
                assert!(code >= lo);
                assert!(code <= hi);
            }
        }
    }

    #[test]
    fn aabb_range_degenerate_point_box() {
        let (lo, hi) = morton_aabb_range_2d((77, 88), (77, 88));
        assert_eq!(lo, hi);
        assert_eq!(lo, morton_encode_2d(77, 88));
    }
}
