//! Spatial locality encodings: Morton (Z-order) and Hilbert curves.
//!
//! These map an integer lattice point to a single sort key whose linear
//! ordering preserves spatial locality. They are the standard primitives for
//! building `BVH`/octree radix-sort keys, spatial-hash slots, large-world cell
//! linearization, and streaming priority order.
//!
//! Everything here is pure integer bit manipulation: no floating point and no
//! allocation, so the results are bit-exact and identical across every
//! platform, which makes them safe to use on the deterministic simulation
//! path.
//!
//! ## Morton (Z-order)
//! [`morton_encode2`] / [`morton_encode3`] interleave the bits of each
//! coordinate so that nearby points usually (but not always, at the "Z" jumps)
//! land near each other in key space. Encoding is a handful of shift-and-mask
//! operations, which is why Morton keys are the go-to choice for `GPU` radix
//! sorts.
//!
//! ## Hilbert
//! [`hilbert_encode3`] trades a little more arithmetic for a stronger locality
//! guarantee: consecutive indices are always lattice neighbours (they differ by
//! one unit along exactly one axis), with none of the long jumps that Morton
//! curves make. That makes Hilbert order preferable for cache-coherent
//! traversal and tiled streaming.

/// Number of bits per axis consumed by the 3D encoders. Three axes of 21 bits
/// pack into the 63 usable bits of a [`u64`] key.
pub const BITS_3D: u32 = 21;

/// Inclusive maximum coordinate accepted (per axis) by the 3D encoders.
pub const MAX_COORD_3D: u32 = (1 << BITS_3D) - 1;

/// Number of bits per axis consumed by the 2D encoders. Two axes of 32 bits
/// pack into the full 64 bits of a [`u64`] key.
pub const BITS_2D: u32 = 32;

// ---------------------------------------------------------------------------
// Morton (Z-order)
// ---------------------------------------------------------------------------

/// Spread the low 21 bits of `value` so each occupies every third bit slot.
#[inline]
const fn part1by2(value: u32) -> u64 {
    let mut x = (value as u64) & 0x001f_ffff;
    x = (x | (x << 32)) & 0x001f_0000_0000_ffff;
    x = (x | (x << 16)) & 0x001f_0000_ff00_00ff;
    x = (x | (x << 8)) & 0x100f_00f0_0f00_f00f;
    x = (x | (x << 4)) & 0x10c3_0c30_c30c_30c3;
    x = (x | (x << 2)) & 0x1249_2492_4924_9249;
    x
}

/// Inverse of [`part1by2`]: gather every third bit back into the low 21 bits.
#[inline]
const fn compact1by2(value: u64) -> u32 {
    let mut x = value & 0x1249_2492_4924_9249;
    x = (x ^ (x >> 2)) & 0x10c3_0c30_c30c_30c3;
    x = (x ^ (x >> 4)) & 0x100f_00f0_0f00_f00f;
    x = (x ^ (x >> 8)) & 0x001f_0000_ff00_00ff;
    x = (x ^ (x >> 16)) & 0x001f_0000_0000_ffff;
    x = (x ^ (x >> 32)) & 0x001f_ffff;
    x as u32
}

/// Spread the low 32 bits of `value` so each occupies every second bit slot.
#[inline]
const fn part1by1(value: u32) -> u64 {
    let mut x = value as u64;
    x = (x | (x << 16)) & 0x0000_ffff_0000_ffff;
    x = (x | (x << 8)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555;
    x
}

/// Inverse of [`part1by1`]: gather every second bit back into the low 32 bits.
#[inline]
const fn compact1by1(value: u64) -> u32 {
    let mut x = value & 0x5555_5555_5555_5555;
    x = (x ^ (x >> 1)) & 0x3333_3333_3333_3333;
    x = (x ^ (x >> 2)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x ^ (x >> 4)) & 0x00ff_00ff_00ff_00ff;
    x = (x ^ (x >> 8)) & 0x0000_ffff_0000_ffff;
    x = (x ^ (x >> 16)) & 0x0000_0000_ffff_ffff;
    x as u32
}

/// Interleave the low 32 bits of `x` and `y` into a 64-bit Morton (Z-order)
/// key. `x` contributes the even bit positions and `y` the odd ones.
#[inline]
#[must_use]
pub const fn morton_encode2(x: u32, y: u32) -> u64 {
    part1by1(x) | (part1by1(y) << 1)
}

/// Recover the `(x, y)` coordinates from a 2D Morton key produced by
/// [`morton_encode2`].
#[inline]
#[must_use]
pub const fn morton_decode2(code: u64) -> (u32, u32) {
    (compact1by1(code), compact1by1(code >> 1))
}

/// Interleave the low 21 bits of `x`, `y`, and `z` into a 63-bit Morton
/// (Z-order) key. Coordinates wider than [`MAX_COORD_3D`] are masked to their
/// low 21 bits.
#[inline]
#[must_use]
pub const fn morton_encode3(x: u32, y: u32, z: u32) -> u64 {
    part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2)
}

/// Recover the `(x, y, z)` coordinates from a 3D Morton key produced by
/// [`morton_encode3`].
#[inline]
#[must_use]
pub const fn morton_decode3(code: u64) -> (u32, u32, u32) {
    (
        compact1by2(code),
        compact1by2(code >> 1),
        compact1by2(code >> 2),
    )
}

// ---------------------------------------------------------------------------
// Hilbert (3D, Skilling's transpose algorithm)
// ---------------------------------------------------------------------------

/// Convert axis coordinates to the Hilbert "transpose" representation in place
/// (Skilling, 2004). Each `axes[i]` holds the bits of the Hilbert index that
/// come from dimension `i`.
#[inline]
fn axes_to_transpose(axes: &mut [u32; 3]) {
    let top = 1_u32 << (BITS_3D - 1);

    // Inverse undo of the Gray-code excess work, from the high bit downwards.
    let mut q = top;
    while q > 1 {
        let p = q - 1;
        let mut i = 0;
        while i < 3 {
            if axes[i] & q != 0 {
                axes[0] ^= p;
            } else {
                let t = (axes[0] ^ axes[i]) & p;
                axes[0] ^= t;
                axes[i] ^= t;
            }
            i += 1;
        }
        q >>= 1;
    }

    // Gray encode across dimensions.
    axes[1] ^= axes[0];
    axes[2] ^= axes[1];

    // Fix up the sign bits that the Gray step propagated.
    let mut t = 0_u32;
    let mut q = top;
    while q > 1 {
        if axes[2] & q != 0 {
            t ^= q - 1;
        }
        q >>= 1;
    }
    axes[0] ^= t;
    axes[1] ^= t;
    axes[2] ^= t;
}

/// Inverse of [`axes_to_transpose`]: turn the transpose representation back
/// into axis coordinates in place.
#[inline]
fn transpose_to_axes(axes: &mut [u32; 3]) {
    // Gray decode.
    let t = axes[2] >> 1;
    axes[2] ^= axes[1];
    axes[1] ^= axes[0];
    axes[0] ^= t;

    // Undo excess work, from the low bit upwards.
    let mut q = 2_u32;
    while q != (1 << BITS_3D) {
        let p = q - 1;
        let mut i = 3;
        while i > 0 {
            i -= 1;
            if axes[i] & q != 0 {
                axes[0] ^= p;
            } else {
                let tt = (axes[0] ^ axes[i]) & p;
                axes[0] ^= tt;
                axes[i] ^= tt;
            }
        }
        q <<= 1;
    }
}

/// Pack a transpose representation into a single Hilbert index, most
/// significant bit first across dimensions.
#[inline]
fn transpose_to_index(axes: &[u32; 3]) -> u64 {
    let mut h = 0_u64;
    let mut b = BITS_3D;
    while b > 0 {
        b -= 1;
        let mut i = 0;
        while i < 3 {
            h = (h << 1) | u64::from((axes[i] >> b) & 1);
            i += 1;
        }
    }
    h
}

/// Unpack a Hilbert index into the transpose representation (inverse of
/// [`transpose_to_index`]).
#[inline]
fn index_to_transpose(index: u64) -> [u32; 3] {
    let mut axes = [0_u32; 3];
    let total = BITS_3D * 3;
    let mut bit = total;
    while bit > 0 {
        bit -= 1;
        let value = ((index >> bit) & 1) as u32;
        // Bits were written groups-of-three, high slot first. Reconstruct the
        // reverse order so dimension `i`'s bit `b` is restored.
        let pos = total - 1 - bit;
        let b = BITS_3D - 1 - (pos / 3);
        let i = (pos % 3) as usize;
        axes[i] |= value << b;
    }
    axes
}

/// Map a 3D lattice point to its position along the 21-bit-per-axis Hilbert
/// curve. Coordinates wider than [`MAX_COORD_3D`] are masked to their low 21
/// bits. Consecutive indices always map to lattice neighbours.
#[inline]
#[must_use]
pub fn hilbert_encode3(x: u32, y: u32, z: u32) -> u64 {
    let mut axes = [
        x & MAX_COORD_3D,
        y & MAX_COORD_3D,
        z & MAX_COORD_3D,
    ];
    axes_to_transpose(&mut axes);
    transpose_to_index(&axes)
}

/// Recover the `(x, y, z)` coordinates from a Hilbert index produced by
/// [`hilbert_encode3`].
#[inline]
#[must_use]
pub fn hilbert_decode3(index: u64) -> (u32, u32, u32) {
    let mut axes = index_to_transpose(index);
    transpose_to_axes(&mut axes);
    (axes[0], axes[1], axes[2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{Rng, SplitMix64};

    #[test]
    fn morton2_round_trip_known() {
        assert_eq!(morton_encode2(0, 0), 0);
        assert_eq!(morton_encode2(1, 0), 0b01);
        assert_eq!(morton_encode2(0, 1), 0b10);
        assert_eq!(morton_encode2(1, 1), 0b11);
        assert_eq!(morton_encode2(3, 0), 0b0101);
        let (x, y) = morton_decode2(morton_encode2(0xdead_beef, 0x1234_5678));
        assert_eq!((x, y), (0xdead_beef, 0x1234_5678));
    }

    #[test]
    fn morton3_round_trip_known() {
        assert_eq!(morton_encode3(0, 0, 0), 0);
        assert_eq!(morton_encode3(1, 0, 0), 0b001);
        assert_eq!(morton_encode3(0, 1, 0), 0b010);
        assert_eq!(morton_encode3(0, 0, 1), 0b100);
        assert_eq!(morton_encode3(1, 1, 1), 0b111);
        let (x, y, z) = morton_decode3(morton_encode3(0x1f_ffff, 0, 0x1234));
        assert_eq!((x, y, z), (0x1f_ffff, 0, 0x1234));
    }

    #[test]
    fn morton3_random_round_trip() {
        let mut rng = SplitMix64::new(0x5eed_1234);
        for _ in 0..10_000 {
            let x = (rng.next_u32()) & MAX_COORD_3D;
            let y = (rng.next_u32()) & MAX_COORD_3D;
            let z = (rng.next_u32()) & MAX_COORD_3D;
            let code = morton_encode3(x, y, z);
            assert!(code < (1 << (BITS_3D * 3)));
            assert_eq!(morton_decode3(code), (x, y, z));
        }
    }

    #[test]
    fn morton2_random_round_trip() {
        let mut rng = SplitMix64::new(0xabcd_ef01);
        for _ in 0..10_000 {
            let x = rng.next_u32();
            let y = rng.next_u32();
            assert_eq!(morton_decode2(morton_encode2(x, y)), (x, y));
        }
    }

    #[test]
    fn hilbert3_random_round_trip() {
        let mut rng = SplitMix64::new(0x1357_9bdf);
        for _ in 0..10_000 {
            let x = rng.next_u32() & MAX_COORD_3D;
            let y = rng.next_u32() & MAX_COORD_3D;
            let z = rng.next_u32() & MAX_COORD_3D;
            let index = hilbert_encode3(x, y, z);
            assert_eq!(hilbert_decode3(index), (x, y, z));
        }
    }

    #[test]
    fn hilbert3_is_a_bijection_on_a_small_cube() {
        // Over a 4x4x4 cube the 64 Hilbert indices must be a permutation of
        // 0..64, confirming the curve visits every cell exactly once.
        let side = 4_u32;
        let n = (side * side * side) as usize;
        let mut seen = alloc::vec![false; n];
        for x in 0..side {
            for y in 0..side {
                for z in 0..side {
                    let h = hilbert_encode3(x, y, z) as usize;
                    assert!(h < n, "index {h} out of range");
                    assert!(!seen[h], "index {h} visited twice");
                    seen[h] = true;
                }
            }
        }
        assert!(seen.into_iter().all(|v| v), "curve skipped a cell");
    }

    #[test]
    fn hilbert3_consecutive_indices_are_neighbours() {
        // The defining Hilbert property: stepping the index by one moves to an
        // adjacent lattice cell (Manhattan distance exactly 1).
        let side = 4_u32;
        let count = side * side * side;
        for h in 0..count - 1 {
            let (x0, y0, z0) = hilbert_decode3(u64::from(h));
            let (x1, y1, z1) = hilbert_decode3(u64::from(h + 1));
            let dist = x0.abs_diff(x1) + y0.abs_diff(y1) + z0.abs_diff(z1);
            assert_eq!(dist, 1, "indices {h} and {} are not neighbours", h + 1);
        }
    }

    #[test]
    fn encoders_are_deterministic() {
        assert_eq!(morton_encode3(5, 9, 17), morton_encode3(5, 9, 17));
        assert_eq!(hilbert_encode3(5, 9, 17), hilbert_encode3(5, 9, 17));
    }
}
