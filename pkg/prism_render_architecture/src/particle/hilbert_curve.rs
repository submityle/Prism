//! `Hilbert` space-filling-curve encode/decode: the tiny, device-free integer
//! contract that maps 2D (and 3D) grid coordinates to a single locality-
//! preserving key and back (design §12 sort-key quantization, §7
//! `PerNeighborCell`).
//!
//! Like a `Morton` (`Z-order`) code, a `Hilbert` index linearizes a
//! multidimensional grid so that spatially close cells land close together
//! along a one-dimensional ordering, which improves `CPU` cache behavior and
//! coalesced `GPU` memory access when particles or `tile` cells are sorted by
//! the key. Unlike `Z-order`, the `Hilbert` curve never makes a long diagonal
//! jump between successive indices: two consecutive indices are always axis
//! neighbors exactly one cell apart. That stronger locality is why production
//! `GPU` schedulers prefer it for `tile` traversal and neighbor gathering.
//!
//! The 2D path uses the classic rotate/reflect iteration (the Wikipedia `rot`
//! algorithm): walk the quadrant bits from most to least significant, and at
//! each level reflect and swap the coordinate frame so the four sub-quadrants
//! are visited in the U-shaped `Hilbert` order. [`xy_to_hilbert`] performs the
//! forward map and [`hilbert_to_xy`] its exact inverse; [`hilbert_key`] is the
//! forward map under the name used by the sort pipeline, and [`tile_order`]
//! materializes the full traversal sequence for a `2^order × 2^order` grid.
//!
//! The optional 3D path ([`hilbert_distance_3d`] / [`hilbert_to_xyz_3d`]) uses
//! the integer `Gray-code` transpose construction (`Skilling`'s algorithm):
//! encode each axis into a `Gray-code`, undo the per-level rotations, then
//! interleave the transposed bits into the linear index. It is a genuine 3D
//! `Hilbert` curve, not three independent 2D curves.
//!
//! Deliberately out of scope, and never imported here: [`super::morton_code`]
//! owns the `Z-order` bit-interleave contract, and [`super::spatial_hash`] owns
//! uniform-grid cell hashing. This module performs neither; it is pure integer
//! bit arithmetic with no floating point and no transcendental functions, so
//! the result is deterministic and platform independent.

use alloc::vec::Vec;
use core::mem::swap;

/// Largest curve order this contract accepts. At `order = 16` a 2D index spans
/// the full `2^32` range of a [`u32`]; higher orders would overflow, so the
/// public entry points clamp their `order` argument to this bound.
pub const MAX_ORDER: u32 = 16;

/// Number of axes handled by the 3D path.
const DIMS_3D: usize = 3;

/// Number of axes handled by the 3D path, as a [`u32`] for bit arithmetic.
const DIMS_3D_U32: u32 = 3;

/// Returns the low-bit mask `2^order - 1` used to fold out-of-range coordinates
/// back into the valid `[0, 2^order)` window for a clamped `order`.
fn coord_mask(order: u32) -> u32 {
    (1u32 << order) - 1
}

/// Rotates and reflects a quadrant frame in place, the shared inner step of the
/// 2D `Hilbert` map. `n` is the side length of the frame being reflected; the
/// forward map passes the full grid side while the inverse passes the current
/// sub-square side, which keeps `n - 1 - *x` non-negative in both directions.
fn rotate_quadrant(n: u32, x: &mut u32, y: &mut u32, rx: u32, ry: u32) {
    if ry == 0 {
        if rx == 1 {
            *x = n - 1 - *x;
            *y = n - 1 - *y;
        }
        swap(x, y);
    }
}

/// Maps 2D grid coordinates to their `Hilbert` index for a curve of the given
/// `order` (grid side `2^order`).
///
/// `order` is clamped to [`MAX_ORDER`], and `x`/`y` are folded into the valid
/// `[0, 2^order)` window, so out-of-range inputs wrap rather than panic. The
/// result is in `[0, 4^order)`.
#[must_use]
pub fn xy_to_hilbert(order: u32, x: u32, y: u32) -> u32 {
    let order = order.min(MAX_ORDER);
    let n = 1u32 << order;
    let mask = n - 1;
    let mut x = x & mask;
    let mut y = y & mask;
    let mut d = 0u32;
    let mut s = n >> 1;
    while s > 0 {
        let rx = u32::from((x & s) > 0);
        let ry = u32::from((y & s) > 0);
        d += s * s * ((3 * rx) ^ ry);
        rotate_quadrant(n, &mut x, &mut y, rx, ry);
        s >>= 1;
    }
    d
}

/// Maps a `Hilbert` index back to its 2D grid coordinates, the exact inverse of
/// [`xy_to_hilbert`] for the same `order`.
///
/// `order` is clamped to [`MAX_ORDER`] and `d` is folded into `[0, 4^order)`.
#[must_use]
pub fn hilbert_to_xy(order: u32, d: u32) -> (u32, u32) {
    let order = order.min(MAX_ORDER);
    let n = 1u32 << order;
    let total_bits = order * 2;
    let mut t = if total_bits >= 32 {
        d
    } else {
        d & ((1u32 << total_bits) - 1)
    };
    let mut x = 0u32;
    let mut y = 0u32;
    let mut s = 1u32;
    while s < n {
        let rx = 1 & (t >> 1);
        let ry = 1 & (t ^ rx);
        rotate_quadrant(s, &mut x, &mut y, rx, ry);
        x += s * rx;
        y += s * ry;
        t >>= 2;
        s <<= 1;
    }
    (x, y)
}

/// Returns the `Hilbert` sort key for a 2D cell, an alias of [`xy_to_hilbert`]
/// under the name the sort pipeline uses.
#[must_use]
pub fn hilbert_key(order: u32, x: u32, y: u32) -> u32 {
    xy_to_hilbert(order, x, y)
}

/// Materializes the full `Hilbert` traversal of a `2^order × 2^order` grid as a
/// sequence of `(x, y)` cells in index order.
///
/// The returned vector has exactly `4^order` entries, each grid cell appearing
/// once, with successive entries being axis neighbors one cell apart. `order`
/// is clamped to [`MAX_ORDER`].
#[must_use]
pub fn tile_order(order: u32) -> Vec<(u32, u32)> {
    let order = order.min(MAX_ORDER);
    let n = 1u32 << order;
    let count = u64::from(n) * u64::from(n);
    let mut out = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
    let mut d = 0u64;
    while d < count {
        let idx = u32::try_from(d).unwrap_or(0);
        out.push(hilbert_to_xy(order, idx));
        d += 1;
    }
    out
}

/// Encodes 3D axes into the `Gray-code` transpose used by `Skilling`'s
/// algorithm, in place. `order` is assumed already clamped and non-zero.
fn axes_to_transpose_3d(coords: &mut [u32; DIMS_3D], order: u32) {
    let top = 1u32 << (order - 1);
    let mut q = top;
    while q > 1 {
        let p = q - 1;
        let mut i = 0;
        while i < DIMS_3D {
            if coords[i] & q != 0 {
                coords[0] ^= p;
            } else {
                let t = (coords[0] ^ coords[i]) & p;
                coords[0] ^= t;
                coords[i] ^= t;
            }
            i += 1;
        }
        q >>= 1;
    }
    let mut i = 1;
    while i < DIMS_3D {
        coords[i] ^= coords[i - 1];
        i += 1;
    }
    let mut acc = 0u32;
    let mut q = top;
    while q > 1 {
        if coords[DIMS_3D - 1] & q != 0 {
            acc ^= q - 1;
        }
        q >>= 1;
    }
    let mut i = 0;
    while i < DIMS_3D {
        coords[i] ^= acc;
        i += 1;
    }
}

/// Inverts [`axes_to_transpose_3d`] in place. `order` is assumed already clamped
/// and non-zero.
fn transpose_to_axes_3d(coords: &mut [u32; DIMS_3D], order: u32) {
    let n = 1u32 << order;
    let lead = coords[DIMS_3D - 1] >> 1;
    let mut i = DIMS_3D - 1;
    while i > 0 {
        coords[i] ^= coords[i - 1];
        i -= 1;
    }
    coords[0] ^= lead;
    let mut q = 2u32;
    while q != n {
        let p = q - 1;
        let mut i = DIMS_3D;
        while i > 0 {
            i -= 1;
            if coords[i] & q != 0 {
                coords[0] ^= p;
            } else {
                let t = (coords[0] ^ coords[i]) & p;
                coords[0] ^= t;
                coords[i] ^= t;
            }
        }
        q <<= 1;
    }
}

/// Interleaves the transposed 3D axis bits into the linear `Hilbert` index,
/// most significant bit level first.
fn interleave_3d(coords: &[u32; DIMS_3D], order: u32) -> u64 {
    let mut h = 0u64;
    let mut bit = order;
    while bit > 0 {
        bit -= 1;
        let mut i = 0;
        while i < DIMS_3D {
            let b = (coords[i] >> bit) & 1;
            h = (h << 1) | u64::from(b);
            i += 1;
        }
    }
    h
}

/// Splits a linear 3D `Hilbert` index back into transposed axis bits, the
/// inverse of [`interleave_3d`].
fn deinterleave_3d(h: u64, order: u32) -> [u32; DIMS_3D] {
    let mut coords = [0u32; DIMS_3D];
    let total = order * DIMS_3D_U32;
    let mut pos = 0u32;
    while pos < total {
        let shift = total - 1 - pos;
        let bitval = u32::try_from((h >> shift) & 1).unwrap_or(0);
        let bit = order - 1 - (pos / DIMS_3D_U32);
        let i = usize::try_from(pos % DIMS_3D_U32).unwrap_or(0);
        coords[i] |= bitval << bit;
        pos += 1;
    }
    coords
}

/// Maps 3D grid coordinates to their `Hilbert` index for a curve of the given
/// `order` (grid side `2^order` per axis), using `Skilling`'s integer
/// `Gray-code` construction.
///
/// `order` is clamped to [`MAX_ORDER`] and each axis is folded into
/// `[0, 2^order)`. The result is in `[0, 8^order)` and is returned as a [`u64`]
/// because a 3D index needs up to `3 * order` bits.
#[must_use]
pub fn hilbert_distance_3d(order: u32, x: u32, y: u32, z: u32) -> u64 {
    let order = order.min(MAX_ORDER);
    if order == 0 {
        return 0;
    }
    let mask = coord_mask(order);
    let mut coords = [x & mask, y & mask, z & mask];
    axes_to_transpose_3d(&mut coords, order);
    interleave_3d(&coords, order)
}

/// Maps a 3D `Hilbert` index back to its grid coordinates, the exact inverse of
/// [`hilbert_distance_3d`] for the same `order`.
///
/// `order` is clamped to [`MAX_ORDER`] and `d` is folded into `[0, 8^order)`.
#[must_use]
pub fn hilbert_to_xyz_3d(order: u32, d: u64) -> (u32, u32, u32) {
    let order = order.min(MAX_ORDER);
    if order == 0 {
        return (0, 0, 0);
    }
    let total = order * DIMS_3D_U32;
    let d = if total >= 64 {
        d
    } else {
        d & ((1u64 << total) - 1)
    };
    let mut coords = deinterleave_3d(d, order);
    transpose_to_axes_3d(&mut coords, order);
    (coords[0], coords[1], coords[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Local `Z-order` (`Morton`) decode used only to contrast the orderings;
    /// this module never imports the `Morton` contract.
    fn morton_xy(d: u32, order: u32) -> (u32, u32) {
        let mut x = 0u32;
        let mut y = 0u32;
        let mut i = 0u32;
        while i < order {
            x |= ((d >> (2 * i)) & 1) << i;
            y |= ((d >> (2 * i + 1)) & 1) << i;
            i += 1;
        }
        (x, y)
    }

    fn manhattan(a: (u32, u32), b: (u32, u32)) -> u32 {
        a.0.abs_diff(b.0) + a.1.abs_diff(b.1)
    }

    fn manhattan_3d(a: (u32, u32, u32), b: (u32, u32, u32)) -> u32 {
        a.0.abs_diff(b.0) + a.1.abs_diff(b.1) + a.2.abs_diff(b.2)
    }

    #[test]
    fn roundtrip_2d_is_exact_all_cells() {
        for order in 1..=4 {
            let n = 1u32 << order;
            for d in 0..(n * n) {
                let (x, y) = hilbert_to_xy(order, d);
                assert!(x < n && y < n);
                assert_eq!(xy_to_hilbert(order, x, y), d);
            }
        }
    }

    #[test]
    fn roundtrip_2d_from_coordinates() {
        for order in 1..=4 {
            let n = 1u32 << order;
            for x in 0..n {
                for y in 0..n {
                    let d = xy_to_hilbert(order, x, y);
                    assert_eq!(hilbert_to_xy(order, d), (x, y));
                }
            }
        }
    }

    #[test]
    fn index_is_within_range() {
        for order in 0..=4 {
            let n = 1u32 << order;
            for x in 0..n {
                for y in 0..n {
                    assert!(xy_to_hilbert(order, x, y) < n * n);
                }
            }
        }
    }

    #[test]
    fn consecutive_indices_are_axis_neighbors() {
        for order in 1..=4 {
            let n = 1u32 << order;
            for d in 1..(n * n) {
                let prev = hilbert_to_xy(order, d - 1);
                let cur = hilbert_to_xy(order, d);
                assert_eq!(manhattan(prev, cur), 1);
            }
        }
    }

    #[test]
    fn maps_every_cell_bijectively() {
        for order in 1..=4 {
            let n = 1u32 << order;
            let mut seen = alloc::vec![false; usize::try_from(n * n).unwrap()];
            for d in 0..(n * n) {
                let (x, y) = hilbert_to_xy(order, d);
                let flat = usize::try_from(y * n + x).unwrap();
                assert!(!seen[flat], "cell visited twice");
                seen[flat] = true;
            }
            assert!(seen.iter().all(|&v| v));
        }
    }

    #[test]
    fn tile_order_has_full_length() {
        for order in 0..=4 {
            let expected = usize::try_from(1u64 << (2 * u64::from(order))).unwrap();
            assert_eq!(tile_order(order).len(), expected);
        }
    }

    #[test]
    fn tile_order_has_no_duplicates() {
        for order in 0..=4 {
            let tiles = tile_order(order);
            let mut sorted = tiles.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), tiles.len());
        }
    }

    #[test]
    fn tile_order_matches_decode_sequence() {
        let order = 3;
        let tiles = tile_order(order);
        for (d, &cell) in tiles.iter().enumerate() {
            let idx = u32::try_from(d).unwrap();
            assert_eq!(cell, hilbert_to_xy(order, idx));
        }
    }

    #[test]
    fn tile_order_is_locally_connected() {
        let tiles = tile_order(4);
        for window in tiles.windows(2) {
            assert_eq!(manhattan(window[0], window[1]), 1);
        }
    }

    #[test]
    fn differs_from_morton_ordering() {
        let order = 2;
        let n = 1u32 << order;
        let mut any_diff = false;
        for d in 0..(n * n) {
            if hilbert_to_xy(order, d) != morton_xy(d, order) {
                any_diff = true;
            }
        }
        assert!(any_diff, "Hilbert order must differ from Z-order");
    }

    #[test]
    fn known_order_one_forward() {
        assert_eq!(xy_to_hilbert(1, 0, 0), 0);
        assert_eq!(xy_to_hilbert(1, 0, 1), 1);
        assert_eq!(xy_to_hilbert(1, 1, 1), 2);
        assert_eq!(xy_to_hilbert(1, 1, 0), 3);
    }

    #[test]
    fn known_order_one_inverse() {
        assert_eq!(hilbert_to_xy(1, 0), (0, 0));
        assert_eq!(hilbert_to_xy(1, 1), (0, 1));
        assert_eq!(hilbert_to_xy(1, 2), (1, 1));
        assert_eq!(hilbert_to_xy(1, 3), (1, 0));
    }

    #[test]
    fn order_zero_is_single_origin_cell() {
        assert_eq!(xy_to_hilbert(0, 0, 0), 0);
        assert_eq!(hilbert_to_xy(0, 0), (0, 0));
        assert_eq!(tile_order(0), alloc::vec![(0, 0)]);
    }

    #[test]
    fn hilbert_key_matches_forward_map() {
        for order in 0..=4 {
            let n = 1u32 << order;
            for x in 0..n {
                for y in 0..n {
                    assert_eq!(hilbert_key(order, x, y), xy_to_hilbert(order, x, y));
                }
            }
        }
    }

    #[test]
    fn order_is_clamped_to_max() {
        let big = xy_to_hilbert(64, 5, 9);
        let clamped = xy_to_hilbert(MAX_ORDER, 5, 9);
        assert_eq!(big, clamped);
        assert_eq!(hilbert_to_xy(64, 123), hilbert_to_xy(MAX_ORDER, 123));
    }

    #[test]
    fn out_of_range_coordinates_wrap() {
        let order = 3;
        let mask = (1u32 << order) - 1;
        assert_eq!(
            xy_to_hilbert(order, 100, 250),
            xy_to_hilbert(order, 100 & mask, 250 & mask)
        );
    }

    #[test]
    fn out_of_range_index_wraps() {
        let order = 3;
        let span = 1u32 << (2 * order);
        assert_eq!(hilbert_to_xy(order, 5), hilbert_to_xy(order, 5 + span));
    }

    #[test]
    fn roundtrip_3d_is_exact_all_cells() {
        for order in 1..=4 {
            let n = 1u32 << order;
            let total = 1u64 << (3 * u64::from(order));
            for d in 0..total {
                let (x, y, z) = hilbert_to_xyz_3d(order, d);
                assert!(x < n && y < n && z < n);
                assert_eq!(hilbert_distance_3d(order, x, y, z), d);
            }
        }
    }

    #[test]
    fn consecutive_indices_are_axis_neighbors_3d() {
        for order in 1..=4 {
            let total = 1u64 << (3 * u64::from(order));
            for d in 1..total {
                let prev = hilbert_to_xyz_3d(order, d - 1);
                let cur = hilbert_to_xyz_3d(order, d);
                assert_eq!(manhattan_3d(prev, cur), 1);
            }
        }
    }

    #[test]
    fn maps_every_cell_bijectively_3d() {
        for order in 1..=3 {
            let n = 1u64 << order;
            let total = n * n * n;
            let mut seen = alloc::vec![false; usize::try_from(total).unwrap()];
            for d in 0..total {
                let (x, y, z) = hilbert_to_xyz_3d(order, d);
                let flat = usize::try_from(u64::from(z) * n * n + u64::from(y) * n + u64::from(x))
                    .unwrap();
                assert!(!seen[flat]);
                seen[flat] = true;
            }
            assert!(seen.iter().all(|&v| v));
        }
    }

    #[test]
    fn index_3d_is_within_range() {
        for order in 0..=4 {
            let n = 1u32 << order;
            let total = 1u64 << (3 * u64::from(order));
            for z in 0..n {
                for y in 0..n {
                    for x in 0..n {
                        assert!(hilbert_distance_3d(order, x, y, z) < total);
                    }
                }
            }
        }
    }

    #[test]
    fn order_zero_3d_is_single_origin_cell() {
        assert_eq!(hilbert_distance_3d(0, 0, 0, 0), 0);
        assert_eq!(hilbert_to_xyz_3d(0, 0), (0, 0, 0));
    }

    #[test]
    fn out_of_range_coordinates_wrap_3d() {
        let order = 2;
        let mask = (1u32 << order) - 1;
        assert_eq!(
            hilbert_distance_3d(order, 40, 41, 42),
            hilbert_distance_3d(order, 40 & mask, 41 & mask, 42 & mask)
        );
    }

    #[test]
    fn known_order_one_forward_3d() {
        // The first `Gray-code` step visits the eight unit-cube corners.
        assert_eq!(hilbert_distance_3d(1, 0, 0, 0), 0);
        assert_eq!(hilbert_distance_3d(1, 1, 0, 0), 7);
        assert_eq!(hilbert_distance_3d(1, 1, 1, 1), 5);
        let mut seen = alloc::vec![false; 8];
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let d = hilbert_distance_3d(1, x, y, z);
                    let idx = usize::try_from(d).unwrap();
                    assert!(!seen[idx]);
                    seen[idx] = true;
                }
            }
        }
        assert!(seen.iter().all(|&v| v));
    }
}
