//! Generic `N-D` array indexing: the device-free integer contract that converts
//! between multi-dimensional coordinates and a flat linear offset using the
//! classic *strides* algebra, for both `row-major` (`C` order) and `col-major`
//! (`Fortran` order) layouts (design §27 buffer addressing, generic tensor and
//! grid storage).
//!
//! A dense multi-dimensional array of a given `shape` is stored in one flat
//! buffer. The *stride* of a dimension is how many flat elements you skip to
//! advance that coordinate by one. The flat offset of a coordinate tuple is the
//! dot product of the coordinates with the strides, and the inverse recovers
//! the coordinates from an offset by repeated integer division and remainder —
//! no floating point anywhere.
//!
//! For `row-major` (`C` order) the *last* dimension is contiguous, so its
//! stride is `1` and each earlier stride is the product of all later extents.
//! For `col-major` (`Fortran` order) the *first* dimension is contiguous, so
//! its stride is `1` and each later stride is the product of all earlier
//! extents. For `shape [2, 3, 4]` the `row-major` strides are `[12, 4, 1]` and
//! the `col-major` strides are `[1, 2, 6]`.
//!
//! # Strict scope
//!
//! This module is **pure, generic `N-D` strides linear algebra** and is bound
//! to no concrete domain. It is deliberately distinct from its neighbors:
//!
//! * `super::fluid` and `super::light_clustered` own *domain-specific* `3D`
//!   grid `linear_index` helpers hard-wired to one fixed layout and purpose
//!   (fluid simulation cells, clustered-shading froxels). This module computes
//!   the strides for *any* rank and either major order and is tied to no grid.
//! * `super::hilbert_curve` and `super::morton_code` are *space-filling-curve*
//!   encodings that reorder points for spatial locality (`Z` order, `Hilbert`
//!   order). Their offset is not a stride dot product; it interleaves or folds
//!   coordinate bits. This module performs no locality reordering — the flat
//!   offset is the plain lexicographic position implied by the strides.
//!
//! In short: those modules pick a *specific* storage layout or locality code;
//! this module is the underlying `strides` dot-product-and-remainder algebra
//! they could all be phrased in terms of.
//!
//! # Overflow
//!
//! All arithmetic is `usize` integer. [`total_elements`] and
//! [`linear_from_coords`] multiply extents and so can overflow for very large
//! shapes; callers must keep the element count within `usize`. The strides and
//! inverse-mapping helpers never multiply beyond that same element-count bound.

use alloc::vec::Vec;

/// Returns the `row-major` (`C` order) strides for `shape`.
///
/// The last dimension is contiguous, so its stride is `1`; every earlier
/// stride is the product of all later extents. The empty shape yields an empty
/// stride list. For `shape [2, 3, 4]` the result is `[12, 4, 1]`.
pub fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut strides = alloc::vec![0usize; n];
    if n == 0 {
        return strides;
    }
    strides[n - 1] = 1;
    let mut i = n - 1;
    while i > 0 {
        i -= 1;
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Returns the `col-major` (`Fortran` order) strides for `shape`.
///
/// The first dimension is contiguous, so its stride is `1`; every later stride
/// is the product of all earlier extents. The empty shape yields an empty
/// stride list. For `shape [2, 3, 4]` the result is `[1, 2, 6]`.
pub fn col_major_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut strides = alloc::vec![0usize; n];
    if n == 0 {
        return strides;
    }
    strides[0] = 1;
    let mut i = 1;
    while i < n {
        strides[i] = strides[i - 1] * shape[i - 1];
        i += 1;
    }
    strides
}

/// Returns the flat linear offset for `coords` under `strides`.
///
/// This is the dot product of the coordinate tuple with the stride tuple. The
/// shorter of the two lengths bounds the sum, so passing matching-length slices
/// is the caller's responsibility. Empty inputs yield `0`.
pub fn linear_from_coords(coords: &[usize], strides: &[usize]) -> usize {
    coords
        .iter()
        .zip(strides.iter())
        .map(|(&c, &s)| c * s)
        .sum()
}

/// Recovers the coordinate tuple for a flat `linear` offset in `shape`.
///
/// When `row_major` is `true` the last dimension varies fastest (`C` order);
/// otherwise the first dimension varies fastest (`Fortran` order). The mapping
/// uses only integer remainder and division. Every extent in `shape` must be
/// non-zero and `linear` must be below [`total_elements`]; the empty shape
/// yields an empty coordinate list.
pub fn coords_from_linear(linear: usize, shape: &[usize], row_major: bool) -> Vec<usize> {
    let n = shape.len();
    let mut coords = alloc::vec![0usize; n];
    let mut rem = linear;
    if row_major {
        let mut i = n;
        while i > 0 {
            i -= 1;
            coords[i] = rem % shape[i];
            rem /= shape[i];
        }
    } else {
        let mut i = 0;
        while i < n {
            coords[i] = rem % shape[i];
            rem /= shape[i];
            i += 1;
        }
    }
    coords
}

/// Returns the total number of elements in `shape`, the product of its extents.
///
/// The empty shape denotes a scalar and yields `1`. A shape containing a `0`
/// extent yields `0`. The product can overflow `usize` for very large shapes.
pub fn total_elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Returns `true` when `coords` names a valid cell of `shape`.
///
/// The ranks must match and every coordinate must be strictly below its extent.
/// Empty `coords` with an empty `shape` is a valid scalar address.
pub fn coords_in_bounds(coords: &[usize], shape: &[usize]) -> bool {
    coords.len() == shape.len() && coords.iter().zip(shape.iter()).all(|(&c, &s)| c < s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_major_strides_3d_reference() {
        assert_eq!(row_major_strides(&[2, 3, 4]), [12, 4, 1]);
    }

    #[test]
    fn test_col_major_strides_3d_reference() {
        assert_eq!(col_major_strides(&[2, 3, 4]), [1, 2, 6]);
    }

    #[test]
    fn test_row_major_strides_2d() {
        assert_eq!(row_major_strides(&[3, 5]), [5, 1]);
    }

    #[test]
    fn test_col_major_strides_2d() {
        assert_eq!(col_major_strides(&[3, 5]), [1, 3]);
    }

    #[test]
    fn test_row_major_strides_4d() {
        assert_eq!(row_major_strides(&[2, 3, 4, 5]), [60, 20, 5, 1]);
    }

    #[test]
    fn test_col_major_strides_4d() {
        assert_eq!(col_major_strides(&[2, 3, 4, 5]), [1, 2, 6, 24]);
    }

    #[test]
    fn test_row_major_strides_1d() {
        assert_eq!(row_major_strides(&[7]), [1]);
    }

    #[test]
    fn test_col_major_strides_1d() {
        assert_eq!(col_major_strides(&[7]), [1]);
    }

    #[test]
    fn test_row_major_strides_empty() {
        assert!(row_major_strides(&[]).is_empty());
    }

    #[test]
    fn test_col_major_strides_empty() {
        assert!(col_major_strides(&[]).is_empty());
    }

    #[test]
    fn test_row_major_strides_all_ones() {
        assert_eq!(row_major_strides(&[1, 1, 1]), [1, 1, 1]);
    }

    #[test]
    fn test_col_major_strides_all_ones() {
        assert_eq!(col_major_strides(&[1, 1, 1]), [1, 1, 1]);
    }

    #[test]
    fn test_last_stride_row_major_is_one() {
        let s = row_major_strides(&[9, 2, 7, 4]);
        assert_eq!(s[s.len() - 1], 1);
    }

    #[test]
    fn test_first_stride_col_major_is_one() {
        let s = col_major_strides(&[9, 2, 7, 4]);
        assert_eq!(s[0], 1);
    }

    #[test]
    fn test_linear_from_coords_row_major_3d() {
        let strides = row_major_strides(&[2, 3, 4]);
        assert_eq!(linear_from_coords(&[1, 2, 3], &strides), 23);
    }

    #[test]
    fn test_linear_from_coords_col_major_3d() {
        let strides = col_major_strides(&[2, 3, 4]);
        assert_eq!(linear_from_coords(&[1, 2, 0], &strides), 5);
    }

    #[test]
    fn test_linear_from_coords_origin() {
        let strides = row_major_strides(&[2, 3, 4]);
        assert_eq!(linear_from_coords(&[0, 0, 0], &strides), 0);
    }

    #[test]
    fn test_linear_from_coords_1d() {
        assert_eq!(linear_from_coords(&[5], &[1]), 5);
    }

    #[test]
    fn test_linear_from_coords_empty() {
        assert_eq!(linear_from_coords(&[], &[]), 0);
    }

    #[test]
    fn test_coords_from_linear_row_major_3d() {
        assert_eq!(coords_from_linear(23, &[2, 3, 4], true), [1, 2, 3]);
    }

    #[test]
    fn test_coords_from_linear_col_major_3d() {
        assert_eq!(coords_from_linear(5, &[2, 3, 4], false), [1, 2, 0]);
    }

    #[test]
    fn test_coords_from_linear_2d_row_major() {
        assert_eq!(coords_from_linear(7, &[3, 5], true), [1, 2]);
    }

    #[test]
    fn test_coords_from_linear_2d_col_major() {
        assert_eq!(coords_from_linear(7, &[3, 5], false), [1, 2]);
    }

    #[test]
    fn test_coords_from_linear_1d() {
        assert_eq!(coords_from_linear(5, &[7], true), [5]);
    }

    #[test]
    fn test_coords_from_linear_single_element() {
        assert_eq!(coords_from_linear(0, &[1], true), [0]);
    }

    #[test]
    fn test_coords_from_linear_4d_row_major() {
        assert_eq!(coords_from_linear(119, &[2, 3, 4, 5], true), [1, 2, 3, 4]);
    }

    #[test]
    fn test_round_trip_row_major_3d() {
        let shape = [2, 3, 4];
        let strides = row_major_strides(&shape);
        let total = total_elements(&shape);
        for linear in 0..total {
            let coords = coords_from_linear(linear, &shape, true);
            assert_eq!(linear_from_coords(&coords, &strides), linear);
        }
    }

    #[test]
    fn test_round_trip_col_major_3d() {
        let shape = [2, 3, 4];
        let strides = col_major_strides(&shape);
        let total = total_elements(&shape);
        for linear in 0..total {
            let coords = coords_from_linear(linear, &shape, false);
            assert_eq!(linear_from_coords(&coords, &strides), linear);
        }
    }

    #[test]
    fn test_round_trip_row_major_4d() {
        let shape = [2, 3, 4, 5];
        let strides = row_major_strides(&shape);
        let total = total_elements(&shape);
        for linear in 0..total {
            let coords = coords_from_linear(linear, &shape, true);
            assert_eq!(linear_from_coords(&coords, &strides), linear);
        }
    }

    #[test]
    fn test_round_trip_col_major_2d() {
        let shape = [6, 7];
        let strides = col_major_strides(&shape);
        let total = total_elements(&shape);
        for linear in 0..total {
            let coords = coords_from_linear(linear, &shape, false);
            assert_eq!(linear_from_coords(&coords, &strides), linear);
        }
    }

    #[test]
    fn test_total_elements_3d() {
        assert_eq!(total_elements(&[2, 3, 4]), 24);
    }

    #[test]
    fn test_total_elements_empty_is_one() {
        assert_eq!(total_elements(&[]), 1);
    }

    #[test]
    fn test_total_elements_zero_dim() {
        assert_eq!(total_elements(&[0]), 0);
    }

    #[test]
    fn test_total_elements_1d() {
        assert_eq!(total_elements(&[5]), 5);
    }

    #[test]
    fn test_total_elements_all_ones() {
        assert_eq!(total_elements(&[1, 1, 1]), 1);
    }

    #[test]
    fn test_total_elements_4d() {
        assert_eq!(total_elements(&[2, 3, 4, 5]), 120);
    }

    #[test]
    fn test_total_elements_overflow_premise() {
        // Precondition for callers: the element count must fit in `usize`.
        // A shape whose product reaches `usize::MAX` is the documented ceiling.
        assert_eq!(total_elements(&[usize::MAX, 1]), usize::MAX);
    }

    #[test]
    fn test_coords_in_bounds_valid() {
        assert!(coords_in_bounds(&[1, 2, 3], &[2, 3, 4]));
    }

    #[test]
    fn test_coords_in_bounds_out_of_range() {
        assert!(!coords_in_bounds(&[2, 0, 0], &[2, 3, 4]));
    }

    #[test]
    fn test_coords_in_bounds_wrong_rank() {
        assert!(!coords_in_bounds(&[1, 2], &[2, 3, 4]));
    }

    #[test]
    fn test_coords_in_bounds_edge() {
        assert!(coords_in_bounds(&[1, 2, 3], &[2, 3, 4]));
    }

    #[test]
    fn test_coords_in_bounds_empty() {
        assert!(coords_in_bounds(&[], &[]));
    }

    #[test]
    fn test_row_major_vs_col_major_differ() {
        let shape = [2, 3, 4];
        assert_ne!(row_major_strides(&shape), col_major_strides(&shape));
    }
}
