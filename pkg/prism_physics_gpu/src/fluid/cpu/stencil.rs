//! Trilinear stencil arithmetic shared by the `CPU` golden transfer twin.
//!
//! The particle-to-grid and grid-to-particle transfers both interpolate a
//! staggered face field at a cell-space coordinate with the same eight-node
//! trilinear stencil. [`axis_stencil`] returns, for one axis, the two clamped
//! node indices bracketing the coordinate and the interpolation fraction; it is
//! a bit-for-bit mirror of `prism_physics_core::fluid::mac_grid`'s private
//! helper and of the `axis_stencil` function in `shaders/fluid_transfer.wgsl`,
//! so the `CPU` twin and the device kernel walk identical nodes with identical
//! weights.
//!
//! # Provenance
//!
//! Trilinear interpolation on a staggered `MAC` grid is a standard, publicly
//! documented `CFD` construct (Harlow and Welch 1965; Bridson). This module
//! contains no Unreal Engine source or derived code.

/// One axis of a trilinear stencil: the low node, the high node, and the
/// interpolation fraction toward the high node, all clamped to `[0, nodes)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxisStencil {
    /// The lower (floor) node index, clamped into range.
    pub lo: usize,
    /// The upper (`lo + 1`) node index, clamped into range.
    pub hi: usize,
    /// The fraction toward the upper node, clamped to `[0, 1]`.
    pub frac: f32,
}

/// Computes the trilinear stencil along one axis for cell-space coordinate
/// `coord`, given the number of `nodes` on that axis.
///
/// Mirrors the reference exactly: floor to the low node, clamp both the low and
/// high node into `[0, nodes - 1]`, and clamp the fraction into `[0, 1]`.
#[must_use]
pub fn axis_stencil(coord: f32, nodes: usize) -> AxisStencil {
    let fi = coord.floor();
    let i0 = fi as i32;
    let frac = coord - fi;
    let max = nodes as i32 - 1;
    let lo = i0.clamp(0, max);
    let hi = (i0 + 1).clamp(0, max);
    AxisStencil {
        lo: lo as usize,
        hi: hi as usize,
        frac: frac.clamp(0.0, 1.0),
    }
}

/// The eight `(index, weight)` node contributions of a trilinear stencil, in
/// the canonical `x`-fastest order used by both engines.
///
/// `stride_y` and `stride_z` are the flat-array strides along the `y` and `z`
/// axes of the field being sampled; `lo` contributes weight `1 - frac` and `hi`
/// contributes `frac` on each axis.
#[must_use]
pub fn trilinear_nodes(
    sx: AxisStencil,
    sy: AxisStencil,
    sz: AxisStencil,
    stride_y: usize,
    stride_z: usize,
) -> [(usize, f32); 8] {
    let xs = [(sx.lo, 1.0 - sx.frac), (sx.hi, sx.frac)];
    let ys = [(sy.lo, 1.0 - sy.frac), (sy.hi, sy.frac)];
    let zs = [(sz.lo, 1.0 - sz.frac), (sz.hi, sz.frac)];
    let mut out = [(0usize, 0.0f32); 8];
    let mut n = 0;
    for &(iz, wz) in &zs {
        for &(iy, wy) in &ys {
            for &(ix, wx) in &xs {
                out[n] = (ix + stride_y * iy + stride_z * iz, wx * wy * wz);
                n += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interior_coordinate_brackets_two_nodes() {
        let s = axis_stencil(2.25, 8);
        assert_eq!(s.lo, 2);
        assert_eq!(s.hi, 3);
        assert!((s.frac - 0.25).abs() < 1e-6);
    }

    #[test]
    fn below_range_collapses_both_nodes_to_zero() {
        // The reference clamps only the node indices, not the fraction, so a
        // negative coordinate collapses `lo == hi == 0`. The retained fraction
        // is irrelevant because both endpoints are the same node, and the axis
        // weight `(1 - frac) + frac` is therefore exactly one.
        let s = axis_stencil(-1.5, 8);
        assert_eq!(s.lo, 0);
        assert_eq!(s.hi, 0);
        assert!((0.0..=1.0).contains(&s.frac));
    }

    #[test]
    fn above_range_collapses_both_nodes_to_last() {
        // Symmetric to the below-range case: a coordinate past the last node
        // clamps `lo == hi == nodes - 1`, with the fraction again irrelevant.
        let s = axis_stencil(9.5, 8);
        assert_eq!(s.lo, 7);
        assert_eq!(s.hi, 7);
        assert!((0.0..=1.0).contains(&s.frac));
    }

    #[test]
    fn collapsed_axis_weights_still_sum_to_one() {
        // Even when an axis is clamped onto a single node, the trilinear
        // weights along it sum to one, so an out-of-range sample stays a
        // convex combination.
        let sx = axis_stencil(-2.0, 8);
        let sy = axis_stencil(9.5, 8);
        let sz = axis_stencil(3.4, 8);
        let nodes = trilinear_nodes(sx, sy, sz, 8, 64);
        let sum: f32 = nodes.iter().map(|&(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn weights_sum_to_one() {
        let sx = axis_stencil(1.3, 8);
        let sy = axis_stencil(2.7, 8);
        let sz = axis_stencil(0.5, 8);
        let nodes = trilinear_nodes(sx, sy, sz, 8, 64);
        let sum: f32 = nodes.iter().map(|&(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
