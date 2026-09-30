//! Generic `2x2` depth / data down-sample `mip`-chain builder: the `CPU`
//! gold-standard for the low-level reduction pyramid that `SSR`, `SSAO`, and
//! other screen-space passes read (design §16-§21).
//!
//! Many screen-space effects want a *pyramid* of an input buffer — a chain of
//! ever-coarser `mip` levels where each texel summarizes a `2x2` block of the
//! finer level. A screen-space-reflection (`SSR`) ray-march reads coarse depth
//! `mip`s to skip empty space; a screen-space-ambient-occlusion (`SSAO`) kernel
//! reads a coarse depth pyramid to bound its sampling radius; a min/max depth
//! pyramid feeds tile classification. All of them need the same primitive: take
//! a base image and fold it down `2x2` at a time, with a chosen reduction
//! operator, until a single `1x1` texel remains.
//!
//! This module owns exactly that primitive and nothing else:
//!
//! * the `mip`-size chain (each level is `div_ceil(dim, 2)` of the previous,
//!   so an odd dimension rounds *up* and keeps its boundary row/column instead
//!   of dropping it);
//! * the packed byte layout of the whole chain (`mip` offsets and total byte
//!   size, expressed through the shared `std430` `u32` stride);
//! * one single-level `2x2` reduction and the full-chain build on top of it;
//! * the reduction-operator enum [`ReduceOp`] (`Min` / `Max` / `Average` /
//!   `CheckerboardMinMax`).
//!
//! # Conservative odd-dimension handling
//!
//! When a level dimension is odd, halving it with `div_ceil` produces one extra
//! coarse texel rather than truncating. That extra texel covers only the lone
//! boundary row or column, so its value is the reduction of a *partial* block
//! (one or two source texels). No source texel is ever discarded: an extreme
//! value sitting on an odd edge survives into the coarser level unchanged. This
//! is the "conservative" rule — the coarse pyramid never loses a `Min` minimum
//! or a `Max` maximum to an odd boundary.
//!
//! # Strict scope — not an `HZB` occlusion contract
//!
//! This file is deliberately *only* a generic reduction data-chain. It is **not**
//! the hierarchical-`Z` occlusion path: the max-depth `HZB` used for
//! occlusion *culling* — with its `select_mip`, `OcclusionQuery`, and `CullStats`
//! API — lives in [`super::occlusion`] and its `HzbPyramid`. That module answers
//! "is this `AABB` provably hidden?"; this module answers "build me the reduced
//! `mip` chain of this buffer under operator X". The operator here is selectable
//! (`Min`/`Max`/`Average`/checkerboard), the size rule is conservative
//! `div_ceil` rather than floored `dim >> 1`, and no culling, query, or
//! `select_mip` semantics are provided or duplicated.
//!
//! # Determinism
//!
//! Only integer arithmetic, `f32` `+ - * /`, and `f32::min`/`f32::max` are used;
//! there are no transcendental functions, no `f32::ceil`/`f32::round`, and no
//! direct `f32` `==`/`!=`. Averages divide an exact sum by an exact small
//! integer sample count, so results are bit-reproducible against a future `GPU`
//! reduction kernel. Only [`super::gpu_layout`] is imported.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE};

/// The reduction operator applied to each `2x2` block when building a coarser
/// `mip` level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    /// Coarse texel is the minimum of its `2x2` block (nearest-depth pyramid).
    Min,
    /// Coarse texel is the maximum of its `2x2` block (farthest-depth pyramid).
    Max,
    /// Coarse texel is the arithmetic mean of its `2x2` block (data pyramid).
    Average,
    /// Coarse texel alternates: `Min` on even `(x + y)` destination texels,
    /// `Max` on odd ones. This packs both a min and a max pyramid into a single
    /// checkerboard chain, a common trick for conservative tile bounds.
    CheckerboardMinMax,
}

impl ReduceOp {
    /// Returns every operator variant, in declaration order.
    ///
    /// Useful for exhaustively driving a reduction over all operators.
    #[must_use]
    pub fn all() -> [ReduceOp; 4] {
        [
            ReduceOp::Min,
            ReduceOp::Max,
            ReduceOp::Average,
            ReduceOp::CheckerboardMinMax,
        ]
    }
}

/// Folds the minimum of a non-empty sample slice without any `f32` `==`.
fn fold_min(values: &[f32]) -> f32 {
    let mut iter = values.iter().copied();
    let Some(mut acc) = iter.next() else {
        return 0.0;
    };
    for value in iter {
        acc = acc.min(value);
    }
    acc
}

/// Folds the maximum of a non-empty sample slice without any `f32` `==`.
fn fold_max(values: &[f32]) -> f32 {
    let mut iter = values.iter().copied();
    let Some(mut acc) = iter.next() else {
        return 0.0;
    };
    for value in iter {
        acc = acc.max(value);
    }
    acc
}

/// Reduces the gathered samples of one destination texel under `op`.
///
/// `take_max` selects the branch of [`ReduceOp::CheckerboardMinMax`] and is
/// ignored by the other operators. The slice is always non-empty because every
/// destination texel covers at least its own top-left source texel.
fn reduce_block(op: ReduceOp, samples: &[f32], take_max: bool) -> f32 {
    match op {
        ReduceOp::Min => fold_min(samples),
        ReduceOp::Max => fold_max(samples),
        ReduceOp::Average => {
            let total: f32 = samples.iter().copied().sum();
            let count = f32::from(u8::try_from(samples.len().max(1)).unwrap_or(1));
            total / count
        }
        ReduceOp::CheckerboardMinMax => {
            if take_max {
                fold_max(samples)
            } else {
                fold_min(samples)
            }
        }
    }
}

/// Down-samples one level `(width, height)` into the next coarser level under
/// `op`, returning the coarse texels in row-major order.
///
/// The coarse extent is `(div_ceil(width, 2), div_ceil(height, 2))`. Each coarse
/// texel gathers the up-to-four source texels of its `2x2` footprint, clamped to
/// the source bounds, so odd boundary texels reduce a partial block and never
/// drop a source value. An empty input (`width == 0` or `height == 0`) yields an
/// empty result.
#[must_use]
pub fn downsample_2x2(src: &[f32], width: usize, height: usize, op: ReduceOp) -> Vec<f32> {
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let dst_w = width.div_ceil(2);
    let dst_h = height.div_ceil(2);
    let mut dst = Vec::with_capacity(dst_w.saturating_mul(dst_h));
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let x0 = dx * 2;
            let y0 = dy * 2;
            let mut samples = [0.0_f32; 4];
            let mut count = 0_usize;
            for oy in 0..2_usize {
                let sy = y0 + oy;
                if sy >= height {
                    continue;
                }
                for ox in 0..2_usize {
                    let sx = x0 + ox;
                    if sx >= width {
                        continue;
                    }
                    samples[count] = src[sy * width + sx];
                    count += 1;
                }
            }
            let take_max = (dx + dy) & 1 == 1;
            dst.push(reduce_block(op, &samples[..count], take_max));
        }
    }
    dst
}

/// Applies the `div_ceil(dim, 2)` halving rule `level` times to a base
/// dimension, yielding the extent of that `mip` level along one axis.
///
/// A base of 0 stays 0; any base of at least 1 never drops below 1, because
/// `div_ceil(1, 2) == 1`.
#[must_use]
pub fn mip_dim(base: usize, level: usize) -> usize {
    let mut dim = base;
    let mut remaining = level;
    while remaining > 0 && dim > 1 {
        dim = dim.div_ceil(2);
        remaining -= 1;
    }
    dim
}

/// Counts the `mip` levels of a `(width, height)` base, including the base level
/// itself, following the conservative `div_ceil` halving down to `1x1`.
///
/// Returns 0 for a degenerate base with a zero dimension.
#[must_use]
pub fn mip_count_for(width: usize, height: usize) -> usize {
    if width == 0 || height == 0 {
        return 0;
    }
    let (mut w, mut h) = (width, height);
    let mut levels = 1_usize;
    while w > 1 || h > 1 {
        w = w.div_ceil(2);
        h = h.div_ceil(2);
        levels += 1;
    }
    levels
}

/// A fully built `2x2` reduction `mip` chain: every level packed back-to-back in
/// one row-major `f32` buffer, plus the size/offset metadata a `GPU` upload
/// needs.
///
/// Construct it with [`DepthMipChain::build`]. The chain always contains at
/// least the base level and ends at a `1x1` top, with each level's extent equal
/// to `div_ceil` of the previous.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthMipChain {
    base_width: usize,
    base_height: usize,
    op: ReduceOp,
    data: Vec<f32>,
    level_dims: Vec<(usize, usize)>,
    level_offsets: Vec<usize>,
}

impl DepthMipChain {
    /// Builds the full `mip` chain from a base image under `op`.
    ///
    /// Returns [`None`] when a dimension is 0, when `width * height` overflows,
    /// or when `base.len()` does not equal `width * height`. Otherwise the base
    /// is copied verbatim as `mip` 0 and every coarser level is produced by
    /// [`downsample_2x2`] until a `1x1` top level is reached.
    #[must_use]
    pub fn build(width: usize, height: usize, base: &[f32], op: ReduceOp) -> Option<Self> {
        if width == 0 || height == 0 {
            return None;
        }
        let base_texels = width.checked_mul(height)?;
        if base.len() != base_texels {
            return None;
        }

        let mut level_dims: Vec<(usize, usize)> = Vec::new();
        let mut level_offsets: Vec<usize> = Vec::new();
        let mut data: Vec<f32> = Vec::new();

        let (mut w, mut h) = (width, height);
        level_dims.push((w, h));
        level_offsets.push(0);
        data.extend_from_slice(base);
        let mut current: Vec<f32> = base.to_vec();

        while w > 1 || h > 1 {
            let next = downsample_2x2(&current, w, h, op);
            let offset = data.len();
            w = w.div_ceil(2);
            h = h.div_ceil(2);
            level_dims.push((w, h));
            level_offsets.push(offset);
            data.extend_from_slice(&next);
            current = next;
        }

        Some(Self {
            base_width: width,
            base_height: height,
            op,
            data,
            level_dims,
            level_offsets,
        })
    }

    /// Number of `mip` levels, including the base level.
    #[must_use]
    pub fn mip_count(&self) -> usize {
        self.level_dims.len()
    }

    /// The `(width, height)` of the base level (`mip` 0).
    #[must_use]
    pub fn base_dims(&self) -> (usize, usize) {
        (self.base_width, self.base_height)
    }

    /// The reduction operator this chain was built with.
    #[must_use]
    pub fn op(&self) -> ReduceOp {
        self.op
    }

    /// The `(width, height)` of `level`, or [`None`] if out of range.
    #[must_use]
    pub fn mip_dims(&self, level: usize) -> Option<(usize, usize)> {
        self.level_dims.get(level).copied()
    }

    /// The texel offset of `level` inside the packed [`DepthMipChain::data`]
    /// buffer, or [`None`] if out of range.
    #[must_use]
    pub fn mip_offset(&self, level: usize) -> Option<usize> {
        self.level_offsets.get(level).copied()
    }

    /// The texel count `width * height` of `level`, or [`None`] if out of range.
    #[must_use]
    pub fn mip_texel_count(&self, level: usize) -> Option<usize> {
        self.level_dims.get(level).map(|&(w, h)| w * h)
    }

    /// Total texel count summed across every level (the packed buffer length).
    #[must_use]
    pub fn total_texel_count(&self) -> usize {
        self.data.len()
    }

    /// Total byte size of the packed chain as a `std430` `u32`-strided storage
    /// buffer, clamped up to one element for an empty chain.
    #[must_use]
    pub fn storage_size_bytes(&self) -> usize {
        storage_bytes(U32_STRIDE, self.data.len())
    }

    /// The row-major texels of `level`, or [`None`] if out of range.
    #[must_use]
    pub fn mip(&self, level: usize) -> Option<&[f32]> {
        let &(w, h) = self.level_dims.get(level)?;
        let offset = *self.level_offsets.get(level)?;
        let count = w * h;
        self.data.get(offset..offset + count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Epsilon for tolerant `f32` comparisons; direct `==`/`!=` is forbidden.
    const CMP_EPS: f32 = 1e-6;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    #[test]
    fn reduce_min_single_level() {
        let src = [3.0, 1.0, 4.0, 2.0];
        let out = downsample_2x2(&src, 2, 2, ReduceOp::Min);
        assert_eq!(out.len(), 1);
        assert!(approx(out[0], 1.0));
    }

    #[test]
    fn reduce_max_single_level() {
        let src = [3.0, 1.0, 4.0, 2.0];
        let out = downsample_2x2(&src, 2, 2, ReduceOp::Max);
        assert_eq!(out.len(), 1);
        assert!(approx(out[0], 4.0));
    }

    #[test]
    fn reduce_average_single_level() {
        let src = [3.0, 1.0, 4.0, 2.0];
        let out = downsample_2x2(&src, 2, 2, ReduceOp::Average);
        assert_eq!(out.len(), 1);
        assert!(approx(out[0], 2.5));
    }

    #[test]
    fn reduce_checkerboard_alternates_min_max() {
        // 4x2 source; destination is 2x1 -> texel (0,0) parity even => Min,
        // texel (1,0) parity odd => Max.
        let src = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let out = downsample_2x2(&src, 4, 2, ReduceOp::CheckerboardMinMax);
        assert_eq!(out.len(), 2);
        assert!(approx(out[0], 1.0), "even texel should take min");
        assert!(approx(out[1], 8.0), "odd texel should take max");
    }

    #[test]
    fn odd_width_preserves_extreme_min() {
        // Width 3 (odd): the lone right column becomes its own coarse texel and
        // must keep the tiny value there under Min.
        let src = [5.0, 5.0, 0.001];
        let out = downsample_2x2(&src, 3, 1, ReduceOp::Min);
        assert_eq!(out.len(), 2);
        assert!(approx(out[0], 5.0));
        assert!(
            approx(out[1], 0.001),
            "boundary minimum must not be dropped"
        );
    }

    #[test]
    fn odd_height_preserves_extreme_max() {
        // Height 3 (odd): the lone bottom row keeps its large value under Max.
        let src = [2.0, 2.0, 99.0];
        let out = downsample_2x2(&src, 1, 3, ReduceOp::Max);
        assert_eq!(out.len(), 2);
        assert!(approx(out[0], 2.0));
        assert!(approx(out[1], 99.0), "boundary maximum must not be dropped");
    }

    #[test]
    fn odd_corner_preserves_extreme_max() {
        // 3x3 with an extreme value at the (2,2) corner; the corner coarse
        // texel (1,1) covers only that single source texel.
        let mut src = [0.0_f32; 9];
        src[2 * 3 + 2] = 42.0;
        let out = downsample_2x2(&src, 3, 3, ReduceOp::Max);
        assert_eq!(out.len(), 4);
        // Destination is 2x2; corner texel index (1,1) = 1*2 + 1 = 3.
        assert!(approx(out[3], 42.0), "corner maximum must survive");
    }

    #[test]
    fn average_partial_boundary_block() {
        // Width 3: first texel averages two samples, boundary texel averages one.
        let src = [2.0, 4.0, 10.0];
        let out = downsample_2x2(&src, 3, 1, ReduceOp::Average);
        assert_eq!(out.len(), 2);
        assert!(approx(out[0], 3.0), "average of the full pair");
        assert!(approx(out[1], 10.0), "single-sample average is the sample");
    }

    #[test]
    fn mip_dim_even_and_odd() {
        assert_eq!(mip_dim(4, 0), 4);
        assert_eq!(mip_dim(4, 1), 2);
        assert_eq!(mip_dim(4, 2), 1);
        // Odd chain rounds up at each step: 5 -> 3 -> 2 -> 1.
        assert_eq!(mip_dim(5, 0), 5);
        assert_eq!(mip_dim(5, 1), 3);
        assert_eq!(mip_dim(5, 2), 2);
        assert_eq!(mip_dim(5, 3), 1);
        // Saturates at 1 and never underflows.
        assert_eq!(mip_dim(1, 10), 1);
        assert_eq!(mip_dim(0, 3), 0);
    }

    #[test]
    fn mip_count_for_various() {
        assert_eq!(mip_count_for(1, 1), 1);
        assert_eq!(mip_count_for(2, 1), 2);
        // 5x3 -> 3x2 -> 2x1 -> 1x1 gives four levels.
        assert_eq!(mip_count_for(5, 3), 4);
        assert_eq!(mip_count_for(0, 5), 0);
        assert_eq!(mip_count_for(5, 0), 0);
    }

    #[test]
    fn mip_count_matches_free_function() {
        let base = [0.0_f32; 15];
        let chain = DepthMipChain::build(5, 3, &base, ReduceOp::Min).expect("valid base");
        assert_eq!(chain.mip_count(), mip_count_for(5, 3));
        assert_eq!(chain.base_dims(), (5, 3));
        assert_eq!(chain.op(), ReduceOp::Min);
    }

    #[test]
    fn chain_ends_at_1x1() {
        let base = [0.0_f32; 15];
        let chain = DepthMipChain::build(5, 3, &base, ReduceOp::Max).expect("valid base");
        let top = chain.mip_count() - 1;
        assert_eq!(chain.mip_dims(top), Some((1, 1)));
        assert_eq!(chain.mip_texel_count(top), Some(1));
    }

    #[test]
    fn chain_offsets_and_texel_counts() {
        // 4x4 -> 2x2 -> 1x1 : 16 + 4 + 1 = 21 texels.
        let base: Vec<f32> = (0..16_u8).map(f32::from).collect();
        let chain = DepthMipChain::build(4, 4, &base, ReduceOp::Max).expect("valid base");
        assert_eq!(chain.mip_count(), 3);
        assert_eq!(chain.mip_dims(0), Some((4, 4)));
        assert_eq!(chain.mip_dims(1), Some((2, 2)));
        assert_eq!(chain.mip_dims(2), Some((1, 1)));
        assert_eq!(chain.mip_offset(0), Some(0));
        assert_eq!(chain.mip_offset(1), Some(16));
        assert_eq!(chain.mip_offset(2), Some(20));
        assert_eq!(chain.mip_texel_count(0), Some(16));
        assert_eq!(chain.mip_texel_count(1), Some(4));
        assert_eq!(chain.mip_texel_count(2), Some(1));
        assert_eq!(chain.total_texel_count(), 21);
    }

    #[test]
    fn storage_size_bytes_matches_layout() {
        let base: Vec<f32> = (0..16_u8).map(f32::from).collect();
        let chain = DepthMipChain::build(4, 4, &base, ReduceOp::Average).expect("valid base");
        // 21 texels * 4 bytes each.
        assert_eq!(chain.storage_size_bytes(), 21 * U32_STRIDE);
    }

    #[test]
    fn mip_slice_roundtrip() {
        let base: Vec<f32> = (0..16_u8).map(f32::from).collect();
        let chain = DepthMipChain::build(4, 4, &base, ReduceOp::Max).expect("valid base");
        assert_eq!(chain.mip(0), Some(base.as_slice()));
        let top = chain.mip(chain.mip_count() - 1).expect("top level");
        assert_eq!(top.len(), 1);
        // Max of 0..16 is 15.
        assert!(approx(top[0], 15.0));
    }

    #[test]
    fn degenerate_1x1_single_level() {
        let chain = DepthMipChain::build(1, 1, &[7.0], ReduceOp::Min).expect("valid base");
        assert_eq!(chain.mip_count(), 1);
        assert_eq!(chain.total_texel_count(), 1);
        assert_eq!(chain.mip(0), Some(&[7.0_f32][..]));
    }

    #[test]
    fn degenerate_zero_dim_returns_none() {
        assert!(DepthMipChain::build(0, 4, &[], ReduceOp::Min).is_none());
        assert!(DepthMipChain::build(4, 0, &[], ReduceOp::Max).is_none());
    }

    #[test]
    fn length_mismatch_returns_none() {
        // 2x2 needs four texels; two is a mismatch.
        assert!(DepthMipChain::build(2, 2, &[1.0, 2.0], ReduceOp::Average).is_none());
    }

    #[test]
    fn reduce_op_all_covers_every_variant() {
        let variants = ReduceOp::all();
        assert_eq!(variants.len(), 4);
        assert!(variants.contains(&ReduceOp::Min));
        assert!(variants.contains(&ReduceOp::Max));
        assert!(variants.contains(&ReduceOp::Average));
        assert!(variants.contains(&ReduceOp::CheckerboardMinMax));

        let src = [1.0, 2.0, 3.0, 4.0];
        for op in variants {
            let out = downsample_2x2(&src, 2, 2, op);
            assert_eq!(out.len(), 1);
            let expected = match op {
                ReduceOp::Min | ReduceOp::CheckerboardMinMax => 1.0,
                ReduceOp::Max => 4.0,
                ReduceOp::Average => 2.5,
            };
            assert!(approx(out[0], expected), "operator {op:?} mismatch");
        }
    }

    #[test]
    fn out_of_range_accessors_return_none() {
        let chain = DepthMipChain::build(1, 1, &[1.0], ReduceOp::Min).expect("valid base");
        assert_eq!(chain.mip_dims(9), None);
        assert_eq!(chain.mip_offset(9), None);
        assert_eq!(chain.mip_texel_count(9), None);
        assert_eq!(chain.mip(9), None);
    }
}
