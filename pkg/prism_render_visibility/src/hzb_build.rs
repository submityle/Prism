//! Reverse-Z HZB pyramid construction (the depth-reduction pass, CPU reference).
//!
//! [`HzbPyramid`](crate::HzbPyramid) and
//! [`gather_occluder`](crate::HzbPyramid::gather_occluder) *consume* a mip
//! chain, and their docs note that "the actual occluder values live in a mip
//! chain built by the depth-reduction pass" — but that pass had no in-crate
//! reference, so every test hand-authored its mips. [`build_hzb_pyramid`]
//! supplies it: it owns a full-resolution reverse-Z depth buffer and reduces it
//! into the finest-first mip chain the pyramid expects, min-reducing each
//! parent `2x2` block into the farthest (smallest reverse-Z) child texel. This
//! is the conservative reduction UE's `FHZBBuilder` performs, including the
//! odd-dimension third tap that keeps a non-power-of-two level from dropping
//! its last row/column.
//!
//! The output owns its texels (so it can outlive any transient source buffer),
//! and [`HzbPyramidStorage::views`] materialises the borrowed
//! [`HzbMip`](crate::HzbMip) slice that [`HzbPyramid::new`](crate::HzbPyramid)
//! wraps — completing the "depth buffer in, cullable pyramid out" chain
//! entirely from classic integer/`min` arithmetic.

use crate::{conservative_occluder_reverse_z, HzbMip};
use alloc::vec::Vec;

/// One owned HZB mip level: a row-major `width x height` grid of reverse-Z
/// depths produced by [`build_hzb_pyramid`].
#[derive(Clone, Debug, PartialEq)]
pub struct OwnedHzbMip {
    /// Mip width in texels (at least `1` for a non-empty pyramid).
    pub width: u32,
    /// Mip height in texels (at least `1` for a non-empty pyramid).
    pub height: u32,
    /// Row-major reverse-Z depths, indexed `texels[y * width + x]`.
    pub texels: Vec<f32>,
}

/// An owned reverse-Z HZB mip chain, finest (mip 0) first, built by
/// [`build_hzb_pyramid`].
///
/// Borrow it as the slice [`HzbPyramid`](crate::HzbPyramid) expects via
/// [`views`](HzbPyramidStorage::views) followed by
/// [`HzbPyramid::new`](crate::HzbPyramid::new).
#[derive(Clone, Debug, PartialEq)]
pub struct HzbPyramidStorage {
    /// Mip levels, finest first; empty for a degenerate (zero-area) source.
    mips: Vec<OwnedHzbMip>,
}

impl HzbPyramidStorage {
    /// Number of mip levels (`0` for an empty pyramid).
    pub fn mip_count(&self) -> u32 {
        self.mips.len() as u32
    }

    /// Borrows mip `index` (finest first), or [`None`] when out of range.
    pub fn mip(&self, index: u32) -> Option<&OwnedHzbMip> {
        self.mips.get(index as usize)
    }

    /// Materialises the borrowed, finest-first [`HzbMip`](crate::HzbMip) slice
    /// to hand to [`HzbPyramid::new`](crate::HzbPyramid::new).
    pub fn views(&self) -> Vec<HzbMip<'_>> {
        self.mips
            .iter()
            .map(|mip| HzbMip {
                width: mip.width,
                height: mip.height,
                texels: &mip.texels,
            })
            .collect()
    }
}

/// Builds the reverse-Z HZB mip chain for a `width x height` depth buffer.
///
/// `depths` is row-major mip-0 reverse-Z depth (`1.0` nearest), indexed
/// `depths[y * width + x]`; it must hold at least `width * height` texels or
/// the build returns [`None`]. A zero-area source (`width == 0 || height == 0`)
/// yields an empty [`HzbPyramidStorage`] with no mips.
///
/// Mip 0 copies the source. Each coarser level halves both dimensions with an
/// integer `>> 1` (clamped to a minimum of `1`), and every destination texel is
/// the conservative occluder — the farthest, i.e. smallest reverse-Z — of its
/// parent block via
/// [`conservative_occluder_reverse_z`](crate::conservative_occluder_reverse_z).
/// The block is the `2x2` quad `(2x, 2y)`, extended to a third tap along any
/// axis whose parent extent is odd so the trailing row/column is never
/// dropped. Non-finite parent taps are ignored; a block with no finite tap
/// stores [`f32::NAN`] ("no occluder"), matching the gather's finite-only
/// contract. The chain stops once both dimensions reach `1`.
pub fn build_hzb_pyramid(width: u32, height: u32, depths: &[f32]) -> Option<HzbPyramidStorage> {
    if width == 0 || height == 0 {
        return Some(HzbPyramidStorage { mips: Vec::new() });
    }
    let area = (width as usize).checked_mul(height as usize)?;
    if depths.len() < area {
        return None;
    }

    let mut mips = Vec::new();
    mips.push(OwnedHzbMip {
        width,
        height,
        texels: depths[..area].to_vec(),
    });

    let (mut w, mut h) = (width, height);
    while w > 1 || h > 1 {
        let parent = mips.last()?;
        let next = reduce_mip(parent);
        w = next.width;
        h = next.height;
        mips.push(next);
    }

    Some(HzbPyramidStorage { mips })
}

/// Reduces one parent mip into the next coarser reverse-Z level.
fn reduce_mip(parent: &OwnedHzbMip) -> OwnedHzbMip {
    let dst_w = (parent.width >> 1).max(1);
    let dst_h = (parent.height >> 1).max(1);
    let extra_x = parent.width & 1 == 1 && parent.width > 1;
    let extra_y = parent.height & 1 == 1 && parent.height > 1;

    let mut texels = Vec::with_capacity((dst_w as usize) * (dst_h as usize));
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            texels.push(reduce_block(parent, dx, dy, extra_x, extra_y));
        }
    }
    OwnedHzbMip {
        width: dst_w,
        height: dst_h,
        texels,
    }
}

/// Conservatively reduces the parent block anchored at `(2 * dx, 2 * dy)` into
/// one child texel: the farthest (smallest reverse-Z) finite tap of the `2x2`
/// quad, widened to `3` taps on an axis flagged odd so no edge texel is lost.
fn reduce_block(parent: &OwnedHzbMip, dx: u32, dy: u32, extra_x: bool, extra_y: bool) -> f32 {
    let last_x = parent.width - 1;
    let last_y = parent.height - 1;
    let base_x = dx << 1;
    let base_y = dy << 1;

    let mut taps = [0.0_f32; 9];
    let mut count = 0_usize;
    let mut y_offset = 0;
    while y_offset < if extra_y { 3 } else { 2 } {
        let sy = (base_y + y_offset).min(last_y);
        let mut x_offset = 0;
        while x_offset < if extra_x { 3 } else { 2 } {
            let sx = (base_x + x_offset).min(last_x);
            let index = (sy as usize) * (parent.width as usize) + sx as usize;
            taps[count] = parent.texels[index];
            count += 1;
            x_offset += 1;
        }
        y_offset += 1;
    }

    // `None` means every tap was non-finite; store NaN so the gather treats the
    // texel as "no occluder" just as it does for a non-finite source tap.
    conservative_occluder_reverse_z(&taps[..count]).unwrap_or(f32::NAN)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HzbFootprint, HzbPyramid};

    // A 4x4 reverse-Z mip-0 (larger = nearer).
    fn mip0() -> [f32; 16] {
        [
            0.90, 0.80, 0.70, 0.60, //
            0.50, 0.40, 0.30, 0.20, //
            0.15, 0.25, 0.35, 0.45, //
            0.55, 0.65, 0.75, 0.85, //
        ]
    }

    #[test]
    fn builds_full_chain_down_to_one_by_one() {
        let data = mip0();
        let storage = build_hzb_pyramid(4, 4, &data).unwrap();
        // 4x4 -> 2x2 -> 1x1 == 3 levels.
        assert_eq!(storage.mip_count(), 3);
        assert_eq!(storage.mip(0).unwrap().texels, data.to_vec());

        // mip-1 is the per-2x2-block farthest (min) reverse-Z of mip-0.
        let mip1 = storage.mip(1).unwrap();
        assert_eq!((mip1.width, mip1.height), (2, 2));
        assert_eq!(mip1.texels, vec![0.40, 0.20, 0.15, 0.35]);

        // mip-2 is the farthest of the four mip-1 texels.
        let mip2 = storage.mip(2).unwrap();
        assert_eq!((mip2.width, mip2.height), (1, 1));
        assert_eq!(mip2.texels, vec![0.15]);
    }

    #[test]
    fn built_chain_feeds_the_pyramid_gather_identically() {
        let data = mip0();
        let storage = build_hzb_pyramid(4, 4, &data).unwrap();
        let views = storage.views();
        let pyramid = HzbPyramid::new(&views);

        // Small footprint stays at mip 0, gathering the (0,0) quad
        // {0.90,0.80,0.50,0.40}: farthest 0.40.
        let small = HzbFootprint::new([0.2, 0.2], [1.2, 1.2]);
        assert_eq!(small.sample_mip(pyramid.mip_count()), 0);
        assert_eq!(pyramid.gather_occluder(small), Some(0.40));

        // A 3-texel footprint needs ceil(log2(3)) = mip 2; the full chain
        // includes the 1x1 mip-2 whose only texel is the farthest of the whole
        // image (0.15), so the gather returns it directly.
        let large = HzbFootprint::new([0.0, 0.0], [3.0, 3.0]);
        assert_eq!(large.sample_mip(pyramid.mip_count()), 2);
        assert_eq!(pyramid.gather_occluder(large), Some(0.15));
    }

    #[test]
    fn odd_dimension_keeps_the_trailing_row_and_column() {
        // 3x3 is odd on both axes: 3 >> 1 == 1, so it collapses straight to
        // 1x1, and the odd third tap must fold in the last row and column.
        let data = [
            0.90, 0.80, 0.10, //
            0.50, 0.40, 0.30, //
            0.70, 0.60, 0.05, //
        ];
        let storage = build_hzb_pyramid(3, 3, &data).unwrap();
        assert_eq!(storage.mip_count(), 2);
        let mip1 = storage.mip(1).unwrap();
        assert_eq!((mip1.width, mip1.height), (1, 1));
        // Farthest of all nine (because odd on both axes extends to a 3x3 tap)
        // is 0.05 — the trailing corner that a 2x2-only reduction would drop.
        assert_eq!(mip1.texels, vec![0.05]);
    }

    #[test]
    fn rectangular_source_halves_each_axis_independently() {
        // 4x2 -> 2x1 -> 1x1.
        let data = [
            0.90, 0.80, 0.70, 0.20, //
            0.50, 0.40, 0.30, 0.60, //
        ];
        let storage = build_hzb_pyramid(4, 2, &data).unwrap();
        assert_eq!(storage.mip_count(), 3);
        let mip1 = storage.mip(1).unwrap();
        assert_eq!((mip1.width, mip1.height), (2, 1));
        // Left 2x2 block {0.90,0.80,0.50,0.40} -> 0.40;
        // right 2x2 block {0.70,0.20,0.30,0.60} -> 0.20.
        assert_eq!(mip1.texels, vec![0.40, 0.20]);
        let mip2 = storage.mip(2).unwrap();
        assert_eq!((mip2.width, mip2.height), (1, 1));
        assert_eq!(mip2.texels, vec![0.20]);
    }

    #[test]
    fn non_finite_taps_are_ignored_and_dead_blocks_store_nan() {
        // Top-left block mixes a NaN with finite taps; the NaN is skipped.
        // Bottom-right block is entirely non-finite and must store NaN.
        let data = [
            f32::NAN,
            0.80,
            0.70,
            0.60,
            0.50,
            0.40,
            0.30,
            0.20,
            0.10,
            0.90,
            f32::INFINITY,
            f32::NAN,
            0.15,
            0.25,
            f32::NAN,
            f32::NEG_INFINITY,
        ];
        let storage = build_hzb_pyramid(4, 4, &data).unwrap();
        let mip1 = storage.mip(1).unwrap();
        // {NaN,0.80,0.50,0.40} -> finite farthest 0.40.
        assert_eq!(mip1.texels[0], 0.40);
        // Bottom-right {inf,NaN,NaN,-inf}: no finite tap -> NaN.
        assert!(mip1.texels[3].is_nan());
    }

    #[test]
    fn zero_area_source_yields_empty_pyramid() {
        assert_eq!(build_hzb_pyramid(0, 4, &[]).unwrap().mip_count(), 0);
        assert_eq!(build_hzb_pyramid(4, 0, &[]).unwrap().mip_count(), 0);
    }

    #[test]
    fn too_short_source_slice_is_rejected() {
        let data = [0.5_f32; 15];
        assert!(build_hzb_pyramid(4, 4, &data).is_none());
    }

    #[test]
    fn single_texel_source_is_its_own_only_mip() {
        let storage = build_hzb_pyramid(1, 1, &[0.42]).unwrap();
        assert_eq!(storage.mip_count(), 1);
        assert_eq!(storage.mip(0).unwrap().texels, vec![0.42]);
    }
}
