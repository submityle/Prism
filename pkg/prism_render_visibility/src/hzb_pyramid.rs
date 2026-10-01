//! A borrowed reverse-Z Hierarchical-Z (HZB) depth pyramid and the `2x2`
//! gather that turns a projected [`HzbFootprint`] into a conservative occluder
//! depth.
//!
//! [`hzb_footprint`](crate::HzbFootprint) picks the mip and
//! [`hzb_projection`](crate::project_world_aabb) produces the footprint, but
//! neither of them stores depth: the actual occluder values live in a mip
//! chain built by the depth-reduction pass. This module models that chain as a
//! set of borrowed mip levels (finest first) and performs the Nanite-style
//! single `2x2` gather at the chosen mip, matching UE's `HZB` bound test: the
//! mip is selected so the footprint fits inside one `2x2` texel quad, and the
//! quad is min-reduced (reverse-Z farthest surface) into one conservative
//! occluder depth.
//!
//! Storage is borrowed and `no_std`/`alloc`-free so the pyramid can wrap either
//! a CPU reference buffer or a mapped GPU readback without copying.

use crate::{conservative_occluder_reverse_z, HzbFootprint};

/// One borrowed HZB mip level: a row-major `width x height` grid of reverse-Z
/// depths (`1.0` nearest). `texels.len()` must be at least `width * height`;
/// extra trailing texels are ignored.
#[derive(Clone, Copy, Debug)]
pub struct HzbMip<'a> {
    /// Mip width in texels.
    pub width: u32,
    /// Mip height in texels.
    pub height: u32,
    /// Row-major reverse-Z depths, indexed `texels[y * width + x]`.
    pub texels: &'a [f32],
}

impl HzbMip<'_> {
    /// Reads one texel, returning [`None`] for out-of-range coordinates or a
    /// grid whose backing slice is too short to hold `(x, y)`.
    pub fn texel(&self, x: u32, y: u32) -> Option<f32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let index = (y as usize)
            .checked_mul(self.width as usize)?
            .checked_add(x as usize)?;
        self.texels.get(index).copied()
    }
}

/// A reverse-Z HZB pyramid: a borrowed slice of mip levels ordered finest
/// (mip 0) to coarsest.
#[derive(Clone, Copy, Debug)]
pub struct HzbPyramid<'a> {
    mips: &'a [HzbMip<'a>],
}

impl<'a> HzbPyramid<'a> {
    /// Wraps a borrowed, finest-first slice of mip levels.
    pub const fn new(mips: &'a [HzbMip<'a>]) -> Self {
        Self { mips }
    }

    /// Number of mip levels; feeds
    /// [`HzbFootprint::sample_mip`](crate::HzbFootprint::sample_mip).
    pub fn mip_count(&self) -> u32 {
        self.mips.len() as u32
    }

    /// Gathers the conservative occluder depth for `footprint`.
    ///
    /// Selects the mip whose `2x2` quad covers the footprint, reads that quad
    /// (clamped to the mip's extent), and min-reduces the finite taps into the
    /// farthest reverse-Z occluder. Returns [`None`] — "no occluder, keep the
    /// candidate visible" — for an empty pyramid or a quad with no finite taps.
    pub fn gather_occluder(&self, footprint: HzbFootprint) -> Option<f32> {
        let mip_index = footprint.sample_mip(self.mip_count());
        let mip = self.mips.get(mip_index as usize)?;

        // Footprint corners are mip-0 texel coordinates; shift them down to
        // this mip. Negative/non-finite corners clamp to texel 0.
        let x0 = texel_at_mip(footprint.min[0], mip_index, mip.width);
        let y0 = texel_at_mip(footprint.min[1], mip_index, mip.height);
        // The chosen mip guarantees the footprint spans at most one 2x2 quad,
        // so the quad anchored at (x0, y0) covers it. Clamp the far corner so a
        // footprint touching the mip edge still gathers in-range texels.
        let x1 = (x0 + 1).min(mip.width.saturating_sub(1));
        let y1 = (y0 + 1).min(mip.height.saturating_sub(1));

        let quad = [
            mip.texel(x0, y0),
            mip.texel(x1, y0),
            mip.texel(x0, y1),
            mip.texel(x1, y1),
        ];
        let mut finite = [0.0_f32; 4];
        let mut count = 0_usize;
        for value in quad.into_iter().flatten() {
            finite[count] = value;
            count += 1;
        }
        conservative_occluder_reverse_z(&finite[..count])
    }
}

/// Maps a mip-0 pixel coordinate to a texel index at `mip_index`, clamped to
/// `[0, extent - 1]`. Non-finite or negative inputs map to texel 0; the shift
/// by `mip_index` is the integer `>> mip` the GPU reduction uses.
fn texel_at_mip(pixel: f32, mip_index: u32, extent: u32) -> u32 {
    if extent == 0 {
        return 0;
    }
    // Truncating cast floors a non-negative coordinate; guard non-finite and
    // negative values to a conservative texel 0.
    let base = if pixel.is_finite() && pixel > 0.0 {
        pixel as u32
    } else {
        0
    };
    let shifted = base >> mip_index.min(u32::BITS - 1);
    shifted.min(extent - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A 4x4 mip-0 and its 2x2 mip-1, reverse-Z (larger = nearer).
    fn mip0() -> [f32; 16] {
        [
            0.90, 0.80, 0.70, 0.60, //
            0.50, 0.40, 0.30, 0.20, //
            0.15, 0.25, 0.35, 0.45, //
            0.55, 0.65, 0.75, 0.85, //
        ]
    }

    #[test]
    fn texel_lookup_rejects_out_of_range_and_short_slices() {
        let data = mip0();
        let mip = HzbMip {
            width: 4,
            height: 4,
            texels: &data,
        };
        assert_eq!(mip.texel(0, 0), Some(0.90));
        assert_eq!(mip.texel(3, 3), Some(0.85));
        assert_eq!(mip.texel(4, 0), None);
        assert_eq!(mip.texel(0, 4), None);
        // A slice too short for its declared extent yields None, not a panic.
        let short = HzbMip {
            width: 4,
            height: 4,
            texels: &data[..3],
        };
        assert_eq!(short.texel(3, 3), None);
    }

    #[test]
    fn gather_reads_the_mip0_quad_for_a_small_footprint() {
        let data = mip0();
        let mips = [HzbMip {
            width: 4,
            height: 4,
            texels: &data,
        }];
        let pyramid = HzbPyramid::new(&mips);
        // A sub-2-texel footprint stays at mip 0 and gathers the 2x2 quad whose
        // anchor is (0, 0): {0.90, 0.80, 0.50, 0.40}. Reverse-Z farthest = 0.40.
        let footprint = HzbFootprint::new([0.2, 0.2], [1.2, 1.2]);
        assert_eq!(footprint.sample_mip(pyramid.mip_count()), 0);
        assert_eq!(pyramid.gather_occluder(footprint), Some(0.40));
    }

    #[test]
    fn gather_descends_to_a_coarser_mip_for_a_large_footprint() {
        let data0 = mip0();
        // mip-1 2x2 min-reduction of each 2x2 block of mip0.
        let data1 = [
            0.40, 0.20, // min of top-left / top-right blocks
            0.15, 0.35, // min of bottom-left / bottom-right blocks
        ];
        let mips = [
            HzbMip {
                width: 4,
                height: 4,
                texels: &data0,
            },
            HzbMip {
                width: 2,
                height: 2,
                texels: &data1,
            },
        ];
        let pyramid = HzbPyramid::new(&mips);
        // A 3-texel-wide footprint needs ceil(log2(3)) = mip 2, clamped to the
        // last available level (mip 1). At mip 1 the anchor texel is (0, 0),
        // quad {0.40, 0.20, 0.15, 0.35}, farthest = 0.15.
        let footprint = HzbFootprint::new([0.0, 0.0], [3.0, 3.0]);
        assert_eq!(footprint.sample_mip(pyramid.mip_count()), 1);
        assert_eq!(pyramid.gather_occluder(footprint), Some(0.15));
    }

    #[test]
    fn empty_pyramid_yields_no_occluder() {
        let pyramid = HzbPyramid::new(&[]);
        assert_eq!(pyramid.mip_count(), 0);
        assert_eq!(
            pyramid.gather_occluder(HzbFootprint::new([0.0, 0.0], [1.0, 1.0])),
            None
        );
    }

    #[test]
    fn gather_clamps_a_footprint_past_the_mip_edge() {
        let data = mip0();
        let mips = [HzbMip {
            width: 4,
            height: 4,
            texels: &data,
        }];
        let pyramid = HzbPyramid::new(&mips);
        // A footprint anchored at the far corner clamps its quad in-range: the
        // anchor texel is (3, 3) and the far corner clamps back to (3, 3), so
        // the quad degenerates to the single corner texel 0.85.
        let footprint = HzbFootprint::new([3.4, 3.4], [3.9, 3.9]);
        assert_eq!(footprint.sample_mip(pyramid.mip_count()), 0);
        assert_eq!(pyramid.gather_occluder(footprint), Some(0.85));
    }

    #[test]
    fn non_finite_footprint_corner_clamps_to_texel_zero() {
        let data = mip0();
        let mips = [HzbMip {
            width: 4,
            height: 4,
            texels: &data,
        }];
        let pyramid = HzbPyramid::new(&mips);
        // A non-finite / negative min corner clamps to texel (0, 0); the
        // single-level pyramid pins the mip to 0, so the quad is anchored at
        // the origin: {0.90, 0.80, 0.50, 0.40}, farthest 0.40.
        let footprint = HzbFootprint::new([f32::NAN, -5.0], [0.5, 0.5]);
        assert_eq!(pyramid.gather_occluder(footprint), Some(0.40));
    }
}
