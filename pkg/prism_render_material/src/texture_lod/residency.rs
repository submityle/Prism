//! Virtual-texture page residency requests derived from a continuous LOD.
//!
//! A streaming / sparse virtual texture (SVT) is tiled into fixed-size pages per
//! mip level. Once [`super::mip`] has picked a *continuous* LOD for a shaded
//! texel, the renderer must translate it into the concrete set of pages that
//! have to be resident for a correct trilinear fetch, and feed any missing ones
//! back to the streaming system. This module performs that translation with
//! pure closed-form integer/`f32` arithmetic -- no hardware sampler feedback and
//! no learning path, so a CPU golden reproduces a GPU twin bit-for-bit.
//!
//! A trilinear fetch samples the two mip levels bracketing the fractional LOD
//! (`floor(lod)` and `floor(lod) + 1`) and blends them by the fractional part.
//! Each level is addressed independently because mip dimensions -- and therefore
//! the page grid -- halve from one level to the next.
//!
//! # Conventions
//! * UVs are in the unit square `[0, 1]^2` before wrapping; they are clamped to
//!   `[0, 1]` here (clamp-to-edge) so out-of-range coordinates still map to a
//!   valid border page instead of panicking.
//! * Mip 0 is the finest level. At mip `m`, each axis dimension is
//!   `max(1, dim >> m)` texels; the page grid is `ceil(dim_m / page_size)`.
//! * `page_size` is defensively clamped to `>= 1`; a zero or absurd value can
//!   never produce a divide-by-zero or out-of-range page index.
//! * All returned page indices are clamped to `[0, pages_per_axis - 1]`.
//!
//! # References
//! * Mittring, "Advanced Virtual Texture Topics" (SIGGRAPH 2008 courses).
//! * van Waveren, "Software Virtual Textures" (2012).
//! * Unreal Engine Streaming Virtual Texturing page-request feedback model.

/// The two mip levels bracketing a continuous LOD plus the trilinear blend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrilinearMip {
    /// Finer (smaller-index) mip level: `floor(lod)` clamped to the pyramid.
    pub fine: u32,
    /// Coarser (larger-index) mip level: `fine + 1`, clamped to `max_mip`.
    pub coarse: u32,
    /// Blend weight toward `coarse` in `[0, 1]` (the fractional part of `lod`).
    pub frac: f32,
}

/// Decompose a continuous LOD into the trilinear mip pair + blend weight.
///
/// The LOD is clamped to `[0, max_mip]` first. When it lands on (or is clamped
/// to) the coarsest level, `fine == coarse == max_mip` and `frac == 0` so the
/// fetch degenerates to a single-level bilinear tap.
#[must_use]
pub fn trilinear_mip(lod: f32, max_mip: u32) -> TrilinearMip {
    let clamped = if lod.is_finite() { lod } else { max_mip as f32 };
    let clamped = clamped.clamp(0.0, max_mip as f32);
    let fine_f = clamped.floor();
    let frac = clamped - fine_f;
    let fine = fine_f as u32;
    let coarse = (fine + 1).min(max_mip);
    // At the top of the pyramid there is no coarser level to blend toward.
    let frac = if coarse == fine { 0.0 } else { frac };
    TrilinearMip { fine, coarse, frac }
}

/// A request that a specific page of a virtual texture be made resident.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PageRequest {
    /// Mip level the page belongs to (0 = finest).
    pub mip: u32,
    /// Page column index within the mip's page grid.
    pub page_x: u32,
    /// Page row index within the mip's page grid.
    pub page_y: u32,
}

/// Immutable description of a streaming virtual texture's tiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VirtualTexture {
    width: u32,
    height: u32,
    page_size: u32,
    max_mip: u32,
}

impl VirtualTexture {
    /// Build a descriptor from base dimensions and a square page size.
    ///
    /// Dimensions are clamped to `>= 1` and `page_size` to `>= 1`. The maximum
    /// mip is `floor(log2(max(width, height)))`, i.e. the level whose largest
    /// axis is a single texel.
    #[must_use]
    pub fn new(width: u32, height: u32, page_size: u32) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let page_size = page_size.max(1);
        let max_axis = width.max(height);
        // ilog2 of a `>= 1` value is well defined; it is the coarsest mip.
        let max_mip = max_axis.ilog2();
        Self {
            width,
            height,
            page_size,
            max_mip,
        }
    }

    /// Coarsest valid mip level (single-texel largest axis).
    #[inline]
    #[must_use]
    pub fn max_mip(self) -> u32 {
        self.max_mip
    }

    /// Coarsest mip as an `f32`, convenient for clamping continuous LODs.
    #[inline]
    #[must_use]
    pub fn max_mip_f32(self) -> f32 {
        self.max_mip as f32
    }

    /// Axis dimension (texels) at `mip`, floored to `>= 1`.
    #[inline]
    #[must_use]
    fn dim_at(dim: u32, mip: u32) -> u32 {
        (dim >> mip.min(31)).max(1)
    }

    /// Number of pages along an axis of `dim_m` texels (ceil division).
    #[inline]
    #[must_use]
    fn pages_along(&self, dim_m: u32) -> u32 {
        dim_m.div_ceil(self.page_size).max(1)
    }

    /// Resolve the page covering `uv` at a single integer `mip`.
    ///
    /// `uv` is clamped to `[0, 1]` (clamp-to-edge) and `mip` to `[0, max_mip]`.
    #[must_use]
    pub fn page_at(&self, uv: [f32; 2], mip: u32) -> PageRequest {
        let mip = mip.min(self.max_mip);
        let w = Self::dim_at(self.width, mip);
        let h = Self::dim_at(self.height, mip);
        let pages_x = self.pages_along(w);
        let pages_y = self.pages_along(h);

        let u = uv[0].clamp(0.0, 1.0);
        let v = uv[1].clamp(0.0, 1.0);
        // Texel coordinate within [0, dim); clamp the right/bottom edge so
        // exactly-1.0 maps to the last texel rather than one past it.
        let tx = (u * w as f32).min((w - 1) as f32).max(0.0) as u32;
        let ty = (v * h as f32).min((h - 1) as f32).max(0.0) as u32;

        let page_x = (tx / self.page_size).min(pages_x - 1);
        let page_y = (ty / self.page_size).min(pages_y - 1);
        PageRequest { mip, page_x, page_y }
    }

    /// Resolve the trilinear page set for a shaded texel at continuous `lod`.
    ///
    /// Returns the fine-mip page followed by the coarse-mip page. When the LOD
    /// sits at the top of the pyramid both entries are identical; callers may
    /// dedup on [`PageRequest`]'s `Eq`.
    #[must_use]
    pub fn residency(&self, uv: [f32; 2], lod: f32) -> [PageRequest; 2] {
        let tri = trilinear_mip(lod, self.max_mip);
        [self.page_at(uv, tri.fine), self.page_at(uv, tri.coarse)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_mip_is_single_texel_largest_axis() {
        // 256 wide -> log2(256) == 8 (mip 8 is 1 texel on the long axis).
        assert_eq!(VirtualTexture::new(256, 64, 128).max_mip(), 8);
        assert_eq!(VirtualTexture::new(1, 1, 128).max_mip(), 0);
    }

    #[test]
    fn trilinear_splits_floor_and_fraction() {
        let t = trilinear_mip(3.25, 10);
        assert_eq!(t.fine, 3);
        assert_eq!(t.coarse, 4);
        assert!((t.frac - 0.25).abs() < 1.0e-6);
    }

    #[test]
    fn trilinear_top_of_pyramid_has_no_blend() {
        let t = trilinear_mip(99.0, 8);
        assert_eq!(t.fine, 8);
        assert_eq!(t.coarse, 8);
        assert_eq!(t.frac, 0.0);
    }

    #[test]
    fn non_finite_lod_falls_back_to_coarsest() {
        let t = trilinear_mip(f32::NAN, 8);
        assert_eq!(t.fine, 8);
        assert_eq!(t.coarse, 8);
    }

    #[test]
    fn center_uv_maps_to_center_page() {
        // 256x256, 128-texel pages -> 2x2 page grid at mip 0.
        let vt = VirtualTexture::new(256, 256, 128);
        let p = vt.page_at([0.5, 0.5], 0);
        assert_eq!(p, PageRequest { mip: 0, page_x: 1, page_y: 1 });
    }

    #[test]
    fn corners_clamp_to_edge_pages() {
        let vt = VirtualTexture::new(256, 256, 128);
        assert_eq!(vt.page_at([0.0, 0.0], 0), PageRequest { mip: 0, page_x: 0, page_y: 0 });
        // u=v=1.0 must land on the last page, not one past the grid.
        assert_eq!(vt.page_at([1.0, 1.0], 0), PageRequest { mip: 0, page_x: 1, page_y: 1 });
        // Out-of-range UV clamps instead of overflowing.
        assert_eq!(vt.page_at([5.0, -5.0], 0), PageRequest { mip: 0, page_x: 1, page_y: 0 });
    }

    #[test]
    fn coarser_mip_has_fewer_pages() {
        // At mip 1 a 256-wide texture is 128 texels -> a single 128 page.
        let vt = VirtualTexture::new(256, 256, 128);
        assert_eq!(vt.page_at([0.9, 0.9], 1), PageRequest { mip: 1, page_x: 0, page_y: 0 });
    }

    #[test]
    fn residency_returns_fine_then_coarse() {
        let vt = VirtualTexture::new(256, 256, 128);
        let [fine, coarse] = vt.residency([0.5, 0.5], 0.5);
        assert_eq!(fine.mip, 0);
        assert_eq!(coarse.mip, 1);
    }

    #[test]
    fn residency_tolerates_zero_page_size() {
        // page_size clamped to >=1; must not divide by zero.
        let vt = VirtualTexture::new(64, 64, 0);
        let [fine, _] = vt.residency([0.3, 0.7], 2.0);
        assert!(fine.page_x < 64 && fine.page_y < 64);
    }
}
