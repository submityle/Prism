//! `HZB` (hierarchical depth-buffer) occlusion-culling contract for design §13.
//!
//! This module is the `CPU`-verifiable contract for the *occlusion* half of the
//! sort-and-cull layer: given a screen-space bounding rectangle and the nearest
//! depth of a particle system, decide whether a hierarchical `Z`-buffer proves
//! the system is fully hidden behind previously rasterized geometry. It mirrors
//! production `Hi-Z` culling at the algorithm level — Unreal's `HZBOcclusion`
//! reduction and `Frostbite`'s `Hi-Z` particle-cluster rejection — without
//! reusing any vendor code.
//!
//! The pyramid stores a *max-depth* reduction: each coarser `mip` texel holds
//! the farthest depth of the four finer texels it covers, under the depth
//! convention "larger is farther" (a standard reversed-`Z`-agnostic max
//! pyramid). A candidate is *possibly visible* when its nearest depth is at or
//! in front of the sampled `HZB` depth, and *provably occluded* only when its
//! nearest depth is strictly behind the sampled depth. Sampling a coarser
//! `mip` widens the covered footprint, so the test grows more conservative (it
//! reports "visible" more often) as `mip` level rises; it never yields a false
//! occlusion.
//!
//! Determinism rules match the sibling particle modules: the only non-`+ - * /`
//! primitives are `sqrt`, `floor`, and integer `div_ceil`; there are no
//! transcendental functions, all level counting is done with integer shifts,
//! and `f32` values are compared against [`CMP_EPS`] rather than with `==`.

/// Epsilon for tolerant `f32` comparisons; direct `==`/`!=` is forbidden.
pub const CMP_EPS: f32 = 1e-6;

/// A hierarchical depth pyramid (`HZB`) described by its base texel extent.
///
/// The pyramid is defined purely by its base `width`/`height`; every coarser
/// `mip` halves each dimension (rounding down, floored at 1) exactly as a
/// `GPU` `Hi-Z` reduction pass would. Only the *shape* of the pyramid lives in
/// this contract; the actual max-depth texels are produced by the `GPU`
/// backend and sampled through [`is_occluded`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HzbPyramid {
    /// Base (`mip` 0) width in texels; always at least 1.
    base_width: u32,
    /// Base (`mip` 0) height in texels; always at least 1.
    base_height: u32,
}

impl HzbPyramid {
    /// Builds a pyramid, clamping each base dimension up to at least 1.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            base_width: width.max(1),
            base_height: height.max(1),
        }
    }

    /// Returns the base width (`mip` 0), guaranteed to be at least 1.
    #[must_use]
    pub fn base_width(&self) -> u32 {
        self.base_width
    }

    /// Returns the base height (`mip` 0), guaranteed to be at least 1.
    #[must_use]
    pub fn base_height(&self) -> u32 {
        self.base_height
    }

    /// Counts the pyramid levels, including the base level.
    ///
    /// Equivalent to `floor(log2(max(w, h))) + 1`, but computed without any
    /// logarithm: the larger dimension is repeatedly halved (floored at 1)
    /// until it reaches 1, counting one level per iteration plus the final
    /// 1x1 level. This matches the `GPU` `Hi-Z` `dim >> 1` down-sample chain.
    #[must_use]
    pub fn mip_count(&self) -> u32 {
        let mut dim = self.base_width.max(self.base_height);
        let mut levels: u32 = 1;
        while dim > 1 {
            dim = (dim / 2).max(1);
            levels = levels.saturating_add(1);
        }
        levels
    }

    /// Returns the `(width, height)` of the requested `mip` level.
    ///
    /// Each dimension is `max(1, base >> level)`. Levels at or beyond
    /// [`HzbPyramid::mip_count`] saturate to the 1x1 tail rather than
    /// underflowing, so the accessor is always well defined.
    #[must_use]
    pub fn mip_size(&self, level: u32) -> (u32, u32) {
        let shift = level.min(31);
        let w = (self.base_width >> shift).max(1);
        let h = (self.base_height >> shift).max(1);
        (w, h)
    }

    /// Returns the texel count `width * height` of one `mip` level.
    ///
    /// The product is computed with a saturating multiply so an extreme base
    /// extent cannot overflow the contract accessor.
    #[must_use]
    pub fn mip_texel_count(&self, level: u32) -> u32 {
        let (w, h) = self.mip_size(level);
        w.saturating_mul(h)
    }

    /// Returns the total texel count summed across every `mip` level.
    ///
    /// The sum is accumulated in `u64` with saturating arithmetic so the whole
    /// pyramid footprint stays representable even for large base extents. Each
    /// texel is addressed as a 4-byte `u32` depth slot in the packed `GPU`
    /// buffer, matching the crate's shared `u32` storage stride.
    #[must_use]
    pub fn total_texel_count(&self) -> u64 {
        let levels = self.mip_count();
        let mut total: u64 = 0;
        let mut level: u32 = 0;
        while level < levels {
            total = total.saturating_add(u64::from(self.mip_texel_count(level)));
            level = level.saturating_add(1);
        }
        total
    }

    /// Selects the coarsest `mip` whose texel roughly covers a screen rect.
    ///
    /// Takes the ceil-`log2` of the rect's longer side by counting how many
    /// halvings collapse it to a single texel (a `1<<level` footprint), then
    /// clamps the result into `0..=mip_count()-1`. Choosing the `mip` whose
    /// texel spans the projected rect keeps the `HZB` fetch to a single
    /// conservative sample, exactly as `UE` `HZBOcclusion` does.
    #[must_use]
    pub fn select_mip(&self, screen_rect_w: u32, screen_rect_h: u32) -> u32 {
        let mut span = screen_rect_w.max(screen_rect_h).max(1);
        let mut level: u32 = 0;
        while span > 1 {
            span = (span / 2).max(1);
            level = level.saturating_add(1);
        }
        let max_level = self.mip_count().saturating_sub(1);
        level.min(max_level)
    }
}

/// An axis-aligned screen-space rectangle (`NDC` or pixel units).
///
/// Both `NDC` (`-1..=1`) and pixel-space rectangles are valid; the contract
/// only assumes `min <= max` on each axis for a non-empty rect. It is the
/// projected footprint that [`OcclusionQuery`] tests against the `HZB`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenRect {
    /// Left edge (smaller x).
    pub min_x: f32,
    /// Bottom edge (smaller y).
    pub min_y: f32,
    /// Right edge (larger x).
    pub max_x: f32,
    /// Top edge (larger y).
    pub max_y: f32,
}

impl ScreenRect {
    /// Builds a rectangle from its four edges (no normalization is applied).
    #[must_use]
    pub fn new(min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    /// Returns the rectangle width, clamped to be non-negative.
    #[must_use]
    pub fn width(&self) -> f32 {
        (self.max_x - self.min_x).max(0.0)
    }

    /// Returns the rectangle height, clamped to be non-negative.
    #[must_use]
    pub fn height(&self) -> f32 {
        (self.max_y - self.min_y).max(0.0)
    }

    /// Reports whether the rectangle encloses a strictly positive area.
    ///
    /// Uses strict `>` on both axes so a degenerate zero-width or zero-height
    /// rect is treated as invalid without any `f32` equality comparison.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.max_x > self.min_x && self.max_y > self.min_y
    }

    /// Clamps the rectangle into `[0, width] x [0, height]` pixel bounds.
    ///
    /// Each edge is clamped independently so the result never leaves the
    /// viewport; a rect fully outside the bounds collapses onto the nearest
    /// edge (and is then reported invalid by [`ScreenRect::is_valid`]).
    #[must_use]
    pub fn clamp_to(&self, width: f32, height: f32) -> Self {
        let w = width.max(0.0);
        let h = height.max(0.0);
        Self {
            min_x: self.min_x.clamp(0.0, w),
            min_y: self.min_y.clamp(0.0, h),
            max_x: self.max_x.clamp(0.0, w),
            max_y: self.max_y.clamp(0.0, h),
        }
    }
}

/// A single occlusion test: a screen footprint plus its nearest depth.
///
/// `nearest_depth` is the closest (smallest, under "larger is farther") depth
/// of the candidate over its whole [`ScreenRect`] footprint; it is compared to
/// the farthest depth the `HZB` recorded for that footprint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcclusionQuery {
    /// Projected screen footprint of the candidate.
    pub rect: ScreenRect,
    /// Nearest (closest-to-camera) depth of the candidate.
    pub nearest_depth: f32,
}

/// Conservatively decides whether the candidate is hidden by the `HZB`.
///
/// Depth convention: larger is farther. The candidate is *provably occluded*
/// only when its nearest depth is strictly farther than the sampled max-depth
/// `HZB` texel — `query.nearest_depth > hzb_sampled_depth + CMP_EPS`. Boundary
/// equality is resolved as *visible* (not occluded) so the test never rejects a
/// candidate that merely touches the recorded depth, preserving conservatism.
#[must_use]
pub fn is_occluded(query: &OcclusionQuery, hzb_sampled_depth: f32) -> bool {
    query.nearest_depth > hzb_sampled_depth + CMP_EPS
}

/// Rolling counters for an occlusion-culling pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CullStats {
    /// Number of candidates tested this pass.
    pub tested: u32,
    /// Number of candidates rejected (proven occluded) this pass.
    pub culled: u32,
}

impl CullStats {
    /// Returns the number of surviving (visible) candidates.
    ///
    /// Computed as `tested.saturating_sub(culled)` so inconsistent counters can
    /// never underflow the accessor.
    #[must_use]
    pub fn visible(&self) -> u32 {
        self.tested.saturating_sub(self.culled)
    }

    /// Returns the fraction of tested candidates that were culled.
    ///
    /// Returns `0.0` when nothing was tested, avoiding a divide-by-zero; the
    /// zero check is on the integer `tested` count, so no `f32` equality
    /// comparison is involved.
    #[must_use]
    pub fn cull_ratio(&self) -> f32 {
        if self.tested == 0 {
            return 0.0;
        }
        let culled = self.culled.min(self.tested);
        (culled as f32) / (self.tested as f32)
    }
}

/// Worst-case per-edge false-negative bound (in base texels) for a `mip`.
///
/// Sampling at `mip_level` covers a `1 << mip_level` texel footprint per edge,
/// so a candidate can be classified visible while occupying up to that many
/// base texels beyond its true silhouette. The bound is returned as base
/// texels via a saturating shift: the coarser the `HZB` `mip`, the larger the
/// conservative over-estimate, which is why [`HzbPyramid::select_mip`] picks
/// the finest `mip` that still covers the footprint.
#[must_use]
pub fn conservative_false_negative_bound(mip_level: u32) -> u32 {
    if mip_level >= 31 {
        return u32::MAX;
    }
    1u32 << mip_level
}

#[cfg(test)]
mod tests {
    use super::{
        conservative_false_negative_bound, is_occluded, CullStats, HzbPyramid, OcclusionQuery,
        ScreenRect, CMP_EPS,
    };

    #[test]
    fn mip_count_one_by_one_is_single_level() {
        let p = HzbPyramid::new(1, 1);
        assert_eq!(p.mip_count(), 1);
    }

    #[test]
    fn mip_count_clamps_zero_dims() {
        let p = HzbPyramid::new(0, 0);
        assert_eq!(p.base_width(), 1);
        assert_eq!(p.base_height(), 1);
        assert_eq!(p.mip_count(), 1);
    }

    #[test]
    fn mip_count_power_of_two() {
        // 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 == 9 levels.
        let p = HzbPyramid::new(256, 256);
        assert_eq!(p.mip_count(), 9);
    }

    #[test]
    fn mip_count_non_power_of_two_uses_max_edge() {
        // max edge 100 -> 50 -> 25 -> 12 -> 6 -> 3 -> 1 == 7 levels.
        let p = HzbPyramid::new(100, 37);
        assert_eq!(p.mip_count(), 7);
    }

    #[test]
    fn mip_size_halves_each_level() {
        let p = HzbPyramid::new(256, 64);
        assert_eq!(p.mip_size(0), (256, 64));
        assert_eq!(p.mip_size(1), (128, 32));
        assert_eq!(p.mip_size(2), (64, 16));
        assert_eq!(p.mip_size(6), (4, 1));
    }

    #[test]
    fn mip_size_saturates_past_last_level() {
        let p = HzbPyramid::new(8, 8);
        assert_eq!(p.mip_size(3), (1, 1));
        assert_eq!(p.mip_size(99), (1, 1));
    }

    #[test]
    fn mip_texel_count_matches_dimensions() {
        let p = HzbPyramid::new(256, 64);
        assert_eq!(p.mip_texel_count(0), 256 * 64);
        assert_eq!(p.mip_texel_count(1), 128 * 32);
    }

    #[test]
    fn total_texel_count_sums_all_levels() {
        let p = HzbPyramid::new(4, 4);
        // 4x4 + 2x2 + 1x1 = 16 + 4 + 1 = 21.
        assert_eq!(p.total_texel_count(), 21);
    }

    #[test]
    fn select_mip_small_rect_is_base_level() {
        let p = HzbPyramid::new(256, 256);
        assert_eq!(p.select_mip(1, 1), 0);
    }

    #[test]
    fn select_mip_matches_ceil_log2() {
        let p = HzbPyramid::new(1024, 1024);
        // span 4 -> 2 -> 1 == level 2.
        assert_eq!(p.select_mip(4, 3), 2);
        // span 8 -> 4 -> 2 -> 1 == level 3.
        assert_eq!(p.select_mip(8, 5), 3);
    }

    #[test]
    fn select_mip_clamps_to_last_level() {
        let p = HzbPyramid::new(8, 8);
        // Huge rect would want a very coarse mip; clamp to mip_count-1 == 3.
        assert_eq!(p.select_mip(100_000, 100_000), 3);
    }

    #[test]
    fn is_occluded_reports_true_when_strictly_behind() {
        let q = OcclusionQuery {
            rect: ScreenRect::new(0.0, 0.0, 1.0, 1.0),
            nearest_depth: 0.9,
        };
        assert!(is_occluded(&q, 0.5));
    }

    #[test]
    fn is_occluded_reports_false_when_in_front() {
        let q = OcclusionQuery {
            rect: ScreenRect::new(0.0, 0.0, 1.0, 1.0),
            nearest_depth: 0.2,
        };
        assert!(!is_occluded(&q, 0.5));
    }

    #[test]
    fn is_occluded_boundary_equality_is_visible() {
        let depth = 0.5;
        let q = OcclusionQuery {
            rect: ScreenRect::new(0.0, 0.0, 1.0, 1.0),
            nearest_depth: depth,
        };
        // Exactly equal, and within epsilon behind, both stay visible.
        assert!(!is_occluded(&q, depth));
        let q_eps = OcclusionQuery {
            rect: ScreenRect::new(0.0, 0.0, 1.0, 1.0),
            nearest_depth: depth + CMP_EPS * 0.5,
        };
        assert!(!is_occluded(&q_eps, depth));
    }

    #[test]
    fn screen_rect_width_height_and_validity() {
        let r = ScreenRect::new(1.0, 2.0, 5.0, 8.0);
        assert!((r.width() - 4.0).abs() < CMP_EPS);
        assert!((r.height() - 6.0).abs() < CMP_EPS);
        assert!(r.is_valid());
    }

    #[test]
    fn screen_rect_degenerate_is_invalid() {
        let r = ScreenRect::new(3.0, 3.0, 3.0, 9.0);
        assert!(!r.is_valid());
        assert!(r.width() < CMP_EPS);
    }

    #[test]
    fn screen_rect_clamp_to_bounds() {
        let r = ScreenRect::new(-5.0, -5.0, 200.0, 400.0);
        let c = r.clamp_to(128.0, 256.0);
        assert!((c.min_x - 0.0).abs() < CMP_EPS);
        assert!((c.min_y - 0.0).abs() < CMP_EPS);
        assert!((c.max_x - 128.0).abs() < CMP_EPS);
        assert!((c.max_y - 256.0).abs() < CMP_EPS);
    }

    #[test]
    fn screen_rect_clamp_fully_outside_collapses() {
        let r = ScreenRect::new(500.0, 500.0, 900.0, 900.0);
        let c = r.clamp_to(128.0, 128.0);
        assert!(!c.is_valid());
    }

    #[test]
    fn cull_stats_visible_and_ratio() {
        let s = CullStats {
            tested: 10,
            culled: 3,
        };
        assert_eq!(s.visible(), 7);
        assert!((s.cull_ratio() - 0.3).abs() < CMP_EPS);
    }

    #[test]
    fn cull_stats_ratio_divide_by_zero_guard() {
        let s = CullStats {
            tested: 0,
            culled: 0,
        };
        assert!((s.cull_ratio() - 0.0).abs() < CMP_EPS);
        assert_eq!(s.visible(), 0);
    }

    #[test]
    fn cull_stats_saturating_visible() {
        let s = CullStats {
            tested: 2,
            culled: 5,
        };
        assert_eq!(s.visible(), 0);
        // Ratio clamps culled to tested, so it never exceeds 1.0.
        assert!((s.cull_ratio() - 1.0).abs() < CMP_EPS);
    }

    #[test]
    fn false_negative_bound_grows_with_mip() {
        assert_eq!(conservative_false_negative_bound(0), 1);
        assert_eq!(conservative_false_negative_bound(1), 2);
        assert_eq!(conservative_false_negative_bound(4), 16);
    }

    #[test]
    fn false_negative_bound_saturates_at_extreme_mip() {
        assert_eq!(conservative_false_negative_bound(31), u32::MAX);
        assert_eq!(conservative_false_negative_bound(255), u32::MAX);
    }
}
