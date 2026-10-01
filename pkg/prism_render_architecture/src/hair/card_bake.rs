//! Groom-to-card atlas auto-bake LOD descriptor contract (design doc §8.6
//! item22).
//!
//! At coarse distance a strand-based groom is far too expensive to raster
//! per-fibre, so production hair engines fall back to **hair cards**: a small
//! number of flat, camera-facing quads textured with a *baked* image of a bundle
//! of strands. The bake takes a cluster of real render strands and renders them
//! once, offline or on a scheduler, into a shared **card atlas** — one rectangle
//! of texels per cluster, holding the strand bundle's appearance so the runtime
//! can draw one textured quad instead of thousands of lines. This is exactly the
//! card/imposter LOD step used by `UE5` Groom's "hair cards" and by the
//! `TressFX`-style card pipelines; the far mesh shell in [`crate::hair::mesh_shell`]
//! is the next rung coarser.
//!
//! This module owns the **pure mapping / contract** half of that bake, not the
//! bake itself. The actual rasterisation of strands into texels is a device /
//! scheduler job; here we deterministically decide, from cheap per-cluster
//! descriptors (bounding-box aspect and screen coverage), *how big* each
//! cluster's card should be, *where* it lands in the atlas, and *which mip* of
//! the card chain that footprint corresponds to. The output is a flat list of
//! [`CardAtlasAllocation`] rectangles plus a per-page bucketing
//! ([`AtlasPageBins`]), in the same array-in / array-out, `golden`-comparable,
//! panic-free style as [`crate::hair::line_coverage`] and
//! [`crate::hair::melanin`]. No device state, no transcendental math.
//!
//! Three co-located channels. Each allocated rectangle addresses **all three**
//! card channel maps at once (see [`CARD_CHANNELS`]): a *density / albedo*
//! channel (the bundle's lit colour and alpha coverage), a *flow / tangent*
//! channel (the dominant strand direction, so the card shades anisotropically
//! like real hair instead of a flat decal), and a *depth* channel (a parallax /
//! self-occlusion offset so the flat card reads with hair thickness). The three
//! maps share one layout, so one `u`,`v` rectangle per cluster is sufficient for
//! all of them; the bake fills the three channel textures at the same texels.
//!
//! Resolution follows coverage. A card seen large on screen needs more texels
//! than one seen tiny, and screen *coverage* is an **area** fraction, so the
//! card's linear side scales with its square root: `side = base * sqrt(coverage)`.
//! That single `sqrt` (allowed; no transcendental) is the only non-trivial math
//! here. The chosen footprint is then mapped to a mip of the full-resolution
//! card chain by integer halving in [`mip_for_footprint`], giving the LOD
//! descriptor the runtime samples with.
//!
//! Packing. Allocation is a deterministic single-pass **shelf / row** packer
//! (the classic rectangle-atlas heuristic): cards are laid left-to-right into a
//! shelf, a shelf wraps to a new row when the page width is exhausted, and the
//! page wraps to the next atlas page when the page height is exhausted. Input
//! order is preserved, so the same requests always produce the same atlas.
//! Degenerate clusters (non-finite / zero bounding box, zero coverage) are
//! skipped as culled, over-sized clusters that cannot fit a page are skipped as
//! over-capacity, and once the page budget [`AtlasConfig::max_pages`] is
//! exhausted the remaining clusters are skipped — never a panic.

use alloc::vec::Vec;

/// Number of co-located channel maps every card allocation addresses: a
/// density/albedo map, a flow/tangent map, and a depth/parallax map. They share
/// one atlas layout, so a single [`CardAtlasAllocation`] rectangle indexes all
/// three.
pub const CARD_CHANNELS: usize = 3;

/// Fixed-point / sizing configuration for one card atlas.
///
/// A page is a `page_width` x `page_height` texel image; the atlas may span up
/// to `max_pages` such pages. `padding` is the texel gutter kept around every
/// card (and around the page edge) so neighbouring bakes do not bleed into each
/// other under bilinear filtering or mip reduction. `base_texels` is the linear
/// resolution a card at full screen coverage (`coverage = 1`) is baked at, and
/// `max_mip` caps the descriptor's mip index. Zero / nonsensical page or base
/// sizes are repaired by [`AtlasConfig::sanitized`] before use, so packing never
/// divides by zero or panics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtlasConfig {
    /// Atlas page width in texels; sanitised to `>= 1`.
    pub page_width: u32,
    /// Atlas page height in texels; sanitised to `>= 1`.
    pub page_height: u32,
    /// Texel gutter kept around each card and the page edge.
    pub padding: u32,
    /// Linear resolution of a card baked at full (`coverage = 1`) coverage;
    /// sanitised to `>= 1`.
    pub base_texels: u32,
    /// Maximum number of atlas pages; clusters past this budget are skipped.
    pub max_pages: u32,
    /// Upper bound on the mip index reported in a [`CardAtlasAllocation`].
    pub max_mip: u8,
}

impl Default for AtlasConfig {
    fn default() -> Self {
        Self {
            page_width: 2048,
            page_height: 2048,
            padding: 2,
            base_texels: 512,
            max_pages: 4,
            max_mip: 12,
        }
    }
}

impl AtlasConfig {
    /// Builds a config from explicit fields.
    #[must_use]
    pub const fn new(
        page_width: u32,
        page_height: u32,
        padding: u32,
        base_texels: u32,
        max_pages: u32,
        max_mip: u8,
    ) -> Self {
        Self {
            page_width,
            page_height,
            padding,
            base_texels,
            max_pages,
            max_mip,
        }
    }

    /// This config with degenerate sizes repaired: page dimensions and
    /// `base_texels` are forced to at least `1` so no division by a zero page
    /// size or a zero base can occur. `padding`, `max_pages` and `max_mip` are
    /// kept as given (a padding wider than the page simply leaves no usable room,
    /// which the packer treats as "nothing fits", and `max_pages == 0` disables
    /// the atlas entirely).
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            page_width: self.page_width.max(1),
            page_height: self.page_height.max(1),
            padding: self.padding,
            base_texels: self.base_texels.max(1),
            max_pages: self.max_pages,
            max_mip: self.max_mip,
        }
    }

    /// Usable interior width of a page after the per-edge gutter, saturating to
    /// `0` when the padding is wider than the page.
    #[must_use]
    fn usable_width(self) -> u32 {
        self.page_width
            .saturating_sub(self.padding.saturating_mul(2))
    }

    /// Usable interior height of a page after the per-edge gutter, saturating to
    /// `0` when the padding is wider than the page.
    #[must_use]
    fn usable_height(self) -> u32 {
        self.page_height
            .saturating_sub(self.padding.saturating_mul(2))
    }
}

/// A cheap per-cluster descriptor driving its card bake, gathered during
/// draw-prep: the cluster's card-space bounding-box aspect and its on-screen
/// coverage. No strand geometry is needed here — the heavy data is what the
/// offline / scheduler bake consumes once this contract has decided the card's
/// size, slot, and mip.
///
/// `bbox_width` / `bbox_height` give the card quad's aspect ratio (only the
/// ratio is used, so any consistent unit works); non-finite or non-positive
/// extents mark a degenerate cluster that [`CardBakeRequest::footprint`] culls.
/// `screen_coverage` is the fraction of the screen the cluster covers, clamped
/// to `[0, 1]`; `0` (or non-finite) coverage culls the cluster as not worth a
/// card this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CardBakeRequest {
    /// Stable identity of the strand cluster this card bakes.
    pub cluster_id: u32,
    /// Card-space bounding-box width; only its ratio to `bbox_height` matters.
    pub bbox_width: f32,
    /// Card-space bounding-box height; only its ratio to `bbox_width` matters.
    pub bbox_height: f32,
    /// On-screen coverage as an area fraction in `[0, 1]` (sanitised).
    pub screen_coverage: f32,
}

impl CardBakeRequest {
    /// Builds a bake request from explicit fields.
    #[must_use]
    pub const fn new(
        cluster_id: u32,
        bbox_width: f32,
        bbox_height: f32,
        screen_coverage: f32,
    ) -> Self {
        Self {
            cluster_id,
            bbox_width,
            bbox_height,
            screen_coverage,
        }
    }

    /// Coverage clamped to a finite value in `[0, 1]` (negative / non-finite ->
    /// `0`), without any floating-point equality test.
    #[must_use]
    fn sanitized_coverage(self) -> f32 {
        if self.screen_coverage.is_finite() && self.screen_coverage > 0.0 {
            self.screen_coverage.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// The desired card footprint in texels, `Some((width, height))`, or `None`
    /// when the cluster is culled.
    ///
    /// The linear side scales with the square root of coverage
    /// (`side = base_texels * sqrt(coverage)`, since coverage is an area
    /// fraction), then the shorter axis is reduced by the bounding-box aspect so
    /// the card is not stretched. A cluster with non-finite / non-positive
    /// bounding-box extents, or zero coverage, yields `None` (culled); otherwise
    /// each side is at least `1` texel so a tiny-but-visible card still gets a
    /// slot.
    #[must_use]
    pub fn footprint(self, config: AtlasConfig) -> Option<(u32, u32)> {
        let cfg = config.sanitized();

        let bw = self.bbox_width;
        let bh = self.bbox_height;
        let bw_ok = bw.is_finite() && bw > 0.0;
        let bh_ok = bh.is_finite() && bh > 0.0;
        if !bw_ok || !bh_ok {
            return None;
        }

        let coverage = self.sanitized_coverage();
        // Area fraction -> linear fraction via sqrt (allowed; non-transcendental).
        let linear = coverage.sqrt();
        let max_side = (cfg.base_texels as f32) * linear;
        if max_side <= 0.0 {
            return None;
        }

        // Preserve the bounding-box aspect: the longer box axis takes the full
        // `max_side`, the shorter axis is scaled down by the aspect ratio.
        let (w_f, h_f) = if bw >= bh {
            (max_side, max_side * (bh / bw))
        } else {
            (max_side * (bw / bh), max_side)
        };

        let w = texels_at_least_one(w_f);
        let h = texels_at_least_one(h_f);
        Some((w, h))
    }
}

/// Rounds a non-negative texel count to the nearest `u32`, flooring the result
/// to at least `1` whenever the input was strictly positive, and to `0` only
/// when the input was `0` / non-finite. Keeps a sub-texel-but-visible card from
/// collapsing to an empty rectangle.
#[must_use]
fn texels_at_least_one(value: f32) -> u32 {
    let positive = value.is_finite() && value > 0.0;
    if !positive {
        return 0;
    }
    let rounded = value.round();
    if rounded < 1.0 {
        1
    } else {
        rounded as u32
    }
}

/// The mip index a footprint corresponds to within a card's full-resolution
/// chain: how many times the footprint's longest side can be doubled before it
/// reaches `base_texels`.
///
/// A card baked at `base_texels` is mip `0`; one baked at half that side is mip
/// `1`, a quarter is mip `2`, and so on, matching a standard power-of-two mip
/// chain. The count is computed by integer doubling (no `log2`), and clamped to
/// `max_mip`. A zero footprint reports `max_mip` (the coarsest level).
#[must_use]
pub fn mip_for_footprint(max_dim: u32, base_texels: u32, max_mip: u8) -> u8 {
    if max_dim == 0 {
        return max_mip;
    }
    let base = base_texels.max(1);
    let mut mip: u8 = 0;
    let mut size = max_dim;
    while size.saturating_mul(2) <= base && mip < max_mip {
        size = size.saturating_mul(2);
        mip += 1;
    }
    mip
}

/// One cluster's reserved rectangle in the card atlas, and therefore the
/// cluster-to-card mapping: `cluster_id` names the strand cluster the card bakes
/// from, `page` names the atlas page, `(u0, v0)`-`(u1, v1)` is the normalised
/// `[0, 1]` texel rectangle on that page, and `mip` is the card-chain level the
/// footprint sits at. The same rectangle addresses all [`CARD_CHANNELS`] channel
/// maps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CardAtlasAllocation {
    /// Strand cluster this card bakes from.
    pub cluster_id: u32,
    /// Atlas page the rectangle lives on.
    pub page: u32,
    /// Left edge of the rectangle in normalised `[0, 1]` page coordinates.
    pub u0: f32,
    /// Top edge of the rectangle in normalised `[0, 1]` page coordinates.
    pub v0: f32,
    /// Right edge of the rectangle in normalised `[0, 1]` page coordinates.
    pub u1: f32,
    /// Bottom edge of the rectangle in normalised `[0, 1]` page coordinates.
    pub v1: f32,
    /// Mip level of the card chain this footprint corresponds to.
    pub mip: u8,
}

/// Card allocations bucketed by atlas page.
///
/// The bake / upload side consumes one page at a time (one atlas texture upload
/// per page), so grouping the flat allocation list by `page` matches how the
/// atlas is actually populated. Buckets preserve the packer's placement order
/// within a page, keeping the layout deterministic. This mirrors the per-path
/// bucketing in [`crate::virtual_geometry::bins`]: `push` routes one allocation,
/// `bucket` reads one page, and `total` / `is_empty` summarise the whole atlas.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AtlasPageBins {
    /// One bucket per atlas page, indexed by page number.
    pub pages: Vec<Vec<CardAtlasAllocation>>,
}

impl AtlasPageBins {
    /// Routes one allocation into its page bucket, growing the page list with
    /// empty buckets as needed so a sparse set of pages never panics.
    pub fn push(&mut self, allocation: CardAtlasAllocation) {
        let page = allocation.page as usize;
        if page >= self.pages.len() {
            self.pages.resize(page + 1, Vec::new());
        }
        self.pages[page].push(allocation);
    }

    /// The allocations on a given page in placement order, or an empty slice when
    /// the page index is out of range (never panics).
    #[must_use]
    pub fn bucket(&self, page: u32) -> &[CardAtlasAllocation] {
        self.pages.get(page as usize).map_or(&[], Vec::as_slice)
    }

    /// Number of atlas pages that have at least been allocated a bucket.
    #[must_use]
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Total number of card allocations across every page.
    #[must_use]
    pub fn total(&self) -> usize {
        self.pages.iter().map(Vec::len).sum()
    }

    /// Returns `true` when no card landed on any page.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pages.iter().all(Vec::is_empty)
    }
}

/// Deterministically packs a set of bake requests into card atlas rectangles.
///
/// A single left-to-right, top-to-bottom shelf packer: each request's
/// [`CardBakeRequest::footprint`] is placed on the current shelf; when it would
/// overflow the page width the shelf wraps to a new row, and when a row would
/// overflow the page height the atlas advances to the next page. Input order is
/// preserved. Culled requests (degenerate box / zero coverage) and requests
/// whose footprint cannot fit a page's usable interior are skipped; once
/// [`AtlasConfig::max_pages`] pages are consumed the remaining requests are
/// skipped. The function never panics and returns the rectangles in placement
/// order, which is also the cluster-to-card mapping.
#[must_use]
pub fn pack_cards(requests: &[CardBakeRequest], config: AtlasConfig) -> Vec<CardAtlasAllocation> {
    let cfg = config.sanitized();
    let mut out = Vec::new();

    let usable_w = cfg.usable_width();
    let usable_h = cfg.usable_height();
    if usable_w == 0 || usable_h == 0 || cfg.max_pages == 0 {
        return out;
    }

    let pad = cfg.padding;
    let right_bound = cfg.page_width.saturating_sub(pad);
    let bottom_bound = cfg.page_height.saturating_sub(pad);
    let page_w = cfg.page_width as f32;
    let page_h = cfg.page_height as f32;

    let mut page: u32 = 0;
    let mut cursor_x = pad;
    let mut shelf_y = pad;
    let mut shelf_h: u32 = 0;

    for request in requests {
        let Some((w, h)) = request.footprint(cfg) else {
            continue;
        };
        // Over-capacity: cannot fit a page's usable interior even when alone.
        if w > usable_w || h > usable_h {
            continue;
        }

        // Wrap to a new shelf when the card overflows the remaining row width.
        if cursor_x.saturating_add(w) > right_bound {
            cursor_x = pad;
            shelf_y = shelf_y.saturating_add(shelf_h).saturating_add(pad);
            shelf_h = 0;
        }
        // Wrap to a new page when the shelf overflows the remaining page height.
        if shelf_y.saturating_add(h) > bottom_bound {
            page = page.saturating_add(1);
            cursor_x = pad;
            shelf_y = pad;
            shelf_h = 0;
        }
        // Page budget exhausted: nothing more can be placed.
        if page >= cfg.max_pages {
            break;
        }

        let x0 = cursor_x;
        let y0 = shelf_y;
        let x1 = x0.saturating_add(w);
        let y1 = y0.saturating_add(h);

        let mip = mip_for_footprint(w.max(h), cfg.base_texels, cfg.max_mip);
        out.push(CardAtlasAllocation {
            cluster_id: request.cluster_id,
            page,
            u0: (x0 as f32) / page_w,
            v0: (y0 as f32) / page_h,
            u1: (x1 as f32) / page_w,
            v1: (y1 as f32) / page_h,
            mip,
        });

        cursor_x = x1.saturating_add(pad);
        shelf_h = shelf_h.max(h);
    }

    out
}

/// Packs bake requests and buckets the resulting allocations by atlas page.
///
/// This is the bin-oriented entry point mirroring
/// [`bin_cut`](crate::virtual_geometry::bins::bin_cut): it runs [`pack_cards`]
/// and routes each rectangle into its [`AtlasPageBins`] page bucket, so an empty
/// request slice yields empty bins without panicking.
#[must_use]
pub fn bin_card_bake(requests: &[CardBakeRequest], config: AtlasConfig) -> AtlasPageBins {
    let mut bins = AtlasPageBins::default();
    for allocation in pack_cards(requests, config) {
        bins.push(allocation);
    }
    bins
}

/// Looks up the card rectangle baked for a given cluster, or `None` when the
/// cluster was culled / skipped (never allocated). The first match in placement
/// order is returned, which is the natural cluster-to-card mapping query.
#[must_use]
pub fn cluster_tile(
    allocations: &[CardAtlasAllocation],
    cluster_id: u32,
) -> Option<CardAtlasAllocation> {
    allocations
        .iter()
        .copied()
        .find(|allocation| allocation.cluster_id == cluster_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const EPS: f32 = 1.0e-6;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    /// A small, exact config: 100x100 pages, no padding, base 64, 2 pages.
    const CFG: AtlasConfig = AtlasConfig::new(100, 100, 0, 64, 2, 8);

    fn req(cluster_id: u32, w: f32, h: f32, coverage: f32) -> CardBakeRequest {
        CardBakeRequest::new(cluster_id, w, h, coverage)
    }

    #[test]
    fn empty_requests_pack_empty() {
        let allocs = pack_cards(&[], CFG);
        assert!(allocs.is_empty());
        let bins = bin_card_bake(&[], CFG);
        assert!(bins.is_empty());
        assert_eq!(bins.total(), 0);
    }

    #[test]
    fn single_full_coverage_card_sits_at_origin() {
        // coverage 1 -> side = base = 64 texels on a 100-texel page.
        let allocs = pack_cards(&[req(7, 1.0, 1.0, 1.0)], CFG);
        assert_eq!(allocs.len(), 1);
        let a = allocs[0];
        assert_eq!(a.cluster_id, 7);
        assert_eq!(a.page, 0);
        assert!(close(a.u0, 0.0));
        assert!(close(a.v0, 0.0));
        assert!(close(a.u1, 0.64));
        assert!(close(a.v1, 0.64));
        assert_eq!(a.mip, 0);
    }

    #[test]
    fn square_aspect_is_preserved() {
        let a = pack_cards(&[req(0, 2.0, 1.0, 1.0)], CFG)[0];
        // width 64, height 32 -> u spans twice the v.
        assert!(close(a.u1 - a.u0, 0.64));
        assert!(close(a.v1 - a.v0, 0.32));
    }

    #[test]
    fn second_card_follows_on_same_shelf() {
        let allocs = pack_cards(&[req(0, 1.0, 1.0, 0.25), req(1, 1.0, 1.0, 0.25)], CFG);
        // coverage 0.25 -> side = 64 * sqrt(0.25) = 32 texels.
        assert_eq!(allocs.len(), 2);
        assert_eq!(allocs[0].page, 0);
        assert_eq!(allocs[1].page, 0);
        assert!(close(allocs[0].u0, 0.0));
        assert!(close(allocs[1].u0, 0.32));
        // Same shelf: identical v extents.
        assert!(close(allocs[0].v0, allocs[1].v0));
        assert!(close(allocs[0].v1, allocs[1].v1));
    }

    #[test]
    fn shelf_wraps_to_next_row() {
        // Four 32-texel cards: three fit the 100-texel width, the fourth wraps.
        let r = req(0, 1.0, 1.0, 0.25);
        let allocs = pack_cards(&[r, r, r, r], CFG);
        assert_eq!(allocs.len(), 4);
        assert!(close(allocs[0].v0, 0.0));
        assert!(close(allocs[2].v0, 0.0));
        // Fourth card dropped to the next shelf (v advanced).
        assert!(allocs[3].v0 > allocs[0].v0);
        assert!(close(allocs[3].u0, 0.0));
    }

    #[test]
    fn page_and_capacity_overflow_skips_rest() {
        // 50-texel cards on a 100x100, 1-page atlas: 2 per shelf, 2 shelves = 4.
        let cfg = AtlasConfig::new(100, 100, 0, 64, 1, 8);
        // coverage so side ~= 50: (50/64)^2 ~= 0.61.
        let r = req(0, 1.0, 1.0, 0.61);
        let allocs = pack_cards(&[r, r, r, r, r, r], cfg);
        // Only one page's worth is placed; the rest are skipped (not panicking).
        assert_eq!(allocs[0].page, 0);
        for a in &allocs {
            assert_eq!(a.page, 0);
        }
        assert!(allocs.len() <= 4);
    }

    #[test]
    fn oversized_card_is_skipped() {
        // A tiny 4-texel page cannot hold a 64-texel full-coverage card.
        let cfg = AtlasConfig::new(4, 4, 0, 64, 2, 8);
        let allocs = pack_cards(&[req(0, 1.0, 1.0, 1.0)], cfg);
        assert!(allocs.is_empty());
    }

    #[test]
    fn degenerate_boxes_and_zero_coverage_are_culled() {
        let reqs = [
            req(0, 0.0, 1.0, 1.0),           // zero width
            req(1, 1.0, -2.0, 1.0),          // negative height
            req(2, f32::NAN, 1.0, 1.0),      // NaN width
            req(3, 1.0, f32::INFINITY, 1.0), // inf height
            req(4, 1.0, 1.0, 0.0),           // zero coverage
            req(5, 1.0, 1.0, f32::NAN),      // NaN coverage
            req(6, 1.0, 1.0, -0.5),          // negative coverage
        ];
        let allocs = pack_cards(&reqs, CFG);
        assert!(allocs.is_empty());
    }

    #[test]
    fn mip_increases_as_coverage_shrinks() {
        // base 64: side 64 -> mip 0, side 32 -> mip 1, side 16 -> mip 2.
        let full = pack_cards(&[req(0, 1.0, 1.0, 1.0)], CFG)[0];
        let half = pack_cards(&[req(0, 1.0, 1.0, 0.25)], CFG)[0];
        let quarter = pack_cards(&[req(0, 1.0, 1.0, 0.0625)], CFG)[0];
        assert_eq!(full.mip, 0);
        assert_eq!(half.mip, 1);
        assert_eq!(quarter.mip, 2);
    }

    #[test]
    fn mip_for_footprint_matches_power_of_two_chain() {
        assert_eq!(mip_for_footprint(64, 64, 8), 0);
        assert_eq!(mip_for_footprint(32, 64, 8), 1);
        assert_eq!(mip_for_footprint(16, 64, 8), 2);
        assert_eq!(mip_for_footprint(1, 64, 8), 6);
        // Clamped to max_mip.
        assert_eq!(mip_for_footprint(1, 1024, 3), 3);
        // Zero footprint -> coarsest.
        assert_eq!(mip_for_footprint(0, 64, 8), 8);
    }

    #[test]
    fn uv_rectangles_stay_in_unit_range() {
        let mut reqs = Vec::new();
        for i in 0..64u32 {
            let coverage = ((i % 8) as f32) * 0.1 + 0.05;
            reqs.push(req(i, 1.0 + (i % 3) as f32, 1.0, coverage));
        }
        for a in pack_cards(&reqs, CFG) {
            assert!(a.u0.is_finite() && a.v0.is_finite());
            assert!(a.u1.is_finite() && a.v1.is_finite());
            assert!(a.u0 >= 0.0 && a.u0 <= 1.0);
            assert!(a.v0 >= 0.0 && a.v0 <= 1.0);
            assert!(a.u1 >= 0.0 && a.u1 <= 1.0);
            assert!(a.v1 >= 0.0 && a.v1 <= 1.0);
            assert!(a.u1 >= a.u0);
            assert!(a.v1 >= a.v0);
        }
    }

    #[test]
    fn bins_bucket_by_page_and_summarise() {
        // 50-texel cards on a 100x100, 2-page atlas spill onto a second page.
        let cfg = AtlasConfig::new(100, 100, 0, 64, 2, 8);
        let r = req(0, 1.0, 1.0, 0.61);
        let many = [r, r, r, r, r, r, r, r];
        let bins = bin_card_bake(&many, cfg);
        assert!(!bins.is_empty());
        assert_eq!(bins.total(), bins.pages.iter().map(Vec::len).sum::<usize>());
        // Each bucket's allocations report the matching page index.
        for page in 0..bins.page_count() as u32 {
            for a in bins.bucket(page) {
                assert_eq!(a.page, page);
            }
        }
        // Out-of-range page is an empty slice, not a panic.
        assert!(bins.bucket(9999).is_empty());
    }

    #[test]
    fn bins_push_grows_sparse_pages() {
        let mut bins = AtlasPageBins::default();
        bins.push(CardAtlasAllocation {
            cluster_id: 1,
            page: 3,
            u0: 0.0,
            v0: 0.0,
            u1: 0.1,
            v1: 0.1,
            mip: 0,
        });
        assert_eq!(bins.page_count(), 4);
        assert_eq!(bins.total(), 1);
        assert!(bins.bucket(0).is_empty());
        assert_eq!(bins.bucket(3).len(), 1);
    }

    #[test]
    fn cluster_tile_maps_cluster_to_rectangle() {
        let allocs = pack_cards(&[req(10, 1.0, 1.0, 0.25), req(20, 1.0, 1.0, 0.25)], CFG);
        let found = cluster_tile(&allocs, 20).expect("cluster 20 allocated");
        assert_eq!(found.cluster_id, 20);
        assert!(cluster_tile(&allocs, 999).is_none());
    }

    #[test]
    fn placement_order_is_deterministic() {
        let reqs = [
            req(2, 1.0, 1.0, 0.25),
            req(0, 1.0, 1.0, 0.25),
            req(1, 1.0, 1.0, 0.25),
        ];
        let first = pack_cards(&reqs, CFG);
        let second = pack_cards(&reqs, CFG);
        assert_eq!(first, second);
        let ids: Vec<u32> = first.iter().map(|a| a.cluster_id).collect();
        assert_eq!(ids, vec![2, 0, 1]);
    }

    #[test]
    fn zero_pages_or_degenerate_config_pack_empty() {
        let no_pages = AtlasConfig::new(100, 100, 0, 64, 0, 8);
        assert!(pack_cards(&[req(0, 1.0, 1.0, 1.0)], no_pages).is_empty());
        // Padding wider than the page leaves no usable interior.
        let all_padding = AtlasConfig::new(10, 10, 20, 64, 2, 8);
        assert!(pack_cards(&[req(0, 1.0, 1.0, 1.0)], all_padding).is_empty());
    }

    #[test]
    fn config_sanitize_repairs_zero_sizes() {
        let cfg = AtlasConfig::new(0, 0, 0, 0, 2, 8).sanitized();
        assert_eq!(cfg.page_width, 1);
        assert_eq!(cfg.page_height, 1);
        assert_eq!(cfg.base_texels, 1);
    }
}
