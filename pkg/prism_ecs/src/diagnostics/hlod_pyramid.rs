//! HLOD proxy-pyramid structure and shown-proxy census (design §13.1 / §16.6).
//!
//! The World-Partition HLOD scheduler ([`Hlod`](crate::partition::hlod::Hlod))
//! owns a stack of [layers](crate::partition::hlod::HlodLayer), each coarser and
//! drawn farther out than the one below it, plus the set of proxies it is
//! currently showing. Its [`push_layer`](crate::partition::hlod::Hlod::push_layer)
//! constructor enforces only the hard geometric contract — `extent >= 2` and a
//! *strictly increasing* `draw_radius` per layer — so the concentric draw bands
//! are always disjoint and ordered.
//!
//! It does **not** enforce the one *semantic* convention a sane pyramid obeys:
//! that layers get **coarser** as they go out, i.e. `extent` never shrinks with
//! layer index. A farther layer with a *smaller* `extent` than a nearer one
//! aggregates less while drawing farther — backwards HLOD that defeats the
//! point of the hierarchy, and the constructor will happily accept it.
//!
//! This report walks the layer table once and, in the same pass, buckets the
//! currently shown proxies by level, so a tuning tool or CI gate can see at a
//! glance:
//!
//! * each layer's draw band geometry (`[band_inner_radius, draw_radius)`), the
//!   source-cell footprint one proxy covers (`extent³`), and whether its
//!   `extent` regresses against the nearer layer;
//! * how many proxies each level is currently showing and the total source-cell
//!   area those impostors stand in for; and
//! * a cross-check between the two independent data sources inside the scheduler
//!   — the layer table and the shown set — surfacing any shown proxy whose level
//!   is not a registered layer (an orphan that should never occur).
//!
//! It is a pure read of the scheduler (`O(layers + shown)`), owns no
//! [`World`](crate::world::World) state, and is deterministic.

use alloc::vec;
use alloc::vec::Vec;

use crate::partition::hlod::Hlod;

/// Per-layer audit of an [`Hlod`] pyramid, pairing the layer's raw geometry with
/// its derived draw band, its currently shown-proxy count, and the one
/// convention smell the constructor does not reject (design §13.1).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HlodLayerAudit {
    /// The layer's index in the pyramid (0 = finest, nearest).
    pub level: u32,
    /// Side length, in source cells, of the block one proxy on this layer
    /// aggregates.
    pub extent: i32,
    /// Outer edge of this layer's draw band, in source cells (Chebyshev
    /// distance from the viewpoint).
    pub draw_radius: i32,
    /// Inner edge of this layer's draw band: the previous (nearer) layer's
    /// `draw_radius`, or `0` for the finest layer.
    pub band_inner_radius: i32,
    /// Width of this layer's draw band (`draw_radius - band_inner_radius`),
    /// always positive because the constructor forces strictly increasing
    /// radii.
    pub band_width: i32,
    /// Number of source cells one proxy on this layer stands in for
    /// (`extent³`).
    pub footprint_cell_count: i64,
    /// Number of proxies this level is currently showing.
    pub shown_proxy_count: usize,
    /// Total source-cell area the level's shown proxies stand in for
    /// (`shown_proxy_count × footprint_cell_count`).
    pub shown_cell_coverage: i64,
    /// Whether this (farther) layer's `extent` is *smaller* than the previous
    /// nearer layer's — a backwards pyramid. Always `false` for the finest
    /// layer.
    pub extent_regresses: bool,
}

impl HlodLayerAudit {
    /// Whether the layer is currently showing at least one proxy.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.shown_proxy_count > 0
    }

    /// Whether the layer is free of the detected smell (its `extent` does not
    /// regress against the nearer layer).
    #[inline]
    pub fn is_regular(&self) -> bool {
        !self.extent_regresses
    }
}

/// Read-only audit of a whole [`Hlod`] proxy pyramid: layer geometry, the
/// coarsening convention, and a census of the currently shown proxies
/// (design §13.1 / §16.6).
#[derive(Clone, Debug)]
pub struct HlodPyramidAudit {
    /// Per-layer audits, finest first (index is the layer level).
    layers: Vec<HlodLayerAudit>,
    /// Shown proxies whose `level` is not a registered layer (should be `0`).
    orphan_shown_count: usize,
}

impl HlodPyramidAudit {
    /// Audit `hlod`. Read-only; walks the layer table once and buckets the
    /// currently shown proxies by level in the same pass (design §13.1).
    pub fn from_hlod(hlod: &Hlod) -> Self {
        let layers = hlod.layers();
        let layer_count = layers.len();

        // Bucket the shown proxies by level; count any whose level has no
        // registered layer as orphans (the two internal data sources
        // disagreeing — a bug the audit surfaces).
        let mut shown_per_level = vec![0usize; layer_count];
        let mut orphan_shown_count = 0usize;
        for id in hlod.shown() {
            let lvl = id.level as usize;
            if lvl < layer_count {
                shown_per_level[lvl] += 1;
            } else {
                orphan_shown_count += 1;
            }
        }

        let mut entries: Vec<HlodLayerAudit> = Vec::with_capacity(layer_count);
        let mut prev_extent: Option<i32> = None;
        let mut prev_radius = 0i32;
        for (index, layer) in layers.iter().enumerate() {
            let extent = layer.extent();
            let draw_radius = layer.draw_radius();
            let e = extent as i64;
            let footprint_cell_count = e * e * e;
            let shown_proxy_count = shown_per_level[index];
            let extent_regresses = prev_extent.is_some_and(|p| extent < p);

            entries.push(HlodLayerAudit {
                level: index as u32,
                extent,
                draw_radius,
                band_inner_radius: prev_radius,
                band_width: draw_radius - prev_radius,
                footprint_cell_count,
                shown_proxy_count,
                shown_cell_coverage: shown_proxy_count as i64 * footprint_cell_count,
                extent_regresses,
            });

            prev_extent = Some(extent);
            prev_radius = draw_radius;
        }

        Self {
            layers: entries,
            orphan_shown_count,
        }
    }

    /// The per-layer audits, finest first.
    #[inline]
    pub fn layers(&self) -> &[HlodLayerAudit] {
        &self.layers
    }

    /// Number of registered layers in the pyramid.
    #[inline]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Total proxies currently shown across every registered layer (orphans
    /// excluded).
    pub fn total_shown_proxy_count(&self) -> usize {
        self.layers.iter().map(|l| l.shown_proxy_count).sum()
    }

    /// Shown proxies whose level is not a registered layer. Should always be
    /// `0`; a non-zero value means the shown set and layer table disagree.
    #[inline]
    pub fn orphan_shown_count(&self) -> usize {
        self.orphan_shown_count
    }

    /// Whether any shown proxy references an unregistered level.
    #[inline]
    pub fn has_orphan_shown(&self) -> bool {
        self.orphan_shown_count > 0
    }

    /// Total source-cell area all shown proxies stand in for, summed across
    /// layers.
    pub fn total_shown_cell_coverage(&self) -> i64 {
        self.layers.iter().map(|l| l.shown_cell_coverage).sum()
    }

    /// Number of layers whose `extent` regresses against the nearer layer.
    pub fn extent_regression_count(&self) -> usize {
        self.layers.iter().filter(|l| l.extent_regresses).count()
    }

    /// Whether `extent` is monotonic non-decreasing from finest to coarsest (no
    /// farther layer aggregates a smaller block than a nearer one).
    #[inline]
    pub fn is_monotonic_extent(&self) -> bool {
        self.extent_regression_count() == 0
    }

    /// Whether any layer's `extent` regresses.
    #[inline]
    pub fn has_extent_regression(&self) -> bool {
        self.extent_regression_count() > 0
    }

    /// Whether the pyramid is well formed: `extent` monotonic and no orphan
    /// shown proxies.
    #[inline]
    pub fn is_well_formed(&self) -> bool {
        self.is_monotonic_extent() && !self.has_orphan_shown()
    }

    /// Number of layers currently showing at least one proxy.
    pub fn active_layer_count(&self) -> usize {
        self.layers.iter().filter(|l| l.is_active()).count()
    }

    /// The layer currently showing the most proxies. Ties resolve to the
    /// nearest (lowest) level; `None` when no proxy is shown.
    pub fn busiest_level(&self) -> Option<&HlodLayerAudit> {
        let mut best: Option<&HlodLayerAudit> = None;
        for layer in &self.layers {
            if layer.shown_proxy_count == 0 {
                continue;
            }
            match best {
                Some(b) if b.shown_proxy_count >= layer.shown_proxy_count => {}
                _ => best = Some(layer),
            }
        }
        best
    }

    /// The finest layer's `extent` (the smallest block aggregated), or `0` for
    /// an empty pyramid.
    pub fn finest_extent(&self) -> i32 {
        self.layers.first().map(|l| l.extent).unwrap_or(0)
    }

    /// The coarsest layer's `extent` (the largest block aggregated), or `0` for
    /// an empty pyramid.
    pub fn coarsest_extent(&self) -> i32 {
        self.layers.last().map(|l| l.extent).unwrap_or(0)
    }

    /// The outermost draw radius: the coarsest layer's `draw_radius`, i.e. how
    /// far HLOD reaches. Returns `0` for an empty pyramid.
    pub fn outer_draw_radius(&self) -> i32 {
        self.layers.last().map(|l| l.draw_radius).unwrap_or(0)
    }

    /// Look up a layer audit by its level. `None` when out of range.
    pub fn entry(&self, level: u32) -> Option<&HlodLayerAudit> {
        self.layers.get(level as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partition::cell::CellCoord;
    use crate::partition::hlod::Hlod;

    #[test]
    fn empty_pyramid_is_vacuously_well_formed() {
        let hlod = Hlod::new();
        let audit = HlodPyramidAudit::from_hlod(&hlod);
        assert_eq!(audit.layer_count(), 0);
        assert!(audit.is_well_formed());
        assert_eq!(audit.total_shown_proxy_count(), 0);
        assert!(audit.busiest_level().is_none());
        assert_eq!(audit.finest_extent(), 0);
        assert_eq!(audit.coarsest_extent(), 0);
        assert_eq!(audit.outer_draw_radius(), 0);
    }

    #[test]
    fn regular_pyramid_geometry_is_computed() {
        let mut hlod = Hlod::new();
        hlod.push_layer(4, 8).push_layer(8, 24).push_layer(16, 64);
        let audit = HlodPyramidAudit::from_hlod(&hlod);
        assert_eq!(audit.layer_count(), 3);
        assert!(audit.is_well_formed());
        assert!(audit.is_monotonic_extent());
        assert!(!audit.has_extent_regression());

        let b0 = audit.entry(0).unwrap();
        assert_eq!(b0.level, 0);
        assert_eq!(b0.extent, 4);
        assert_eq!(b0.draw_radius, 8);
        assert_eq!(b0.band_inner_radius, 0);
        assert_eq!(b0.band_width, 8);
        assert_eq!(b0.footprint_cell_count, 64); // 4³

        let b1 = audit.entry(1).unwrap();
        assert_eq!(b1.band_inner_radius, 8);
        assert_eq!(b1.band_width, 16); // 24 - 8
        assert_eq!(b1.footprint_cell_count, 512); // 8³
        assert!(!b1.extent_regresses);

        let b2 = audit.entry(2).unwrap();
        assert_eq!(b2.band_inner_radius, 24);
        assert_eq!(b2.band_width, 40); // 64 - 24
        assert_eq!(b2.footprint_cell_count, 4096); // 16³

        assert_eq!(audit.finest_extent(), 4);
        assert_eq!(audit.coarsest_extent(), 16);
        assert_eq!(audit.outer_draw_radius(), 64);
    }

    #[test]
    fn extent_regression_is_detected() {
        // Farther layer aggregates a *smaller* block (2) than the nearer (4):
        // a backwards pyramid the constructor still accepts.
        let mut hlod = Hlod::new();
        hlod.push_layer(4, 8).push_layer(2, 16);
        let audit = HlodPyramidAudit::from_hlod(&hlod);
        assert_eq!(audit.extent_regression_count(), 1);
        assert!(!audit.is_monotonic_extent());
        assert!(audit.has_extent_regression());
        assert!(audit.entry(1).unwrap().extent_regresses);
        assert!(!audit.entry(0).unwrap().extent_regresses);
        assert!(!audit.is_well_formed());
    }

    #[test]
    fn shown_proxies_are_bucketed_by_level() {
        let mut hlod = Hlod::new();
        hlod.push_layer(4, 8).push_layer(8, 24);
        // Nothing resident: every candidate proxy in-band is shown.
        let delta = hlod.resolve(CellCoord::new(0, 0, 0), |_| false);
        assert!(!delta.to_show.is_empty());
        let total_shown = hlod.shown_count();
        assert!(total_shown > 0);

        let audit = HlodPyramidAudit::from_hlod(&hlod);
        assert_eq!(audit.total_shown_proxy_count(), total_shown);
        assert!(!audit.has_orphan_shown());
        assert_eq!(audit.orphan_shown_count(), 0);

        // Per-level counts sum to the whole shown set.
        let summed: usize = audit.layers().iter().map(|l| l.shown_proxy_count).sum();
        assert_eq!(summed, total_shown);

        // Coverage equals each level's shown count times its footprint.
        let expected: i64 = audit
            .layers()
            .iter()
            .map(|l| l.shown_proxy_count as i64 * l.footprint_cell_count)
            .sum();
        assert_eq!(audit.total_shown_cell_coverage(), expected);

        // At least one layer is active, and the busiest is a shown layer.
        assert!(audit.active_layer_count() >= 1);
        let busiest = audit.busiest_level().unwrap();
        assert!(busiest.shown_proxy_count > 0);
        for layer in audit.layers() {
            assert!(busiest.shown_proxy_count >= layer.shown_proxy_count);
        }
    }

    #[test]
    fn cleared_scheduler_shows_nothing() {
        let mut hlod = Hlod::new();
        hlod.push_layer(4, 8).push_layer(8, 24);
        hlod.resolve(CellCoord::new(0, 0, 0), |_| false);
        hlod.clear();
        let audit = HlodPyramidAudit::from_hlod(&hlod);
        assert_eq!(audit.total_shown_proxy_count(), 0);
        assert_eq!(audit.active_layer_count(), 0);
        assert!(audit.busiest_level().is_none());
        assert_eq!(audit.total_shown_cell_coverage(), 0);
        // Structure is still intact and well formed.
        assert_eq!(audit.layer_count(), 2);
        assert!(audit.is_well_formed());
    }

    #[test]
    fn entry_out_of_range_is_none() {
        let mut hlod = Hlod::new();
        hlod.push_layer(4, 8);
        let audit = HlodPyramidAudit::from_hlod(&hlod);
        assert!(audit.entry(1).is_none());
        assert!(audit.entry(99).is_none());
    }
}
