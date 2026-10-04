//! World Partition **HLOD** — hierarchical level-of-detail proxies (design §13.1).
//!
//! Cell streaming ([`cell`](crate::partition::cell)) keeps only the cells near
//! an interest source resident; everything beyond the unload ball is evicted
//! and simply *disappears*. In a large open world that pop-out is unacceptable:
//! a distant mountain range, skyline, or terrain shelf must stay visible even
//! though its full-detail cells are on disk. UE5 World Partition solves this
//! with **HLOD** (Hierarchical Level Of Detail): offline, groups of source
//! actors are baked into a single cheap *proxy* — a merged impostor mesh or a
//! statistical stand-in — and that proxy is drawn in place of the detail
//! whenever the detail is streamed out but the region is still within view.
//!
//! This module is the CPU-side *scheduler* for those proxies, exactly as
//! [`CellStreamer`](crate::partition::cell::CellStreamer) is the scheduler for
//! detail cells. It owns no [`World`](crate::world::World) state, bakes nothing,
//! and draws nothing: it decides, each frame, **which proxies should be shown
//! and which hidden**, and emits that as an [`HlodDelta`] the render/scene crate
//! turns into actual impostor draws.
//!
//! # Hierarchy of layers
//!
//! HLOD is *hierarchical*: several [layers](HlodLayer) nest, each coarser and
//! visible farther away than the one below it.
//!
//! * **Layer 0** (finest) aggregates a small block of source cells — e.g. a
//!   4×4×4 cube — into one proxy, shown in the near-to-mid distance band just
//!   beyond the streamed detail.
//! * **Layer 1** aggregates a larger block and is shown farther out, where even
//!   layer-0 proxies would be too dense to be worth drawing.
//! * …and so on. The coarsest layer typically covers the whole far horizon in a
//!   handful of proxies.
//!
//! Each layer owns an [`extent`](HlodLayer::extent) (the side length, in source
//! cells, of the block one proxy represents) and a
//! [`draw_radius`](HlodLayer::draw_radius) (how far, in source cells, that layer
//! is allowed to draw). Draw radii strictly increase with layer index, carving
//! the world into concentric **distance bands**: the gap between layer `L-1`'s
//! radius and layer `L`'s radius is exactly the band in which layer `L` draws.
//! A point in space is therefore represented by at most one proxy — the finest
//! layer whose band contains it — so proxies never stack or double-draw.
//!
//! # Visibility rule
//!
//! A proxy for layer `L`, HLOD cell `C`, is **shown** iff *all* hold:
//!
//! 1. the viewpoint's Chebyshev distance (in source cells) to `C`'s footprint
//!    falls in layer `L`'s band `[draw_radius(L-1), draw_radius(L))`
//!    (`draw_radius(-1)` ≡ `0`); and
//! 2. **none** of `C`'s source cells are currently resident — if any detail
//!    under the proxy is loaded, the detail is drawn and the proxy is
//!    suppressed, mirroring UE5 hiding an HLOD actor once its cell loads.
//!
//! Because residency itself hides proxies, the near band collapses naturally:
//! cells close enough to be streamed in are resident, so their proxies are
//! suppressed without any special-casing.
//!
//! # Determinism
//!
//! [`resolve`](Hlod::resolve) enumerates candidate proxies in a fixed grid order
//! and sorts both halves of the returned [`HlodDelta`] by
//! `(level, x, y, z)`, so the show/hide schedule is frame-stable and
//! independent of hash-map iteration order (design §14).

use alloc::vec::Vec;

use crate::collections::HashMap;
use crate::partition::cell::CellCoord;

/// One level of the HLOD hierarchy (design §13.1).
///
/// A layer merges every `extent × extent × extent` block of source cells into a
/// single proxy and is permitted to draw those proxies out to `draw_radius`
/// source cells from the viewpoint. Layers are registered finest-first with
/// strictly increasing `draw_radius`, so consecutive layers form disjoint
/// concentric distance bands (see the [module docs](self)).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HlodLayer {
    /// Side length, in source cells, of the block one proxy represents.
    /// Must be at least `2` (a 1-cell proxy would be the detail itself).
    extent: i32,
    /// Outer edge of this layer's draw band, in source cells (Chebyshev
    /// distance from the viewpoint). Must be strictly greater than the previous
    /// layer's `draw_radius`.
    draw_radius: i32,
}

impl HlodLayer {
    /// Side length, in source cells, of the block one proxy aggregates.
    #[inline]
    pub const fn extent(self) -> i32 {
        self.extent
    }

    /// Outer edge of this layer's draw band, in source cells.
    #[inline]
    pub const fn draw_radius(self) -> i32 {
        self.draw_radius
    }
}

/// Stable identity of a single HLOD proxy: its layer index plus the proxy's
/// coordinate on that layer's coarse grid (design §13.1).
///
/// The coordinate is in *HLOD-cell* units for its layer, i.e. a source
/// [`CellCoord`] floor-divided by the layer's [`extent`](HlodLayer::extent).
/// Cheap, `Copy`, hashable, and totally ordered for deterministic scheduling.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct HlodProxyId {
    /// Index of the owning [`HlodLayer`] (0 = finest).
    pub level: u32,
    /// The proxy's cell on its layer's coarse grid.
    pub coord: CellCoord,
}

impl HlodProxyId {
    /// Builds a proxy id from a layer index and coarse-grid coordinate.
    #[inline]
    pub const fn new(level: u32, coord: CellCoord) -> Self {
        Self { level, coord }
    }
}

impl HlodProxyId {
    /// The totally-ordered sort key `(level, x, y, z)` used to make the
    /// show/hide schedule deterministic (design §14).
    #[inline]
    const fn sort_key(self) -> (u32, i32, i32, i32) {
        (self.level, self.coord.x, self.coord.y, self.coord.z)
    }
}

impl Ord for HlodProxyId {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.sort_key().cmp(&other.sort_key())
    }
}

impl PartialOrd for HlodProxyId {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// The source-cell block a proxy stands in for (design §13.1).
///
/// `origin` is the minimum-corner source [`CellCoord`] of the block and
/// `extent` its side length, so the proxy covers source cells
/// `origin.x .. origin.x + extent` on each axis. The render/scene owner uses
/// this to place and scale the baked impostor over the region it replaces.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HlodFootprint {
    /// Minimum-corner source cell of the covered block.
    pub origin: CellCoord,
    /// Side length of the block, in source cells.
    pub extent: i32,
}

impl HlodFootprint {
    /// The exclusive maximum-corner source cell (`origin + extent` per axis).
    #[inline]
    pub const fn max_corner(self) -> CellCoord {
        CellCoord::new(
            self.origin.x + self.extent,
            self.origin.y + self.extent,
            self.origin.z + self.extent,
        )
    }

    /// Number of source cells in the block (`extent³`).
    #[inline]
    pub const fn cell_count(self) -> i64 {
        let e = self.extent as i64;
        e * e * e
    }
}

/// The set of proxy show/hide requests produced by one
/// [`Hlod::resolve`] (design §13.1).
///
/// Both vectors are sorted by `(level, x, y, z)` so the schedule is
/// deterministic and frame-stable (design §14). The owner turns `to_show` into
/// impostor draws and `to_hide` into their removal; proxies already in the
/// desired state appear in neither list.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct HlodDelta {
    /// Proxies that newly became visible this frame.
    pub to_show: Vec<HlodProxyId>,
    /// Proxies that newly became hidden this frame.
    pub to_hide: Vec<HlodProxyId>,
}

impl HlodDelta {
    /// Whether this delta requests no change at all.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.to_show.is_empty() && self.to_hide.is_empty()
    }
}

/// Floor-divides a source coordinate axis onto a coarse grid of the given
/// `extent`, rounding toward negative infinity so negative cells map
/// consistently (design §13.1).
#[inline]
const fn coarsen_axis(v: i32, extent: i32) -> i32 {
    v.div_euclid(extent)
}

/// Chebyshev distance, in source cells, from `viewpoint` to the nearest source
/// cell of the axis-aligned block `[origin, origin + extent)`.
///
/// Returns `0` when the viewpoint lies inside the block.
#[inline]
fn footprint_distance(viewpoint: CellCoord, origin: CellCoord, extent: i32) -> i32 {
    #[inline]
    fn axis(v: i32, lo: i32, extent: i32) -> i32 {
        let hi = lo + extent - 1; // inclusive far cell
        if v < lo {
            lo - v
        } else if v > hi {
            v - hi
        } else {
            0
        }
    }
    let dx = axis(viewpoint.x, origin.x, extent);
    let dy = axis(viewpoint.y, origin.y, extent);
    let dz = axis(viewpoint.z, origin.z, extent);
    let m = if dx > dy { dx } else { dy };
    if m > dz { m } else { dz }
}

/// Hierarchical LOD proxy scheduler (design §13.1).
///
/// Holds the ordered [layers](HlodLayer) of the hierarchy and remembers which
/// proxies are currently shown, so [`resolve`](Self::resolve) can emit a
/// minimal [`HlodDelta`] each frame. It is `World`-independent and does no I/O:
/// it consumes a viewpoint plus a residency predicate (sourced from the
/// [`CellStreamer`](crate::partition::cell::CellStreamer)) and produces the
/// proxy show/hide schedule the renderer executes.
#[derive(Clone, Debug, Default)]
pub struct Hlod {
    /// Layers finest-first; `draw_radius` strictly increasing.
    layers: Vec<HlodLayer>,
    /// Proxies currently shown, used to diff against the next frame's desired
    /// set. A `HashMap<_, ()>` stands in for a hash set.
    shown: HashMap<HlodProxyId, ()>,
}

impl Hlod {
    /// Creates an empty hierarchy with no layers. With no layers,
    /// [`resolve`](Self::resolve) always yields an empty delta.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a layer that merges `extent³` source-cell blocks and draws them
    /// out to `draw_radius` source cells.
    ///
    /// Layers must be added finest-first: `extent` at least `2`, and
    /// `draw_radius` strictly greater than the previous layer's. This keeps the
    /// bands disjoint and ordered.
    ///
    /// # Panics
    /// Panics if `extent < 2`, if `draw_radius < 1`, or if `draw_radius` does
    /// not strictly exceed the previously added layer's `draw_radius`.
    pub fn push_layer(&mut self, extent: i32, draw_radius: i32) -> &mut Self {
        assert!(
            extent >= 2,
            "HLOD layer extent must be >= 2 (got {extent}); a 1-cell proxy is the detail itself"
        );
        assert!(
            draw_radius >= 1,
            "HLOD layer draw_radius must be >= 1 (got {draw_radius})"
        );
        if let Some(prev) = self.layers.last() {
            assert!(
                draw_radius > prev.draw_radius,
                "HLOD layers must be added finest-first with strictly increasing draw_radius \
                 (got {draw_radius} after {})",
                prev.draw_radius
            );
        }
        self.layers.push(HlodLayer {
            extent,
            draw_radius,
        });
        self
    }

    /// The registered layers, finest-first.
    #[inline]
    pub fn layers(&self) -> &[HlodLayer] {
        &self.layers
    }

    /// Number of registered layers.
    #[inline]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// The source-cell [`HlodFootprint`] a proxy stands in for, or `None` if
    /// `id`'s layer is not registered.
    pub fn footprint(&self, id: HlodProxyId) -> Option<HlodFootprint> {
        let layer = self.layers.get(id.level as usize)?;
        let e = layer.extent;
        Some(HlodFootprint {
            origin: CellCoord::new(id.coord.x * e, id.coord.y * e, id.coord.z * e),
            extent: e,
        })
    }

    /// Whether `id` is currently shown.
    #[inline]
    pub fn is_shown(&self, id: HlodProxyId) -> bool {
        self.shown.contains_key(&id)
    }

    /// Number of proxies currently shown.
    #[inline]
    pub fn shown_count(&self) -> usize {
        self.shown.len()
    }

    /// Iterates the currently shown proxies in unspecified order.
    #[inline]
    pub fn shown(&self) -> impl Iterator<Item = HlodProxyId> + '_ {
        self.shown.keys().copied()
    }

    /// Recomputes proxy visibility for `viewpoint` and returns the minimal
    /// [`HlodDelta`] against the previously shown set (design §13.1).
    ///
    /// `viewpoint` is the source cell the camera (or primary interest) sits in,
    /// in the same lattice the [`CellStreamer`](crate::partition::cell::CellStreamer)
    /// uses. `resident` reports whether a given source cell is currently loaded;
    /// pass [`CellStreamer::is_loaded`](crate::partition::cell::CellStreamer::is_loaded)
    /// (or any predicate that captures the resident set). A proxy is shown when
    /// it falls in its layer's distance band *and* none of its covered source
    /// cells are resident.
    ///
    /// The internal shown-set is updated in place, so successive calls emit only
    /// the frame-over-frame change.
    pub fn resolve<F>(&mut self, viewpoint: CellCoord, resident: F) -> HlodDelta
    where
        F: Fn(CellCoord) -> bool,
    {
        let mut desired: HashMap<HlodProxyId, ()> = HashMap::new();

        let mut inner = 0; // lower edge of the current layer's band (source cells)
        for (idx, layer) in self.layers.iter().enumerate() {
            let level = idx as u32;
            let e = layer.extent;
            let outer = layer.draw_radius;

            // The viewpoint's coarse cell on this layer's grid, and how many
            // coarse cells reach `outer` source cells away (ceil division).
            let v = CellCoord::new(
                coarsen_axis(viewpoint.x, e),
                coarsen_axis(viewpoint.y, e),
                coarsen_axis(viewpoint.z, e),
            );
            let reach = (outer + e - 1) / e;

            for dz in -reach..=reach {
                for dy in -reach..=reach {
                    for dx in -reach..=reach {
                        let coord = CellCoord::new(v.x + dx, v.y + dy, v.z + dz);
                        let origin = CellCoord::new(coord.x * e, coord.y * e, coord.z * e);
                        let dist = footprint_distance(viewpoint, origin, e);
                        // Band test: [inner, outer). The finer layer owns
                        // everything closer than `inner`.
                        if dist < inner || dist >= outer {
                            continue;
                        }
                        if footprint_has_resident(origin, e, &resident) {
                            continue; // detail is drawn; suppress the proxy
                        }
                        desired.insert(HlodProxyId { level, coord }, ());
                    }
                }
            }

            inner = outer;
        }

        let mut delta = HlodDelta::default();
        for &id in desired.keys() {
            if !self.shown.contains_key(&id) {
                delta.to_show.push(id);
            }
        }
        for &id in self.shown.keys() {
            if !desired.contains_key(&id) {
                delta.to_hide.push(id);
            }
        }
        delta.to_show.sort_unstable();
        delta.to_hide.sort_unstable();

        self.shown = desired;
        delta
    }

    /// Hides every shown proxy, returning them (sorted) as a single
    /// [`HlodDelta::to_hide`]. Use when streaming is torn down so the renderer
    /// drops all impostors.
    pub fn clear(&mut self) -> HlodDelta {
        let mut to_hide: Vec<HlodProxyId> = self.shown.keys().copied().collect();
        to_hide.sort_unstable();
        self.shown.clear();
        HlodDelta {
            to_show: Vec::new(),
            to_hide,
        }
    }
}

/// Whether any source cell in the block `[origin, origin + extent)` satisfies
/// `resident`.
#[inline]
fn footprint_has_resident<F>(origin: CellCoord, extent: i32, resident: &F) -> bool
where
    F: Fn(CellCoord) -> bool,
{
    for z in 0..extent {
        for y in 0..extent {
            for x in 0..extent {
                if resident(CellCoord::new(origin.x + x, origin.y + y, origin.z + z)) {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collections::HashMap;

    /// A resident set backed by a hash map, mirroring what the streamer tracks.
    fn resident_set(cells: &[CellCoord]) -> impl Fn(CellCoord) -> bool + '_ {
        let mut set: HashMap<CellCoord, ()> = HashMap::new();
        for &c in cells {
            set.insert(c, ());
        }
        move |c| set.contains_key(&c)
    }

    #[test]
    fn empty_hierarchy_yields_nothing() {
        let mut hlod = Hlod::new();
        let delta = hlod.resolve(CellCoord::ORIGIN, |_| false);
        assert!(delta.is_empty());
        assert_eq!(hlod.shown_count(), 0);
    }

    #[test]
    fn coarsen_axis_floors_toward_negative_infinity() {
        assert_eq!(coarsen_axis(0, 4), 0);
        assert_eq!(coarsen_axis(3, 4), 0);
        assert_eq!(coarsen_axis(4, 4), 1);
        assert_eq!(coarsen_axis(-1, 4), -1);
        assert_eq!(coarsen_axis(-4, 4), -1);
        assert_eq!(coarsen_axis(-5, 4), -2);
    }

    #[test]
    fn footprint_distance_is_zero_inside_and_grows_outside() {
        // Block of extent 4 anchored at origin (0,0,0): covers 0..=3 per axis.
        assert_eq!(footprint_distance(CellCoord::new(1, 1, 1), CellCoord::ORIGIN, 4), 0);
        assert_eq!(footprint_distance(CellCoord::new(5, 1, 1), CellCoord::ORIGIN, 4), 2); // 5 - 3
        assert_eq!(footprint_distance(CellCoord::new(-2, 0, 0), CellCoord::ORIGIN, 4), 2); // 0 - (-2)
    }

    #[test]
    fn footprint_reports_covered_cells() {
        let mut hlod = Hlod::new();
        hlod.push_layer(4, 8);
        let fp = hlod.footprint(HlodProxyId::new(0, CellCoord::new(2, -1, 0))).unwrap();
        assert_eq!(fp.origin, CellCoord::new(8, -4, 0));
        assert_eq!(fp.extent, 4);
        assert_eq!(fp.max_corner(), CellCoord::new(12, 0, 4));
        assert_eq!(fp.cell_count(), 64);
        // Unregistered layer -> None.
        assert!(hlod.footprint(HlodProxyId::new(1, CellCoord::ORIGIN)).is_none());
    }

    #[test]
    #[should_panic(expected = "extent must be >= 2")]
    fn push_layer_rejects_degenerate_extent() {
        Hlod::new().push_layer(1, 4);
    }

    #[test]
    #[should_panic(expected = "strictly increasing draw_radius")]
    fn push_layer_rejects_non_increasing_radius() {
        let mut hlod = Hlod::new();
        hlod.push_layer(4, 8);
        hlod.push_layer(8, 8); // not > 8
    }

    #[test]
    fn resident_detail_suppresses_its_proxy() {
        let mut hlod = Hlod::new();
        hlod.push_layer(2, 10); // one layer, big band

        // Nothing resident: the proxy over the viewpoint's own block is in the
        // band (distance 0) and shown.
        let viewpoint = CellCoord::new(1, 0, 0);
        let delta = hlod.resolve(viewpoint, |_| false);
        let self_cell = HlodProxyId::new(0, CellCoord::new(0, 0, 0)); // 1.div_euclid(2) == 0
        assert!(delta.to_show.contains(&self_cell));
        assert!(hlod.is_shown(self_cell));

        // Now mark one source cell of that block resident: the proxy must hide.
        let delta = hlod.resolve(viewpoint, resident_set(&[CellCoord::new(0, 0, 0)]));
        assert!(delta.to_hide.contains(&self_cell));
        assert!(!hlod.is_shown(self_cell));
    }

    #[test]
    fn bands_are_disjoint_across_layers() {
        // Layer 0: extent 2, draws in [0, 4). Layer 1: extent 4, draws in [4, 12).
        let mut hlod = Hlod::new();
        hlod.push_layer(2, 4);
        hlod.push_layer(4, 12);

        let delta = hlod.resolve(CellCoord::ORIGIN, |_| false);

        // Every shown proxy's footprint distance must respect its layer's band,
        // so no source region is claimed by two layers at once.
        for id in delta.to_show.iter().copied() {
            let fp = hlod.footprint(id).unwrap();
            let dist = footprint_distance(CellCoord::ORIGIN, fp.origin, fp.extent);
            match id.level {
                0 => assert!((0..4).contains(&dist), "L0 proxy out of band: {dist}"),
                1 => assert!((4..12).contains(&dist), "L1 proxy out of band: {dist}"),
                other => panic!("unexpected level {other}"),
            }
        }
        // Both layers contribute at least one proxy.
        assert!(delta.to_show.iter().any(|id| id.level == 0));
        assert!(delta.to_show.iter().any(|id| id.level == 1));
    }

    #[test]
    fn beyond_coarsest_radius_nothing_is_shown() {
        let mut hlod = Hlod::new();
        hlod.push_layer(2, 3);
        // A far source cell well outside the coarsest band: its own proxy is
        // out of range, so with nothing resident it is still not shown.
        let far = CellCoord::new(100, 0, 0);
        let delta = hlod.resolve(CellCoord::ORIGIN, |_| false);
        let far_id = HlodProxyId::new(0, CellCoord::new(coarsen_axis(far.x, 2), 0, 0));
        assert!(!delta.to_show.contains(&far_id));
        assert!(!hlod.is_shown(far_id));
    }

    #[test]
    fn delta_is_minimal_and_stable_across_frames() {
        let mut hlod = Hlod::new();
        hlod.push_layer(2, 6);

        let first = hlod.resolve(CellCoord::ORIGIN, |_| false);
        assert!(!first.to_show.is_empty());
        assert!(first.to_hide.is_empty());

        // Same viewpoint, same residency: no change at all.
        let second = hlod.resolve(CellCoord::ORIGIN, |_| false);
        assert!(second.is_empty(), "steady state must emit an empty delta");
    }

    #[test]
    fn deltas_are_sorted_deterministically() {
        let mut hlod = Hlod::new();
        hlod.push_layer(2, 4);
        hlod.push_layer(4, 10);
        let delta = hlod.resolve(CellCoord::ORIGIN, |_| false);

        let mut sorted = delta.to_show.clone();
        sorted.sort_unstable();
        assert_eq!(delta.to_show, sorted, "to_show must be sorted by (level, x, y, z)");
    }

    #[test]
    fn clear_hides_everything() {
        let mut hlod = Hlod::new();
        hlod.push_layer(2, 6);
        hlod.resolve(CellCoord::ORIGIN, |_| false);
        assert!(hlod.shown_count() > 0);

        let delta = hlod.clear();
        assert!(delta.to_show.is_empty());
        assert!(!delta.to_hide.is_empty());
        assert_eq!(hlod.shown_count(), 0);
        // Sorted.
        let mut sorted = delta.to_hide.clone();
        sorted.sort_unstable();
        assert_eq!(delta.to_hide, sorted);
    }

    #[test]
    fn moving_viewpoint_shifts_the_shown_band() {
        let mut hlod = Hlod::new();
        hlod.push_layer(2, 4);

        hlod.resolve(CellCoord::ORIGIN, |_| false);
        let before: Vec<_> = {
            let mut v: Vec<_> = hlod.shown().collect();
            v.sort_unstable();
            v
        };

        // Move far enough that the shown set must change.
        let delta = hlod.resolve(CellCoord::new(50, 0, 0), |_| false);
        let after: Vec<_> = {
            let mut v: Vec<_> = hlod.shown().collect();
            v.sort_unstable();
            v
        };
        assert_ne!(before, after, "shown set must track the viewpoint");
        assert!(!delta.is_empty());
    }
}
