//! Virtual shadow maps (VSM): the CPU golden reference for the sparse,
//! demand-paged shadow atlas modelled on Unreal Engine 5's virtual shadow maps.
//!
//! A virtual shadow map exposes an enormous virtual resolution but only ever
//! keeps a bounded working set of fixed-size *pages* resident in a physical
//! pool.  Each frame the renderer figures out which pages the visible surfaces
//! actually sample, makes exactly those resident (evicting the coldest pages
//! when the pool is full), re-renders only the pages a moving caster dirtied,
//! and reuses everything else.  That page lifecycle is what this module
//! reproduces on the CPU so the result is a byte-for-byte twin of the GPU
//! page-table, feedback and physical-atlas passes.
//!
//! The pipeline is split by concern, each piece an independently tested pure
//! function or small state machine:
//!
//! * [`page_table`] - the virtual->physical directory and per-page residency
//!   (resident / pending / evicted) with hit/miss statistics.
//! * [`clipmap`] - directional-light clipmap levels: level selection from view
//!   distance and world-space texel/page snapping for cache stability.
//! * [`allocator`] - the physical page pool with least-recently-used eviction
//!   under the [`VirtualShadowSettings`] budget.
//! * [`request`] - receiver-driven page requests, de-duplication and PCF/PCSS
//!   filter-footprint expansion.
//! * [`invalidation`] - caster-movement page invalidation and the camera-static
//!   reuse rule, aligned with the architecture history-invalidation concept.
//!
//! [`VirtualShadowMap`] is the top-level orchestrator: given this frame's
//! receiver footprints, camera position and caster movements it drives the
//! whole set of subsystems and reports the resident set, the pages to render,
//! the evictions and the budget statistics.

pub mod allocator;
pub mod clipmap;
pub mod invalidation;
pub mod page_table;
pub mod request;

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use bevy_math::Vec2;

pub use allocator::{Allocation, AllocatorStats, PhysicalPageAllocator};
pub use clipmap::{ClipmapConfig, ClipmapLevel};
pub use invalidation::{
    camera_move_invalidates_pages, invalidate_casters, CasterMovement, Invalidation,
};
pub use page_table::{
    key_from_order, page_order, PageOrder, PageTableStats, Residency, VirtualPageTable,
};
pub use request::{filter_page_radius, generate_page_requests, PageRequestSet, Receiver};

// Re-export the architecture contracts so callers use one canonical type.
pub use prism_render_architecture::virtual_shadow::{ShadowPageKey, VirtualShadowSettings};

/// Everything the frame driver needs to resolve one light's working set.
#[derive(Clone, Copy, Debug)]
pub struct FrameInput<'a> {
    /// Light being driven this frame.
    pub light: u32,
    /// Camera position in the light's clipmap plane (drives window snapping).
    pub camera_light_space: Vec2,
    /// Whether the camera moved since the previous frame (informational: it
    /// never invalidates pages, see [`camera_move_invalidates_pages`]).
    pub camera_moved: bool,
    /// Shadow receivers whose footprints request pages.
    pub receivers: &'a [Receiver],
    /// Casters that moved and dirtied the pages they swept over.
    pub caster_movements: &'a [CasterMovement],
}

/// Per-frame budget and residency statistics.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BudgetStats {
    /// Unique pages requested this frame.
    pub requested: usize,
    /// Requests served from an already-resident, non-invalidated page.
    pub hits: usize,
    /// Requests that needed a (re-)render: fresh, evicted-in or invalidated.
    pub misses: usize,
    /// Fresh physical claims made this frame.
    pub allocations: usize,
    /// Pages evicted this frame to make room.
    pub evictions: usize,
    /// Total physical pages in the pool.
    pub physical_capacity: u32,
    /// Physical pages occupied after the frame resolved.
    pub physical_live: u32,
    /// How far the unique request count exceeded the physical budget.
    pub over_budget: usize,
    /// Requests that could not be served at all (zero-capacity pool).
    pub unserved: usize,
}

impl BudgetStats {
    /// Fraction of requests served as hits, in `[0, 1]`; `0.0` when nothing was
    /// requested.
    pub fn hit_rate(&self) -> f32 {
        if self.requested == 0 {
            0.0
        } else {
            self.hits as f32 / self.requested as f32
        }
    }
}

/// One frame's resolved virtual-shadow working set.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrameResult {
    /// Pages reused from the resident set (cache hits), in key order.
    pub resident: Vec<ShadowPageKey>,
    /// Pages that were (re-)rendered this frame, in key order.
    pub to_render: Vec<ShadowPageKey>,
    /// Pages evicted to make room this frame, in key order.
    pub evicted: Vec<ShadowPageKey>,
    /// The caster invalidation computed for this light.
    pub invalidation: Invalidation,
    /// Budget and residency statistics for this frame.
    pub budget: BudgetStats,
    /// The resident clipmap window of every clip level touched this frame,
    /// snapped to the camera position (from `FrameInput::camera_light_space`).
    /// Callers use these to bound sampling to each level's covered region.
    pub windows: Vec<ClipmapLevel>,
}

/// Stateful driver tying the VSM subsystems together across frames.
#[derive(Clone, Debug)]
pub struct VirtualShadowMap {
    clipmap: ClipmapConfig,
    table: VirtualPageTable,
    allocator: PhysicalPageAllocator,
    frame: u64,
}

impl VirtualShadowMap {
    /// Builds a driver from the architecture [`VirtualShadowSettings`] budget
    /// and a fully specified clipmap layout.  The physical pool is sized to
    /// `settings.physical_pages`.
    pub fn from_settings(
        settings: &VirtualShadowSettings,
        pages_per_level_edge: u16,
        level0_texel_world_size: f32,
        level0_max_distance: f32,
        page_coord_bias: i32,
    ) -> Self {
        let clipmap = ClipmapConfig::from_settings(
            settings,
            pages_per_level_edge,
            level0_texel_world_size,
            level0_max_distance,
            page_coord_bias,
        );
        Self {
            clipmap,
            table: VirtualPageTable::new(),
            allocator: PhysicalPageAllocator::new(settings.physical_pages),
            frame: 0,
        }
    }

    /// Builds a driver from an explicit clipmap config and physical capacity.
    pub fn new(clipmap: ClipmapConfig, physical_pages: u32) -> Self {
        Self {
            clipmap,
            table: VirtualPageTable::new(),
            allocator: PhysicalPageAllocator::new(physical_pages),
            frame: 0,
        }
    }

    /// The clipmap layout in use.
    pub fn clipmap(&self) -> &ClipmapConfig {
        &self.clipmap
    }

    /// The virtual-page directory.
    pub fn table(&self) -> &VirtualPageTable {
        &self.table
    }

    /// The physical page pool.
    pub fn allocator(&self) -> &PhysicalPageAllocator {
        &self.allocator
    }

    /// The number of frames driven so far.
    pub fn frame_index(&self) -> u64 {
        self.frame
    }

    /// Drives one frame: generates the requested pages, applies caster
    /// invalidation, makes the working set resident under the LRU budget and
    /// reports what was reused, rendered and evicted.
    ///
    /// Rendered pages are promoted to resident before returning (a synchronous
    /// render), so the following frame sees them as hits.
    pub fn drive_frame(&mut self, input: FrameInput<'_>) -> FrameResult {
        self.frame += 1;
        let frame = self.frame;

        // Camera translation never invalidates cached pages: world-space page
        // addressing keeps a static caster's page identity stable as the camera
        // pans, so its depth is reused.  A camera *cut* would drop the whole
        // cache; `camera_move_invalidates_pages` encodes that this frame's move
        // does not, so this reset stays dormant under the reuse rule.
        if input.camera_moved && camera_move_invalidates_pages() {
            self.table = VirtualPageTable::new();
            self.allocator = PhysicalPageAllocator::new(self.allocator.capacity());
        }

        let requests = generate_page_requests(&self.clipmap, input.light, input.receivers);

        // Levels actually touched this frame bound the invalidation work.
        let mut levels: Vec<u16> = requests.keys.iter().map(|k| k.level).collect();
        levels.sort_unstable();
        levels.dedup();

        // Snap each touched level's resident window to the camera position; this
        // is where `camera_light_space` drives window snapping.
        let windows: Vec<ClipmapLevel> = levels
            .iter()
            .map(|&level| self.clipmap.build_level(level, input.camera_light_space))
            .collect();

        let invalidation =
            invalidate_casters(&self.clipmap, input.light, input.caster_movements, &levels);
        let invalid_set: BTreeSet<PageOrder> =
            invalidation.pages.iter().map(page_order).collect();

        let stats_before = self.allocator.stats();

        let mut resident = Vec::new();
        let mut to_render = Vec::new();
        let mut evicted = BTreeSet::new();
        let mut unserved = 0usize;

        for key in &requests.keys {
            let dirty = invalid_set.contains(&page_order(key));
            if !dirty
                && let Some(_physical) = self.table.query(key, frame)
            {
                self.allocator.touch(key, frame);
                resident.push(*key);
                continue;
            }
            match self.allocator.request(key, frame) {
                Some(alloc) => {
                    if let Some(victim) = alloc.evicted {
                        self.table.mark_evicted(&victim);
                        evicted.insert(page_order(&victim));
                    }
                    self.table.insert_pending(key, alloc.physical_page, frame);
                    to_render.push(*key);
                }
                None => unserved += 1,
            }
        }

        // Synchronous render: promote the freshly paged-in set to resident.
        for key in &to_render {
            self.table.mark_resident(key, frame);
        }

        let stats_after = self.allocator.stats();
        let capacity = self.allocator.capacity() as usize;
        let evicted: Vec<ShadowPageKey> = evicted.into_iter().map(key_from_order).collect();

        let budget = BudgetStats {
            requested: requests.keys.len(),
            hits: resident.len(),
            misses: to_render.len(),
            allocations: (stats_after.allocations - stats_before.allocations) as usize,
            evictions: (stats_after.evictions - stats_before.evictions) as usize,
            physical_capacity: self.allocator.capacity(),
            physical_live: self.allocator.live_pages(),
            over_budget: requests.keys.len().saturating_sub(capacity),
            unserved,
        };

        FrameResult {
            resident,
            to_render,
            evicted,
            invalidation,
            budget,
            windows,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(physical_pages: u32) -> VirtualShadowSettings {
        VirtualShadowSettings {
            physical_pages,
            page_size: 128,
            max_clip_levels: 4,
        }
    }

    fn driver(physical_pages: u32) -> VirtualShadowMap {
        VirtualShadowMap::from_settings(&settings(physical_pages), 8, 0.1, 10.0, 32_768)
    }

    fn receiver(x: f32, y: f32) -> Receiver {
        Receiver {
            light_space_xy: Vec2::new(x, y),
            view_distance: 1.0,
            filter_radius_texels: 0.0,
        }
    }

    /// The first frame is all misses (nothing resident yet); the identical
    /// second frame is all hits (the working set was cached).
    #[test]
    fn cold_frame_misses_then_warm_frame_hits() {
        let mut vsm = driver(64);
        let receivers = [receiver(0.0, 0.0), receiver(50.0, 0.0), receiver(0.0, 50.0)];
        let input = FrameInput {
            light: 0,
            camera_light_space: Vec2::ZERO,
            camera_moved: false,
            receivers: &receivers,
            caster_movements: &[],
        };
        let cold = vsm.drive_frame(input);
        assert_eq!(cold.budget.hits, 0);
        assert_eq!(cold.budget.misses, cold.budget.requested);
        assert!(cold.evicted.is_empty());

        let warm = vsm.drive_frame(input);
        assert_eq!(warm.budget.misses, 0);
        assert_eq!(warm.budget.hits, warm.budget.requested);
        assert!((warm.budget.hit_rate() - 1.0).abs() < 1.0e-6);
    }

    /// Panning the camera does not invalidate a static caster's pages: the
    /// pages a receiver still occupies stay resident and are reused.
    #[test]
    fn camera_pan_reuses_static_pages() {
        let mut vsm = driver(256);
        let receivers = [receiver(0.0, 0.0)];
        let first = vsm.drive_frame(FrameInput {
            light: 0,
            camera_light_space: Vec2::ZERO,
            camera_moved: false,
            receivers: &receivers,
            caster_movements: &[],
        });
        assert_eq!(first.budget.misses, 1);

        // Move the camera but keep the same receiver: its page identity is
        // world-locked, so it is a hit.
        let second = vsm.drive_frame(FrameInput {
            light: 0,
            camera_light_space: Vec2::new(3.0, 3.0),
            camera_moved: true,
            receivers: &receivers,
            caster_movements: &[],
        });
        assert_eq!(second.budget.hits, 1);
        assert!(second.invalidation.is_empty());
    }

    /// A moving caster invalidates the page it sits on, forcing a re-render even
    /// though the page was resident.
    #[test]
    fn moving_caster_forces_a_rerender() {
        let mut vsm = driver(256);
        let receivers = [receiver(1.0, 1.0)];
        let base = FrameInput {
            light: 0,
            camera_light_space: Vec2::ZERO,
            camera_moved: false,
            receivers: &receivers,
            caster_movements: &[],
        };
        vsm.drive_frame(base); // page now resident

        let with_caster = vsm.drive_frame(FrameInput {
            caster_movements: &[CasterMovement {
                light_space_min: Vec2::new(0.5, 0.5),
                light_space_max: Vec2::new(1.5, 1.5),
            }],
            ..base
        });
        assert!(!with_caster.invalidation.is_empty());
        assert_eq!(with_caster.budget.hits, 0);
        assert_eq!(with_caster.budget.misses, 1);
        assert_eq!(with_caster.to_render.len(), 1);
    }

    /// Requesting more unique pages than the pool holds evicts the coldest ones
    /// and reports the over-budget overflow.
    #[test]
    fn over_budget_evicts_and_reports_overflow() {
        let mut vsm = driver(4);
        // Six well-separated receivers -> six unique pages, pool holds four.
        let receivers = [
            receiver(0.0, 0.0),
            receiver(50.0, 0.0),
            receiver(100.0, 0.0),
            receiver(150.0, 0.0),
            receiver(200.0, 0.0),
            receiver(250.0, 0.0),
        ];
        let result = vsm.drive_frame(FrameInput {
            light: 0,
            camera_light_space: Vec2::ZERO,
            camera_moved: false,
            receivers: &receivers,
            caster_movements: &[],
        });
        assert_eq!(result.budget.requested, 6);
        assert_eq!(result.budget.physical_capacity, 4);
        assert_eq!(result.budget.over_budget, 2);
        // The pool never exceeds its capacity.
        assert!(result.budget.physical_live <= 4);
        // Filling a full pool of four then paging in two more evicts two.
        assert_eq!(result.evicted.len(), 2);
    }

    /// A zero-capacity pool serves nothing and records every request as
    /// unserved.
    #[test]
    fn zero_budget_pool_serves_nothing() {
        let mut vsm = driver(0);
        let receivers = [receiver(0.0, 0.0)];
        let result = vsm.drive_frame(FrameInput {
            light: 0,
            camera_light_space: Vec2::ZERO,
            camera_moved: false,
            receivers: &receivers,
            caster_movements: &[],
        });
        assert_eq!(result.budget.unserved, 1);
        assert!(result.to_render.is_empty());
        assert!(result.resident.is_empty());
    }

    /// The frame reports one camera-snapped resident window per clip level it
    /// touched, and every window is centred so it contains the camera's own
    /// page (exercising the window membership helpers).
    #[test]
    fn windows_report_camera_snapped_levels() {
        let mut vsm = driver(64);
        let camera = Vec2::new(1.0, 1.0);
        let receivers = [receiver(0.0, 0.0), receiver(50.0, 0.0)];
        let result = vsm.drive_frame(FrameInput {
            light: 0,
            camera_light_space: camera,
            camera_moved: false,
            receivers: &receivers,
            caster_movements: &[],
        });

        let touched: BTreeSet<u16> = result.to_render.iter().map(|k| k.level).collect();
        assert_eq!(result.windows.len(), touched.len());

        for window in &result.windows {
            // Deterministically rebuilt from the same camera position.
            assert_eq!(*window, vsm.clipmap().build_level(window.level, camera));
            // The camera's own page always lies inside its centred window.
            let cam_page = vsm.clipmap().world_page_coords(window.level, camera);
            assert!(window.contains(cam_page));
            assert!(window.local_page(cam_page).is_some());
            // Origin snaps to a whole page boundary (no sub-page shimmer).
            let ratio = window.snapped_origin.x / window.page_world_size;
            assert!((ratio - bevy_math::ops::round(ratio)).abs() < 1.0e-4);
        }
    }

    /// The camera-cut gate stays dormant under the world-space reuse rule: even
    /// with `camera_moved` set, a static receiver's page is reused, not flushed.
    #[test]
    fn camera_move_does_not_flush_the_cache() {
        assert!(!camera_move_invalidates_pages());
        let mut vsm = driver(64);
        let receivers = [receiver(2.0, 2.0)];
        let warm = FrameInput {
            light: 0,
            camera_light_space: Vec2::ZERO,
            camera_moved: false,
            receivers: &receivers,
            caster_movements: &[],
        };
        assert_eq!(vsm.drive_frame(warm).budget.misses, 1);

        // A large camera move with camera_moved=true must not evict the page.
        let moved = vsm.drive_frame(FrameInput {
            camera_light_space: Vec2::new(500.0, -500.0),
            camera_moved: true,
            ..warm
        });
        assert_eq!(moved.budget.hits, 1);
        assert!(moved.evicted.is_empty());
    }
}
