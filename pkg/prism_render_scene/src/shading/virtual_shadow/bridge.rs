//! Same-frame **bridge** feeding the page-table readback outputs to the
//! resolve-pass sampler and the physical-atlas caster-depth raster fill.
//!
//! # Why a bridge exists (and why it adds no extra frame)
//! [`collect_vsm_page_readback`](super::page_table::collect_vsm_page_readback)
//! consumes the previous frame's page-request readback, drives the golden
//! allocator and uploads the virtual->physical page table. It is scheduled in
//! [`RenderSystems::PrepareResources`](bevy_render::RenderSystems::PrepareResources),
//! one phase *before* its two consumers, which run in `PrepareBindGroups`:
//!
//! * [`prepare_shading_resolve_bind_groups`](super::super::resolve) needs the
//!   per-view page-table buffer to build the resolve pass's group 6, and
//! * [`prepare_vsm_caster_depth_views`](super::atlas) needs the resident page
//!   list to pack the per-page light projections.
//!
//! Those consumers live in other modules and cannot see the `page_table`-private
//! `ViewVsmPageTable` / `ViewVsmRenderPages` types directly, so this module owns
//! the translation. `collect` records each view's driven page table + resident
//! pages into [`VsmBridgeCache`] (a plain `ResMut`, immediately visible within
//! the set), and [`bridge_vsm_view_resources`] -- ordered *after* `collect` in
//! the same `PrepareResources` set -- re-attaches them as the resolve/atlas
//! components. The `PrepareResourcesFlush` sync point then applies those inserts
//! before `PrepareBindGroups`, so the consumers see them the **same frame**.
//!
//! The cache is keyed by the stable [`RetainedViewEntity`] (the render-world
//! `Entity` is rebuilt every frame) purely to re-match `collect`'s per-view
//! outputs to this frame's view entities; it is not a cross-frame carry. The
//! page-table buffer itself is persisted in
//! [`VsmPageTableBufferCache`](super::page_table::VsmPageTableBufferCache) keyed
//! by the same [`RetainedViewEntity`], so the cache only clones a cheap handle.
//!
//! The net latency is therefore just the readback's own unavoidable one-frame
//! copy (residency requested on frame *N* becomes resident on frame *N+1*), with
//! no additional bridge frame on top. Removing that last frame would require a
//! fully GPU-driven page table (no CPU readback), a separate larger effort.

use bevy_ecs::prelude::*;
use bevy_math::{UVec2, Vec2};
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    render_resource::Buffer,
    view::{ExtractedView, RetainedViewEntity},
};

use super::super::resolve::ViewResolveVsmPageTable;
use super::super::runtime::PrismShadingSettings;
use super::atlas::{ViewVsmCasterPages, VsmCasterPage};

/// One resident clipmap page carried across the phase gap.
///
/// Field-for-field the intersection of the page-table crate's `VsmRenderPage`
/// and the atlas crate's [`VsmCasterPage`]; kept as a bridge-local type so
/// neither of those private, non-re-exported module types has to leak across the
/// `page_table` / `atlas` boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BridgePage {
    /// Physical page index whose atlas tile stores this page's depth.
    pub physical_page: u32,
    /// Top-left texel of the page's tile in the physical atlas texture.
    pub atlas_tile_origin: UVec2,
    /// Lower-left world corner of the page footprint, in light-plane coordinates.
    pub world_origin: Vec2,
    /// World-space edge length of the (square) page footprint.
    pub world_size: f32,
    /// Clipmap level the page belongs to (0 = finest).
    pub level: u16,
}

/// One view's most recently driven page-table + resident-page outputs.
#[derive(Clone)]
struct BridgeEntry {
    /// The flat virtual->physical page-table storage buffer (a cheap handle into
    /// the persistent [`VsmPageTableBufferCache`](super::page_table::VsmPageTableBufferCache)).
    page_table: Buffer,
    /// Number of `u32` slots the page table covers (`levels * edge * edge`).
    slot_count: u32,
    /// Resident clipmap pages the atlas raster fill repaints this frame.
    pages: Vec<BridgePage>,
}

/// Render-world resource handing [`collect_vsm_page_readback`](super::page_table::collect_vsm_page_readback)'s
/// per-view outputs to [`bridge_vsm_view_resources`] within the same
/// `PrepareResources` set.
#[derive(Resource, Default)]
pub(crate) struct VsmBridgeCache {
    entries: HashMap<RetainedViewEntity, BridgeEntry>,
}

impl VsmBridgeCache {
    /// Records (or overwrites) one view's driven page-table + resident pages.
    pub(crate) fn store(
        &mut self,
        retained: RetainedViewEntity,
        page_table: Buffer,
        slot_count: u32,
        pages: Vec<BridgePage>,
    ) {
        self.entries.insert(
            retained,
            BridgeEntry {
                page_table,
                slot_count,
                pages,
            },
        );
    }

    /// Drops cached entries for views not seen this frame, mirroring the page
    /// table / driver caches so a removed view leaks no stale residency.
    pub(crate) fn retain_seen(&mut self, live: &HashSet<RetainedViewEntity>) {
        self.entries.retain(|retained, _| live.contains(retained));
    }

    /// Clears every entry (VSM disabled): the next re-enable re-derives residency.
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }
}

/// `PrepareResources` system (ordered after `collect_vsm_page_readback`)
/// re-attaching each view's cached page table + resident pages as the
/// resolve/atlas-visible components, so both `PrepareBindGroups` consumers see
/// them the same frame.
///
/// Gated on [`PrismShadingSettings::enable_virtual_shadow`]; a view without a
/// cached entry (feature just enabled, or the readback has not produced a frame
/// yet) is skipped, and the resolve pass falls back to CSM while the atlas raster
/// fill skips it.
pub(crate) fn bridge_vsm_view_resources(
    settings: Res<PrismShadingSettings>,
    cache: Res<VsmBridgeCache>,
    views: Query<(Entity, &ExtractedView)>,
    mut commands: Commands,
) {
    if !settings.enable_virtual_shadow {
        return;
    }
    for (entity, view) in &views {
        let Some(entry) = cache.entries.get(&view.retained_view_entity) else {
            continue;
        };
        let pages = entry
            .pages
            .iter()
            .map(|page| VsmCasterPage {
                physical_page: page.physical_page,
                atlas_tile_origin: page.atlas_tile_origin,
                world_origin: page.world_origin,
                world_size: page.world_size,
                level: page.level,
            })
            .collect();
        commands.entity(entity).insert((
            ViewResolveVsmPageTable {
                buffer: entry.page_table.clone(),
                slot_count: entry.slot_count,
            },
            ViewVsmCasterPages { pages },
        ));
    }
}
