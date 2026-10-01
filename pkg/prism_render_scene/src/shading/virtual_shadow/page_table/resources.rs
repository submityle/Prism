//! Render-world resources for the virtual-shadow-map **page-table upload
//! bridge**: the per-view golden driver plus the GPU page-table buffer cache
//! that turn one frame's decoded page requests into the resident virtual->
//! physical page table the resolve pass samples and the render-page list the
//! physical-atlas raster fill pass repaints.
//!
//! [`readback`](super::readback) reads the GPU page-mark request bitmap back one
//! frame late, decodes it with [`request_keys`](super::decode::request_keys), drives
//! the golden [`VirtualShadowMap`] here to make the requested pages resident, and
//! uploads the resulting flat page table with [`VsmPageTableBufferCache::upload`].
//! The pure residency math lives in [`super::decode`] / [`super::pages`]; this
//! module owns only the render-world state (`Resource`s keyed by
//! [`RetainedViewEntity`], since the render-world `Entity` is rebuilt every frame)
//! and the device buffer lifetime.

use bevy_ecs::prelude::*;
use bevy_math::Vec2;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    render_resource::{Buffer, BufferDescriptor, BufferUsages},
    renderer::{RenderDevice, RenderQueue},
    view::RetainedViewEntity,
};
use prism_render_shading::{ShadowPageKey, VirtualShadowMap};

use super::super::abi::window_slot_count;
use super::super::settings::PrismVirtualShadowSettings;
use super::decode::build_page_table;
use super::pages::{build_render_pages, VsmRenderPage};

/// Per-view component carrying the resident virtual->physical page table for
/// this frame: the flat `window_slot_count`-entry `u32` buffer the
/// `vsm_sample.wesl` shader indexes by window slot, plus its slot count.
///
/// Produced by [`super::readback::collect_vsm_page_readback`] and consumed by the
/// resolve-pass VSM sample integration wired in a later slice.
#[derive(Component)]
pub(crate) struct ViewVsmPageTable {
    /// `STORAGE | COPY_DST` buffer holding the flat page table (binding for the
    /// resolve pass's `vsm_sample` shader).
    pub buffer: Buffer,
    /// Number of `u32` entries in `buffer` (== `window_slot_count`).
    #[expect(
        dead_code,
        reason = "slot count is read by the resolve-pass vsm_sample integration wired in a later slice"
    )]
    pub slot_count: u32,
}

/// Per-view component listing the resident clipmap pages the physical-atlas
/// raster fill pass repaints caster depth into this frame.
///
/// Produced by [`super::readback::collect_vsm_page_readback`] and consumed by the
/// physical-atlas raster fill pass wired in a later slice.
#[derive(Component)]
#[expect(
    dead_code,
    reason = "pages are read by the physical-atlas raster fill pass wired in a later slice"
)]
pub(crate) struct ViewVsmRenderPages {
    /// The pages to repaint this frame, each resolved to its backing physical
    /// atlas tile and world-space light-plane footprint.
    pub pages: Vec<VsmRenderPage>,
}

/// One driven frame's CPU outputs: the flat page table to upload and the list
/// of pages the raster fill pass repaints.
pub(crate) struct FrameDrive {
    /// Flat `window_slot_count`-entry virtual->physical page table.
    pub page_table: Vec<u32>,
    /// Resident clipmap pages the raster fill pass repaints this frame.
    pub render_pages: Vec<VsmRenderPage>,
}

/// One view's persistent golden driver state, rebuilt when its settings change
/// so a live retune re-snaps the clipmap without carrying stale residency.
struct DriverState {
    vsm: VirtualShadowMap,
    settings: PrismVirtualShadowSettings,
}

impl DriverState {
    fn new(settings: &PrismVirtualShadowSettings) -> Self {
        let vsm = VirtualShadowMap::from_settings(
            &settings.contract(),
            settings.pages_per_level_edge,
            settings.level0_texel_world_size,
            settings.level0_max_distance,
            settings.page_coord_bias,
        );
        Self {
            vsm,
            settings: *settings,
        }
    }
}

/// Render-world resource owning the golden [`VirtualShadowMap`] driver per view.
///
/// The GPU page-mark request bitmap is read back one frame late; this driver
/// replays the decoded requests through the golden allocator to produce the
/// resident page table and render-page list. It addresses pages in world space
/// (`camera_moved = false`), so a static world plus a moving camera resolves
/// correctly. Per-caster invalidation of dynamic occluders (`caster_movements`)
/// is not driven yet -- an honest gap the raster fill / invalidation slice
/// closes; static casters are already correct.
#[derive(Resource, Default)]
pub(crate) struct VirtualShadowMapDriver {
    views: HashMap<RetainedViewEntity, DriverState>,
}

impl VirtualShadowMapDriver {
    /// Drives one view's frame: replays `keys` through the golden allocator (making
    /// them resident), then resolves the flat page table and the render-page
    /// list from the resulting residency.
    ///
    /// Rebuilds the view's driver from scratch when `settings` changed since the
    /// last frame, so a live retune never mixes two clipmap layouts.
    pub(crate) fn drive(
        &mut self,
        retained: RetainedViewEntity,
        settings: PrismVirtualShadowSettings,
        light: u32,
        camera_light_space: Vec2,
        keys: &[ShadowPageKey],
    ) -> FrameDrive {
        let state = self
            .views
            .entry(retained)
            .or_insert_with(|| DriverState::new(&settings));
        if state.settings != settings {
            *state = DriverState::new(&settings);
        }

        let result =
            state
                .vsm
                .drive_frame_with_requests(light, camera_light_space, false, keys, &[]);
        let clipmap = *state.vsm.clipmap();
        let page_table = build_page_table(
            &clipmap,
            camera_light_space,
            light,
            window_slot_count(&clipmap),
            state.vsm.table(),
        );
        let render_pages = build_render_pages(
            &clipmap,
            state.vsm.table(),
            &result.to_render,
            settings.physical_pages_per_edge(),
            u32::from(settings.page_size),
        );
        FrameDrive {
            page_table,
            render_pages,
        }
    }

    /// Drops driver state for views absent this frame.
    pub(crate) fn retain_seen(&mut self, live: &HashSet<RetainedViewEntity>) {
        self.views.retain(|retained, _| live.contains(retained));
    }

    /// Drops all driver state (VSM disabled).
    pub(crate) fn clear(&mut self) {
        self.views.clear();
    }
}

/// One cached page-table buffer plus the slot count it was sized for, so a
/// clipmap retune that changes the slot space rebuilds the buffer.
struct CachedPageTable {
    buffer: Buffer,
    slot_count: u32,
}

/// Render-world resource caching the per-view GPU page-table buffer so a
/// stable clipmap layout reuses one buffer across frames.
#[derive(Resource, Default)]
pub(crate) struct VsmPageTableBufferCache {
    tables: HashMap<RetainedViewEntity, CachedPageTable>,
}

impl VsmPageTableBufferCache {
    /// Uploads `table` for `retained`, (re)allocating the `STORAGE | COPY_DST`
    /// buffer only when the slot count changes, and returns the buffer to bind.
    pub(crate) fn upload(
        &mut self,
        device: &RenderDevice,
        queue: &RenderQueue,
        retained: RetainedViewEntity,
        table: &[u32],
    ) -> Buffer {
        let slot_count = table.len() as u32;
        let needs_new = self
            .tables
            .get(&retained)
            .is_none_or(|cached| cached.slot_count != slot_count);
        if needs_new {
            // A u32 is 4 bytes; never size the buffer to zero so a degenerate
            // (empty) table still yields a bindable buffer.
            let byte_size = (table.len() * 4).max(4) as u64;
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some("prism VSM page table"),
                size: byte_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.tables.insert(
                retained,
                CachedPageTable {
                    buffer: buffer.clone(),
                    slot_count,
                },
            );
        }
        let buffer = self
            .tables
            .get(&retained)
            .expect("just inserted or already present")
            .buffer
            .clone();
        if !table.is_empty() {
            queue.write_buffer(&buffer, 0, bytemuck::cast_slice(table));
        }
        buffer
    }

    /// Drops cached buffers for views absent this frame.
    pub(crate) fn retain_seen(&mut self, live: &HashSet<RetainedViewEntity>) {
        self.tables.retain(|retained, _| live.contains(retained));
    }

    /// Drops all cached buffers (VSM disabled).
    pub(crate) fn clear(&mut self) {
        self.tables.clear();
    }
}
