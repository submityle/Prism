//! Data records, per-view components, and transient render targets for the
//! virtual-shadow-map caster depth pass.
//!
//! The pass has three moving parts, split so each is independently testable and
//! the render-graph node stays a thin replay:
//!
//! * The **input** [`ViewVsmCasterPages`] component -- one resident clipmap page
//!   per element -- is the per-view work list the allocator produced. It mirrors
//!   the page-table crate's `VsmRenderPage` field-for-field
//!   ([`VsmCasterPage`]); the parent slice bridges the page-table's
//!   `ViewVsmRenderPages` into this component (see the module report), keeping
//!   the atlas subsystem free of a page-table dependency it cannot name.
//! * [`prepare_vsm_caster_depth_views`](super::dispatch::prepare_vsm_caster_depth_views)
//!   turns each page into a per-page orthographic light projection uniform slice
//!   and records the resulting **output** [`ViewVsmCasterDepth`] component: one
//!   [`VsmCasterDepthEntry`] per page carrying the slice's dynamic offset and the
//!   atlas tile the page is rasterized into.
//! * The scene's shadow-caster geometry is specialized once into a flat
//!   [`VsmCasterDepthDrawList`] and replayed per page, exactly as the classic
//!   shadow depth pass replays [`ShadowDepthDrawList`] per atlas layer.
//!
//! Each page is rasterized *directly* into its tile of the shared physical atlas:
//! the node opens one render pass per view whose colour target is the atlas
//! itself and sets a per-page viewport / scissor to the page's tile, so a page
//! can only ever touch its own texels. Because resident pages occupy disjoint
//! tiles, the single atlas-wide colour clear (to the far value) at the start of
//! the pass gives every rendered tile a clean far-depth background without the
//! pages interfering with one another. Occlusion within each page is resolved by
//! an atlas-sized transient depth buffer ([`VsmCasterDepthTargets`]) that shares
//! the framebuffer geometry and is cleared once per pass; only the atlas colour
//! is persisted.
//!
//! [`ShadowDepthDrawList`]: super::super::super::shadow

use bevy_ecs::prelude::*;
use bevy_math::{UVec2, Vec2};
use bevy_render::{
    render_resource::{
        CachedRenderPipelineId, Extent3d, TextureDescriptor, TextureDimension,
        TextureUsages, TextureView, TextureViewDescriptor,
    },
    renderer::RenderDevice,
    sync_world::MainEntity,
};

use super::pipeline::VSM_CASTER_DEPTH_FORMAT;

/// One resident clipmap page to rasterize this frame.
///
/// Field-for-field identical to the page-table crate's `VsmRenderPage` (which
/// this atlas subsystem cannot name -- it lives in a private, non-re-exported
/// page-table submodule). The parent slice bridges each `VsmRenderPage` into a
/// `VsmCasterPage`; until that bridge lands nothing in this crate *constructs*
/// one outside the tests, hence the guard.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VsmCasterPage {
    /// Physical page index whose atlas tile this page's depth is stored in.
    pub physical_page: u32,
    /// Texel coordinate of the page's tile top-left corner in the atlas.
    pub atlas_tile_origin: UVec2,
    /// Lower-left world corner of the page footprint, in light-plane
    /// coordinates `(world . light_right, world . light_up)`.
    pub world_origin: Vec2,
    /// World-space edge length of the (square) page footprint.
    pub world_size: f32,
    /// Clipmap level the page belongs to (0 = finest).
    pub level: u16,
}

/// Per-view input: the resident clipmap pages to rasterize this frame.
///
/// Attached to a shaded 3D view by the parent slice, which bridges the
/// page-table's per-view `ViewVsmRenderPages` into this atlas-owned type. Read
/// by [`prepare_vsm_caster_depth_views`](super::dispatch::prepare_vsm_caster_depth_views);
/// until the bridge lands no in-crate system inserts it, hence the guard.
#[derive(Component, Default)]
pub(crate) struct ViewVsmCasterPages {
    /// Resident clipmap pages to render into the atlas this frame.
    pub pages: Vec<VsmCasterPage>,
}

/// One rendered page: the dynamic-offset slice of its projection uniform and the
/// atlas tile it is rasterized into.
#[derive(Clone, Copy)]
pub(crate) struct VsmCasterDepthEntry {
    /// Byte offset into [`VsmCasterDepthViewUniform`](super::pipeline::VsmCasterDepthViewUniform)'s
    /// buffer selecting this page's `world -> light-clip` matrix.
    pub dynamic_offset: u32,
    /// Texel coordinate of this page's tile top-left corner in the atlas; the
    /// origin of the per-page viewport / scissor the pass rasterizes into.
    pub atlas_tile_origin: UVec2,
}

/// Per-view output: one [`VsmCasterDepthEntry`] per page rendered this frame, in
/// the order [`prepare_vsm_caster_depth_views`](super::dispatch::prepare_vsm_caster_depth_views)
/// pushed the pages' uniform slices.
///
/// Consumed by [`vsm_caster_depth_pass`](super::dispatch::vsm_caster_depth_pass):
/// each entry sets its atlas tile as the per-page viewport / scissor and replays
/// the geometry draw list with its projection slice bound.
#[derive(Component, Default)]
pub(crate) struct ViewVsmCasterDepth {
    /// The pages rendered this frame for this view.
    pub entries: Vec<VsmCasterDepthEntry>,
}

/// One replayable shadow-caster geometry draw for the caster depth pass.
///
/// The same list is replayed into every resident page; only the bound per-page
/// uniform (selected by [`VsmCasterDepthEntry::dynamic_offset`]) changes between
/// pages. Mirrors the classic shadow depth pass's per-layer replay.
#[derive(Clone, Copy)]
pub(crate) struct VsmCasterDepthDrawEntry {
    /// The mesh-specialized caster depth pipeline for this instance.
    pub pipeline_id: CachedRenderPipelineId,
    /// Main-world entity used to resolve the mesh asset id at draw time.
    pub main_entity: MainEntity,
    /// Row of this instance in the GPU-scene transform / instance tables.
    pub instance_index: u32,
}

/// The per-frame flat list of shadow-caster geometry draws, rebuilt in
/// [`queue_vsm_caster_depth`](super::dispatch::queue_vsm_caster_depth) and
/// replayed per resident page by
/// [`vsm_caster_depth_pass`](super::dispatch::vsm_caster_depth_pass).
#[derive(Resource, Default)]
pub(crate) struct VsmCasterDepthDrawList {
    /// The replayable geometry draws, one per admitted GPU-scene instance.
    draws: Vec<VsmCasterDepthDrawEntry>,
}

impl VsmCasterDepthDrawList {
    /// Clears the list for a fresh frame.
    pub(crate) fn clear(&mut self) {
        self.draws.clear();
    }

    /// Appends one specialized geometry draw.
    pub(crate) fn push(&mut self, entry: VsmCasterDepthDrawEntry) {
        self.draws.push(entry);
    }

    /// The recorded geometry draws, replayed per resident page.
    pub(crate) fn draws(&self) -> &[VsmCasterDepthDrawEntry] {
        &self.draws
    }
}

/// The atlas-sized transient depth-stencil buffer the caster depth pass tests
/// against while rasterizing resident pages directly into the shared physical
/// atlas.
///
/// The atlas colour texture ([`ViewVsmPhysicalAtlas`](super::resources::ViewVsmPhysicalAtlas))
/// is the render target the pass writes NDC depth into; wgpu still requires a
/// matching depth-stencil attachment to resolve per-page occlusion. Because the
/// pass sets a per-page viewport / scissor into the *shared* atlas, that depth
/// attachment must cover the whole atlas framebuffer, so a single
/// atlas-edge-square [`VSM_CASTER_DEPTH_FORMAT`] buffer is built and reused every
/// frame, rebuilt only when the atlas edge changes (a steady-state camera never
/// churns the allocation). It is cleared once per pass; because resident pages
/// occupy disjoint tiles a single clear gives every rendered tile a clean
/// far-depth start.
#[derive(Resource, Default)]
pub(crate) struct VsmCasterDepthTargets {
    /// Atlas edge length (in texels) the current depth buffer was built for; `0`
    /// until the first [`ensure`](Self::ensure).
    atlas_edge: u32,
    /// The atlas-edge-square transient depth-stencil view the pass clears and
    /// depth-tests against; reused across pages and frames.
    depth_view: Option<TextureView>,
}

impl VsmCasterDepthTargets {
    /// Ensures the transient depth buffer exists and is sized to `atlas_edge`
    /// texels square, rebuilding it when the atlas edge changed.
    ///
    /// `atlas_edge` is clamped to at least one texel so a degenerate setting can
    /// never request a zero-sized texture.
    pub(crate) fn ensure(&mut self, device: &RenderDevice, atlas_edge: u32) {
        let size = atlas_edge.max(1);
        if self.atlas_edge == size && self.depth_view.is_some() {
            return;
        }

        let depth_texture = device.create_texture(&TextureDescriptor {
            label: Some("prism vsm caster depth transient depth"),
            size: Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: VSM_CASTER_DEPTH_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&TextureViewDescriptor::default());

        self.atlas_edge = size;
        self.depth_view = Some(depth_view);
    }

    /// Atlas edge length (in texels) the current transient depth buffer was
    /// built for; `0` until the first [`ensure`](Self::ensure).
    pub(crate) fn atlas_edge(&self) -> u32 {
        self.atlas_edge
    }

    /// The transient depth attachment view, once [`ensure`](Self::ensure) has run.
    pub(crate) fn depth_view(&self) -> Option<&TextureView> {
        self.depth_view.as_ref()
    }
}
