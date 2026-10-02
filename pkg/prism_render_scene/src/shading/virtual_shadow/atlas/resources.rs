//! Per-view virtual-shadow-map physical page atlas: the depth texture the
//! resident physical pages are packed into, its sampling view, the filtering
//! sampler, and the tiling math that maps a physical page index to its texel
//! origin in the atlas.
//!
//! The atlas is a single square `R32Float` texture whose `.r` channel stores
//! each resident page's NDC depth -- exactly what `shaders/vsm_sample.wesl`
//! reads through its `physical_atlas` binding. `physical_pages` pages are laid
//! out in a `physical_pages_per_edge`-on-a-side grid, so the texture edge is
//! `physical_pages_per_edge * page_size` texels. The geometry is driven purely
//! by [`PrismVirtualShadowSettings`] (`physical_pages` and `page_size`), so it
//! is identical for every view and only changes when those settings are retuned.
//!
//! Following the sibling page-mark cache
//! ([`super::super::page_mark`]'s request bitmap), the texture is cached across
//! frames keyed by [`RetainedViewEntity` ] and rebuilt only when its edge
//! changes, so a steady-state camera never churns the (potentially large) atlas
//! allocation. The prepare step is gated on
//! [`PrismShadingSettings::enable_virtual_shadow`] and on a primary directional
//! light being present, mirroring the rest of the VSM subsystem; when either is
//! missing every cached atlas is dropped so nothing lingers resident.

use bevy_ecs::prelude::*;
use bevy_math::UVec2;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    render_resource::{
        Extent3d, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
        TextureView, TextureViewDescriptor,
    },
    renderer::RenderDevice,
    view::{ExtractedView, RetainedViewEntity},
};

use super::super::super::runtime::PrismShadingSettings;
use super::super::extract::VsmPrimaryLight;
use super::super::resources::ViewVsmReceivers;
use super::super::settings::PrismVirtualShadowSettings;

/// Depth format of the physical page atlas: one 32-bit float channel holding
/// each resident page's stored NDC depth, matching the `.r` read in
/// `shaders/vsm_sample.wesl`.
pub(crate) const VSM_PHYSICAL_ATLAS_FORMAT: TextureFormat = TextureFormat::R32Float;

/// Number of physical pages along one edge of the square atlas grid:
/// `ceil(sqrt(physical_pages))`, always at least `1` so the atlas is never
/// zero-sized.
///
/// This is the same edge count [`PrismVirtualShadowSettings::physical_pages_per_edge`]
/// derives; it is kept here as a free function so the tiling math is unit
/// testable in isolation and so `shaders/vsm_sample.wesl`'s
/// `physical_pages_per_edge` immediate field has one authoritative source.
pub(crate) fn physical_pages_per_edge(physical_pages: u32) -> u32 {
    if physical_pages == 0 {
        return 1;
    }
    let edge = f64::from(physical_pages).sqrt().ceil() as u32;
    edge.max(1)
}

/// Texel coordinate of the top-left corner of `physical_page`'s tile in the
/// atlas.
///
/// Physical page `P` occupies grid cell
/// `(P % physical_pages_per_edge, P / physical_pages_per_edge)`, and each cell
/// is `page_size` texels on a side, so the origin is that cell scaled by the
/// page size. This is the CPU twin of the tile addressing
/// `shaders/vsm_sample.wesl` performs before it offsets by the in-page texel.
pub(crate) fn atlas_tile_origin(
    physical_page: u32,
    physical_pages_per_edge: u32,
    page_size: u32,
) -> UVec2 {
    let edge = physical_pages_per_edge.max(1);
    UVec2::new(
        (physical_page % edge) * page_size,
        (physical_page / edge) * page_size,
    )
}

/// Per-view physical page atlas: the depth texture the resident pages are
/// packed into, its sampling view, and the sampler `shaders/vsm_sample.wesl`
/// binds alongside it.
///
/// Present only on views the prepare step ran this frame (VSM enabled and a
/// primary directional light present, with a resident
/// [`ViewVsmReceivers`] marking a shaded 3D view).
#[derive(Component)]
pub(crate) struct ViewVsmPhysicalAtlas {
    /// Sampling view of the `R32Float` atlas depth texture bound at
    /// `physical_atlas` (binding 1) in `shaders/vsm_sample.wesl`; its `.r`
    /// channel stores resident pages' NDC depth. The texture itself lives in
    /// [`VsmPhysicalAtlasCache`], which owns it across frames.
    view: TextureView,
    /// Physical page budget the atlas backs (`physical_pages` immediate field).
    physical_pages: u32,
    /// Physical pages along one atlas edge (`physical_pages_per_edge` immediate
    /// field); the atlas is `physical_pages_per_edge * page_size` texels square.
    physical_pages_per_edge: u32,
}

impl ViewVsmPhysicalAtlas {
    /// Sampling view of the atlas depth texture (`vsm_sample.wesl` binding 1).
    pub(crate) fn atlas_view(&self) -> &TextureView {
        &self.view
    }

    /// The physical page budget the atlas backs.
    pub(crate) fn physical_pages(&self) -> u32 {
        self.physical_pages
    }

    /// The number of physical pages along one atlas edge.
    pub(crate) fn physical_pages_per_edge(&self) -> u32 {
        self.physical_pages_per_edge
    }
}

/// One cached atlas texture plus the edge (in texels) it was sized for, so a
/// settings retune that changes `physical_pages` or `page_size` can detect the
/// mismatch and rebuild.
struct CachedAtlas {
    texture: Texture,
    view: TextureView,
    edge_texels: u32,
}

/// Render-world cache of each view's persistent physical page atlas, keyed by
/// its stable [`RetainedViewEntity`].
///
/// A view that persists across frames with an unchanged atlas edge reuses the
/// same texture; a retune rebuilds it and a vanished view is dropped so
/// textures never leak.
#[derive(Resource, Default)]
pub(crate) struct VsmPhysicalAtlasCache {
    atlases: HashMap<RetainedViewEntity, CachedAtlas>,
}

impl VsmPhysicalAtlasCache {
    /// Drops every cached atlas (used when the feature is disabled or no light
    /// drives the subsystem, so nothing lingers resident).
    fn clear(&mut self) {
        self.atlases.clear();
    }

    /// Returns the cached atlas view for `retained`, (re)allocating the texture
    /// when absent or sized for a different edge. `edge_texels` is the required
    /// `physical_pages_per_edge * page_size`.
    fn get_or_create(
        &mut self,
        device: &RenderDevice,
        retained: RetainedViewEntity,
        edge_texels: u32,
    ) -> (Texture, TextureView) {
        let needs_new = self
            .atlases
            .get(&retained)
            .is_none_or(|cached| cached.edge_texels != edge_texels);
        if needs_new {
            let texture = device.create_texture(&TextureDescriptor {
                label: Some("prism VSM physical atlas"),
                size: Extent3d {
                    width: edge_texels,
                    height: edge_texels,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: VSM_PHYSICAL_ATLAS_FORMAT,
                // RENDER_ATTACHMENT so the shadow-depth draw can render page
                // tiles into it; TEXTURE_BINDING so vsm_sample.wesl can sample
                // the stored depth back.
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&TextureViewDescriptor::default());
            self.atlases.insert(
                retained,
                CachedAtlas {
                    texture: texture.clone(),
                    view: view.clone(),
                    edge_texels,
                },
            );
            (texture, view)
        } else {
            let cached = self
                .atlases
                .get(&retained)
                .expect("atlas present after the is_none_or check");
            (cached.texture.clone(), cached.view.clone())
        }
    }

    /// Drops any cached atlas whose view was not seen this frame.
    fn retain_seen(&mut self, seen: &HashSet<RetainedViewEntity>) {
        self.atlases.retain(|retained, _| seen.contains(retained));
    }
}

/// `PrepareResources` system: for every shaded 3D view (a resident
/// [`ViewVsmReceivers`]) ensure a correctly-sized physical page atlas texture
/// and attach it as a [`ViewVsmPhysicalAtlas`] component.
///
/// Gated on [`PrismShadingSettings::enable_virtual_shadow`] and on the presence
/// of a primary directional light ([`VsmPrimaryLight`]); when either is missing
/// the cache is cleared and no component is inserted. The atlas edge is derived
/// solely from [`PrismVirtualShadowSettings`] (`physical_pages` and
/// `page_size`), so it is identical for every view and rebuilt only on a retune.
pub(crate) fn prepare_vsm_physical_atlas(
    mut commands: Commands,
    device: Res<RenderDevice>,
    settings: Res<PrismShadingSettings>,
    vsm_settings: Res<PrismVirtualShadowSettings>,
    primary_light: Res<VsmPrimaryLight>,
    mut cache: ResMut<VsmPhysicalAtlasCache>,
    views: Query<(Entity, &ExtractedView, &ViewVsmReceivers)>,
) {
    if !settings.enable_virtual_shadow {
        cache.clear();
        return;
    }
    if primary_light.direction.is_none() {
        cache.clear();
        return;
    }

    let physical_pages = vsm_settings.physical_pages;
    let page_size = u32::from(vsm_settings.page_size).max(1);
    let pages_per_edge = physical_pages_per_edge(physical_pages);
    let edge_texels = pages_per_edge.saturating_mul(page_size).max(1);

    let mut seen: HashSet<RetainedViewEntity> = HashSet::default();
    for (entity, view, _receivers) in &views {
        let retained = view.retained_view_entity;
        let (_texture, atlas_view) = cache.get_or_create(&device, retained, edge_texels);

        commands.entity(entity).insert(ViewVsmPhysicalAtlas {
            view: atlas_view,
            physical_pages,
            physical_pages_per_edge: pages_per_edge,
        });
        seen.insert(retained);
    }
    cache.retain_seen(&seen);
}
