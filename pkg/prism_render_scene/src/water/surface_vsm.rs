//! The water-surface pass's **virtual-shadow-map** `@group(2)` plumbing: the
//! `CPU` half that lets the §5 lighting fork sample the same demand-paged
//! virtual shadow map the opaque `shading_resolve` pass reads for the primary
//! directional light.
//!
//! The sibling [`super::surface_pipeline`] slice owns the raster pipelines and
//! the `@group(0)`/`@group(1)` layouts (per-body surface data + the shared
//! engine light table). This slice adds the third bind group — a byte-for-byte
//! mirror of the resolve pass's VSM group (`resolve/bind_groups.rs`) — so the
//! water surface's directional term is shadowed by the identical page
//! table + physical atlas instead of being left fully lit.
//!
//! ## Why a separate fallback resource
//!
//! [`WaterSurfacePipelines`](super::surface_pipeline::WaterSurfacePipelines) is
//! deliberately a *pure-`CPU`* specializer: it stores only layout descriptors
//! and a shader handle, so its unit-test fixture can build it with no
//! [`RenderDevice`]. The VSM group, however, must bind **real device objects**
//! even on a view that has no resident page table or atlas yet (feature off, or
//! the upload bridge / raster fill has not produced them this frame). Those
//! fallbacks are format-correct dummies, exactly like
//! [`ShadingResolvePipeline`](crate::shading)'s `vsm_dummy_*`:
//!
//! * a four-entry `VSM_PAGE_UNMAPPED` page table (any accidental slot read is a
//!   clean page miss, never an out-of-bounds physical index),
//! * a 1x1 `R32Float` atlas view (format-matched to
//!   [`ViewVsmPhysicalAtlas`](crate::shading::ViewVsmPhysicalAtlas)), and
//! * a `Nearest`/`ClampToEdge` `NonFiltering` sampler (the twin compares the
//!   stored `R32Float` depth manually, so it must not require the optional
//!   `FLOAT32_FILTERABLE` feature).
//!
//! Keeping them in this standalone [`WaterVsmFallback`] resource — built once at
//! `RenderStartup` by [`init_water_vsm_fallback`] — lets the specializer stay a
//! device-free, unit-testable descriptor bag while the draw node
//! ([`super::surface_draw`]) still has real objects to bind every frame.

use bevy_ecs::prelude::*;
use bevy_material::bind_group_layout_entries::{
    binding_types::{sampler, storage_buffer_read_only_sized, texture_2d, uniform_buffer_sized},
    BindGroupLayoutEntries,
};
use bevy_render::{
    render_resource::{
        AddressMode, Buffer, BufferInitDescriptor, BufferUsages, Extent3d, FilterMode,
        MipmapFilterMode, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
        TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
        TextureView, TextureViewDescriptor,
    },
    renderer::RenderDevice,
};

/// Builds the water-surface `@group(2)` VSM layout entries (four bindings), in
/// the exact `@binding(n)` order `water_surface_raster.wesl` declares and
/// byte-identical to the opaque resolve pass's VSM layout:
///
/// 0. the flat resident virtual->physical page table (`array<u32>`, read-only
///    storage; `None` min-size keeps the layout agnostic to the run-time slot
///    count — the shader guards the index against the resident window),
/// 1. the physical atlas depth texture (non-filterable float; the twin compares
///    its `.r` channel manually),
/// 2. the atlas sampler (`NonFiltering` for the reason above), and
/// 3. the [`GpuVsmResolveParams`](crate::shading::GpuVsmResolveParams) uniform.
///
/// Declared [`ShaderStages::FRAGMENT`] (the resolve pass uses `COMPUTE`): the
/// water surface samples the shadow in its fragment stage, not a compute pass.
pub(crate) fn vsm_layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::FRAGMENT,
        (
            storage_buffer_read_only_sized(false, None),
            texture_2d(TextureSampleType::Float { filterable: false }),
            sampler(SamplerBindingType::NonFiltering),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// Format-correct fallback device objects bound into the surface pass's
/// `@group(2)` on any view without a resident page table / physical atlas.
///
/// Built once at `RenderStartup` by [`init_water_vsm_fallback`]; read by the
/// raster draw node ([`super::surface_draw`]). Every object is a real, valid
/// resource so the bind group is always complete; the shader only *samples*
/// them when the params uniform's `sample_enable` bit is set, which the draw
/// node clears unless both the real page table and atlas are resident.
#[derive(Resource)]
pub(crate) struct WaterVsmFallback {
    /// Four `VSM_PAGE_UNMAPPED` sentinels (`STORAGE | COPY_DST`): any accidental
    /// slot read is a clean page miss, never an out-of-bounds physical index.
    pub(crate) page_table: Buffer,
    /// A 1x1 `R32Float` atlas view, format-matched to the real
    /// [`ViewVsmPhysicalAtlas`](crate::shading::ViewVsmPhysicalAtlas) so the
    /// bind group is valid when no atlas is resident.
    pub(crate) atlas_view: TextureView,
    /// `Nearest`/`ClampToEdge` `NonFiltering` sampler shared by the real and
    /// fallback atlas paths (the twin reads `R32Float` depth manually).
    pub(crate) sampler: Sampler,
}

/// `RenderStartup` initializer that builds the surface pass's VSM fallback
/// objects and inserts [`WaterVsmFallback`].
///
/// Mirrors the opaque resolve pipeline's `vsm_dummy_*` construction verbatim so
/// the two passes bind structurally identical fallbacks.
pub(crate) fn init_water_vsm_fallback(mut commands: Commands, device: Res<RenderDevice>) {
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("prism water surface vsm atlas sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        mipmap_filter: MipmapFilterMode::Nearest,
        ..Default::default()
    });

    // Four `VSM_PAGE_UNMAPPED` sentinels: any accidental slot read is a clean
    // page miss rather than an out-of-bounds physical index.
    let unmapped = [u32::MAX; 4];
    let page_table = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("prism water surface vsm dummy page table"),
        contents: bytemuck::cast_slice(&unmapped),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    let dummy_atlas = device.create_texture(&TextureDescriptor {
        label: Some("prism water surface vsm dummy atlas"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        // Format-matched to `ViewVsmPhysicalAtlas`'s `R32Float` depth texture.
        format: TextureFormat::R32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let atlas_view = dummy_atlas.create_view(&TextureViewDescriptor::default());

    commands.insert_resource(WaterVsmFallback {
        page_table,
        atlas_view,
        sampler,
    });
}
