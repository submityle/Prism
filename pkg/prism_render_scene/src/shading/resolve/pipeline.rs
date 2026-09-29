//! Compute pipeline + bind-group layouts for the shading-resolve pass.
//!
//! The resolve entry point (`shaders/shading_resolve.wesl`) is dispatched once
//! per [`MaterialShadingClass`](prism_render_shading::MaterialShadingClass)
//! with an indirect argument buffer.  It reads four bind groups:
//!
//! * **group 0** — visibility inputs + the HDR storage-texture output.  Owned
//!   here because it is unique to this pass.
//! * **group 1** — the shared material tables, reusing [`MaterialBindGroup`]'s
//!   layout so the same buffers bind byte-for-byte.
//! * **group 2** — the compacted per-class worklist plus the scene/geometry
//!   tables the surface reconstruction walks.  Owned here.
//! * **group 3** — the analytic light tables, reusing [`LightBindGroup`]'s
//!   layout.
//! * **group 4** — the shadow atlas + shadow tables, reusing
//!   [`ShadowBindGroup`]'s layout so the resolve pass samples the exact atlas
//!   the depth pass fills.
//! * **group 5** — the clustered-light tables, reusing [`ClusterBindGroup`]'s
//!   layout so each fragment iterates only the punctual lights assigned to its
//!   froxel instead of the whole scene.  Falls back to the full light list when
//!   a neutral single-cluster grid is bound.
//!
//! Reusing the material/light *layout descriptors* (rather than re-declaring
//! them) guarantees the resolve pipeline and those bind groups can never drift
//! out of sync.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{
            sampler, storage_buffer_read_only_sized, texture_2d, texture_cube,
            texture_storage_2d, uniform_buffer_sized,
        },
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        AddressMode, BindGroupLayout, Buffer, BufferInitDescriptor, BufferUsages,
        CachedComputePipelineId, ComputePipelineDescriptor, Extent3d, FilterMode,
        MipmapFilterMode, PipelineCache,
        Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages, StorageTextureAccess,
        TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
        TextureView, TextureViewDescriptor,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;

use crate::{ClusterBindGroup, LightBindGroup, MaterialBindGroup};

use super::super::shadow::ShadowBindGroup;

use super::abi::GpuShadingResolveParams;
use super::super::resources::{MOTION_VECTOR_FORMAT, SCENE_COLOR_FORMAT};

/// Compute pipeline and the two owned bind-group layouts for the resolve pass.
#[derive(Resource)]
pub(crate) struct ShadingResolvePipeline {
    /// `shading_resolve` compute entry point, specialized against all four
    /// group layouts and the 32-byte immediate block.
    pub(crate) resolve: CachedComputePipelineId,
    /// group 0: visibility ids/metadata textures + HDR storage-texture output.
    pub(crate) view_layout: BindGroupLayout,
    /// group 2: compacted worklist + scene/geometry tables.
    pub(crate) scene_layout: BindGroupLayout,
    /// group 6: virtual-shadow-map page table + physical atlas + sampler +
    /// [`GpuVsmResolveParams`](super::abi::GpuVsmResolveParams) uniform, sampled
    /// by the inline `sample_virtual_shadow` twin in `shading_resolve.wesl`.
    pub(crate) vsm_layout: BindGroupLayout,
    /// Non-filtering (nearest, clamp) sampler for the `R32Float` physical atlas.
    ///
    /// The atlas stores raw NDC depth and the twin compares it manually, so it
    /// binds a `NonFiltering` sampler rather than depending on the optional
    /// `FLOAT32_FILTERABLE` device feature a linear `R32Float` sampler needs.
    pub(crate) vsm_sampler: Sampler,
    /// Fallback page-table storage buffer bound when a view has no resident VSM
    /// page table (feature off, or the upload bridge has not run yet).  Filled
    /// with the `VSM_PAGE_UNMAPPED` sentinel so a stray read is a clean miss.
    pub(crate) vsm_dummy_page_table: Buffer,
    /// Fallback 1x1 `R32Float` atlas view bound when a view has no physical
    /// atlas; format-matched to the real atlas so the bind group is always valid.
    pub(crate) vsm_dummy_atlas: TextureView,
}

/// Builds the group-0 layout entries:
///
/// 0. `texture_2d<u32>` visibility ids,
/// 1. `texture_2d<u32>` visibility metadata,
/// 2. write-only `rgba16float` storage texture (the HDR output),
/// 3. the sampled screen-space GTAO visibility texture (non-filterable float; a
///    1x1 white fallback is bound when GTAO is disabled),
/// 4. the GGX-prefiltered environment radiance cube (filterable float),
/// 5. its trilinear clamp `sampler` (filtering),
/// 6. the split-sum DFG table (filterable float), plus
/// 7. its linear clamp `sampler` (filtering).
///
/// Entries 4-7 are always bound (the global IBL textures are resident from
/// `RenderStartup`); the `RESOLVE_FLAG_IBL_SPECULAR` immediate bit gates whether
/// the shader actually samples them.
///
/// Entries 8-9 are the write-only SSR energy exports (`ssr_env_specular`,
/// `ssr_spec_weight`); always bound because the pass runs regardless of SSR.
///
/// Entry 10 is the write-only `Rg16Float` motion-vector G-buffer and entry 11
/// the 128-byte current/previous view-projection uniform that projects it; both
/// always bound because the resolve writes a motion vector for every pixel.
///
/// Entries 12-13 are the write-only SSGI exports (`ssgi_ambient`,
/// `ssgi_albedo`); always bound because the resolve writes them every pixel.
fn view_layout_entries() -> BindGroupLayoutEntries<14> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Uint),
            texture_2d(TextureSampleType::Uint),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_cube(TextureSampleType::Float { filterable: true }),
            sampler(SamplerBindingType::Filtering),
            texture_2d(TextureSampleType::Float { filterable: true }),
            sampler(SamplerBindingType::Filtering),
            // 8-9: SSR energy-conservation exports (IBL specular + env-BRDF
            // weight), write-only storage the SSR composite later samples.
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
            // 10: motion-vector G-buffer (write-only rg16float).
            texture_storage_2d(MOTION_VECTOR_FORMAT, StorageTextureAccess::WriteOnly),
            // 11: current/previous view-projection uniform (128 bytes).
            uniform_buffer_sized(false, None),
            // 12-13: screen-space GI exports (pre-albedo ambient irradiance +
            // Lambertian albedo), write-only storage the SSGI trace/composite
            // sample; always bound because the resolve writes them every pixel.
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// Builds the group-2 layout entries: nine read-only storage buffers (work
/// items, class offsets, class counts, scene instances, geometry
/// headers/vertices/primitives, the per-instance current `world_from_local`
/// transforms the resolve uses to lift local-space geometry into world space,
/// and the matching *previous*-frame transforms used only to place that surface
/// in last frame's world space for the motion vector).
/// `None` min-binding-size keeps the layout agnostic to the run-time array
/// length; the shader guards every index.
fn scene_layout_entries() -> BindGroupLayoutEntries<9> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
        ),
    )
}

/// Builds the group-6 layout entries for the virtual-shadow-map sample:
///
/// 0. the flat virtual->physical page table (`array<u32>`, read-only storage),
/// 1. the physical page atlas depth texture (non-filterable `f32`; the twin
///    compares depth manually, so it never needs a filtering sample), plus
/// 2. its `NonFiltering` (nearest/clamp) sampler, and
/// 3. the [`GpuVsmResolveParams`](super::abi::GpuVsmResolveParams) uniform
///    carrying the clipmap/atlas geometry, the enable switch and the light basis.
///
/// `None` min-binding-size keeps the layout agnostic to the page table's
/// run-time length; the shader guards the slot index against the resident window.
fn vsm_layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            texture_2d(TextureSampleType::Float { filterable: false }),
            sampler(SamplerBindingType::NonFiltering),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// `RenderStartup` initializer.  Must run after both [`MaterialBindGroup`] and
/// [`LightBindGroup`] exist so their reflected layout descriptors are available
/// to clone into the pipeline's layout list.
pub(crate) fn init_shading_resolve_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    material_bindings: Res<MaterialBindGroup>,
    light_bindings: Res<LightBindGroup>,
    shadow_bindings: Res<ShadowBindGroup>,
    cluster_bindings: Res<ClusterBindGroup>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let view_entries = view_layout_entries();
    let scene_entries = scene_layout_entries();
    let vsm_entries = vsm_layout_entries();
    let view_descriptor = BindGroupLayoutDescriptor::new("prism resolve view", &view_entries);
    let scene_descriptor = BindGroupLayoutDescriptor::new("prism resolve scene", &scene_entries);
    let vsm_descriptor = BindGroupLayoutDescriptor::new("prism resolve vsm", &vsm_entries);
    let view_layout = device.create_bind_group_layout("prism resolve view", &view_entries);
    let scene_layout = device.create_bind_group_layout("prism resolve scene", &scene_entries);
    let vsm_layout = device.create_bind_group_layout("prism resolve vsm", &vsm_entries);

    // Fallback VSM resources bound when a view has no resident page table /
    // physical atlas (feature off, or the upload bridge / raster fill has not
    // produced them yet). They are real, format-correct objects so the bind
    // group is always valid; the shader only samples them when the uniform's
    // `enable` bit is set, which the bind-group builder clears unless both the
    // real page table and atlas are present.
    let vsm_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("prism resolve vsm atlas sampler"),
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
    let vsm_dummy_page_table = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("prism resolve vsm dummy page table"),
        contents: bytemuck::cast_slice(&unmapped),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let dummy_atlas = device.create_texture(&TextureDescriptor {
        label: Some("prism resolve vsm dummy atlas"),
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
    let vsm_dummy_atlas = dummy_atlas.create_view(&TextureViewDescriptor::default());

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/shading_resolve.wesl");

    let resolve = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism shading resolve".into()),
        layout: vec![
            view_descriptor,
            material_bindings.layout_descriptor.clone(),
            scene_descriptor,
            light_bindings.layout_descriptor.clone(),
            shadow_bindings.layout_descriptor.clone(),
            cluster_bindings.layout_descriptor.clone(),
            // group 6: virtual-shadow-map sample bindings.
            vsm_descriptor,
        ],
        immediate_size: size_of::<GpuShadingResolveParams>() as u32,
        shader,
        entry_point: Some("shading_resolve".into()),
        ..Default::default()
    });

    commands.insert_resource(ShadingResolvePipeline {
        resolve,
        view_layout,
        scene_layout,
        vsm_layout,
        vsm_sampler,
        vsm_dummy_page_table,
        vsm_dummy_atlas,
    });
}
