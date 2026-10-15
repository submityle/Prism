//! Compute pipeline + bind-group layouts for the shading-resolve pass.
//!
//! The resolve entry point (`shaders/shading_resolve.wesl`) is dispatched once
//! per [`ResponseBucketKey`](prism_render_shading::ResponseBucketKey)
//! with an indirect argument buffer. It reads eight bind groups:
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
//! * **group 6** — virtual-shadow-map resources and their disabled fallbacks.
//! * **group 7** — area lights and the independent shade-cache consumer ABI.
//!   They share a physical group to stay within the eight-group device limit.
//!
//! Reusing the material/light *layout descriptors* (rather than re-declaring
//! them) guarantees the resolve pipeline and those bind groups can never drift
//! out of sync.

use bevy_asset::{Assets, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{
            sampler, storage_buffer_read_only_sized, storage_buffer_sized, texture_2d,
            texture_cube, texture_storage_2d, uniform_buffer_sized,
        },
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        AddressMode, BindGroupLayout, Buffer, BufferInitDescriptor, BufferUsages,
        CachedComputePipelineId, ComputePipelineDescriptor, Extent3d, FilterMode, MipmapFilterMode,
        PipelineCache, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
        StorageTextureAccess, TextureDataOrder, TextureDescriptor, TextureDimension, TextureFormat,
        TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor,
    },
    renderer::{RenderDevice, RenderQueue},
};
use bevy_shader::{Shader, ShaderDefVal};
use bytemuck::Zeroable;

use super::super::texel_cache::gpu::atlas::SHADE_ATLAS_FORMAT;
use crate::material::runtime::RenderMaterialRegistry;
use crate::shading::texel_cache::{GpuShadeChartInfo, GpuShadePageLookup, GpuShadeResolvePage};
use crate::{ClusterBindGroup, LightBindGroup, MaterialBindGroup};

use super::super::shadow::ShadowBindGroup;

use super::super::area_light::GpuAreaLight;
use super::super::resources::{MOTION_VECTOR_FORMAT, SCENE_COLOR_FORMAT};
use super::abi::GpuShadingResolveParams;
use super::registered_source;

/// Compute pipeline and the four owned bind-group layouts for the resolve pass.
#[derive(Resource)]
pub(crate) struct ShadingResolvePipeline {
    /// `shading_resolve` compute entry point, specialized against all eight
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
    /// Fallback light-routing visibility mask (group 2, binding 9) bound when a
    /// view has no resident [`ViewLightRouting`](super::super::light_routing::ViewLightRouting)
    /// (the opt-in Lighting Channels subsystem is off). A single all-ones `u32`
    /// word so the resolve's channel gate treats every punctual light as
    /// visible, preserving byte-for-byte behaviour when routing is disabled.
    pub(crate) scene_dummy_visible_lights: Buffer,
    /// Fallback one-element response output bound when a view has not yet
    /// received its resident response allocation. The shader guards the pixel
    /// index against this buffer's length, so it is safe even for large views.
    pub(crate) response_dummy: Buffer,
    /// group 7: the per-frame `GpuAreaLight` storage array + the two baked
    /// Linearly Transformed Cosines (`LTC`) `Rgba32Float` coefficient LUTs the
    /// `accumulate_area_lights` twin fetches with a manual bilinear `textureLoad`.
    pub(crate) area_layout: BindGroupLayout,
    /// Fallback one-element area-light buffer bound when the opt-in subsystem is
    /// off or the per-frame buffer has not uploaded yet. Holds a single
    /// degenerate record (`half_width == 0`) the shader's extent guard skips.
    pub(crate) area_dummy_lights: Buffer,
    /// Fallback 1x1 `Rgba32Float` `LTC` coeffs LUT (one identity `M⁻¹` texel)
    /// bound when no real `AreaLightLtcLut` is resident.
    pub(crate) area_dummy_coeffs: TextureView,
    /// Fallback 1x1 `Rgba32Float` `LTC` amplitude LUT (identity amplitude texel)
    /// bound when no real `AreaLightLtcLut` is resident.
    pub(crate) area_dummy_amp: TextureView,
    pub(crate) shade_cache_dummy_lookup: Buffer,
    pub(crate) shade_cache_dummy_page_table: Buffer,
    pub(crate) shade_cache_dummy_pages: Buffer,
    pub(crate) shade_cache_dummy_charts: Buffer,
    pub(crate) shade_cache_dummy_atlas: TextureView,
    /// The bind-group descriptor list used to queue the current resolve PSO.
    /// Keeping the descriptors beside the cached pipeline lets a newly
    /// installed execution table relink only the shader module without
    /// rebuilding any resident layouts or fallback resources.
    pub(crate) layout_descriptors: [BindGroupLayoutDescriptor; 8],
    /// Shader module for the currently linked execution table.
    pub(crate) shader: Handle<Shader>,
    /// Whether the material group uses binding arrays on this device.
    pub(crate) bindless: bool,
    /// Execution-table version captured when `resolve` was queued.
    pub(crate) execution_table_version: u32,
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
/// `ssgi_albedo`); entry 14 is the write-only world-space ReSTIR direct
/// export (`world_restir_direct`); all always bound because the resolve
/// writes them every pixel. Entry 15 is the structured response output written
/// by the resolve producer and consumed by the resident response compositor.
fn view_layout_entries() -> BindGroupLayoutEntries<16> {
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
            // 14: world-space ReSTIR direct export (raw cosine-weighted punctual
            // irradiance), write-only storage the world_restir composite reads to
            // subtract the clustered punctual diffuse before folding in the
            // reservoir estimate; always bound because the resolve writes it.
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
            // 15: structured response output (16 f32 words = 64 bytes). The
            // resolve writes one element per covered pixel; a one-element
            // dummy is bound until the per-view allocation is prepared.
            storage_buffer_sized(false, None),
        ),
    )
}

/// Builds the group-2 layout entries: nine read-only storage buffers (work
/// items, class offsets, class counts, scene instances, geometry
/// headers/vertices/primitives, the per-instance current `world_from_local`
/// transforms the resolve uses to lift local-space geometry into world space,
/// and the matching *previous*-frame transforms used only to place that surface
/// in last frame's world space for the motion vector, followed by the compiled
/// response-bucket table used to translate a dynamic bucket id to its current
/// response entry).
/// `None` min-binding-size keeps the layout agnostic to the run-time array
/// length; the shader guards every index.
fn scene_layout_entries() -> BindGroupLayoutEntries<11> {
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
            // 9: the light-routing channel-gated visibility mask (one `u32`
            // per 32-light cluster word). Read-only here; the light-routing
            // cull pass writes it. A pipeline-owned all-ones dummy is bound
            // when the opt-in subsystem is off so every light stays visible.
            storage_buffer_read_only_sized(false, None),
            // 10: dynamic response-bucket table. The resolve keeps the bucket
            // id for worklist addressing and projects it to the legacy shader
            // arm only at the final switch.
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

/// Builds the shared group-7 layout for area lights and the shade-cache consumer:
///
/// 0. the per-frame `GpuAreaLight` storage array (read-only), and
/// 1-2. the two baked `LTC` `Rgba32Float` coefficient LUTs,
/// 3-6. independent shade-cache lookup, writer page table, page and chart buffers,
/// 7. the shade-cache atlas, and
/// 8. the shade-cache enable/count uniform.
///
/// The LUTs are declared non-filterable: the resolve twin fetches them with a
/// manual bilinear `textureLoad` because the device may lack the optional
/// `FLOAT32_FILTERABLE` feature a linear `Rgba32Float` sampler would need.
/// `None` min-binding-size keeps the layout agnostic to the array length; the
/// shader guards every index with `arrayLength`.
/// The cache keeps its independent buffers and metadata ABI; only the physical
/// group changes. A ninth group would exceed Metal's eight-group device limit.
fn area_light_layout_entries() -> BindGroupLayoutEntries<9> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            texture_2d(TextureSampleType::Float { filterable: false }),
            uniform_buffer_sized(false, None),
        ),
    )
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    use bevy_render::render_resource::{BindingType, BufferBindingType, TextureViewDimension};

    #[test]
    fn resolve_auxiliary_layout_matches_shader_bindings() {
        let entries = area_light_layout_entries();
        let shader = include_str!("../../shaders/shading_resolve.wesl");
        let names = [
            "area_lights",
            "area_light_ltc_coeffs",
            "area_light_ltc_amp",
            "shade_page_lookup",
            "shade_page_table",
            "shade_resolve_pages",
            "shade_chart_infos",
            "shade_atlas",
            "shade_cache_params",
        ];
        for (index, (entry, name)) in entries.iter().zip(names).enumerate() {
            assert_eq!(entry.binding, index as u32);
            assert_eq!(entry.visibility, ShaderStages::COMPUTE);
            assert_eq!(entry.count, None);
            let (ty, qualifier) = match index {
                0 | 3..=6 => (
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    "<storage, read>",
                ),
                1 | 2 | 7 => (
                    BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    "",
                ),
                8 => (
                    BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    "<uniform>",
                ),
                _ => unreachable!(),
            };
            assert_eq!(entry.ty, ty, "binding {index}: {name}");
            assert!(shader.contains(&format!(
                "@group(7) @binding({index}) var{qualifier} {name}:"
            )));
        }
    }

    #[test]
    fn resolve_shader_and_dispatch_stay_within_eight_groups() {
        let shader = include_str!("../../shaders/shading_resolve.wesl");
        let mut bindings = std::collections::BTreeSet::new();
        for declaration in shader
            .lines()
            .filter_map(|line| line.strip_prefix("@group("))
        {
            let (group, rest) = declaration.split_once(") @binding(").unwrap();
            let group: u32 = group.parse().unwrap();
            let binding: u32 = rest.split_once(')').unwrap().0.parse().unwrap();
            assert!(group < 8, "resolve needs a ninth bind group: {group}");
            assert!(
                bindings.insert((group, binding)),
                "duplicate shader binding"
            );
        }
        assert_eq!(
            bindings
                .iter()
                .filter(|(group, _)| *group == 7)
                .copied()
                .collect::<Vec<_>>(),
            (0..9).map(|binding| (7, binding)).collect::<Vec<_>>()
        );

        let dispatch = include_str!("dispatch.rs");
        let groups: Vec<u32> = dispatch
            .lines()
            .filter_map(|line| line.trim().strip_prefix("pass.set_bind_group("))
            .map(|args| args.split_once(',').unwrap().0.parse().unwrap())
            .collect();
        assert_eq!(groups, (0..8).collect::<Vec<_>>());

        // Preparation must bind the same sequence as the shader/host layout.
        let preparation = include_str!("bind_groups.rs");
        let packed = preparation
            .split_once("let area = device.create_bind_group(")
            .unwrap()
            .1;
        let packed = packed.split_once(")),").unwrap().0;
        let actual: Vec<&str> = packed
            .split_once("BindGroupEntries::sequential((")
            .unwrap()
            .1
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect();
        assert_eq!(
            actual,
            [
                "area_lights_binding,",
                "area_coeffs,",
                "area_amp,",
                "cache_lookup,",
                "cache_page_table,",
                "cache_pages,",
                "cache_charts,",
                "cache_atlas,",
                "cache_params_buffer.as_entire_binding(),",
            ]
        );
    }
}

/// Builds a 1x1 `Rgba32Float` fallback `LTC` LUT view carrying one identity
/// texel, bound at group 7 when no real baked [`AreaLightLtcLut`] is resident.
fn create_area_dummy_lut(
    device: &RenderDevice,
    queue: &RenderQueue,
    label: &'static str,
    texel: &[f32; 4],
) -> TextureView {
    let texture = device.create_texture_with_data(
        queue,
        &TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        TextureDataOrder::default(),
        bytemuck::cast_slice(texel),
    );
    texture.create_view(&TextureViewDescriptor::default())
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
    materials: Res<RenderMaterialRegistry>,
    mut shaders: ResMut<Assets<Shader>>,
    queue: Res<RenderQueue>,
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
    let area_entries = area_light_layout_entries();
    let area_descriptor =
        BindGroupLayoutDescriptor::new("prism resolve area lights", &area_entries);
    let area_layout = device.create_bind_group_layout("prism resolve area lights", &area_entries);

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

    // All-ones fallback for the light-routing channel gate: one 32-light word
    // with every bit set, so a view without a resident `ViewLightRouting`
    // (routing disabled) keeps every punctual light visible in the resolve's
    // channel gate.
    let scene_dummy_visible_lights = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("prism resolve light routing dummy visible"),
        contents: bytemuck::cast_slice(&[u32::MAX]),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let response_dummy = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("prism resolve response dummy"),
        contents: bytemuck::cast_slice(&[0.0_f32; 16]),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    // Fallback area-light resources bound when the opt-in subsystem is off (no
    // `AreaLightLtcLut` baked) or the per-frame buffer has not uploaded yet.
    // One degenerate record (`half_width == 0`, skipped by the shader's extent
    // guard) plus two 1x1 identity LUTs, so group 7 is always valid and
    // contributes nothing when disabled.
    let area_dummy_lights = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("prism resolve area lights dummy"),
        contents: bytemuck::bytes_of(&GpuAreaLight::rect(
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            0.0,
            0.0,
            [0.0; 3],
            0.0,
        )),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    // Identity `M⁻¹` row (a00=1, a02=0, a11=1, a20=0) and identity amplitude
    // (a22=1, amplitude=1), matching the shader's non-finite sanitize fallback.
    let area_dummy_coeffs = create_area_dummy_lut(
        &device,
        &queue,
        "prism resolve area ltc coeffs dummy",
        &[1.0, 0.0, 1.0, 0.0],
    );
    let area_dummy_amp = create_area_dummy_lut(
        &device,
        &queue,
        "prism resolve area ltc amp dummy",
        &[1.0, 1.0, 0.0, 0.0],
    );

    let shade_cache_dummy_lookup = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("prism resolve shade cache dummy lookup"),
        contents: bytemuck::bytes_of(&GpuShadePageLookup {
            object_id: u32::MAX,
            material_id: u32::MAX,
            material_epoch_low: 0,
            material_epoch_high: 0,
            face_mip_channel: 0,
            page_x: 0,
            page_y: 0,
            page_index: u32::MAX,
        }),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let shade_cache_dummy_page_table = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("prism resolve shade cache dummy page table"),
        contents: bytemuck::cast_slice(&[u32::MAX; 4]),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let shade_cache_dummy_pages = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("prism resolve shade cache dummy pages"),
        contents: bytemuck::bytes_of(&GpuShadeResolvePage::zeroed()),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let shade_cache_dummy_charts = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("prism resolve shade cache dummy charts"),
        contents: bytemuck::bytes_of(&GpuShadeChartInfo::zeroed()),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let shade_cache_dummy_texture = device.create_texture(&TextureDescriptor {
        label: Some("prism resolve shade cache dummy atlas"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: SHADE_ATLAS_FORMAT,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let shade_cache_dummy_atlas =
        shade_cache_dummy_texture.create_view(&TextureViewDescriptor::default());
    let modules = materials
        .registry
        .execution_tables()
        .map(registered_source::catalogue)
        .transpose()
        .unwrap_or_else(|error| panic!("invalid registered response catalogue: {error:?}"))
        .unwrap_or_default();
    let linked_source =
        registered_source::link(include_str!("../../shaders/shading_resolve.wesl"), &modules)
            .unwrap_or_else(|error| panic!("failed to link registered response shader: {error:?}"));
    let shader: Handle<Shader> = shaders.add(Shader::from_wesl(
        linked_source,
        "embedded://prism_render_scene/shaders/shading_resolve_linked.wesl",
    ));

    // Keep the physical group budget explicit: adding a ninth descriptor must
    // not silently make the pipeline unconstructible on eight-group devices.
    let layout_descriptors: [BindGroupLayoutDescriptor; 8] = [
        view_descriptor,
        material_bindings.layout_descriptor.clone(),
        scene_descriptor,
        light_bindings.layout_descriptor.clone(),
        shadow_bindings.layout_descriptor.clone(),
        cluster_bindings.layout_descriptor.clone(),
        // group 6: virtual-shadow-map sample bindings.
        vsm_descriptor,
        // group 7: area lights (bindings 0..2) + shade cache (3..8).
        area_descriptor,
    ];
    let layout = layout_descriptors.clone();
    let resolve =
        queue_resolve_pipeline(&cache, layout, shader.clone(), material_bindings.bindless());

    commands.insert_resource(ShadingResolvePipeline {
        resolve,
        view_layout,
        scene_layout,
        vsm_layout,
        vsm_sampler,
        vsm_dummy_page_table,
        vsm_dummy_atlas,
        scene_dummy_visible_lights,
        response_dummy,
        area_layout,
        area_dummy_lights,
        area_dummy_coeffs,
        area_dummy_amp,
        shade_cache_dummy_lookup,
        shade_cache_dummy_page_table,
        shade_cache_dummy_pages,
        shade_cache_dummy_charts,
        shade_cache_dummy_atlas,
        layout_descriptors,
        shader,
        bindless: material_bindings.bindless(),
        execution_table_version: materials.registry.execution_table_version(),
    });
}

fn queue_resolve_pipeline(
    cache: &PipelineCache,
    layout: [BindGroupLayoutDescriptor; 8],
    shader: Handle<Shader>,
    bindless: bool,
) -> CachedComputePipelineId {
    cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism shading resolve".into()),
        layout: layout.into(),
        immediate_size: size_of::<GpuShadingResolveParams>() as u32,
        shader,
        shader_defs: vec![ShaderDefVal::Bool("MATERIAL_BINDLESS".into(), bindless)],
        entry_point: Some("shading_resolve".into()),
        ..Default::default()
    })
}

/// Re-links the finite registered-response switch when execution tables change.
/// The hot path only observes the new cached pipeline id; bind-group layouts and
/// fallback resources remain resident, so a material-table update costs one
/// cold shader link and never leaves the resolve pass using an older ABI.
pub(crate) fn refresh_shading_resolve_pipeline(
    materials: Res<RenderMaterialRegistry>,
    cache: Res<PipelineCache>,
    mut shaders: ResMut<Assets<Shader>>,
    mut pipeline: ResMut<ShadingResolvePipeline>,
) {
    let version = materials.registry.execution_table_version();
    if version == pipeline.execution_table_version {
        return;
    }
    let modules = materials
        .registry
        .execution_tables()
        .map(registered_source::catalogue)
        .transpose()
        .unwrap_or_else(|error| panic!("invalid registered response catalogue: {error:?}"))
        .unwrap_or_default();
    let linked_source =
        registered_source::link(include_str!("../../shaders/shading_resolve.wesl"), &modules)
            .unwrap_or_else(|error| {
                panic!("failed to relink registered response shader: {error:?}")
            });
    let shader = shaders.add(Shader::from_wesl(
        linked_source,
        format!("embedded://prism_render_scene/shaders/shading_resolve_linked_{version}.wesl"),
    ));
    pipeline.resolve = queue_resolve_pipeline(
        &cache,
        pipeline.layout_descriptors.clone(),
        shader.clone(),
        // The layout is fixed for the lifetime of the render device. The
        // existing shader was queued with the same bindless choice.
        pipeline.bindless,
    );
    pipeline.shader = shader;
    pipeline.execution_table_version = version;
}
