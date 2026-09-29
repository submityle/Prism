//! Volumetric-cloud compute pipelines and their owned `@group(0)` layouts.
//!
//! The eight `@compute` entries of `shaders/volumetric_clouds.wesl` realise the
//! subsystem's frame graph — semi-Lagrangian weather advection, the
//! Perlin-Worley density bake, the coverage/type/height modelling composition,
//! the multiple-scatter `LUT` bake, the adaptive view ray-march, the
//! HG-double-lobe scatter resolve, the light-space `AVSM` cloud-shadow march
//! and the temporal reprojection upsample — one distinct pipeline per entry
//! (mirroring the sibling froxel [`super::super::volumetrics`] and
//! [`super::super::dof`] multi-entry precedents).
//!
//! The shader assigns every resource a *unique* `@group(0)` binding across the
//! whole file (bindings `0`–`17`), and each kernel binds only its own subset of
//! those slots. Each pipeline therefore owns a bind-group layout built with
//! [`BindGroupLayoutEntries::with_indices`] so the layout pins the exact global
//! binding numbers the shader declares (the same non-zero-based idiom
//! [`super::super::dof::pipeline`] uses), rather than the sequential `0..N` a
//! per-pass group would imply.
//!
//! No sampler, uniform or storage buffer is bound: every sampled texture is
//! read with `textureLoad` (so it is declared non-filterable) and every
//! per-dispatch constant travels as a `var<immediate>` push-constant block
//! (the [`super::abi`] host mirrors), so a pipeline's `immediate_size` is the
//! `size_of` of its param block and its layout is textures only. Every storage
//! texture is the `rgba16float` [`VC_STORAGE_FORMAT`], matching the shader's
//! `texture_storage_2d/3d<rgba16float, write>` declarations.

#![allow(
    dead_code,
    reason = "the eight volumetric-cloud compute pipelines and their owned `@group(0)` layouts are the render-resource foundation of the GPU cloud subsystem; the resident-resource, bind-group and Core3d dispatch slices that consume `VolumetricCloudPipelines`, its `pipeline`/`layout` accessors and `init_volumetric_cloud_pipelines` land in the following slices, and the kernel-index ordering is exercised now by the contract test below"
)]

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{texture_2d, texture_3d, texture_storage_2d, texture_storage_3d},
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroupLayout, CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache,
        ShaderStages, StorageTextureAccess, TextureFormat, TextureSampleType,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;
use prism_render_architecture::volumetric::gpu::kernels::VolumetricKernel;

use super::abi::{
    GpuModelingParams, GpuMsLutParams, GpuNoiseBakeParams, GpuRaymarchParams,
    GpuScatterResolveParams, GpuShadowMarchParams, GpuUpsampleParams, GpuWeatherAdvectParams,
};

/// The `rgba16float` storage format shared by every write-only volumetric-cloud
/// storage texture (the advected weather map, the baked density cache, the
/// multi-scatter `LUT`, the low-res ray-march / resolve targets, the light-space
/// cloud-shadow map and the full-res upsample output). Matches every
/// `texture_storage_2d/3d<rgba16float, write>` in `volumetric_clouds.wesl`.
pub(crate) const VC_STORAGE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// The eight volumetric-cloud compute pipelines and their owned `@group(0)`
/// layouts, both indexed in [`VolumetricKernel::ALL`] order via
/// [`kernel_index`]. A kernel's pipeline is specialized against
/// `volumetric_clouds.wesl` at [`VolumetricKernel::wesl_entry_point`] with its
/// own layout and the `size_of` of its [`super::abi`] push-constant block.
#[derive(Resource)]
pub(crate) struct VolumetricCloudPipelines {
    /// Compute pipeline ids, one per kernel, in [`VolumetricKernel::ALL`] order.
    pipelines: [CachedComputePipelineId; 8],
    /// `@group(0)` bind-group layouts, one per kernel, in the same order.
    layouts: [BindGroupLayout; 8],
}

impl VolumetricCloudPipelines {
    /// The compute pipeline id for `kernel`.
    pub(crate) fn pipeline(&self, kernel: VolumetricKernel) -> CachedComputePipelineId {
        self.pipelines[kernel_index(kernel)]
    }

    /// The `@group(0)` bind-group layout `kernel`'s bind group must satisfy.
    pub(crate) fn layout(&self, kernel: VolumetricKernel) -> &BindGroupLayout {
        &self.layouts[kernel_index(kernel)]
    }
}

/// Maps a [`VolumetricKernel`] to its slot in the pipeline/layout arrays, which
/// are kept in [`VolumetricKernel::ALL`] order so the dispatch node can walk the
/// kernels in frame order and index straight into either table.
pub(crate) const fn kernel_index(kernel: VolumetricKernel) -> usize {
    match kernel {
        VolumetricKernel::WeatherAdvect => 0,
        VolumetricKernel::NoiseBake => 1,
        VolumetricKernel::Modeling => 2,
        VolumetricKernel::MultiscatterLutBake => 3,
        VolumetricKernel::Raymarch => 4,
        VolumetricKernel::ScatterResolve => 5,
        VolumetricKernel::ShadowMarch => 6,
        VolumetricKernel::Upsample => 7,
    }
}

/// A non-filterable sampled `texture_2d<f32>` binding (`textureLoad`ed).
fn sampled_2d() -> bevy_render::render_resource::BindGroupLayoutEntryBuilder {
    texture_2d(TextureSampleType::Float { filterable: false })
}

/// A non-filterable sampled `texture_3d<f32>` binding (`textureLoad`ed).
fn sampled_3d() -> bevy_render::render_resource::BindGroupLayoutEntryBuilder {
    texture_3d(TextureSampleType::Float { filterable: false })
}

/// A write-only `rgba16float` `texture_storage_2d` binding.
fn storage_2d() -> bevy_render::render_resource::BindGroupLayoutEntryBuilder {
    texture_storage_2d(VC_STORAGE_FORMAT, StorageTextureAccess::WriteOnly)
}

/// A write-only `rgba16float` `texture_storage_3d` binding.
fn storage_3d() -> bevy_render::render_resource::BindGroupLayoutEntryBuilder {
    texture_storage_3d(VC_STORAGE_FORMAT, StorageTextureAccess::WriteOnly)
}

/// `volumetric_weather_advect` group 0: the previous weather map sampled at
/// binding `0`, the advected weather map written at binding `1`.
fn weather_advect_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::with_indices(
        ShaderStages::COMPUTE,
        ((0, sampled_2d()), (1, storage_2d())),
    )
}

/// `volumetric_noise_bake` group 0: the baked density cache written at binding
/// `2`.
fn noise_bake_entries() -> BindGroupLayoutEntries<1> {
    BindGroupLayoutEntries::with_indices(ShaderStages::COMPUTE, ((2, storage_3d()),))
}

/// `volumetric_modeling` group 0: the baked noise volume sampled at binding `3`
/// and the weather map at binding `4`, the composed density cache written at
/// binding `5`.
fn modeling_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::with_indices(
        ShaderStages::COMPUTE,
        ((3, sampled_3d()), (4, sampled_2d()), (5, storage_3d())),
    )
}

/// `volumetric_multiscatter_lut_bake` group 0: the multi-scatter `LUT` written
/// at binding `6`.
fn ms_lut_entries() -> BindGroupLayoutEntries<1> {
    BindGroupLayoutEntries::with_indices(ShaderStages::COMPUTE, ((6, storage_3d()),))
}

/// `volumetric_raymarch` group 0: the density cache sampled at binding `7` and
/// the cloud-shadow map at binding `8`, the low-res scattering/transmittance
/// target written at binding `9`.
fn raymarch_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::with_indices(
        ShaderStages::COMPUTE,
        ((7, sampled_3d()), (8, sampled_2d()), (9, storage_2d())),
    )
}

/// `volumetric_scatter_resolve` group 0: the low-res ray-march target sampled at
/// binding `10` and the multi-scatter `LUT` at binding `11`, the resolved
/// scattering written at binding `12`.
fn scatter_resolve_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::with_indices(
        ShaderStages::COMPUTE,
        ((10, sampled_2d()), (11, sampled_3d()), (12, storage_2d())),
    )
}

/// `volumetric_shadow_march` group 0: the density cache sampled at binding `13`,
/// the light-space cloud-shadow map written at binding `14`.
fn shadow_march_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::with_indices(
        ShaderStages::COMPUTE,
        ((13, sampled_3d()), (14, storage_2d())),
    )
}

/// `volumetric_upsample` group 0: the low-res resolved target sampled at binding
/// `15` and the previous-frame history at binding `16`, the full-res cloud
/// buffer written at binding `17`.
fn upsample_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::with_indices(
        ShaderStages::COMPUTE,
        ((15, sampled_2d()), (16, sampled_2d()), (17, storage_2d())),
    )
}

/// `RenderStartup` initializer for [`VolumetricCloudPipelines`]. Queues all
/// eight compute pipelines against the shared `volumetric_clouds.wesl` shader,
/// each with its own `with_indices` `@group(0)` layout and the `size_of` of its
/// push-constant block, and stores both tables in [`VolumetricKernel::ALL`]
/// order.
pub(crate) fn init_volumetric_cloud_pipelines(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let weather_advect_entries = weather_advect_entries();
    let noise_bake_entries = noise_bake_entries();
    let modeling_entries = modeling_entries();
    let ms_lut_entries = ms_lut_entries();
    let raymarch_entries = raymarch_entries();
    let scatter_resolve_entries = scatter_resolve_entries();
    let shadow_march_entries = shadow_march_entries();
    let upsample_entries = upsample_entries();

    let weather_advect_layout = device.create_bind_group_layout(
        "prism volumetric-cloud weather-advect",
        &weather_advect_entries,
    );
    let noise_bake_layout =
        device.create_bind_group_layout("prism volumetric-cloud noise-bake", &noise_bake_entries);
    let modeling_layout =
        device.create_bind_group_layout("prism volumetric-cloud modeling", &modeling_entries);
    let ms_lut_layout =
        device.create_bind_group_layout("prism volumetric-cloud multiscatter-lut", &ms_lut_entries);
    let raymarch_layout =
        device.create_bind_group_layout("prism volumetric-cloud raymarch", &raymarch_entries);
    let scatter_resolve_layout = device.create_bind_group_layout(
        "prism volumetric-cloud scatter-resolve",
        &scatter_resolve_entries,
    );
    let shadow_march_layout = device
        .create_bind_group_layout("prism volumetric-cloud shadow-march", &shadow_march_entries);
    let upsample_layout =
        device.create_bind_group_layout("prism volumetric-cloud upsample", &upsample_entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/volumetric_clouds.wesl");

    let weather_advect = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism volumetric-cloud weather-advect".into()),
        layout: vec![BindGroupLayoutDescriptor::new(
            "prism volumetric-cloud weather-advect",
            &weather_advect_entries,
        )],
        immediate_size: size_of::<GpuWeatherAdvectParams>() as u32,
        shader: shader.clone(),
        entry_point: Some(VolumetricKernel::WeatherAdvect.wesl_entry_point().into()),
        ..Default::default()
    });

    let noise_bake = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism volumetric-cloud noise-bake".into()),
        layout: vec![BindGroupLayoutDescriptor::new(
            "prism volumetric-cloud noise-bake",
            &noise_bake_entries,
        )],
        immediate_size: size_of::<GpuNoiseBakeParams>() as u32,
        shader: shader.clone(),
        entry_point: Some(VolumetricKernel::NoiseBake.wesl_entry_point().into()),
        ..Default::default()
    });

    let modeling = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism volumetric-cloud modeling".into()),
        layout: vec![BindGroupLayoutDescriptor::new(
            "prism volumetric-cloud modeling",
            &modeling_entries,
        )],
        immediate_size: size_of::<GpuModelingParams>() as u32,
        shader: shader.clone(),
        entry_point: Some(VolumetricKernel::Modeling.wesl_entry_point().into()),
        ..Default::default()
    });

    let ms_lut = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism volumetric-cloud multiscatter-lut".into()),
        layout: vec![BindGroupLayoutDescriptor::new(
            "prism volumetric-cloud multiscatter-lut",
            &ms_lut_entries,
        )],
        immediate_size: size_of::<GpuMsLutParams>() as u32,
        shader: shader.clone(),
        entry_point: Some(
            VolumetricKernel::MultiscatterLutBake
                .wesl_entry_point()
                .into(),
        ),
        ..Default::default()
    });

    let raymarch = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism volumetric-cloud raymarch".into()),
        layout: vec![BindGroupLayoutDescriptor::new(
            "prism volumetric-cloud raymarch",
            &raymarch_entries,
        )],
        immediate_size: size_of::<GpuRaymarchParams>() as u32,
        shader: shader.clone(),
        entry_point: Some(VolumetricKernel::Raymarch.wesl_entry_point().into()),
        ..Default::default()
    });

    let scatter_resolve = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism volumetric-cloud scatter-resolve".into()),
        layout: vec![BindGroupLayoutDescriptor::new(
            "prism volumetric-cloud scatter-resolve",
            &scatter_resolve_entries,
        )],
        immediate_size: size_of::<GpuScatterResolveParams>() as u32,
        shader: shader.clone(),
        entry_point: Some(VolumetricKernel::ScatterResolve.wesl_entry_point().into()),
        ..Default::default()
    });

    let shadow_march = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism volumetric-cloud shadow-march".into()),
        layout: vec![BindGroupLayoutDescriptor::new(
            "prism volumetric-cloud shadow-march",
            &shadow_march_entries,
        )],
        immediate_size: size_of::<GpuShadowMarchParams>() as u32,
        shader: shader.clone(),
        entry_point: Some(VolumetricKernel::ShadowMarch.wesl_entry_point().into()),
        ..Default::default()
    });

    let upsample = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism volumetric-cloud upsample".into()),
        layout: vec![BindGroupLayoutDescriptor::new(
            "prism volumetric-cloud upsample",
            &upsample_entries,
        )],
        immediate_size: size_of::<GpuUpsampleParams>() as u32,
        shader,
        entry_point: Some(VolumetricKernel::Upsample.wesl_entry_point().into()),
        ..Default::default()
    });

    // Both tables are stored in `VolumetricKernel::ALL` order so `kernel_index`
    // is a straight positional lookup.
    commands.insert_resource(VolumetricCloudPipelines {
        pipelines: [
            weather_advect,
            noise_bake,
            modeling,
            ms_lut,
            raymarch,
            scatter_resolve,
            shadow_march,
            upsample,
        ],
        layouts: [
            weather_advect_layout,
            noise_bake_layout,
            modeling_layout,
            ms_lut_layout,
            raymarch_layout,
            scatter_resolve_layout,
            shadow_march_layout,
            upsample_layout,
        ],
    });
}

#[cfg(test)]
mod tests {
    use super::kernel_index;
    use prism_render_architecture::volumetric::gpu::kernels::VolumetricKernel;

    /// [`kernel_index`] must be the exact inverse of [`VolumetricKernel::ALL`]'s
    /// ordering, so the pipeline/layout arrays the dispatch node indexes stay
    /// aligned with the frame-order walk over `ALL`.
    #[test]
    fn kernel_index_matches_all_order() {
        for (slot, kernel) in VolumetricKernel::ALL.into_iter().enumerate() {
            assert_eq!(
                kernel_index(kernel),
                slot,
                "{kernel:?} mapped to wrong slot"
            );
        }
    }
}
