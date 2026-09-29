//! Depth-of-field compute pipelines, their owned group-0 layouts, and the
//! `RenderStartup` initializer that queues them.
//!
//! Mirrors [`super::super::motion_blur::pipeline`]: three compute entry points
//! from the single `dof.wesl` module, each specialized against its own group-0
//! layout and its own immediate block (unlike motion blur, `DoF` carries one
//! immediate block *per pass* — `naga` prunes the immediate globals an entry
//! point does not reference). The three passes chain
//! `dof_coc` -> `dof_gather` -> `dof_composite`:
//!
//! * `dof_coc` (layout `{0,1}`): reads the SSR reverse-Z device depth, writes
//!   the per-pixel near/far `CoC` gather radii in pixels (`rg16float`).
//! * `dof_gather` (layout `{2,3,4}`): reads the pre-exposed scene colour and the
//!   `CoC` field, writes the disk-bokeh blurred HDR image (`rgba16float`).
//! * `dof_composite` (layout `{2,3,5,6}`): reads the sharp scene colour, the `CoC`
//!   field and the blurred image, writes the blended HDR output (`rgba16float`)
//!   the dispatch copies back over `scene_color`.
//!
//! The bindings use the shader's explicit indices (each entry point uses only a
//! subset of the module-scope bindings), so `dof_coc` is `sequential` `{0,1}`
//! while `dof_gather` and `dof_composite` are `with_indices`.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{texture_2d, texture_storage_2d},
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

use super::abi::{GpuDofCocParams, GpuDofCompositeParams, GpuDofGatherParams};
use super::super::resources::SCENE_COLOR_FORMAT;

/// Half-float RG storage/sampled format of the `CoC` field: two channels hold the
/// near/far gather radii in pixels, half precision is ample for the clamped
/// radius budget, and it matches the shader's
/// `texture_storage_2d<rg16float, write>` binding.
pub(crate) const DOF_COC_FORMAT: TextureFormat = TextureFormat::Rg16Float;

/// The three depth-of-field compute pipelines and their owned group-0 layouts.
#[derive(Resource)]
pub(crate) struct DofPipeline {
    /// `dof_coc` entry: per-pixel near/far circle-of-confusion prepass.
    coc: CachedComputePipelineId,
    /// `dof_gather` entry: near/far separated disk-bokeh gather.
    gather: CachedComputePipelineId,
    /// `dof_composite` entry: sharp/blurred blend by the `CoC`.
    composite: CachedComputePipelineId,
    /// group 0 for `dof_coc`: depth read + `CoC` storage write.
    coc_layout: BindGroupLayout,
    /// group 0 for `dof_gather`: scene-colour + `CoC` reads, blurred storage write.
    gather_layout: BindGroupLayout,
    /// group 0 for `dof_composite`: scene-colour + `CoC` + blurred reads, output
    /// storage write.
    composite_layout: BindGroupLayout,
}

impl DofPipeline {
    /// The `dof_coc` compute pipeline id.
    pub(crate) fn coc(&self) -> CachedComputePipelineId {
        self.coc
    }

    /// The `dof_gather` compute pipeline id.
    pub(crate) fn gather(&self) -> CachedComputePipelineId {
        self.gather
    }

    /// The `dof_composite` compute pipeline id.
    pub(crate) fn composite(&self) -> CachedComputePipelineId {
        self.composite
    }

    /// group-0 layout for the `dof_coc` dispatch.
    pub(crate) fn coc_layout(&self) -> &BindGroupLayout {
        &self.coc_layout
    }

    /// group-0 layout for the `dof_gather` dispatch.
    pub(crate) fn gather_layout(&self) -> &BindGroupLayout {
        &self.gather_layout
    }

    /// group-0 layout for the `dof_composite` dispatch.
    pub(crate) fn composite_layout(&self) -> &BindGroupLayout {
        &self.composite_layout
    }
}

/// `dof_coc` group-0 layout: the SSR reverse-Z device depth (non-filterable
/// float, `textureLoad`ed per pixel) at binding `0`, then the write-only
/// `rg16float` `CoC` output at binding `1`.
fn coc_layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(DOF_COC_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `dof_gather` group-0 layout at the shader's explicit indices: the pre-exposed
/// scene colour (`2`) and the `CoC` field (`3`), both non-filterable floats
/// `textureLoad`ed, then the write-only `rgba16float` blurred output (`4`).
fn gather_layout_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::with_indices(
        ShaderStages::COMPUTE,
        (
            (2, texture_2d(TextureSampleType::Float { filterable: false })),
            (3, texture_2d(TextureSampleType::Float { filterable: false })),
            (
                4,
                texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
            ),
        ),
    )
}

/// `dof_composite` group-0 layout at the shader's explicit indices: the sharp
/// scene colour (`2`), the `CoC` field (`3`) and the blurred field (`5`), all
/// non-filterable floats `textureLoad`ed, then the write-only `rgba16float`
/// composited output (`6`).
fn composite_layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::with_indices(
        ShaderStages::COMPUTE,
        (
            (2, texture_2d(TextureSampleType::Float { filterable: false })),
            (3, texture_2d(TextureSampleType::Float { filterable: false })),
            (5, texture_2d(TextureSampleType::Float { filterable: false })),
            (
                6,
                texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
            ),
        ),
    )
}

/// `RenderStartup` initializer for [`DofPipeline`].
pub(crate) fn init_dof_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let coc_entries = coc_layout_entries();
    let gather_entries = gather_layout_entries();
    let composite_entries = composite_layout_entries();

    let coc_descriptor = BindGroupLayoutDescriptor::new("prism dof coc", &coc_entries);
    let gather_descriptor = BindGroupLayoutDescriptor::new("prism dof gather", &gather_entries);
    let composite_descriptor =
        BindGroupLayoutDescriptor::new("prism dof composite", &composite_entries);

    let coc_layout = device.create_bind_group_layout("prism dof coc", &coc_entries);
    let gather_layout = device.create_bind_group_layout("prism dof gather", &gather_entries);
    let composite_layout =
        device.create_bind_group_layout("prism dof composite", &composite_entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/dof.wesl");

    let coc = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism dof coc".into()),
        layout: vec![coc_descriptor],
        immediate_size: size_of::<GpuDofCocParams>() as u32,
        shader: shader.clone(),
        entry_point: Some("dof_coc".into()),
        ..Default::default()
    });

    let gather = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism dof gather".into()),
        layout: vec![gather_descriptor],
        immediate_size: size_of::<GpuDofGatherParams>() as u32,
        shader: shader.clone(),
        entry_point: Some("dof_gather".into()),
        ..Default::default()
    });

    let composite = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism dof composite".into()),
        layout: vec![composite_descriptor],
        immediate_size: size_of::<GpuDofCompositeParams>() as u32,
        shader,
        entry_point: Some("dof_composite".into()),
        ..Default::default()
    });

    commands.insert_resource(DofPipeline {
        coc,
        gather,
        composite,
        coc_layout,
        gather_layout,
        composite_layout,
    });
}
