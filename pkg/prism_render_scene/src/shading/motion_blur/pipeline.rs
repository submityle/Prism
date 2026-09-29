//! Motion-blur compute pipelines, their owned group-0 layouts, and the
//! `RenderStartup` initializer that queues them.
//!
//! Mirrors [`super::super::ssr::reconstruct`]'s pipeline plumbing: three compute
//! entry points from the single `motion_blur.wesl` module, each specialized
//! against its own group-0 layout and the 112-byte [`MotionBlurParams`]
//! immediate block. The three passes chain `TileMax` -> `NeighborMax` ->
//! reconstruction:
//!
//! * `tile_max` (layout `{0,1}`): reads the resolved motion-vector G-buffer,
//!   writes the per-tile longest velocity (`rg16float`).
//! * `neighbor_max` (layout `{2,3}`): reads the tile field, writes the
//!   3x3-dilated dominant velocity (`rg16float`).
//! * `reconstruct` (layout `{0,4,5,6,7}`): reads the motion vectors, the
//!   composited scene colour, the dilated tile field and the SSR device depth,
//!   writes the blurred HDR output (`rgba16float`).
//!
//! The bindings use the shader's explicit indices (each entry point uses only a
//! subset of the module-scope bindings; `naga` prunes the rest per entry point),
//! so `tile_max` is `sequential` `{0,1}` while `neighbor_max` and `reconstruct`
//! are `with_indices`.

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

use super::abi::MotionBlurParams;
use super::super::resources::SCENE_COLOR_FORMAT;

/// Half-float RG storage/sampled format of the `TileMax` and `NeighborMax` tile
/// fields: two channels hold a pixel-space velocity, half precision is ample
/// for the clamped blur budget, and it matches the shader's
/// `texture_storage_2d<rg16float, write>` bindings.
pub(crate) const MOTION_BLUR_TILE_FORMAT: TextureFormat = TextureFormat::Rg16Float;

/// The three motion-blur compute pipelines and their owned group-0 layouts.
#[derive(Resource)]
pub(crate) struct MotionBlurPipeline {
    /// `tile_max` entry: per-tile longest shuttered/clamped velocity.
    tile_max: CachedComputePipelineId,
    /// `neighbor_max` entry: 3x3 tile dilation.
    neighbor_max: CachedComputePipelineId,
    /// `reconstruct` entry: depth-aware scatter-as-gather blur.
    reconstruct: CachedComputePipelineId,
    /// group 0 for `tile_max`: motion-vector read + tile-max storage write.
    tile_max_layout: BindGroupLayout,
    /// group 0 for `neighbor_max`: tile-field read + dilated storage write.
    neighbor_max_layout: BindGroupLayout,
    /// group 0 for `reconstruct`: motion + scene colour + dilated tiles + depth
    /// reads, blurred storage write.
    reconstruct_layout: BindGroupLayout,
}

impl MotionBlurPipeline {
    /// The `tile_max` compute pipeline id.
    pub(crate) fn tile_max(&self) -> CachedComputePipelineId {
        self.tile_max
    }

    /// The `neighbor_max` compute pipeline id.
    pub(crate) fn neighbor_max(&self) -> CachedComputePipelineId {
        self.neighbor_max
    }

    /// The `reconstruct` compute pipeline id.
    pub(crate) fn reconstruct(&self) -> CachedComputePipelineId {
        self.reconstruct
    }

    /// group-0 layout for the `tile_max` dispatch.
    pub(crate) fn tile_max_layout(&self) -> &BindGroupLayout {
        &self.tile_max_layout
    }

    /// group-0 layout for the `neighbor_max` dispatch.
    pub(crate) fn neighbor_max_layout(&self) -> &BindGroupLayout {
        &self.neighbor_max_layout
    }

    /// group-0 layout for the `reconstruct` dispatch.
    pub(crate) fn reconstruct_layout(&self) -> &BindGroupLayout {
        &self.reconstruct_layout
    }
}

/// `tile_max` group-0 layout: the resolved motion-vector G-buffer (non-filterable
/// float, `textureLoad`ed per pixel) then the write-only `rg16float` tile-max
/// output.
fn tile_max_layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(MOTION_BLUR_TILE_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `neighbor_max` group-0 layout at the shader's explicit indices: the tile
/// field read at binding `2` and the dilated `rg16float` output written at
/// binding `3`.
fn neighbor_max_layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::with_indices(
        ShaderStages::COMPUTE,
        (
            (2, texture_2d(TextureSampleType::Float { filterable: false })),
            (
                3,
                texture_storage_2d(MOTION_BLUR_TILE_FORMAT, StorageTextureAccess::WriteOnly),
            ),
        ),
    )
}

/// `reconstruct` group-0 layout at the shader's explicit indices: the
/// motion-vector G-buffer (`0`), the composited scene colour (`4`), the dilated
/// `NeighborMax` field (`5`) and the SSR reverse-Z device depth (`6`), all
/// non-filterable floats `textureLoad`ed, then the write-only `rgba16float`
/// blurred output (`7`).
fn reconstruct_layout_entries() -> BindGroupLayoutEntries<5> {
    BindGroupLayoutEntries::with_indices(
        ShaderStages::COMPUTE,
        (
            (0, texture_2d(TextureSampleType::Float { filterable: false })),
            (4, texture_2d(TextureSampleType::Float { filterable: false })),
            (5, texture_2d(TextureSampleType::Float { filterable: false })),
            (6, texture_2d(TextureSampleType::Float { filterable: false })),
            (
                7,
                texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
            ),
        ),
    )
}

/// `RenderStartup` initializer for [`MotionBlurPipeline`].
pub(crate) fn init_motion_blur_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let tile_max_entries = tile_max_layout_entries();
    let neighbor_max_entries = neighbor_max_layout_entries();
    let reconstruct_entries = reconstruct_layout_entries();

    let tile_max_descriptor =
        BindGroupLayoutDescriptor::new("prism motion blur tile max", &tile_max_entries);
    let neighbor_max_descriptor =
        BindGroupLayoutDescriptor::new("prism motion blur neighbor max", &neighbor_max_entries);
    let reconstruct_descriptor =
        BindGroupLayoutDescriptor::new("prism motion blur reconstruct", &reconstruct_entries);

    let tile_max_layout =
        device.create_bind_group_layout("prism motion blur tile max", &tile_max_entries);
    let neighbor_max_layout =
        device.create_bind_group_layout("prism motion blur neighbor max", &neighbor_max_entries);
    let reconstruct_layout =
        device.create_bind_group_layout("prism motion blur reconstruct", &reconstruct_entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/motion_blur.wesl");

    let immediate_size = size_of::<MotionBlurParams>() as u32;

    let tile_max = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism motion blur tile max".into()),
        layout: vec![tile_max_descriptor],
        immediate_size,
        shader: shader.clone(),
        entry_point: Some("tile_max".into()),
        ..Default::default()
    });

    let neighbor_max = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism motion blur neighbor max".into()),
        layout: vec![neighbor_max_descriptor],
        immediate_size,
        shader: shader.clone(),
        entry_point: Some("neighbor_max".into()),
        ..Default::default()
    });

    let reconstruct = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism motion blur reconstruct".into()),
        layout: vec![reconstruct_descriptor],
        immediate_size,
        shader,
        entry_point: Some("reconstruct".into()),
        ..Default::default()
    });

    commands.insert_resource(MotionBlurPipeline {
        tile_max,
        neighbor_max,
        reconstruct,
        tile_max_layout,
        neighbor_max_layout,
        reconstruct_layout,
    });
}
