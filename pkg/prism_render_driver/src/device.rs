//! The backend trait surface: what a concrete GPU backend must implement.
//!
//! The RHI describes resources and commands as plain data (the rest of this
//! crate). Those descriptions are inert until a *backend* turns them into real
//! GPU objects. This module declares the two traits every backend implements —
//! [`RenderDevice`] for resource creation/destruction and [`RenderQueue`] for
//! uploads and submission — plus the small value types queue uploads need.
//!
//! These are trait *declarations*: the method bodies live in backend crates
//! (wgpu, Vulkan, …), which may use `unsafe` and therefore live outside this
//! `unsafe`-free crate. Nothing here is a placeholder; the trait is the
//! contract.

use crate::binding::{BindGroupDescriptor, BindGroupLayoutDescriptor};
use crate::buffer::BufferDescriptor;
use crate::capabilities::DeviceCapabilities;
use crate::command::CommandBuffer;
use crate::pipeline::{
    ComputePipelineDescriptor, PipelineLayoutDescriptor, RenderPipelineDescriptor,
};
use crate::resource::{
    BindGroupId, BindGroupLayoutId, BufferId, ComputePipelineId, PipelineLayoutId,
    RenderPipelineId, SamplerId, ShaderModuleId, TextureId, TextureViewId,
};
use crate::sampler::SamplerDescriptor;
use crate::shader::ShaderModuleDescriptor;
use crate::texture::{Extent3d, TextureDescriptor, TextureViewDescriptor};

/// The byte layout of a linear image buffer used for texture uploads.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ImageDataLayout {
    /// The byte offset of the first texel in the source data.
    pub offset: u64,
    /// Bytes between the start of consecutive rows, or `None` for tightly
    /// packed rows.
    pub bytes_per_row: Option<u32>,
    /// Rows between the start of consecutive layers/depth slices, or `None`
    /// for tightly packed layers.
    pub rows_per_image: Option<u32>,
}

/// The destination region of a [`RenderQueue::write_texture`] upload.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TextureWrite {
    /// The destination texture.
    pub texture: TextureId,
    /// The destination mip level.
    pub mip_level: u32,
    /// The `(x, y, z)` texel origin of the destination region.
    pub origin: [u32; 3],
    /// The size of the written region in texels.
    pub size: Extent3d,
}

/// A backend device: the factory for every GPU resource.
///
/// Methods take borrowed descriptors (plain RHI data) and return typed,
/// generational ids. A backend owns the real objects behind those ids and is
/// responsible for validating each descriptor against its reported
/// [`DeviceCapabilities`]. Each `create_*` has a matching `destroy_*`; dropping
/// an id without destroying it leaks the backing resource until the device is
/// torn down.
pub trait RenderDevice {
    /// The features and limits this device reports.
    fn capabilities(&self) -> &DeviceCapabilities;

    /// Creates a GPU buffer.
    fn create_buffer(&self, descriptor: &BufferDescriptor) -> BufferId;
    /// Creates a GPU texture.
    fn create_texture(&self, descriptor: &TextureDescriptor) -> TextureId;
    /// Creates a view onto an existing texture.
    fn create_texture_view(
        &self,
        texture: TextureId,
        descriptor: &TextureViewDescriptor,
    ) -> TextureViewId;
    /// Creates a sampler.
    fn create_sampler(&self, descriptor: &SamplerDescriptor) -> SamplerId;
    /// Compiles a shader module.
    fn create_shader_module(&self, descriptor: &ShaderModuleDescriptor) -> ShaderModuleId;
    /// Creates a bind group layout.
    fn create_bind_group_layout(
        &self,
        descriptor: &BindGroupLayoutDescriptor,
    ) -> BindGroupLayoutId;
    /// Creates a bind group binding concrete resources to a layout.
    fn create_bind_group(&self, descriptor: &BindGroupDescriptor) -> BindGroupId;
    /// Creates a pipeline layout.
    fn create_pipeline_layout(&self, descriptor: &PipelineLayoutDescriptor) -> PipelineLayoutId;
    /// Creates a render pipeline.
    fn create_render_pipeline(&self, descriptor: &RenderPipelineDescriptor) -> RenderPipelineId;
    /// Creates a compute pipeline.
    fn create_compute_pipeline(
        &self,
        descriptor: &ComputePipelineDescriptor,
    ) -> ComputePipelineId;

    /// Releases a buffer. Using its id afterwards is a stale-id error.
    fn destroy_buffer(&self, id: BufferId);
    /// Releases a texture.
    fn destroy_texture(&self, id: TextureId);
    /// Releases a texture view.
    fn destroy_texture_view(&self, id: TextureViewId);
    /// Releases a sampler.
    fn destroy_sampler(&self, id: SamplerId);
    /// Releases a shader module.
    fn destroy_shader_module(&self, id: ShaderModuleId);
    /// Releases a bind group layout.
    fn destroy_bind_group_layout(&self, id: BindGroupLayoutId);
    /// Releases a bind group.
    fn destroy_bind_group(&self, id: BindGroupId);
    /// Releases a pipeline layout.
    fn destroy_pipeline_layout(&self, id: PipelineLayoutId);
    /// Releases a render pipeline.
    fn destroy_render_pipeline(&self, id: RenderPipelineId);
    /// Releases a compute pipeline.
    fn destroy_compute_pipeline(&self, id: ComputePipelineId);
}

/// A backend submission queue: uploads data and submits recorded work.
///
/// Uploads are ordered with respect to subsequent submissions on the same
/// queue, so data written before a submission is visible to it.
pub trait RenderQueue {
    /// Uploads `data` into `buffer` starting at `offset` bytes.
    fn write_buffer(&self, buffer: BufferId, offset: u64, data: &[u8]);
    /// Uploads `data` into the texture region described by `destination`,
    /// interpreting the source bytes according to `layout`.
    fn write_texture(&self, destination: TextureWrite, data: &[u8], layout: ImageDataLayout);
    /// Submits finished command buffers for execution in order.
    fn submit(&self, command_buffers: &[CommandBuffer]);
}
