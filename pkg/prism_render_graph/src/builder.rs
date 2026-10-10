//! Setup-phase and execute-phase pass contexts.
//!
//! [`PassBuilder`] is handed to a pass's *setup* closure: it creates virtual
//! resources and records how the pass reads and writes them, mutating the
//! graph's resource table through SSA versioning. [`ExecuteContext`] is handed
//! to the *execute* closure after compilation: it resolves virtual handles to
//! realized driver ids and accumulates the pass's draw/dispatch commands.

use alloc::string::String;
use alloc::vec::Vec;
use core::ops::Range;

use prism_render_driver::{
    BindGroupId, BufferId, Color, ComputeCommand, ComputePipelineId, IndexBufferBinding,
    IndexFormat, RenderCommand, RenderPipelineId, ScissorRect, SubresourceRange, TextureId,
    TextureViewId, Viewport,
};

use crate::access::{full_range, BufferUse, TextureUse};
use crate::blackboard::Blackboard;
use crate::desc::{BufferDesc, TextureDesc};
use crate::handle::{BufferHandle, ResourceIndex, TextureHandle};
use crate::pass::{BufferAccessRecord, PassKind, TextureAccessRecord};
use crate::resource::{BufferResource, Lifetime, TextureResource};

/// Builds a single pass during the setup phase.
///
/// Every `create_*` registers a new transient in the graph's resource table;
/// every `read_*` / `write_*` records an access and (for writes) mints a fresh
/// SSA version, returning the handle later passes must use to observe that
/// write. Usage bits are inferred into each resource as a side effect, so
/// authors seldom set them by hand.
pub struct PassBuilder<'g> {
    textures: &'g mut Vec<TextureResource>,
    buffers: &'g mut Vec<BufferResource>,
    blackboard: &'g mut Blackboard,
    kind: PassKind,
    tex_accesses: Vec<TextureAccessRecord>,
    buf_accesses: Vec<BufferAccessRecord>,
}

impl<'g> PassBuilder<'g> {
    pub(crate) fn new(
        textures: &'g mut Vec<TextureResource>,
        buffers: &'g mut Vec<BufferResource>,
        blackboard: &'g mut Blackboard,
        kind: PassKind,
    ) -> Self {
        Self {
            textures,
            buffers,
            blackboard,
            kind,
            tex_accesses: Vec::new(),
            buf_accesses: Vec::new(),
        }
    }

    /// The kind of pass being built.
    #[must_use]
    pub fn kind(&self) -> PassKind {
        self.kind
    }

    /// Borrows the frame blackboard for sharing handles with later passes.
    #[must_use]
    pub fn blackboard(&self) -> &Blackboard {
        self.blackboard
    }

    /// Mutably borrows the frame blackboard.
    pub fn blackboard_mut(&mut self) -> &mut Blackboard {
        self.blackboard
    }

    /// Registers a transient texture (frame-local, aliasable) and returns a
    /// version-0 handle to it.
    pub fn create_texture(&mut self, name: impl Into<String>, desc: TextureDesc) -> TextureHandle {
        self.push_texture(name, desc, Lifetime::Transient)
    }

    /// Registers a persistent texture (graph-allocated, preserved across
    /// frames, never aliased) and returns a version-0 handle to it.
    pub fn create_persistent_texture(
        &mut self,
        name: impl Into<String>,
        desc: TextureDesc,
    ) -> TextureHandle {
        self.push_texture(name, desc, Lifetime::Persistent)
    }

    fn push_texture(
        &mut self,
        name: impl Into<String>,
        desc: TextureDesc,
        lifetime: Lifetime,
    ) -> TextureHandle {
        let index = ResourceIndex(self.textures.len() as u32);
        self.textures.push(TextureResource {
            name: name.into(),
            desc,
            lifetime,
            imported: None,
            inferred_usage: prism_render_driver::TextureUsages::NONE,
            version: 0,
            realized: None,
        });
        TextureHandle::new(index, 0)
    }

    /// Registers a transient buffer (frame-local, aliasable) and returns a
    /// version-0 handle to it.
    pub fn create_buffer(&mut self, name: impl Into<String>, desc: BufferDesc) -> BufferHandle {
        self.push_buffer(name, desc, Lifetime::Transient)
    }

    /// Registers a persistent buffer (graph-allocated, preserved across frames,
    /// never aliased) and returns a version-0 handle to it.
    pub fn create_persistent_buffer(
        &mut self,
        name: impl Into<String>,
        desc: BufferDesc,
    ) -> BufferHandle {
        self.push_buffer(name, desc, Lifetime::Persistent)
    }

    fn push_buffer(
        &mut self,
        name: impl Into<String>,
        desc: BufferDesc,
        lifetime: Lifetime,
    ) -> BufferHandle {
        let index = ResourceIndex(self.buffers.len() as u32);
        self.buffers.push(BufferResource {
            name: name.into(),
            desc,
            lifetime,
            imported: None,
            inferred_usage: prism_render_driver::BufferUsages::NONE,
            version: 0,
            realized: None,
        });
        BufferHandle::new(index, 0)
    }

    // --- texture accesses -------------------------------------------------

    /// Records a texture access with an explicit subresource range, returning
    /// the post-access handle (a fresh version for writes, the same handle for
    /// reads).
    pub fn access_texture(
        &mut self,
        handle: TextureHandle,
        usage: TextureUse,
        range: SubresourceRange,
    ) -> TextureHandle {
        let res = &mut self.textures[handle.resource().get() as usize];
        res.inferred_usage = res.inferred_usage.union(usage.required_usage());
        if usage.is_write() {
            let output = res.version + 1;
            res.version = output;
            self.tex_accesses.push(TextureAccessRecord {
                resource: handle.resource(),
                input_version: handle.version(),
                produces: true,
                output_version: output,
                usage,
                range,
            });
            TextureHandle::new(handle.resource(), output)
        } else {
            self.tex_accesses.push(TextureAccessRecord {
                resource: handle.resource(),
                input_version: handle.version(),
                produces: false,
                output_version: 0,
                usage,
                range,
            });
            handle
        }
    }

    /// Records a whole-resource texture access.
    pub fn use_texture(&mut self, handle: TextureHandle, usage: TextureUse) -> TextureHandle {
        self.access_texture(handle, usage, full_range())
    }

    /// Samples a texture in a shader (read-only).
    pub fn sample(&mut self, handle: TextureHandle) -> TextureHandle {
        self.use_texture(handle, TextureUse::Sampled)
    }

    /// Reads a texture as a read-only storage image.
    pub fn read_storage_texture(&mut self, handle: TextureHandle) -> TextureHandle {
        self.use_texture(handle, TextureUse::StorageRead)
    }

    /// Writes a texture as a storage image (mints a new version).
    pub fn write_storage_texture(&mut self, handle: TextureHandle) -> TextureHandle {
        self.use_texture(handle, TextureUse::StorageWrite)
    }

    /// Declares a texture as a written color attachment (mints a new version).
    pub fn color_attachment(&mut self, handle: TextureHandle) -> TextureHandle {
        self.use_texture(handle, TextureUse::ColorAttachment)
    }

    /// Declares a texture as a written depth attachment (mints a new version).
    pub fn depth_attachment(&mut self, handle: TextureHandle) -> TextureHandle {
        self.use_texture(handle, TextureUse::DepthAttachment)
    }

    /// Declares a read-only depth attachment (depth test, no writes).
    pub fn read_depth_attachment(&mut self, handle: TextureHandle) -> TextureHandle {
        self.use_texture(handle, TextureUse::DepthAttachmentRead)
    }

    /// Reads a texture as a transfer source.
    pub fn copy_src_texture(&mut self, handle: TextureHandle) -> TextureHandle {
        self.use_texture(handle, TextureUse::CopySrc)
    }

    /// Writes a texture as a transfer destination (mints a new version).
    pub fn copy_dst_texture(&mut self, handle: TextureHandle) -> TextureHandle {
        self.use_texture(handle, TextureUse::CopyDst)
    }

    /// Declares a texture presented to a surface (a terminal read).
    pub fn present_texture(&mut self, handle: TextureHandle) -> TextureHandle {
        self.use_texture(handle, TextureUse::Present)
    }

    // --- buffer accesses --------------------------------------------------

    /// Records a buffer access, returning the post-access handle.
    pub fn use_buffer(&mut self, handle: BufferHandle, usage: BufferUse) -> BufferHandle {
        let res = &mut self.buffers[handle.resource().get() as usize];
        res.inferred_usage = res.inferred_usage.union(usage.required_usage());
        if usage.is_write() {
            let output = res.version + 1;
            res.version = output;
            self.buf_accesses.push(BufferAccessRecord {
                resource: handle.resource(),
                input_version: handle.version(),
                produces: true,
                output_version: output,
                usage,
            });
            BufferHandle::new(handle.resource(), output)
        } else {
            self.buf_accesses.push(BufferAccessRecord {
                resource: handle.resource(),
                input_version: handle.version(),
                produces: false,
                output_version: 0,
                usage,
            });
            handle
        }
    }

    /// Reads a buffer as a uniform buffer.
    pub fn read_uniform(&mut self, handle: BufferHandle) -> BufferHandle {
        self.use_buffer(handle, BufferUse::Uniform)
    }

    /// Reads a buffer as a read-only storage buffer.
    pub fn read_storage_buffer(&mut self, handle: BufferHandle) -> BufferHandle {
        self.use_buffer(handle, BufferUse::StorageRead)
    }

    /// Writes a buffer as a storage buffer (mints a new version).
    pub fn write_storage_buffer(&mut self, handle: BufferHandle) -> BufferHandle {
        self.use_buffer(handle, BufferUse::StorageWrite)
    }

    /// Reads a buffer as an index buffer.
    pub fn read_index(&mut self, handle: BufferHandle) -> BufferHandle {
        self.use_buffer(handle, BufferUse::Index)
    }

    /// Reads a buffer as a vertex buffer.
    pub fn read_vertex(&mut self, handle: BufferHandle) -> BufferHandle {
        self.use_buffer(handle, BufferUse::Vertex)
    }

    /// Reads a buffer as an indirect argument source.
    pub fn read_indirect(&mut self, handle: BufferHandle) -> BufferHandle {
        self.use_buffer(handle, BufferUse::Indirect)
    }

    /// Reads a buffer as a transfer source.
    pub fn copy_src_buffer(&mut self, handle: BufferHandle) -> BufferHandle {
        self.use_buffer(handle, BufferUse::CopySrc)
    }

    /// Writes a buffer as a transfer destination (mints a new version).
    pub fn copy_dst_buffer(&mut self, handle: BufferHandle) -> BufferHandle {
        self.use_buffer(handle, BufferUse::CopyDst)
    }

    /// Consumes the builder, yielding the recorded accesses.
    pub(crate) fn finish(self) -> (Vec<TextureAccessRecord>, Vec<BufferAccessRecord>) {
        (self.tex_accesses, self.buf_accesses)
    }
}

/// Resolves virtual handles and records driver commands during execution.
///
/// The executor builds one of these per surviving pass, runs the pass's
/// execute closure against it, then drains the recorded commands into a driver
/// pass. Handle resolution reads the already-realized resource table, so a
/// closure never sees an unrealized resource.
pub struct ExecuteContext<'e> {
    textures: &'e [TextureResource],
    buffers: &'e [BufferResource],
    kind: PassKind,
    render_commands: Vec<RenderCommand>,
    compute_commands: Vec<ComputeCommand>,
}

impl<'e> ExecuteContext<'e> {
    pub(crate) fn new(
        textures: &'e [TextureResource],
        buffers: &'e [BufferResource],
        kind: PassKind,
    ) -> Self {
        Self {
            textures,
            buffers,
            kind,
            render_commands: Vec::new(),
            compute_commands: Vec::new(),
        }
    }

    /// The kind of pass being executed.
    #[must_use]
    pub fn kind(&self) -> PassKind {
        self.kind
    }

    /// Resolves a texture handle to its realized driver texture.
    ///
    /// # Panics
    /// Panics if the resource was culled or never realized, which cannot happen
    /// for a resource a surviving pass accesses.
    #[must_use]
    pub fn texture(&self, handle: TextureHandle) -> TextureId {
        self.textures[handle.resource().get() as usize]
            .realized
            .expect("texture accessed by a surviving pass must be realized")
            .texture
    }

    /// Resolves a texture handle to its realized default view.
    ///
    /// # Panics
    /// Panics if the resource was culled or never realized.
    #[must_use]
    pub fn texture_view(&self, handle: TextureHandle) -> TextureViewId {
        self.textures[handle.resource().get() as usize]
            .realized
            .expect("texture accessed by a surviving pass must be realized")
            .default_view
    }

    /// Resolves a buffer handle to its realized driver buffer.
    ///
    /// # Panics
    /// Panics if the resource was culled or never realized.
    #[must_use]
    pub fn buffer(&self, handle: BufferHandle) -> BufferId {
        self.buffers[handle.resource().get() as usize]
            .realized
            .expect("buffer accessed by a surviving pass must be realized")
    }

    // --- raster recording -------------------------------------------------

    /// Binds the active render pipeline.
    pub fn set_render_pipeline(&mut self, pipeline: RenderPipelineId) -> &mut Self {
        self.render_commands
            .push(RenderCommand::SetPipeline(pipeline));
        self
    }

    /// Binds a bind group for raster work at `@group(index)`.
    pub fn set_render_bind_group(
        &mut self,
        index: u32,
        bind_group: BindGroupId,
        dynamic_offsets: Vec<u32>,
    ) -> &mut Self {
        self.render_commands.push(RenderCommand::SetBindGroup {
            index,
            bind_group,
            dynamic_offsets,
        });
        self
    }

    /// Binds a vertex buffer at `slot`.
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: BufferId, offset: u64) -> &mut Self {
        self.render_commands.push(RenderCommand::SetVertexBuffer {
            slot,
            buffer,
            offset,
        });
        self
    }

    /// Binds the index buffer.
    pub fn set_index_buffer(
        &mut self,
        buffer: BufferId,
        format: IndexFormat,
        offset: u64,
    ) -> &mut Self {
        self.render_commands
            .push(RenderCommand::SetIndexBuffer(IndexBufferBinding {
                buffer,
                format,
                offset,
            }));
        self
    }

    /// Restricts rasterization to a viewport.
    pub fn set_viewport(&mut self, viewport: Viewport) -> &mut Self {
        self.render_commands
            .push(RenderCommand::SetViewport(viewport));
        self
    }

    /// Restricts rasterization to a scissor rectangle.
    pub fn set_scissor(&mut self, scissor: ScissorRect) -> &mut Self {
        self.render_commands
            .push(RenderCommand::SetScissor(scissor));
        self
    }

    /// Sets the blend constant color.
    pub fn set_blend_constant(&mut self, color: Color) -> &mut Self {
        self.render_commands
            .push(RenderCommand::SetBlendConstant(color));
        self
    }

    /// Sets the stencil reference value.
    pub fn set_stencil_reference(&mut self, reference: u32) -> &mut Self {
        self.render_commands
            .push(RenderCommand::SetStencilReference(reference));
        self
    }

    /// Records a non-indexed draw.
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) -> &mut Self {
        self.render_commands.push(RenderCommand::Draw {
            vertices,
            instances,
        });
        self
    }

    /// Records an indexed draw.
    pub fn draw_indexed(
        &mut self,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) -> &mut Self {
        self.render_commands.push(RenderCommand::DrawIndexed {
            indices,
            base_vertex,
            instances,
        });
        self
    }

    /// Records an indirect draw.
    pub fn draw_indirect(&mut self, buffer: BufferId, offset: u64) -> &mut Self {
        self.render_commands
            .push(RenderCommand::DrawIndirect { buffer, offset });
        self
    }

    // --- compute recording ------------------------------------------------

    /// Binds the active compute pipeline.
    pub fn set_compute_pipeline(&mut self, pipeline: ComputePipelineId) -> &mut Self {
        self.compute_commands
            .push(ComputeCommand::SetPipeline(pipeline));
        self
    }

    /// Binds a bind group for compute work at `@group(index)`.
    pub fn set_compute_bind_group(
        &mut self,
        index: u32,
        bind_group: BindGroupId,
        dynamic_offsets: Vec<u32>,
    ) -> &mut Self {
        self.compute_commands.push(ComputeCommand::SetBindGroup {
            index,
            bind_group,
            dynamic_offsets,
        });
        self
    }

    /// Dispatches a grid of workgroups.
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) -> &mut Self {
        self.compute_commands
            .push(ComputeCommand::Dispatch { x, y, z });
        self
    }

    /// Dispatches with workgroup counts sourced from a buffer.
    pub fn dispatch_indirect(&mut self, buffer: BufferId, offset: u64) -> &mut Self {
        self.compute_commands
            .push(ComputeCommand::DispatchIndirect { buffer, offset });
        self
    }

    /// Drains the recorded render commands.
    pub(crate) fn take_render_commands(&mut self) -> Vec<RenderCommand> {
        core::mem::take(&mut self.render_commands)
    }

    /// Drains the recorded compute commands.
    pub(crate) fn take_compute_commands(&mut self) -> Vec<ComputeCommand> {
        core::mem::take(&mut self.compute_commands)
    }
}
