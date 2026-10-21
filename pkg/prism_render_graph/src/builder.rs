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
    BindGroupId, BufferId, Color, ComputeCommand, ComputePipelineId, CopyCommand, Extent3d,
    IndexBufferBinding, IndexFormat, RenderCommand, RenderPipelineId, ScissorRect,
    SubresourceRange, TextureId, TextureViewId, Viewport,
};

use crate::access::{full_range, BufferUse, TextureUse};
use crate::blackboard::Blackboard;
use crate::capacity::GraphLimits;
use crate::desc::{BufferDesc, TextureDesc};
use crate::handle::{BufferHandle, ResourceIndex, TextureHandle};
use crate::pass::{BufferAccessRecord, PassKind, TextureAccessRecord};
use crate::plan::CompileError;
use crate::resource::{BufferResource, Lifetime, TextureResource};

/// Builds a single pass during the setup phase.
///
/// Every `create_*` registers a new transient in the graph's resource table;
/// every `read_*` / `write_*` records an access and (for writes) mints a fresh
/// SSA version, returning the handle later passes must use to observe that
/// write. Usage bits are inferred into each resource as a side effect, so
/// authors seldom set them by hand.
///
/// Infallible registration/access helpers are intended for bounded startup
/// declarations. Their checked `try_*` counterparts reject before changing
/// resource usage, versions or access tables. Convenience access helpers share
/// the preconditions of [`Self::access_texture`] or [`Self::use_buffer`].
pub struct PassBuilder<'g> {
    textures: &'g mut Vec<TextureResource>,
    buffers: &'g mut Vec<BufferResource>,
    blackboard: &'g mut Blackboard,
    kind: PassKind,
    limits: GraphLimits,
    tex_accesses: Vec<TextureAccessRecord>,
    buf_accesses: Vec<BufferAccessRecord>,
}

impl<'g> PassBuilder<'g> {
    pub(crate) fn new(
        textures: &'g mut Vec<TextureResource>,
        buffers: &'g mut Vec<BufferResource>,
        blackboard: &'g mut Blackboard,
        kind: PassKind,
        limits: GraphLimits,
    ) -> Self {
        Self {
            textures,
            buffers,
            blackboard,
            kind,
            limits,
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
    ///
    /// # Panics
    /// Panics if the texture limit is exhausted. Use [`Self::try_create_texture`]
    /// for dynamic declarations.
    pub fn create_texture(&mut self, name: impl Into<String>, desc: TextureDesc) -> TextureHandle {
        self.push_texture(name, desc, Lifetime::Transient)
    }

    /// Registers a persistent texture (graph-allocated, preserved across
    /// frames, never aliased) and returns a version-0 handle to it.
    ///
    /// # Panics
    /// Panics if the texture limit is exhausted. Use
    /// [`Self::try_create_persistent_texture`] for dynamic declarations.
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
        self.try_push_texture(name, desc, lifetime)
            .expect("graph texture capacity precondition; use try_create_texture for dynamic input")
    }

    fn try_push_texture(
        &mut self,
        name: impl Into<String>,
        desc: TextureDesc,
        lifetime: Lifetime,
    ) -> Result<TextureHandle, CompileError> {
        let index = ResourceIndex(GraphLimits::next_id(
            "textures",
            self.textures.len(),
            self.limits.resources,
        )?);
        self.textures.push(TextureResource {
            name: name.into(),
            desc,
            lifetime,
            imported: None,
            inferred_usage: prism_render_driver::TextureUsages::NONE,
            version: 0,
            realized: None,
        });
        Ok(TextureHandle::new(index, 0))
    }

    /// Admits a transient texture before growing the table or narrowing its ID.
    pub fn try_create_texture(
        &mut self,
        name: impl Into<String>,
        desc: TextureDesc,
    ) -> Result<TextureHandle, CompileError> {
        self.try_push_texture(name, desc, Lifetime::Transient)
    }

    /// Checked persistent registration; rejects before table growth.
    pub fn try_create_persistent_texture(
        &mut self,
        name: impl Into<String>,
        desc: TextureDesc,
    ) -> Result<TextureHandle, CompileError> {
        self.try_push_texture(name, desc, Lifetime::Persistent)
    }

    /// Registers a transient buffer (frame-local, aliasable) and returns a
    /// version-0 handle to it.
    ///
    /// # Panics
    /// Panics if the buffer limit is exhausted. Use [`Self::try_create_buffer`]
    /// for dynamic declarations.
    pub fn create_buffer(&mut self, name: impl Into<String>, desc: BufferDesc) -> BufferHandle {
        self.push_buffer(name, desc, Lifetime::Transient)
    }

    /// Registers a persistent buffer (graph-allocated, preserved across frames,
    /// never aliased) and returns a version-0 handle to it.
    ///
    /// # Panics
    /// Panics if the buffer limit is exhausted. Use
    /// [`Self::try_create_persistent_buffer`] for dynamic declarations.
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
        self.try_push_buffer(name, desc, lifetime)
            .expect("graph buffer capacity precondition; use try_create_buffer for dynamic input")
    }

    fn try_push_buffer(
        &mut self,
        name: impl Into<String>,
        desc: BufferDesc,
        lifetime: Lifetime,
    ) -> Result<BufferHandle, CompileError> {
        let index = ResourceIndex(GraphLimits::next_id(
            "buffers",
            self.buffers.len(),
            self.limits.resources,
        )?);
        self.buffers.push(BufferResource {
            name: name.into(),
            desc,
            lifetime,
            imported: None,
            inferred_usage: prism_render_driver::BufferUsages::NONE,
            version: 0,
            realized: None,
        });
        Ok(BufferHandle::new(index, 0))
    }

    /// Admits a transient buffer before growing the table or narrowing its ID.
    pub fn try_create_buffer(
        &mut self,
        name: impl Into<String>,
        desc: BufferDesc,
    ) -> Result<BufferHandle, CompileError> {
        self.try_push_buffer(name, desc, Lifetime::Transient)
    }

    /// Checked persistent registration; rejects before table growth.
    pub fn try_create_persistent_buffer(
        &mut self,
        name: impl Into<String>,
        desc: BufferDesc,
    ) -> Result<BufferHandle, CompileError> {
        self.try_push_buffer(name, desc, Lifetime::Persistent)
    }

    // --- texture accesses -------------------------------------------------

    /// Records a texture access with an explicit subresource range, returning
    /// the post-access handle (a fresh version for writes, the same handle for
    /// reads).
    ///
    /// # Panics
    /// Panics on an out-of-bounds/future handle or an exhausted access/version
    /// limit. The handle must belong to this graph; in-range foreign handles
    /// cannot be distinguished. Use [`Self::try_access_texture`] for dynamic
    /// input. Stale input versions are allowed and preserve SSA dependencies.
    pub fn access_texture(
        &mut self,
        handle: TextureHandle,
        usage: TextureUse,
        range: SubresourceRange,
    ) -> TextureHandle {
        self.try_access_texture(handle, usage, range)
            .expect("graph texture access capacity/version precondition")
    }

    /// Checked access; leaves usage, version and access table unchanged on error.
    ///
    /// # Errors
    /// Returns a capacity, invalid-handle or version-overflow error before
    /// mutation. The handle must belong to this graph; this API validates its
    /// index and version, not graph identity or backend subresource legality.
    pub fn try_access_texture(
        &mut self,
        handle: TextureHandle,
        usage: TextureUse,
        range: SubresourceRange,
    ) -> Result<TextureHandle, CompileError> {
        self.check_next_access()?;
        let invalid = CompileError::InvalidHandle {
            kind: "texture",
            index: handle.resource().get(),
            version: handle.version(),
        };
        let res = self
            .textures
            .get_mut(handle.resource().get() as usize)
            .ok_or(invalid.clone())?;
        if handle.version() > res.version {
            return Err(invalid);
        }
        let output = if usage.is_write() {
            Some(self.limits.next_version(res.version)?)
        } else {
            None
        };
        res.inferred_usage = res.inferred_usage.union(usage.required_usage());
        if let Some(output) = output {
            res.version = output;
            self.tex_accesses.push(TextureAccessRecord {
                resource: handle.resource(),
                input_version: handle.version(),
                produces: true,
                output_version: output,
                usage,
                range,
            });
            Ok(TextureHandle::new(handle.resource(), output))
        } else {
            self.tex_accesses.push(TextureAccessRecord {
                resource: handle.resource(),
                input_version: handle.version(),
                produces: false,
                output_version: 0,
                usage,
                range,
            });
            Ok(handle)
        }
    }

    /// Records a whole-resource texture access.
    ///
    /// # Panics
    /// Shares the handle and admission preconditions of [`Self::access_texture`].
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
    ///
    /// # Panics
    /// Panics on an out-of-bounds/future handle or an exhausted access/version
    /// limit. The handle must belong to this graph; in-range foreign handles
    /// cannot be distinguished. Use [`Self::try_use_buffer`] for dynamic input.
    /// Stale input versions are allowed and preserve SSA dependencies.
    pub fn use_buffer(&mut self, handle: BufferHandle, usage: BufferUse) -> BufferHandle {
        self.try_use_buffer(handle, usage)
            .expect("graph buffer access capacity/version precondition")
    }

    /// Checked buffer access; does not change versions on rejection.
    ///
    /// # Errors
    /// Returns a capacity, invalid-handle or version-overflow error before
    /// mutation. The handle must belong to this graph; this API validates its
    /// index and version, not graph identity.
    pub fn try_use_buffer(
        &mut self,
        handle: BufferHandle,
        usage: BufferUse,
    ) -> Result<BufferHandle, CompileError> {
        self.check_next_access()?;
        let invalid = CompileError::InvalidHandle {
            kind: "buffer",
            index: handle.resource().get(),
            version: handle.version(),
        };
        let res = self
            .buffers
            .get_mut(handle.resource().get() as usize)
            .ok_or(invalid.clone())?;
        if handle.version() > res.version {
            return Err(invalid);
        }
        let output = if usage.is_write() {
            Some(self.limits.next_version(res.version)?)
        } else {
            None
        };
        res.inferred_usage = res.inferred_usage.union(usage.required_usage());
        if let Some(output) = output {
            res.version = output;
            self.buf_accesses.push(BufferAccessRecord {
                resource: handle.resource(),
                input_version: handle.version(),
                produces: true,
                output_version: output,
                usage,
            });
            Ok(BufferHandle::new(handle.resource(), output))
        } else {
            self.buf_accesses.push(BufferAccessRecord {
                resource: handle.resource(),
                input_version: handle.version(),
                produces: false,
                output_version: 0,
                usage,
            });
            Ok(handle)
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

    fn check_next_access(&self) -> Result<(), CompileError> {
        let count = self
            .tex_accesses
            .len()
            .checked_add(self.buf_accesses.len())
            .ok_or(CompileError::Capacity {
                kind: "accesses",
                count: usize::MAX,
                limit: self.limits.accesses_per_pass,
            })?;
        GraphLimits::next_id("accesses", count, self.limits.accesses_per_pass)?;
        Ok(())
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
    copy_commands: Vec<CopyCommand>,
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
            copy_commands: Vec::new(),
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

    // --- indirect recording ----------------------------------------------
    //
    // Every indirect command takes a [`BufferHandle`] and resolves it through
    // [`Self::buffer`], the same realized table a pass's declared accesses were
    // checked against. That keeps the graph's "declared accesses only" contract
    // unbroken: a pass cannot drive an indirect draw from a buffer it never
    // declared (and therefore never synchronized against), because it has no
    // handle for it.

    /// Records an indirect (non-indexed) draw from a declared argument buffer.
    pub fn draw_indirect(&mut self, args: BufferHandle, offset: u64) -> &mut Self {
        let buffer = self.buffer(args);
        self.render_commands
            .push(RenderCommand::DrawIndirect { buffer, offset });
        self
    }

    /// Records an indexed indirect draw from a declared argument buffer.
    ///
    /// The buffer holds one packed
    /// [`DRAW_INDEXED_INDIRECT_ARGS_SIZE`](prism_render_driver::DRAW_INDEXED_INDIRECT_ARGS_SIZE)-byte
    /// record; letting a compute pass author its `instance_count` is what makes
    /// the draw GPU-driven.
    pub fn draw_indexed_indirect(&mut self, args: BufferHandle, offset: u64) -> &mut Self {
        let buffer = self.buffer(args);
        self.render_commands
            .push(RenderCommand::DrawIndexedIndirect { buffer, offset });
        self
    }

    /// Records `count` packed indexed indirect draws from a declared argument
    /// array.
    ///
    /// `stride` is the byte distance between consecutive records. This is the
    /// multi-draw primitive that collapses many GPU-authored draws into one
    /// command.
    pub fn multi_draw_indirect(
        &mut self,
        args: BufferHandle,
        offset: u64,
        count: u32,
        stride: u32,
    ) -> &mut Self {
        let buffer = self.buffer(args);
        self.render_commands.push(RenderCommand::MultiDrawIndirect {
            buffer,
            offset,
            count,
            stride,
        });
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

    /// Writes owned push-constant bytes after binding the compute pipeline.
    /// No payload copy is made. Offset and length must meet the backend's
    /// alignment, enabled device limits and pipeline range requirements; this
    /// command-only context cannot reflect the bound device or pipeline layout.
    pub fn set_compute_push_constants(&mut self, offset: u32, data: Vec<u8>) -> &mut Self {
        self.compute_commands
            .push(ComputeCommand::SetPushConstants { offset, data });
        self
    }

    /// Dispatches a grid of workgroups.
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) -> &mut Self {
        self.compute_commands
            .push(ComputeCommand::Dispatch { x, y, z });
        self
    }

    /// Dispatches with workgroup counts sourced from a declared argument buffer.
    pub fn dispatch_indirect(&mut self, args: BufferHandle, offset: u64) -> &mut Self {
        let buffer = self.buffer(args);
        self.compute_commands
            .push(ComputeCommand::DispatchIndirect { buffer, offset });
        self
    }

    // --- transfer recording ----------------------------------------------

    /// Records a buffer-to-buffer copy, resolving both handles to their
    /// realized driver buffers.
    pub fn copy_buffer_to_buffer(
        &mut self,
        src: BufferHandle,
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        size: u64,
    ) -> &mut Self {
        let src = self.buffer(src);
        let dst = self.buffer(dst);
        self.copy_commands.push(CopyCommand::CopyBufferToBuffer {
            src,
            src_offset,
            dst,
            dst_offset,
            size,
        });
        self
    }

    /// Records a buffer-to-texture copy, resolving the source buffer and
    /// destination texture handles to their realized driver objects.
    pub fn copy_buffer_to_texture(
        &mut self,
        src: BufferHandle,
        src_offset: u64,
        bytes_per_row: Option<u32>,
        rows_per_image: Option<u32>,
        dst: TextureHandle,
        dst_mip_level: u32,
        dst_origin: [u32; 3],
        size: Extent3d,
    ) -> &mut Self {
        let src = self.buffer(src);
        let dst = self.texture(dst);
        self.copy_commands.push(CopyCommand::CopyBufferToTexture {
            src,
            src_offset,
            bytes_per_row,
            rows_per_image,
            dst,
            dst_mip_level,
            dst_origin,
            size,
        });
        self
    }

    /// Records a texture-to-buffer copy, resolving the source texture and
    /// destination buffer handles to their realized driver objects.
    pub fn copy_texture_to_buffer(
        &mut self,
        src: TextureHandle,
        src_mip_level: u32,
        src_origin: [u32; 3],
        dst: BufferHandle,
        dst_offset: u64,
        bytes_per_row: Option<u32>,
        rows_per_image: Option<u32>,
        size: Extent3d,
    ) -> &mut Self {
        let src = self.texture(src);
        let dst = self.buffer(dst);
        self.copy_commands.push(CopyCommand::CopyTextureToBuffer {
            src,
            src_mip_level,
            src_origin,
            dst,
            dst_offset,
            bytes_per_row,
            rows_per_image,
            size,
        });
        self
    }

    /// Records a texture-to-texture copy, resolving both handles to their
    /// realized driver textures.
    pub fn copy_texture_to_texture(
        &mut self,
        src: TextureHandle,
        src_mip_level: u32,
        src_origin: [u32; 3],
        dst: TextureHandle,
        dst_mip_level: u32,
        dst_origin: [u32; 3],
        size: Extent3d,
    ) -> &mut Self {
        let src = self.texture(src);
        let dst = self.texture(dst);
        self.copy_commands.push(CopyCommand::CopyTextureToTexture {
            src,
            src_mip_level,
            src_origin,
            dst,
            dst_mip_level,
            dst_origin,
            size,
        });
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

    /// Drains the recorded copy commands.
    pub(crate) fn take_copy_commands(&mut self) -> Vec<CopyCommand> {
        core::mem::take(&mut self.copy_commands)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn compute_push_constants_move_exact_bytes_in_command_order() {
        let pipeline = ComputePipelineId::from_parts(2, 3);
        let group = BindGroupId::from_parts(4, 5);
        let payload: Vec<u8> = (0..20).collect();
        let allocation = payload.as_ptr();
        let mut context = ExecuteContext::new(&[], &[], PassKind::Compute);
        context
            .set_compute_pipeline(pipeline)
            .set_compute_bind_group(0, group, Vec::new())
            .set_compute_push_constants(4, payload)
            .dispatch(1, 2, 3);
        let commands = context.take_compute_commands();
        let ComputeCommand::SetPushConstants { offset, data } = &commands[2] else {
            panic!("push constants must precede dispatch");
        };
        assert_eq!(*offset, 4);
        assert_eq!(
            data.as_ptr(),
            allocation,
            "payload must move without copying"
        );
        assert_eq!(
            commands,
            vec![
                ComputeCommand::SetPipeline(pipeline),
                ComputeCommand::SetBindGroup {
                    index: 0,
                    bind_group: group,
                    dynamic_offsets: Vec::new()
                },
                ComputeCommand::SetPushConstants {
                    offset: 4,
                    data: (0..20).collect()
                },
                ComputeCommand::Dispatch { x: 1, y: 2, z: 3 },
            ]
        );
        assert!(context.take_compute_commands().is_empty());
    }

    #[test]
    fn compute_without_push_constants_keeps_existing_command_stream() {
        let mut context = ExecuteContext::new(&[], &[], PassKind::Compute);
        context.dispatch(1, 1, 1);
        assert_eq!(
            context.take_compute_commands(),
            vec![ComputeCommand::Dispatch { x: 1, y: 1, z: 1 }]
        );
    }
}
