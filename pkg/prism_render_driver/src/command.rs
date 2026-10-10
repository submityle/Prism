//! Render-pass attachments and the recorded draw/dispatch command model.

use crate::color::Color;
use crate::resource::{BindGroupId, BufferId, ComputePipelineId, RenderPipelineId, TextureViewId};
use crate::state::IndexFormat;
use alloc::string::String;
use alloc::vec::Vec;
use core::ops::Range;

/// What to do with an attachment's existing contents at the start of a pass.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum LoadOp {
    /// Clear to the given value.
    Clear(Color),
    /// Preserve and load the existing contents.
    Load,
}

/// What to do with an attachment's contents at the end of a pass.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum StoreOp {
    /// Write the results back to the attachment.
    #[default]
    Store,
    /// Discard the results.
    Discard,
}

/// Depth-plane load/store for a depth attachment.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DepthOperations {
    /// What to do with existing depth at pass start.
    pub load: DepthLoadOp,
    /// What to do with depth at pass end.
    pub store: StoreOp,
}

/// Load operation for the depth plane (clear value is a scalar).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DepthLoadOp {
    /// Clear to the given depth value.
    Clear(f32),
    /// Preserve the existing depth.
    Load,
}

/// Stencil-plane load/store for a depth/stencil attachment.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StencilOperations {
    /// What to do with existing stencil at pass start.
    pub load: StencilLoadOp,
    /// What to do with stencil at pass end.
    pub store: StoreOp,
}

/// Load operation for the stencil plane.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StencilLoadOp {
    /// Clear to the given stencil value.
    Clear(u32),
    /// Preserve the existing stencil.
    Load,
}

/// One color attachment of a render pass.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ColorAttachment {
    /// The view rendered into.
    pub view: TextureViewId,
    /// The resolve target for MSAA, if any.
    pub resolve_target: Option<TextureViewId>,
    /// What to do at pass start.
    pub load: LoadOp,
    /// What to do at pass end.
    pub store: StoreOp,
}

/// The depth/stencil attachment of a render pass.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DepthStencilAttachment {
    /// The depth/stencil view.
    pub view: TextureViewId,
    /// Depth-plane operations, if the attachment has depth.
    pub depth: Option<DepthOperations>,
    /// Stencil-plane operations, if the attachment has stencil.
    pub stencil: Option<StencilOperations>,
}

/// A description of the attachments a render pass renders into.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct RenderPassDescriptor {
    /// A debug label surfaced in GPU tooling.
    pub label: Option<String>,
    /// The color attachments, indexed by `@location(n)`; `None` skips a slot.
    pub color_attachments: Vec<Option<ColorAttachment>>,
    /// The optional depth/stencil attachment.
    pub depth_stencil_attachment: Option<DepthStencilAttachment>,
}

/// A rectangular viewport mapping clip space to framebuffer pixels.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Viewport {
    /// Left pixel coordinate.
    pub x: f32,
    /// Top pixel coordinate.
    pub y: f32,
    /// Width in pixels.
    pub width: f32,
    /// Height in pixels.
    pub height: f32,
    /// Minimum depth (usually 0).
    pub min_depth: f32,
    /// Maximum depth (usually 1).
    pub max_depth: f32,
}

/// A scissor rectangle in framebuffer pixels.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ScissorRect {
    /// Left pixel coordinate.
    pub x: u32,
    /// Top pixel coordinate.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// A binding of an index buffer for indexed draws.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IndexBufferBinding {
    /// The buffer holding indices.
    pub buffer: BufferId,
    /// The index width.
    pub format: IndexFormat,
    /// The byte offset into the buffer.
    pub offset: u64,
}

/// A single recorded command inside a render pass.
///
/// Recording to a data model (rather than issuing immediately) keeps the RHI
/// backend-agnostic and lets passes be validated, reordered, or replayed. A
/// backend walks this list and translates each command to its native API.
#[derive(Clone, PartialEq, Debug)]
pub enum RenderCommand {
    /// Bind the active render pipeline.
    SetPipeline(RenderPipelineId),
    /// Bind a bind group at `@group(index)` with optional dynamic offsets.
    SetBindGroup {
        /// The group index.
        index: u32,
        /// The bind group.
        bind_group: BindGroupId,
        /// Dynamic offsets, one per dynamic binding in the group.
        dynamic_offsets: Vec<u32>,
    },
    /// Bind a vertex buffer at `slot`.
    SetVertexBuffer {
        /// The vertex buffer slot.
        slot: u32,
        /// The buffer.
        buffer: BufferId,
        /// The byte offset into the buffer.
        offset: u64,
    },
    /// Bind the index buffer.
    SetIndexBuffer(IndexBufferBinding),
    /// Restrict rasterization to the viewport.
    SetViewport(Viewport),
    /// Restrict rasterization to the scissor rectangle.
    SetScissor(ScissorRect),
    /// Set the blend constant color.
    SetBlendConstant(Color),
    /// Set the stencil reference value.
    SetStencilReference(u32),
    /// Draw `vertices` for each instance in `instances`.
    Draw {
        /// The vertex range.
        vertices: Range<u32>,
        /// The instance range.
        instances: Range<u32>,
    },
    /// Draw indexed, with `base_vertex` added to each index.
    DrawIndexed {
        /// The index range.
        indices: Range<u32>,
        /// A value added to every index before fetching.
        base_vertex: i32,
        /// The instance range.
        instances: Range<u32>,
    },
    /// Issue an indirect draw sourced from a buffer.
    DrawIndirect {
        /// The buffer holding draw arguments.
        buffer: BufferId,
        /// The byte offset of the arguments.
        offset: u64,
    },
}

/// A single recorded command inside a compute pass.
#[derive(Clone, PartialEq, Debug)]
pub enum ComputeCommand {
    /// Bind the active compute pipeline.
    SetPipeline(ComputePipelineId),
    /// Bind a bind group at `@group(index)` with optional dynamic offsets.
    SetBindGroup {
        /// The group index.
        index: u32,
        /// The bind group.
        bind_group: BindGroupId,
        /// Dynamic offsets, one per dynamic binding in the group.
        dynamic_offsets: Vec<u32>,
    },
    /// Dispatch a grid of `(x, y, z)` workgroups.
    Dispatch {
        /// Workgroups along x.
        x: u32,
        /// Workgroups along y.
        y: u32,
        /// Workgroups along z.
        z: u32,
    },
    /// Dispatch with workgroup counts sourced from a buffer.
    DispatchIndirect {
        /// The buffer holding dispatch arguments.
        buffer: BufferId,
        /// The byte offset of the arguments.
        offset: u64,
    },
}

/// One recorded pass inside a command buffer.
///
/// A pass bundles its attachment description (for render passes) with the
/// ordered list of commands recorded into it. Keeping passes as data lets a
/// command buffer be validated, inspected, or replayed before a backend
/// translates it to native API calls.
#[derive(Clone, PartialEq, Debug)]
pub enum Pass {
    /// A render pass with its attachments and recorded draw commands.
    Render {
        /// The attachments the pass renders into.
        descriptor: RenderPassDescriptor,
        /// The commands recorded in submission order.
        commands: Vec<RenderCommand>,
    },
    /// A compute pass with its recorded dispatch commands.
    Compute {
        /// A debug label surfaced in GPU tooling.
        label: Option<String>,
        /// The commands recorded in submission order.
        commands: Vec<ComputeCommand>,
    },
}

/// A finished, backend-agnostic recording of passes ready for submission.
///
/// Produced by [`CommandEncoder::finish`] and consumed by a backend's queue.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct CommandBuffer {
    /// A debug label surfaced in GPU tooling.
    pub label: Option<String>,
    /// The passes in submission order.
    pub passes: Vec<Pass>,
}

/// Records passes into a [`CommandBuffer`].
///
/// The encoder is pure data assembly: it performs no GPU work, so it is fully
/// deterministic and testable. Backends consume the finished buffer rather than
/// driving the encoder directly, which keeps command construction identical
/// across every backend.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct CommandEncoder {
    buffer: CommandBuffer,
}

impl CommandEncoder {
    /// Creates an empty encoder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty encoder carrying a debug label.
    #[must_use]
    pub fn labeled(label: impl Into<String>) -> Self {
        Self {
            buffer: CommandBuffer {
                label: Some(label.into()),
                passes: Vec::new(),
            },
        }
    }

    /// Records a render pass from its attachment description and command list.
    pub fn push_render_pass(
        &mut self,
        descriptor: RenderPassDescriptor,
        commands: Vec<RenderCommand>,
    ) -> &mut Self {
        self.buffer.passes.push(Pass::Render {
            descriptor,
            commands,
        });
        self
    }

    /// Records a compute pass from its command list.
    pub fn push_compute_pass(
        &mut self,
        label: Option<String>,
        commands: Vec<ComputeCommand>,
    ) -> &mut Self {
        self.buffer.passes.push(Pass::Compute { label, commands });
        self
    }

    /// The number of passes recorded so far.
    #[must_use]
    pub fn pass_count(&self) -> usize {
        self.buffer.passes.len()
    }

    /// Consumes the encoder, yielding the finished command buffer.
    #[must_use]
    pub fn finish(self) -> CommandBuffer {
        self.buffer
    }
}
