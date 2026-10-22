//! Render-pass attachments and the recorded draw/dispatch command model.

use crate::color::Color;
use crate::flags::ShaderStages;
use crate::resource::{
    BindGroupId, BufferId, ComputePipelineId, RenderPipelineId, TextureId, TextureViewId,
};
use crate::state::IndexFormat;
use crate::texture::Extent3d;
use alloc::string::String;
use alloc::vec::Vec;
use core::ops::Range;

/// The byte size of one packed indirect-draw argument record, shared by
/// [`RenderCommand::DrawIndexedIndirect`] (a single record) and
/// [`RenderCommand::MultiDrawIndirect`] (a tightly packed array of records).
///
/// A record is the five 32-bit fields the GPU consumes when resolving a draw:
/// `index_count`, `instance_count`, `first_index`, `base_vertex` and
/// `first_instance` — 20 bytes in that order. Backends that require tightly
/// packed records (wgpu does) treat this as the mandatory
/// [`RenderCommand::MultiDrawIndirect::stride`].
pub const DRAW_INDEXED_INDIRECT_ARGS_SIZE: u32 = 20;

/// The byte size of the WebGPU-compatible non-indexed indirect draw record
/// (`vertex_count`, `instance_count`, `first_vertex`, `first_instance`).
pub const DRAW_INDIRECT_ARGS_SIZE: u32 = 16;

/// The byte size of the WebGPU-compatible indirect dispatch record
/// (`workgroup_count_x`, `workgroup_count_y`, `workgroup_count_z`).
pub const DISPATCH_INDIRECT_ARGS_SIZE: u32 = 12;

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
    /// Draw indexed, sourcing the arguments from a buffer instead of the
    /// command stream.
    ///
    /// The buffer holds one packed `DRAW_INDEXED_INDIRECT_ARGS_SIZE`-byte
    /// record — `index_count`, `instance_count`, `first_index`, `base_vertex`,
    /// `first_instance` — at `offset`. Letting the GPU fill those bytes (a
    /// compute pass writing them is the usual producer) is what makes a draw
    /// "GPU-driven": the CPU never learns the instance count, so the work it
    /// dispatches is bounded by a readback or by a budget rather than by the
    /// exact surviving set. Requires the
    /// `Features::INDIRECT_DRAW` capability and an index buffer bound with
    /// [`SetIndexBuffer`](Self::SetIndexBuffer).
    DrawIndexedIndirect {
        /// The buffer holding the packed argument record.
        buffer: BufferId,
        /// The byte offset of the record.
        offset: u64,
    },
    /// Issue an indirect draw sourced from a buffer.
    DrawIndirect {
        /// The buffer holding draw arguments.
        buffer: BufferId,
        /// The byte offset of the arguments.
        offset: u64,
    },
    /// Issue `count` indexed indirect draws from one tightly packed array.
    ///
    /// This is the multi-draw-indexed-indirect primitive: starting at `offset`,
    /// `count` records of `stride` bytes are read in order and each resolved as
    /// an indexed draw (exactly what
    /// [`DrawIndexedIndirect`](Self::DrawIndexedIndirect) does for one record).
    /// It collapses many small GPU-authored draws into a single command, which
    /// is the point of a GPU-driven pipeline: the CPU emits one call regardless
    /// of how many batches the GPU decided to draw.
    ///
    /// `stride` is the byte distance between records. Backends that cannot
    /// honor an arbitrary stride require the tightly packed
    /// `DRAW_INDEXED_INDIRECT_ARGS_SIZE`; passing any other value there is a
    /// validation error. Requires
    /// `Features::MULTI_DRAW_INDIRECT` and an index buffer bound with
    /// [`SetIndexBuffer`](Self::SetIndexBuffer).
    MultiDrawIndirect {
        /// The buffer holding the packed argument array.
        buffer: BufferId,
        /// The byte offset of the first record.
        offset: u64,
        /// The number of records (draws) to issue.
        count: u32,
        /// The byte distance between consecutive records.
        stride: u32,
    },
    /// Upload push constants for the given stages.
    ///
    /// The bytes are written verbatim starting at `offset`; the pipeline's
    /// [`crate::pipeline::PushConstantRange`]s define the legal window and the
    /// stage visibility. Backends map this to their native immediate/push
    /// constant mechanism; `stages` is recorded so validation and native
    /// backends that need an explicit stage mask can honor it.
    SetPushConstants {
        /// The stages that may read these constants.
        stages: ShaderStages,
        /// The byte offset to write at.
        offset: u32,
        /// The constant bytes to upload.
        data: Vec<u8>,
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
    /// Upload push constants for the compute stage.
    ///
    /// The bytes are written verbatim starting at `offset`; the pipeline's
    /// [`crate::pipeline::PushConstantRange`]s define the legal window.
    SetPushConstants {
        /// The byte offset to write at.
        offset: u32,
        /// The constant bytes to upload.
        data: Vec<u8>,
    },
}

/// A single recorded copy/transfer command inside a transfer pass.
///
/// Transfers move bytes between buffers and textures without running a shader.
/// They are recorded into a [`Pass::Transfer`], which a backend translates to
/// its native `copy_*` calls. Buffer regions are expressed in bytes and texture
/// regions in texels; `bytes_per_row` / `rows_per_image` are optional block
/// layout hints (`None` means tightly packed / a single row), matching the
/// RHI's upload layout vocabulary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CopyCommand {
    /// Copy `size` bytes from `src` to `dst` at the given byte offsets.
    CopyBufferToBuffer {
        /// The buffer to read from.
        src: BufferId,
        /// The byte offset into the source buffer.
        src_offset: u64,
        /// The buffer to write to.
        dst: BufferId,
        /// The byte offset into the destination buffer.
        dst_offset: u64,
        /// The number of bytes to copy.
        size: u64,
    },
    /// Copy texels from a buffer into a texture region.
    CopyBufferToTexture {
        /// The buffer to read from.
        src: BufferId,
        /// The byte offset into the source buffer.
        src_offset: u64,
        /// Bytes per row of the source image, or `None` for tightly packed rows.
        bytes_per_row: Option<u32>,
        /// Rows per image of the source image, or `None` for a single image.
        rows_per_image: Option<u32>,
        /// The texture to write to.
        dst: TextureId,
        /// The destination mip level.
        dst_mip_level: u32,
        /// The destination texel origin `[x, y, z]`.
        dst_origin: [u32; 3],
        /// The texel extent to copy.
        size: Extent3d,
    },
    /// Copy texels from a texture region into a buffer.
    CopyTextureToBuffer {
        /// The texture to read from.
        src: TextureId,
        /// The source mip level.
        src_mip_level: u32,
        /// The source texel origin `[x, y, z]`.
        src_origin: [u32; 3],
        /// The buffer to write to.
        dst: BufferId,
        /// The byte offset into the destination buffer.
        dst_offset: u64,
        /// Bytes per row of the destination image, or `None` for tightly packed rows.
        bytes_per_row: Option<u32>,
        /// Rows per image of the destination image, or `None` for a single image.
        rows_per_image: Option<u32>,
        /// The texel extent to copy.
        size: Extent3d,
    },
    /// Copy texels from one texture region into another.
    CopyTextureToTexture {
        /// The texture to read from.
        src: TextureId,
        /// The source mip level.
        src_mip_level: u32,
        /// The source texel origin `[x, y, z]`.
        src_origin: [u32; 3],
        /// The texture to write to.
        dst: TextureId,
        /// The destination mip level.
        dst_mip_level: u32,
        /// The destination texel origin `[x, y, z]`.
        dst_origin: [u32; 3],
        /// The texel extent to copy.
        size: Extent3d,
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
    /// A transfer pass with recorded copy commands and no shader work.
    Transfer {
        /// A debug label surfaced in GPU tooling.
        label: Option<String>,
        /// The copy commands recorded in submission order.
        commands: Vec<CopyCommand>,
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

    /// Records a transfer pass from its copy command list.
    ///
    /// Transfer passes carry only [`CopyCommand`]s: no attachments, pipelines,
    /// or dispatches. A backend translates each command to its native
    /// buffer/texture copy call in submission order.
    pub fn push_transfer_pass(
        &mut self,
        label: Option<String>,
        commands: Vec<CopyCommand>,
    ) -> &mut Self {
        self.buffer.passes.push(Pass::Transfer { label, commands });
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn transfer_pass_records_copy_commands() {
        let src = BufferId::from_parts(1, 0);
        let dst = BufferId::from_parts(2, 0);
        let texture = TextureId::from_parts(3, 0);
        let extent = Extent3d::new_2d(4, 4);

        let mut encoder = CommandEncoder::labeled("frame");
        encoder.push_transfer_pass(
            Some(String::from("upload")),
            vec![
                CopyCommand::CopyBufferToBuffer {
                    src,
                    src_offset: 0,
                    dst,
                    dst_offset: 256,
                    size: 1024,
                },
                CopyCommand::CopyBufferToTexture {
                    src,
                    src_offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(4),
                    dst: texture,
                    dst_mip_level: 0,
                    dst_origin: [0, 0, 0],
                    size: extent,
                },
                CopyCommand::CopyTextureToBuffer {
                    src: texture,
                    src_mip_level: 0,
                    src_origin: [1, 2, 0],
                    dst,
                    dst_offset: 0,
                    bytes_per_row: None,
                    rows_per_image: None,
                    size: extent,
                },
                CopyCommand::CopyTextureToTexture {
                    src: texture,
                    src_mip_level: 0,
                    src_origin: [0, 0, 0],
                    dst: texture,
                    dst_mip_level: 1,
                    dst_origin: [0, 0, 0],
                    size: extent,
                },
            ],
        );

        assert_eq!(encoder.pass_count(), 1);
        let cb = encoder.finish();
        match &cb.passes[0] {
            Pass::Transfer { label, commands } => {
                assert_eq!(label.as_deref(), Some("upload"));
                assert_eq!(commands.len(), 4);
                assert_eq!(
                    commands[0],
                    CopyCommand::CopyBufferToBuffer {
                        src,
                        src_offset: 0,
                        dst,
                        dst_offset: 256,
                        size: 1024,
                    }
                );
                assert_eq!(
                    commands[1],
                    CopyCommand::CopyBufferToTexture {
                        src,
                        src_offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(4),
                        dst: texture,
                        dst_mip_level: 0,
                        dst_origin: [0, 0, 0],
                        size: extent,
                    }
                );
            }
            other => panic!("expected a transfer pass, got {other:?}"),
        }
    }

    #[test]
    fn push_constant_commands_round_trip() {
        let mut encoder = CommandEncoder::new();
        encoder.push_render_pass(
            RenderPassDescriptor::default(),
            vec![RenderCommand::SetPushConstants {
                stages: ShaderStages::vertex_fragment(),
                offset: 0,
                data: vec![1, 2, 3, 4],
            }],
        );
        encoder.push_compute_pass(
            None,
            vec![ComputeCommand::SetPushConstants {
                offset: 16,
                data: vec![5, 6],
            }],
        );

        let cb = encoder.finish();
        match &cb.passes[0] {
            Pass::Render { commands, .. } => assert_eq!(
                commands[0],
                RenderCommand::SetPushConstants {
                    stages: ShaderStages::vertex_fragment(),
                    offset: 0,
                    data: Vec::from([1u8, 2, 3, 4]),
                }
            ),
            other => panic!("expected a render pass, got {other:?}"),
        }
        match &cb.passes[1] {
            Pass::Compute { commands, .. } => assert_eq!(
                commands[0],
                ComputeCommand::SetPushConstants {
                    offset: 16,
                    data: Vec::from([5u8, 6]),
                }
            ),
            other => panic!("expected a compute pass, got {other:?}"),
        }
    }

    #[test]
    fn indirect_draw_commands_round_trip() {
        let buffer = BufferId::from_parts(7, 1);
        let mut encoder = CommandEncoder::labeled("indirect");
        encoder.push_render_pass(
            RenderPassDescriptor::default(),
            vec![
                RenderCommand::DrawIndexedIndirect { buffer, offset: 0 },
                RenderCommand::MultiDrawIndirect {
                    buffer,
                    offset: 256,
                    count: 3,
                    stride: DRAW_INDEXED_INDIRECT_ARGS_SIZE,
                },
            ],
        );

        let cb = encoder.finish();
        match &cb.passes[0] {
            Pass::Render { commands, .. } => {
                assert_eq!(
                    commands[0],
                    RenderCommand::DrawIndexedIndirect { buffer, offset: 0 }
                );
                assert_eq!(
                    commands[1],
                    RenderCommand::MultiDrawIndirect {
                        buffer,
                        offset: 256,
                        count: 3,
                        stride: DRAW_INDEXED_INDIRECT_ARGS_SIZE,
                    }
                );
            }
            other => panic!("expected a render pass, got {other:?}"),
        }
    }

    #[test]
    fn packed_indirect_args_record_is_twenty_bytes() {
        // Five 32-bit fields: index_count, instance_count, first_index,
        // base_vertex, first_instance.
        assert_eq!(DRAW_INDEXED_INDIRECT_ARGS_SIZE, 5 * 4);
    }

    #[test]
    fn packed_indirect_draw_and_dispatch_record_sizes_match_webgpu_abi() {
        assert_eq!(DRAW_INDIRECT_ARGS_SIZE, 4 * 4);
        assert_eq!(DISPATCH_INDIRECT_ARGS_SIZE, 3 * 4);
    }
}
