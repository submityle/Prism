//! Small RHI assembly helpers shared by the stages in this crate.
//!
//! Every stage in the slice builds the same few shapes — a whole-buffer bind
//! group entry, a bind-group layout entry, the command prefix of a single-quad
//! draw, and a black-cleared single-pass render target. Keeping them here means
//! the cull, the coverage draw, the scene draw and the naive baseline describe
//! those shapes once.

use prism_render_driver as rhi;
use rhi::{
    BindGroupEntry, BindGroupId, BindGroupLayoutEntry, BindingResource, BindingType,
    BufferBindingType, BufferId, Color, ColorAttachment, CommandBuffer, CommandEncoder,
    IndexBufferBinding, IndexFormat, LoadOp, RenderCommand, RenderPassDescriptor, RenderPipelineId,
    ShaderStages, StoreOp, TextureViewId,
};

/// A bind-group-layout entry for a whole-buffer binding visible to `visibility`.
pub(crate) fn buffer_entry(
    binding: u32,
    ty: BufferBindingType,
    visibility: ShaderStages,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A whole-buffer bind-group entry.
pub(crate) fn bind_buffer(binding: u32, buffer: BufferId) -> BindGroupEntry {
    BindGroupEntry {
        binding,
        resource: BindingResource::Buffer {
            buffer,
            offset: 0,
            size: None,
        },
    }
}

/// The command prefix every one-quad draw in this crate shares: bind the
/// pipeline, its single bind group, and the shared 16-bit index buffer.
pub(crate) fn quad_prefix(
    pipeline: RenderPipelineId,
    bind_group: BindGroupId,
    index_buffer: BufferId,
) -> Vec<RenderCommand> {
    vec![
        RenderCommand::SetPipeline(pipeline),
        RenderCommand::SetBindGroup {
            index: 0,
            bind_group,
            dynamic_offsets: Vec::new(),
        },
        RenderCommand::SetIndexBuffer(IndexBufferBinding {
            buffer: index_buffer,
            format: IndexFormat::Uint16,
            offset: 0,
        }),
    ]
}

/// Records a single render pass that clears `view` to opaque black, replays
/// `commands`, and stores the result.
///
/// Clearing to a known black frame is what lets a pixel readback distinguish a
/// drawn instance from the background: nothing but a draw can produce a
/// non-black pixel.
pub(crate) fn black_pass(
    label: &str,
    view: TextureViewId,
    commands: Vec<RenderCommand>,
) -> CommandBuffer {
    let mut encoder = CommandEncoder::labeled(label);
    push_black_pass(&mut encoder, label, view, commands);
    encoder.finish()
}

/// Records the same black-cleared render pass into an encoder a caller owns, so
/// several stages can share one command buffer.
pub(crate) fn push_black_pass(
    encoder: &mut CommandEncoder,
    label: &str,
    view: TextureViewId,
    commands: Vec<RenderCommand>,
) {
    encoder.push_render_pass(
        RenderPassDescriptor {
            label: Some(label.into()),
            color_attachments: vec![Some(ColorAttachment {
                view,
                resolve_target: None,
                load: LoadOp::Clear(Color::new(0.0, 0.0, 0.0, 1.0)),
                store: StoreOp::Store,
            })],
            depth_stencil_attachment: None,
        },
        commands,
    );
}
