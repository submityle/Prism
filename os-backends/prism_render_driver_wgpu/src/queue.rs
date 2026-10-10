//! The [`WgpuQueue`]: uploads and command-buffer submission against the shared
//! [`crate::device::DeviceInner`].
//!
//! The queue co-owns the device's slot maps (through an [`Arc`]) so it can
//! resolve the same ids the device minted. [`WgpuQueue::submit`] walks each
//! backend-agnostic [`rhi::CommandBuffer`], encodes it into a real
//! `wgpu::CommandEncoder`, replays every recorded pass/command, and hands the
//! finished command buffers to the wgpu queue in submission order.

use alloc::vec::Vec;

use prism_render_driver as rhi;
use rhi::RenderQueue;

use crate::convert;
use crate::device::DeviceInner;
use alloc::sync::Arc;

/// A wgpu-backed [`rhi::RenderQueue`].
///
/// Shares the device's resource tables so submitted command buffers resolve
/// their ids against the exact objects the device created.
pub struct WgpuQueue {
    /// The underlying wgpu queue.
    queue: wgpu::Queue,
    /// The shared device state, used to resolve ids during submission.
    inner: Arc<DeviceInner>,
}

/// A color attachment with its referenced views owned for the pass duration.
struct OwnedColorAttachment {
    /// The attachment view rendered into.
    view: wgpu::TextureView,
    /// The optional MSAA resolve target.
    resolve: Option<wgpu::TextureView>,
    /// The load/store operations for the attachment.
    ops: wgpu::Operations<wgpu::Color>,
}

/// A depth/stencil attachment with its view owned for the pass duration.
struct OwnedDepthAttachment {
    /// The depth/stencil view.
    view: wgpu::TextureView,
    /// Depth-plane operations, if present.
    depth: Option<wgpu::Operations<f32>>,
    /// Stencil-plane operations, if present.
    stencil: Option<wgpu::Operations<u32>>,
}

impl WgpuQueue {
    /// Wraps a ready wgpu queue sharing `inner` with its device.
    #[must_use]
    pub(crate) fn from_parts(queue: wgpu::Queue, inner: Arc<DeviceInner>) -> Self {
        Self { queue, inner }
    }

    /// Borrows the underlying wgpu queue, e.g. to present a surface frame.
    #[must_use]
    pub fn wgpu_queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Encodes one backend-agnostic command buffer into a wgpu command buffer.
    fn encode(&self, command_buffer: &rhi::CommandBuffer) -> wgpu::CommandBuffer {
        let mut encoder =
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: command_buffer.label.as_deref(),
                });
        for pass in &command_buffer.passes {
            match pass {
                rhi::Pass::Render {
                    descriptor,
                    commands,
                } => self.encode_render_pass(&mut encoder, descriptor, commands),
                rhi::Pass::Compute { label, commands } => {
                    self.encode_compute_pass(&mut encoder, label.as_deref(), commands);
                }
            }
        }
        encoder.finish()
    }

    /// Replays a render pass: resolves attachments, opens the pass, and walks
    /// every recorded [`rhi::RenderCommand`].
    fn encode_render_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        descriptor: &rhi::RenderPassDescriptor,
        commands: &[rhi::RenderCommand],
    ) {
        // Resolve attachment views into owned handles first so the borrowed
        // `RenderPassColorAttachment`s below outlive the resolution locks.
        let owned_colors: Vec<Option<OwnedColorAttachment>> = descriptor
            .color_attachments
            .iter()
            .map(|slot| {
                slot.map(|attachment| OwnedColorAttachment {
                    view: self.inner.resolve_texture_view(attachment.view),
                    resolve: attachment
                        .resolve_target
                        .map(|id| self.inner.resolve_texture_view(id)),
                    ops: convert::color_operations(attachment.load, attachment.store),
                })
            })
            .collect();

        let owned_depth: Option<OwnedDepthAttachment> =
            descriptor
                .depth_stencil_attachment
                .map(|attachment| OwnedDepthAttachment {
                    view: self.inner.resolve_texture_view(attachment.view),
                    depth: attachment.depth.map(convert::depth_operations),
                    stencil: attachment.stencil.map(convert::stencil_operations),
                });

        let color_attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = owned_colors
            .iter()
            .map(|slot| {
                slot.as_ref().map(|owned| wgpu::RenderPassColorAttachment {
                    view: &owned.view,
                    depth_slice: None,
                    resolve_target: owned.resolve.as_ref(),
                    ops: owned.ops,
                })
            })
            .collect();

        let depth_stencil_attachment =
            owned_depth
                .as_ref()
                .map(|owned| wgpu::RenderPassDepthStencilAttachment {
                    view: &owned.view,
                    depth_ops: owned.depth,
                    stencil_ops: owned.stencil,
                });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: descriptor.label.as_deref(),
            color_attachments: &color_attachments,
            depth_stencil_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        for command in commands {
            self.replay_render_command(&mut render_pass, command);
        }
    }

    /// Replays a single render command against an open render pass.
    fn replay_render_command(&self, pass: &mut wgpu::RenderPass<'_>, command: &rhi::RenderCommand) {
        match command {
            rhi::RenderCommand::SetPipeline(id) => {
                let pipeline = self.inner.resolve_render_pipeline(*id);
                pass.set_pipeline(&pipeline);
            }
            rhi::RenderCommand::SetBindGroup {
                index,
                bind_group,
                dynamic_offsets,
            } => {
                let group = self.inner.resolve_bind_group(*bind_group);
                pass.set_bind_group(*index, Some(&group), dynamic_offsets);
            }
            rhi::RenderCommand::SetVertexBuffer {
                slot,
                buffer,
                offset,
            } => {
                let buf = self.inner.resolve_buffer(*buffer);
                pass.set_vertex_buffer(*slot, buf.slice(*offset..));
            }
            rhi::RenderCommand::SetIndexBuffer(binding) => {
                let buf = self.inner.resolve_buffer(binding.buffer);
                pass.set_index_buffer(
                    buf.slice(binding.offset..),
                    convert::index_format(binding.format),
                );
            }
            rhi::RenderCommand::SetViewport(viewport) => {
                pass.set_viewport(
                    viewport.x,
                    viewport.y,
                    viewport.width,
                    viewport.height,
                    viewport.min_depth,
                    viewport.max_depth,
                );
            }
            rhi::RenderCommand::SetScissor(rect) => {
                pass.set_scissor_rect(rect.x, rect.y, rect.width, rect.height);
            }
            rhi::RenderCommand::SetBlendConstant(c) => {
                pass.set_blend_constant(convert::color(*c));
            }
            rhi::RenderCommand::SetStencilReference(reference) => {
                pass.set_stencil_reference(*reference);
            }
            rhi::RenderCommand::Draw {
                vertices,
                instances,
            } => {
                pass.draw(vertices.clone(), instances.clone());
            }
            rhi::RenderCommand::DrawIndexed {
                indices,
                base_vertex,
                instances,
            } => {
                pass.draw_indexed(indices.clone(), *base_vertex, instances.clone());
            }
            rhi::RenderCommand::DrawIndirect { buffer, offset } => {
                let buf = self.inner.resolve_buffer(*buffer);
                pass.draw_indirect(&buf, *offset);
            }
        }
    }

    /// Replays a compute pass: opens the pass and walks every recorded
    /// [`rhi::ComputeCommand`].
    fn encode_compute_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: Option<&str>,
        commands: &[rhi::ComputeCommand],
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label,
            timestamp_writes: None,
        });
        for command in commands {
            match command {
                rhi::ComputeCommand::SetPipeline(id) => {
                    let pipeline = self.inner.resolve_compute_pipeline(*id);
                    pass.set_pipeline(&pipeline);
                }
                rhi::ComputeCommand::SetBindGroup {
                    index,
                    bind_group,
                    dynamic_offsets,
                } => {
                    let group = self.inner.resolve_bind_group(*bind_group);
                    pass.set_bind_group(*index, Some(&group), dynamic_offsets);
                }
                rhi::ComputeCommand::Dispatch { x, y, z } => {
                    pass.dispatch_workgroups(*x, *y, *z);
                }
                rhi::ComputeCommand::DispatchIndirect { buffer, offset } => {
                    let buf = self.inner.resolve_buffer(*buffer);
                    pass.dispatch_workgroups_indirect(&buf, *offset);
                }
            }
        }
    }
}

impl RenderQueue for WgpuQueue {
    fn write_buffer(&self, buffer: rhi::BufferId, offset: u64, data: &[u8]) {
        let buf = self.inner.resolve_buffer(buffer);
        self.queue.write_buffer(&buf, offset, data);
    }

    fn write_texture(
        &self,
        destination: rhi::TextureWrite,
        data: &[u8],
        layout: rhi::ImageDataLayout,
    ) {
        let texture = self.inner.resolve_texture(destination.texture);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: destination.mip_level,
                origin: wgpu::Origin3d {
                    x: destination.origin[0],
                    y: destination.origin[1],
                    z: destination.origin[2],
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: layout.offset,
                bytes_per_row: layout.bytes_per_row,
                rows_per_image: layout.rows_per_image,
            },
            convert::extent3d(destination.size),
        );
    }

    fn submit(&self, command_buffers: &[rhi::CommandBuffer]) {
        let encoded: Vec<wgpu::CommandBuffer> =
            command_buffers.iter().map(|cb| self.encode(cb)).collect();
        self.queue.submit(encoded);
    }
}
