//! The [`WgpuQueue`]: uploads and command-buffer submission against the shared
//! [`crate::device::DeviceInner`].
//!
//! The queue co-owns the device's slot maps (through an [`Arc`]) so it can
//! resolve the same ids the device minted. [`WgpuQueue::submit`] walks each
//! backend-agnostic [`rhi::CommandBuffer`], encodes it into a real
//! `wgpu::CommandEncoder`, replays every recorded pass/command, and hands the
//! finished command buffers to the wgpu queue in submission order.

use alloc::vec::Vec;
use core::any::Any;

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

/// Builds a texel copy buffer layout from the RHI's optional block hints.
fn buffer_layout(
    offset: u64,
    bytes_per_row: Option<u32>,
    rows_per_image: Option<u32>,
) -> wgpu::TexelCopyBufferLayout {
    wgpu::TexelCopyBufferLayout {
        offset,
        bytes_per_row,
        rows_per_image,
    }
}

/// Builds a texel copy texture region covering the whole `aspect`.
fn texture_copy_info(
    texture: &wgpu::Texture,
    mip_level: u32,
    origin: [u32; 3],
) -> wgpu::TexelCopyTextureInfo<'_> {
    wgpu::TexelCopyTextureInfo {
        texture,
        mip_level,
        origin: wgpu::Origin3d {
            x: origin[0],
            y: origin[1],
            z: origin[2],
        },
        aspect: wgpu::TextureAspect::All,
    }
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
                rhi::Pass::Transfer { label, commands } => {
                    self.encode_transfer_pass(&mut encoder, label.as_deref(), commands);
                }
            }
        }
        encoder.finish()
    }

    /// Replays a transfer pass: issues each recorded [`rhi::CopyCommand`] as
    /// its native wgpu `copy_*` call, in submission order.
    fn encode_transfer_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: Option<&str>,
        commands: &[rhi::CopyCommand],
    ) {
        // wgpu has no first-class "transfer pass"; a debug group bounds the
        // copies so the label still appears in GPU capture tooling.
        if let Some(label) = label {
            encoder.push_debug_group(label);
        }
        for command in commands {
            self.replay_copy_command(encoder, command);
        }
        if label.is_some() {
            encoder.pop_debug_group();
        }
    }

    /// Translates one [`rhi::CopyCommand`] into the matching wgpu copy call.
    fn replay_copy_command(&self, encoder: &mut wgpu::CommandEncoder, command: &rhi::CopyCommand) {
        match command {
            rhi::CopyCommand::CopyBufferToBuffer {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } => {
                let src = self.inner.resolve_buffer(*src);
                let dst = self.inner.resolve_buffer(*dst);
                encoder.copy_buffer_to_buffer(&src, *src_offset, &dst, *dst_offset, *size);
            }
            rhi::CopyCommand::CopyBufferToTexture {
                src,
                src_offset,
                bytes_per_row,
                rows_per_image,
                dst,
                dst_mip_level,
                dst_origin,
                size,
            } => {
                let src = self.inner.resolve_buffer(*src);
                let dst = self.inner.resolve_texture(*dst);
                encoder.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer: &src,
                        layout: buffer_layout(*src_offset, *bytes_per_row, *rows_per_image),
                    },
                    texture_copy_info(&dst, *dst_mip_level, *dst_origin),
                    convert::extent3d(*size),
                );
            }
            rhi::CopyCommand::CopyTextureToBuffer {
                src,
                src_mip_level,
                src_origin,
                dst,
                dst_offset,
                bytes_per_row,
                rows_per_image,
                size,
            } => {
                let src = self.inner.resolve_texture(*src);
                let dst = self.inner.resolve_buffer(*dst);
                encoder.copy_texture_to_buffer(
                    texture_copy_info(&src, *src_mip_level, *src_origin),
                    wgpu::TexelCopyBufferInfo {
                        buffer: &dst,
                        layout: buffer_layout(*dst_offset, *bytes_per_row, *rows_per_image),
                    },
                    convert::extent3d(*size),
                );
            }
            rhi::CopyCommand::CopyTextureToTexture {
                src,
                src_mip_level,
                src_origin,
                dst,
                dst_mip_level,
                dst_origin,
                size,
            } => {
                let src = self.inner.resolve_texture(*src);
                let dst = self.inner.resolve_texture(*dst);
                encoder.copy_texture_to_texture(
                    texture_copy_info(&src, *src_mip_level, *src_origin),
                    texture_copy_info(&dst, *dst_mip_level, *dst_origin),
                    convert::extent3d(*size),
                );
            }
        }
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
            rhi::RenderCommand::DrawIndexedIndirect { buffer, offset } => {
                let buf = self.inner.resolve_buffer(*buffer);
                pass.draw_indexed_indirect(&buf, *offset);
            }
            rhi::RenderCommand::DrawIndirect { buffer, offset } => {
                let buf = self.inner.resolve_buffer(*buffer);
                pass.draw_indirect(&buf, *offset);
            }
            rhi::RenderCommand::MultiDrawIndirect {
                buffer,
                offset,
                count,
                stride,
            } => {
                // wgpu has no stride parameter: it always reads tightly packed
                // `DrawIndexedIndirectArgs`. Anything else would silently read
                // the wrong offsets, so surface the mismatch in debug builds
                // (release builds keep the RHI's documented status quo).
                debug_assert_eq!(
                    *stride,
                    rhi::DRAW_INDEXED_INDIRECT_ARGS_SIZE,
                    "wgpu requires tightly packed indirect draw arguments"
                );
                let buf = self.inner.resolve_buffer(*buffer);
                pass.multi_draw_indexed_indirect(&buf, *offset, *count);
            }
            rhi::RenderCommand::SetPushConstants { offset, data, .. } => {
                pass.set_immediates(*offset, data);
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
                rhi::ComputeCommand::SetPushConstants { offset, data } => {
                    pass.set_immediates(*offset, data);
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
        // The frozen trait is fire-and-forget; discard the submission index.
        let _ = self.submit_tracked(command_buffers);
    }
}

impl WgpuQueue {
    /// Encodes and submits `command_buffers`, returning `wgpu`'s completion
    /// token (its submission index) for the batch.
    ///
    /// This is the tracked counterpart to the frozen
    /// [`RenderQueue::submit`], which discards the index. Frames-in-flight
    /// pacing holds the returned index and waits on it with
    /// [`wait_for_submission`](Self::wait_for_submission) to know when a frame
    /// has actually retired.
    #[must_use]
    pub fn submit_tracked(&self, command_buffers: &[rhi::CommandBuffer]) -> wgpu::SubmissionIndex {
        let encoded: Vec<wgpu::CommandBuffer> =
            command_buffers.iter().map(|cb| self.encode(cb)).collect();
        self.queue.submit(encoded)
    }

    /// Blocks the calling thread until the submission named by `index` has
    /// completed.
    ///
    /// Returns `true` once the submission is done, or `false` if the device was
    /// lost or `wgpu` rejected the index. Prefer this over a queue-wide wait
    /// when pacing frames: it waits for exactly one frame rather than for every
    /// later submission too.
    pub fn wait_for_submission(&self, index: &wgpu::SubmissionIndex) -> bool {
        self.inner.wait_for_submission(index)
    }

    /// Type-erased [`wait_for_submission`](Self::wait_for_submission).
    ///
    /// Accepts a token boxed through the [`Any`] boundary by a `FrameSync`
    /// owner, downcasting it back to a submission index internally so that owner
    /// can carry tokens without naming `wgpu` types. Returns `false` if `token`
    /// did not originate from this backend.
    pub fn wait_for_submission_erased(&self, token: &(dyn Any + Send + Sync)) -> bool {
        token
            .downcast_ref::<wgpu::SubmissionIndex>()
            .is_some_and(|index| self.wait_for_submission(index))
    }

    /// Blocks the calling thread until every submission on this queue has
    /// completed.
    ///
    /// Returns `true` on success, or `false` if the device was lost. Use this at
    /// idle points (readback, surface resize, shutdown) that need the GPU fully
    /// caught up.
    pub fn wait_idle(&self) -> bool {
        self.inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .is_ok()
    }
}

impl WgpuQueue {
    /// Reads an `Rgba8Unorm` 2D texture back into tightly-packed RGBA bytes.
    ///
    /// Blocks the calling thread until the copy completes and returns
    /// `width * height * 4` bytes in row-major order with the GPU's row padding
    /// stripped. The texture must be a single-mip, single-layer 2D texture in an
    /// 8-bit-per-channel RGBA format created with [`rhi::TextureUsages::COPY_SRC`].
    ///
    /// This is the backend hook golden-image and headless tests use to observe
    /// what the RHI actually rendered; it is not a hot-path API.
    #[must_use]
    pub fn read_texture_rgba8(
        &self,
        texture: rhi::TextureId,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        const BYTES_PER_TEXEL: u32 = 4;
        let unpadded_bytes_per_row = width * BYTES_PER_TEXEL;
        let padded_bytes_per_row =
            unpadded_bytes_per_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer_size = u64::from(padded_bytes_per_row) * u64::from(height);

        let texture = self.inner.resolve_texture(texture);
        let staging = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rhi-texture-readback"),
            size: buffer_size.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rhi-texture-readback"),
                });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        let _ = self.inner.device.poll(wgpu::PollType::wait_indefinitely());

        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.inner.device.poll(wgpu::PollType::wait_indefinitely());

        let view = staging
            .slice(..)
            .get_mapped_range()
            .expect("readback range is available after the device poll");
        let mut out = Vec::with_capacity((unpadded_bytes_per_row * height) as usize);
        for row in 0..height {
            let start = (row * padded_bytes_per_row) as usize;
            out.extend_from_slice(&view[start..start + unpadded_bytes_per_row as usize]);
        }
        drop(view);
        staging.unmap();
        out
    }

    /// Reads `size` bytes back from a buffer into a tightly packed `Vec<u8>`.
    ///
    /// Blocks the calling thread until the copy completes. The buffer must be
    /// created with [`rhi::BufferUsages::COPY_SRC`]; `size` is rounded up to
    /// wgpu's 4-byte copy alignment internally, and only the first `size` bytes
    /// are returned. This is the backend hook compute/indirect tests use to
    /// observe what the RHI actually produced on the GPU (a culling result, a
    /// GPU-written draw argument record); it is not a hot-path API.
    #[must_use]
    pub fn read_buffer(&self, buffer: rhi::BufferId, size: u64) -> Vec<u8> {
        if size == 0 {
            return Vec::new();
        }
        // Buffer copies must be a multiple of `COPY_BUFFER_ALIGNMENT` (4).
        let copy_size = size.next_multiple_of(4);
        let buffer = self.inner.resolve_buffer(buffer);
        let staging = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rhi-buffer-readback"),
            size: copy_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rhi-buffer-readback"),
                });
        encoder.copy_buffer_to_buffer(&buffer, 0, &staging, 0, copy_size);
        self.queue.submit([encoder.finish()]);
        let _ = self.inner.device.poll(wgpu::PollType::wait_indefinitely());

        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.inner.device.poll(wgpu::PollType::wait_indefinitely());

        let view = staging
            .slice(..)
            .get_mapped_range()
            .expect("readback range is available after the device poll");
        let mut out = Vec::with_capacity(size as usize);
        out.extend_from_slice(&view[..size as usize]);
        drop(view);
        staging.unmap();
        out
    }
}

impl rhi::FrameSync for WgpuQueue {
    fn submit_tracked(&self, command_buffers: &[rhi::CommandBuffer]) -> rhi::CompletionToken {
        // The inherent `submit_tracked` returns the native submission index and
        // takes precedence over this trait method; boxing type-erases it.
        Box::new(self.submit_tracked(command_buffers))
    }

    fn wait_for(&self, token: &(dyn Any + Send + Sync)) -> bool {
        self.wait_for_submission_erased(token)
    }

    fn wait_idle(&self) -> bool {
        self.wait_idle()
    }

    fn on_submitted_work_done(&self, callback: Box<dyn FnOnce() + Send + 'static>) {
        self.queue.on_submitted_work_done(callback);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{WgpuDevice, WgpuInstance};
    use prism_render_driver::RenderDevice as _;

    /// Brings up a device/queue pair, or `None` when no GPU is available so the
    /// suite passes without hardware.
    fn device_or_skip() -> Option<(WgpuDevice, WgpuQueue)> {
        WgpuInstance::new().request_default_device().ok()
    }

    /// Whether a missing adapter must fail the test instead of skipping it.
    fn require_gpu() -> bool {
        std::env::var_os("PRISM_REQUIRE_GPU").is_some()
    }

    #[test]
    fn gpu_transfer_pass_copies_buffer_to_buffer() {
        const PAYLOAD: [u8; 16] = [
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
            0x1e, 0x1f,
        ];
        let Some((device, queue)) = device_or_skip() else {
            assert!(
                !require_gpu(),
                "PRISM_REQUIRE_GPU is set but no adapter was found"
            );
            return;
        };

        let src = device.create_buffer(&rhi::BufferDescriptor {
            label: Some("transfer-src".into()),
            size: PAYLOAD.len() as u64,
            usage: rhi::BufferUsages::COPY_SRC | rhi::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dst = device.create_buffer(&rhi::BufferDescriptor {
            label: Some("transfer-dst".into()),
            size: PAYLOAD.len() as u64,
            usage: rhi::BufferUsages::COPY_SRC | rhi::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(src, 0, &PAYLOAD);

        let mut encoder = rhi::CommandEncoder::labeled("transfer-cb");
        encoder.push_transfer_pass(
            Some("copy".into()),
            vec![rhi::CopyCommand::CopyBufferToBuffer {
                src,
                src_offset: 0,
                dst,
                dst_offset: 0,
                size: PAYLOAD.len() as u64,
            }],
        );
        queue.submit(&[encoder.finish()]);

        // Read the destination back through the queue's readback hook: the
        // bytes the transfer pass wrote must survive verbatim.
        assert_eq!(
            queue.read_buffer(dst, PAYLOAD.len() as u64),
            PAYLOAD.to_vec(),
            "the transfer pass must have copied the source bytes verbatim"
        );
        // A short read returns exactly the requested prefix.
        assert_eq!(
            queue.read_buffer(dst, 6),
            PAYLOAD[..6].to_vec(),
            "the readback honors a partial size"
        );

        device.destroy_buffer(src);
        device.destroy_buffer(dst);
    }

    #[test]
    fn gpu_tracked_submission_reports_completion() {
        let Some((device, queue)) = device_or_skip() else {
            assert!(
                !require_gpu(),
                "PRISM_REQUIRE_GPU is set but no adapter was found"
            );
            return;
        };
        // An empty command buffer is a valid, side-effect-free submission; its
        // readiness is exactly what frames-in-flight pacing waits on.
        let commands = [rhi::CommandEncoder::new().finish()];

        let index = queue.submit_tracked(&commands);
        assert!(
            queue.wait_for_submission(&index),
            "the queue's typed completion hook confirms the submission finished"
        );
        assert!(
            device.wait_for_submission(&index),
            "the device's completion hook confirms the same submission"
        );

        // The type-erased route (what the runtime's pacer carries) must accept a
        // token from this backend and reject a foreign one rather than block.
        let erased: Box<dyn Any + Send + Sync> = Box::new(queue.submit_tracked(&commands));
        assert!(
            queue.wait_for_submission_erased(erased.as_ref()),
            "the erased token route completes a real submission"
        );
        assert!(
            !queue.wait_for_submission_erased(&0_u32),
            "a token from another source is rejected, not silently accepted"
        );

        assert!(queue.wait_idle(), "the queue drains to idle");
    }
}
