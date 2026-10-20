//! Bounded rectangular GPU copy and padded-row ID parsing.

use alloc::{collections::BTreeSet, vec::Vec};
use core::fmt;
use std::sync::mpsc;

use super::{
    validate_correlation, EncodedPickId, GpuPickingBackend, GpuPickingConfig, GpuPickingError,
    GpuPickingFrame, ID_BUFFER_BYTES_PER_TEXEL, READBACK_ROW_ALIGNMENT,
};
use crate::{BoxSelectionRequest, BoxSelectionResult, PixelRect};

/// Explicit per-copy limits. The caller also bounds concurrent in-flight GPU
/// buffers; the coordinator's pending bound alone does not own these buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuBoxSelectionLimits {
    /// Maximum number of actual pixels copied (not row padding).
    pub max_pixels: u64,
    /// Maximum padded staging-buffer bytes, also capped by device limits.
    pub max_bytes: u64,
    /// Maximum unique nonzero IDs returned. Exceeding it fails, never truncates.
    /// Set no higher than the associated coordinator's `max_hits`.
    pub max_ids: usize,
}

impl Default for GpuBoxSelectionLimits {
    fn default() -> Self {
        Self {
            max_pixels: 1_048_576,
            max_bytes: 32 * 1024 * 1024,
            max_ids: 4096,
        }
    }
}

/// Rectangular copy/map/parse failure. No partial selection is returned.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuBoxSelectionError {
    /// Frame identity or GPU readback failure.
    Picking(GpuPickingError),
    /// The requested inclusive-exclusive rectangle exceeds the target.
    RectangleOutOfBounds,
    /// Row alignment, buffer sizing, or native address space overflowed.
    LayoutOverflow,
    /// Pixel count exceeds the configured bound.
    PixelLimitExceeded {
        /// Requested pixels.
        pixels: u64,
        /// Configured maximum.
        limit: u64,
    },
    /// Padded byte count exceeds the configured or native device bound.
    ByteLimitExceeded {
        /// Required bytes.
        bytes: u64,
        /// Effective maximum.
        limit: u64,
    },
    /// The rectangle contains more unique IDs than permitted.
    IdLimitExceeded {
        /// Configured maximum.
        limit: usize,
    },
}

impl fmt::Display for GpuBoxSelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GPU box selection failed: {self:?}")
    }
}

impl core::error::Error for GpuBoxSelectionError {}

impl From<GpuPickingError> for GpuBoxSelectionError {
    fn from(error: GpuPickingError) -> Self {
        Self::Picking(error)
    }
}

#[derive(Clone, Copy, Debug)]
struct CopyLayout {
    width: u32,
    height: u32,
    row_bytes: u32,
    buffer_bytes: u64,
}

/// Encoded rectangle copy, waiting for a shared-queue submission.
#[derive(Debug)]
pub struct PendingBoxSelectionReadback {
    request: BoxSelectionRequest,
    buffer: wgpu::Buffer,
    layout: CopyLayout,
    max_ids: usize,
}

/// Submitted rectangle copy, polled without blocking. Owns its device/buffer
/// and can complete after resize or closure; current-state resolution is still
/// mandatory. Drop cancels mapping and releases the staging allocation.
pub struct InFlightBoxSelectionReadback {
    pending: PendingBoxSelectionReadback,
    device: wgpu::Device,
    receiver: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    completed: bool,
}

impl fmt::Debug for InFlightBoxSelectionReadback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InFlightBoxSelectionReadback")
            .field("request", &self.pending.request)
            .field("completed", &self.completed)
            .finish_non_exhaustive()
    }
}

impl InFlightBoxSelectionReadback {
    /// Returns immutable request correlation, even after the target closes.
    #[must_use]
    pub const fn request(&self) -> BoxSelectionRequest {
        self.pending.request
    }

    /// Polls device progress and delivers a result or error at most once.
    /// IDs are deduplicated on CPU from only the requested rectangle, ignoring
    /// zero/miss and row padding. This is not a GPU reduction algorithm.
    pub fn poll(&mut self) -> Result<Option<BoxSelectionResult>, GpuBoxSelectionError> {
        if self.completed {
            return Ok(None);
        }
        if let Err(error) = self.device.poll(wgpu::PollType::Poll) {
            self.completed = true;
            self.pending.buffer.destroy();
            return Err(GpuPickingError::DevicePollFailed(error.to_string()).into());
        }
        match self.receiver.try_recv() {
            Ok(result) => {
                self.completed = true;
                if let Err(error) = result {
                    self.pending.buffer.destroy();
                    return Err(GpuPickingError::ReadbackMapFailed(error.to_string()).into());
                }
                read_mapped(&self.pending).map(Some)
            }
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => {
                self.completed = true;
                self.pending.buffer.destroy();
                Err(GpuPickingError::ReadbackCallbackClosed.into())
            }
        }
    }
}

impl Drop for InFlightBoxSelectionReadback {
    fn drop(&mut self) {
        self.pending.buffer.destroy();
    }
}

impl GpuPickingFrame<'_> {
    /// Copies the requested rectangle into a bounded, 256-byte-row-aligned
    /// staging buffer. Rejects stale identity, out-of-bounds region and resource
    /// limits before allocating or recording commands. No full-target fallback.
    pub fn encode_box_readback(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        request: BoxSelectionRequest,
        limits: GpuBoxSelectionLimits,
    ) -> Result<PendingBoxSelectionReadback, GpuBoxSelectionError> {
        validate_correlation(
            self.state,
            self.scene_generation,
            request.viewport,
            request.generation,
            request.scene_generation,
            request.target_generation,
            request.target_handle,
        )?;
        let layout = copy_layout(
            self.backend.config,
            request.rectangle,
            limits,
            self.backend.device.limits().max_buffer_size,
        )?;
        let buffer = self.backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prism-editor-box-readback"),
            size: layout.buffer_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let min = request.rectangle.min();
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.backend.target,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: min.x,
                    y: min.y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(layout.row_bytes),
                    rows_per_image: Some(layout.height),
                },
            },
            wgpu::Extent3d {
                width: layout.width,
                height: layout.height,
                depth_or_array_layers: 1,
            },
        );
        Ok(PendingBoxSelectionReadback {
            request,
            buffer,
            layout,
            max_ids: limits.max_ids,
        })
    }
}

impl GpuPickingBackend {
    /// Submits an encoder containing the rectangle copy and starts mapping.
    pub fn submit_box_readback(
        &self,
        encoder: wgpu::CommandEncoder,
        pending: PendingBoxSelectionReadback,
    ) -> InFlightBoxSelectionReadback {
        self.queue.submit([encoder.finish()]);
        self.start_box_readback(pending)
    }

    /// Starts mapping after the host has submitted the rectangle copy to the
    /// shared queue. Supports batching several copies with surface/color work.
    pub fn start_box_readback(
        &self,
        pending: PendingBoxSelectionReadback,
    ) -> InFlightBoxSelectionReadback {
        let (sender, receiver) = mpsc::sync_channel(1);
        pending
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        InFlightBoxSelectionReadback {
            pending,
            device: self.device.clone(),
            receiver,
            completed: false,
        }
    }

    /// Explicit blocking policy for tools/tests, not per-frame presentation.
    pub fn submit_box_and_read_sync(
        &self,
        encoder: wgpu::CommandEncoder,
        pending: PendingBoxSelectionReadback,
    ) -> Result<BoxSelectionResult, GpuBoxSelectionError> {
        let mut in_flight = self.submit_box_readback(encoder, pending);
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| GpuPickingError::DevicePollFailed(error.to_string()))?;
        in_flight
            .poll()?
            .ok_or_else(|| GpuPickingError::ReadbackCallbackClosed.into())
    }
}

fn copy_layout(
    config: GpuPickingConfig,
    rectangle: PixelRect,
    limits: GpuBoxSelectionLimits,
    max_buffer_size: u64,
) -> Result<CopyLayout, GpuBoxSelectionError> {
    let max = rectangle.max();
    if max.x > config.width || max.y > config.height {
        return Err(GpuBoxSelectionError::RectangleOutOfBounds);
    }
    let width = rectangle.width();
    let height = rectangle.height();
    let pixels = u64::from(width) * u64::from(height);
    if pixels > limits.max_pixels {
        return Err(GpuBoxSelectionError::PixelLimitExceeded {
            pixels,
            limit: limits.max_pixels,
        });
    }
    let alignment = u64::from(READBACK_ROW_ALIGNMENT);
    let row_bytes =
        (u64::from(width) * u64::from(ID_BUFFER_BYTES_PER_TEXEL)).div_ceil(alignment) * alignment;
    let row_bytes = u32::try_from(row_bytes).map_err(|_| GpuBoxSelectionError::LayoutOverflow)?;
    let buffer_bytes = u64::from(row_bytes) * u64::from(height);
    let limit = limits.max_bytes.min(max_buffer_size).min(usize::MAX as u64);
    if buffer_bytes > limit {
        return Err(GpuBoxSelectionError::ByteLimitExceeded {
            bytes: buffer_bytes,
            limit,
        });
    }
    Ok(CopyLayout {
        width,
        height,
        row_bytes,
        buffer_bytes,
    })
}

fn parse_ids(
    bytes: &[u8],
    layout: CopyLayout,
    max_ids: usize,
) -> Result<Vec<EncodedPickId>, GpuBoxSelectionError> {
    if (bytes.len() as u64) < layout.buffer_bytes {
        return Err(GpuPickingError::ReadbackBufferTooSmall {
            actual: bytes.len(),
            required: layout.buffer_bytes as usize,
        }
        .into());
    }
    let mut ids = BTreeSet::new();
    for row in bytes
        .chunks_exact(layout.row_bytes as usize)
        .take(layout.height as usize)
    {
        for texel in
            row[..layout.width as usize * ID_BUFFER_BYTES_PER_TEXEL as usize].chunks_exact(16)
        {
            let mut channels = [0; 4];
            for (word, channel) in texel.chunks_exact(4).zip(&mut channels) {
                *channel = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
            }
            if channels != [0; 4] && !ids.contains(&channels) {
                if ids.len() >= max_ids {
                    return Err(GpuBoxSelectionError::IdLimitExceeded { limit: max_ids });
                }
                ids.insert(channels);
            }
        }
    }
    Ok(ids.into_iter().map(EncodedPickId::from_channels).collect())
}

fn read_mapped(
    pending: &PendingBoxSelectionReadback,
) -> Result<BoxSelectionResult, GpuBoxSelectionError> {
    let slice = pending.buffer.slice(..);
    let mapped = match slice.get_mapped_range() {
        Ok(mapped) => mapped,
        Err(error) => {
            pending.buffer.unmap();
            return Err(GpuPickingError::ReadbackMapFailed(error.to_string()).into());
        }
    };
    let ids = parse_ids(&mapped, pending.layout, pending.max_ids);
    drop(mapped);
    pending.buffer.unmap();
    let request = pending.request;
    Ok(BoxSelectionResult {
        id: request.id,
        viewport: request.viewport,
        generation: request.generation,
        scene_generation: request.scene_generation,
        target_generation: request.target_generation,
        target_handle: request.target_handle,
        ids: ids?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangle_layout_honors_pixel_padded_byte_and_native_limits() {
        let rectangle = PixelRect::new(1, 2, 4, 4).expect("rectangle");
        let config = GpuPickingConfig::new(8, 8);
        let limits = GpuBoxSelectionLimits {
            max_pixels: 6,
            max_bytes: 512,
            max_ids: 1,
        };
        let layout = copy_layout(config, rectangle, limits, 512).expect("padded rows");
        assert_eq!(
            (
                layout.width,
                layout.height,
                layout.row_bytes,
                layout.buffer_bytes
            ),
            (3, 2, 256, 512)
        );
        assert_eq!(
            copy_layout(
                config,
                rectangle,
                GpuBoxSelectionLimits {
                    max_pixels: 5,
                    ..limits
                },
                512
            )
            .unwrap_err(),
            GpuBoxSelectionError::PixelLimitExceeded {
                pixels: 6,
                limit: 5
            }
        );
        assert_eq!(
            copy_layout(config, rectangle, limits, 511).unwrap_err(),
            GpuBoxSelectionError::ByteLimitExceeded {
                bytes: 512,
                limit: 511
            }
        );
        assert_eq!(
            copy_layout(GpuPickingConfig::new(3, 8), rectangle, limits, 512).unwrap_err(),
            GpuBoxSelectionError::RectangleOutOfBounds
        );
    }

    #[test]
    fn parser_ignores_padding_and_miss_deduplicates_and_never_truncates() {
        let layout = CopyLayout {
            width: 2,
            height: 2,
            row_bytes: 256,
            buffer_bytes: 512,
        };
        let mut bytes = vec![255; 512]; // Padding would produce a bogus identity.
        for (offset, id) in [
            (0, [5, 1, 0, 0]),
            (16, [5, 1, 0, 0]),
            (256, [0; 4]),
            (272, [9, 2, 0, 0]),
        ] {
            for (word, channel) in id.into_iter().enumerate() {
                bytes[offset + word * 4..offset + word * 4 + 4]
                    .copy_from_slice(&u32::to_le_bytes(channel));
            }
        }
        let ids = parse_ids(&bytes, layout, 2).expect("unique IDs");
        assert_eq!(
            ids,
            vec![
                EncodedPickId::from_channels([5, 1, 0, 0]),
                EncodedPickId::from_channels([9, 2, 0, 0])
            ]
        );
        assert_eq!(
            parse_ids(&bytes, layout, 1),
            Err(GpuBoxSelectionError::IdLimitExceeded { limit: 1 })
        );
        assert!(parse_ids(&bytes[..511], layout, 2).is_err());
    }
}
