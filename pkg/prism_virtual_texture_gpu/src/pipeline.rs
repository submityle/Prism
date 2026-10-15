//! Recoverable CPU/GPU virtual-texture frame orchestration.

use alloc::vec::Vec;
use core::fmt;

use prism_render_architecture::texture_streaming::{
    decode_feedback, FeedbackInput, PageDemand, StreamerConfig, StreamerFrame,
    VirtualTextureStreamer,
};

use crate::{
    context::{GpuContext, GpuError},
    feedback::GpuFeedbackDecode,
    page_table::{GpuPageTableStorage, PageTableStorageError},
};

/// Backend selected for the current frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VirtualTextureBackend {
    /// Deterministic device-free decode and residency path.
    Cpu,
    /// Real-device feedback decode and page-table upload path.
    Gpu,
}

/// Failure that caused a GPU frame to fall back to the CPU path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VirtualTextureError {
    /// The device or readback operation failed.
    Gpu(GpuError),
    /// The persistent page-table upload failed.
    PageTable(PageTableStorageError),
    /// The pipeline invariant that a GPU context owns page-table storage was
    /// broken before an upload.
    StorageUnavailable,
}

impl fmt::Display for VirtualTextureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gpu(error) => error.fmt(formatter),
            Self::PageTable(error) => error.fmt(formatter),
            Self::StorageUnavailable => {
                formatter.write_str("GPU page-table storage is unavailable")
            }
        }
    }
}

impl std::error::Error for VirtualTextureError {}

/// One frame result from feedback through residency and page-table upload.
#[derive(Clone, Debug)]
pub struct VirtualTextureFrame {
    /// Backend that successfully produced this frame.
    pub backend: VirtualTextureBackend,
    /// Deterministic residency decision and upload/eviction plan.
    pub streamer: StreamerFrame,
    /// Deduplicated demands consumed by the streamer.
    pub demands: Vec<PageDemand>,
    /// Error that caused a GPU attempt to fall back to CPU, if any.
    pub fallback: Option<VirtualTextureError>,
}

/// CPU-first VT coordinator with optional real-GPU feedback decode and a
/// persistent page-table allocation. GPU failures switch the frame to CPU and
/// leave the deterministic streamer state as the source of truth.
pub struct VirtualTexturePipeline {
    streamer: VirtualTextureStreamer,
    gpu: Option<GpuContext>,
    decode: Option<GpuFeedbackDecode>,
    storage: Option<GpuPageTableStorage>,
    backend: VirtualTextureBackend,
}

impl VirtualTexturePipeline {
    /// Creates a CPU-first pipeline with no device allocation.
    ///
    /// The CPU streamer remains the source of truth even after a GPU context is
    /// attached, so the pipeline can always continue when a device is absent or
    /// lost.
    #[must_use]
    pub fn new(config: StreamerConfig, pool_capacity: u32) -> Self {
        Self {
            streamer: VirtualTextureStreamer::new(config, pool_capacity),
            gpu: None,
            decode: None,
            storage: None,
            backend: VirtualTextureBackend::Cpu,
        }
    }

    /// Attaches only a real integrated/discrete adapter. CPU mode remains valid
    /// when the host has no adapter or only a software adapter.
    pub fn attach_real_gpu(&mut self, ctx: GpuContext) -> bool {
        if !ctx.is_real_gpu() || ctx.device_lost().is_some() {
            return false;
        }
        self.decode = Some(GpuFeedbackDecode::new(&ctx));
        self.gpu = Some(ctx);
        self.storage = None;
        self.backend = VirtualTextureBackend::Gpu;
        true
    }

    /// Disables the GPU path and releases its device-owned helpers.
    pub fn disable_gpu(&mut self) {
        self.gpu = None;
        self.decode = None;
        self.storage = None;
        self.backend = VirtualTextureBackend::Cpu;
    }

    /// Returns the backend selected for the next frame.
    #[must_use]
    pub const fn backend(&self) -> VirtualTextureBackend {
        self.backend
    }

    /// Returns the deterministic CPU residency streamer.
    #[must_use]
    pub const fn streamer(&self) -> &VirtualTextureStreamer {
        &self.streamer
    }

    /// Returns the persistent GPU page-table allocation, when GPU mode has
    /// successfully uploaded at least one table.
    #[must_use]
    pub fn page_table_storage(&self) -> Option<&GpuPageTableStorage> {
        self.storage.as_ref()
    }

    /// Processes one frame. A failed GPU decode or table upload falls back to
    /// the CPU path and returns the CPU frame with the failure in
    /// [`VirtualTextureFrame::fallback`].
    pub fn frame(&mut self, inputs: &[FeedbackInput<'_>]) -> VirtualTextureFrame {
        if self.backend == VirtualTextureBackend::Gpu
            && let (Some(ctx), Some(decode)) = (&self.gpu, &self.decode)
        {
            let mut demands = Vec::new();
            let pool = self.streamer.pool();
            let mut fallback = None;
            for input in inputs {
                match decode.try_decode(
                    ctx,
                    &input.desc,
                    input.grid,
                    |key| pool.contains(key).then_some(key.mip),
                    self.streamer.frame().saturating_add(1),
                ) {
                    Ok(mut decoded) => demands.append(&mut decoded),
                    Err(error) => {
                        fallback = Some(VirtualTextureError::Gpu(error));
                        break;
                    }
                }
            }
            if fallback.is_none() {
                let streamer = self.streamer.stream_demands(&demands);
                match self.upload_table() {
                    Ok(()) => {
                        return VirtualTextureFrame {
                            backend: VirtualTextureBackend::Gpu,
                            streamer,
                            demands,
                            fallback: None,
                        };
                    }
                    Err(error) => {
                        self.backend = VirtualTextureBackend::Cpu;
                        return VirtualTextureFrame {
                            backend: VirtualTextureBackend::Cpu,
                            streamer,
                            demands,
                            fallback: Some(error),
                        };
                    }
                }
            }
            self.backend = VirtualTextureBackend::Cpu;
            let demands = inputs
                .iter()
                .flat_map(|input| {
                    let pool = self.streamer.pool();
                    decode_feedback(
                        &input.desc,
                        input.grid,
                        |key| pool.contains(key).then_some(key.mip),
                        self.streamer.frame().saturating_add(1),
                    )
                })
                .collect::<Vec<_>>();
            let streamer = self.streamer.stream_demands(&demands);
            return VirtualTextureFrame {
                backend: VirtualTextureBackend::Cpu,
                streamer,
                demands,
                fallback,
            };
        }
        let demands = inputs
            .iter()
            .flat_map(|input| {
                let pool = self.streamer.pool();
                decode_feedback(
                    &input.desc,
                    input.grid,
                    |key| pool.contains(key).then_some(key.mip),
                    self.streamer.frame().saturating_add(1),
                )
            })
            .collect::<Vec<_>>();
        let streamer = self.streamer.stream_demands(&demands);
        VirtualTextureFrame {
            backend: VirtualTextureBackend::Cpu,
            streamer,
            demands,
            fallback: None,
        }
    }

    fn upload_table(&mut self) -> Result<(), VirtualTextureError> {
        let Some(ctx) = &self.gpu else {
            return Ok(());
        };
        let table = self.streamer.page_table();
        if self.storage.is_none() {
            self.storage = Some(
                GpuPageTableStorage::new(ctx, table.len().max(1))
                    .map_err(VirtualTextureError::PageTable)?,
            );
        }
        let Some(storage) = self.storage.as_mut() else {
            return Err(VirtualTextureError::StorageUnavailable);
        };
        storage
            .upload_resizing(ctx, table)
            .map_err(VirtualTextureError::PageTable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::texture_streaming::{
        FeedbackTextureDesc, TextureSemantic, NOT_REQUESTED,
    };

    #[test]
    fn cpu_fallback_closes_feedback_residency_and_table() {
        let mut pipeline = VirtualTexturePipeline::new(StreamerConfig::new(4096), 4);
        let desc = FeedbackTextureDesc {
            texture: 1,
            layer: 0,
            semantic: TextureSemantic::Color,
            base_mip: 0,
            mip_count: 1,
            pages_x: 1,
            pages_y: 1,
            page_byte_cost: 1,
            screen_importance: 1,
        };
        let frame = pipeline.frame(&[FeedbackInput::new(desc, &[0])]);
        assert_eq!(frame.backend, VirtualTextureBackend::Cpu);
        assert!(frame.fallback.is_none());
        assert!(!frame.demands.is_empty());
        assert_eq!(
            pipeline.streamer().page_table().len(),
            frame.streamer.resident_count
        );
        assert!(pipeline
            .frame(&[FeedbackInput::new(desc, &[NOT_REQUESTED])])
            .demands
            .is_empty());
    }

    #[test]
    fn malformed_feedback_is_a_renderable_cpu_frame() {
        let mut pipeline = VirtualTexturePipeline::new(StreamerConfig::new(4096), 4);
        let desc = FeedbackTextureDesc {
            texture: 1,
            layer: 0,
            semantic: TextureSemantic::Color,
            base_mip: 0,
            mip_count: 1,
            pages_x: 2,
            pages_y: 2,
            page_byte_cost: 1,
            screen_importance: 1,
        };
        let frame = pipeline.frame(&[FeedbackInput::new(desc, &[0, 1, 2])]);
        assert_eq!(frame.backend, VirtualTextureBackend::Cpu);
        assert!(frame.demands.is_empty());
        assert!(frame.fallback.is_none());
    }
}
