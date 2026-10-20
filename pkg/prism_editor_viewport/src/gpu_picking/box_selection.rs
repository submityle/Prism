//! Bounded rectangular GPU scan and compact ID readback.

use alloc::vec::Vec;
use core::fmt;
use std::sync::mpsc;

use super::{
    validate_correlation, EncodedPickId, GpuPickingBackend, GpuPickingConfig, GpuPickingError,
    GpuPickingFrame,
};
use crate::{BoxSelectionRequest, BoxSelectionResult, PixelRect};

/// Explicit per-scan limits. The caller also bounds concurrent in-flight GPU
/// buffers; the coordinator's pending bound alone does not own these buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuBoxSelectionLimits {
    /// Maximum number of rectangle pixels scanned by the compute pass.
    pub max_pixels: u64,
    /// Maximum compact reduction staging-buffer bytes, also capped by device limits.
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

/// Rectangular scan/map/parse failure. No partial selection is returned.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuBoxSelectionError {
    /// Frame identity or GPU readback failure.
    Picking(GpuPickingError),
    /// The requested inclusive-exclusive rectangle exceeds the target.
    RectangleOutOfBounds,
    /// Buffer sizing, dispatch sizing, or native address space overflowed.
    LayoutOverflow,
    /// Pixel count exceeds the configured bound.
    PixelLimitExceeded {
        /// Requested pixels.
        pixels: u64,
        /// Configured maximum.
        limit: u64,
    },
    /// Reduction or staging byte count exceeds the configured or native device bound.
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
    /// The compute dispatch would exceed the device's two-dimensional workgroup grid.
    DispatchLimitExceeded {
        /// Required workgroups.
        groups: u64,
        /// Device workgroup limit.
        limit: u32,
    },
    /// GPU reduction output did not match its reported insertion count.
    ReductionInvariantViolation {
        /// Count reported by the GPU.
        reported: u32,
        /// Non-empty slots observed in the mapped reduction table.
        observed: usize,
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
struct ReductionLayout {
    width: u32,
    height: u32,
    dispatch: [u32; 2],
    table_capacity: u32,
    values_bytes: u64,
    readback_bytes: u64,
}

/// The cached GPU compute pipeline used for all box selections on one backend.
pub(crate) struct BoxSelectionPipeline {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl fmt::Debug for BoxSelectionPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BoxSelectionPipeline")
            .finish_non_exhaustive()
    }
}

impl BoxSelectionPipeline {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-editor-box-selection-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../box_selection.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prism-editor-box-selection-bind-group-layout"),
            entries: &[
                buffer_layout(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                buffer_layout(
                    2,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    3,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    4,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-editor-box-selection-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prism-editor-box-selection-pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline, layout }
    }
}

fn buffer_layout(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

/// Encoded rectangle reduction, waiting for a shared-queue submission.
#[derive(Debug)]
pub struct PendingBoxSelectionReadback {
    request: BoxSelectionRequest,
    buffer: wgpu::Buffer,
    layout: ReductionLayout,
    max_ids: usize,
}

/// Submitted rectangle reduction, polled without blocking. Owns its device/buffer
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
    /// The mapped buffer contains the GPU's unique-ID hash table; the host only
    /// extracts its non-empty slots and never scans the source rectangle.
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
    /// Scans the requested rectangle on the GPU and records a bounded compact
    /// reduction readback. Rejects stale identity, empty/out-of-bounds regions,
    /// and resource limits before allocating or recording commands.
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
        let layout = reduction_layout(
            self.backend.config,
            request.rectangle,
            limits,
            self.backend.device.limits().max_buffer_size,
            self.backend.device.limits().max_storage_buffer_binding_size,
            self.backend
                .device
                .limits()
                .max_compute_workgroups_per_dimension,
        )?;
        let params = create_params_buffer(&self.backend.device, request.rectangle, &layout, limits);
        let representatives = create_zeroed_buffer(
            &self.backend.device,
            "prism-editor-box-selection-representatives",
            u64::from(layout.table_capacity) * 4,
            wgpu::BufferUsages::STORAGE,
        );
        let values = create_zeroed_buffer(
            &self.backend.device,
            "prism-editor-box-selection-values",
            layout.values_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let stats = create_zeroed_buffer(
            &self.backend.device,
            "prism-editor-box-selection-stats",
            8,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let buffer = self.backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prism-editor-box-selection-readback"),
            size: layout.readback_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = self
            .backend
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prism-editor-box-selection-bind-group"),
                layout: &self.backend.box_selection_pipeline.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.backend.target_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: representatives.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: values.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: stats.as_entire_binding(),
                    },
                ],
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prism-editor-box-selection-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.backend.box_selection_pipeline.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(layout.dispatch[0], layout.dispatch[1], 1);
        }
        encoder.copy_buffer_to_buffer(&values, 0, &buffer, 0, layout.values_bytes);
        encoder.copy_buffer_to_buffer(&stats, 0, &buffer, layout.values_bytes, 8);
        Ok(PendingBoxSelectionReadback {
            request,
            buffer,
            layout,
            max_ids: limits.max_ids,
        })
    }
}

impl GpuPickingBackend {
    /// Submits an encoder containing the rectangle reduction and starts mapping.
    pub fn submit_box_readback(
        &self,
        encoder: wgpu::CommandEncoder,
        pending: PendingBoxSelectionReadback,
    ) -> InFlightBoxSelectionReadback {
        self.queue.submit([encoder.finish()]);
        self.start_box_readback(pending)
    }

    /// Starts mapping after the host has submitted the rectangle reduction to the
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

fn create_zeroed_buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    // Fresh WebGPU buffers are zero-initialized. Avoid a second host-side
    // allocation and mapping/copying a table whose only writer is the GPU.
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn create_params_buffer(
    device: &wgpu::Device,
    rectangle: PixelRect,
    layout: &ReductionLayout,
    limits: GpuBoxSelectionLimits,
) -> wgpu::Buffer {
    let min = rectangle.min();
    let words = [
        min.x,
        min.y,
        layout.width,
        layout.height,
        layout.table_capacity - 1,
        u32::try_from(limits.max_ids).expect("reduction layout validates max_ids"),
        0,
        0,
    ];
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("prism-editor-box-selection-params"),
        size: 32,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: true,
    });
    {
        let mut mapped = buffer
            .slice(..)
            .get_mapped_range_mut()
            .expect("newly mapped uniform buffer");
        for (index, word) in words.into_iter().enumerate() {
            mapped
                .slice(index * 4..index * 4 + 4)
                .copy_from_slice(&word.to_le_bytes());
        }
    }
    buffer.unmap();
    buffer
}

fn reduction_layout(
    config: GpuPickingConfig,
    rectangle: PixelRect,
    limits: GpuBoxSelectionLimits,
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u64,
    max_compute_workgroups_per_dimension: u32,
) -> Result<ReductionLayout, GpuBoxSelectionError> {
    let max = rectangle.max();
    if rectangle.width() == 0 || rectangle.height() == 0 {
        return Err(GpuBoxSelectionError::RectangleOutOfBounds);
    }
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
    let groups = pixels.div_ceil(64);
    let dispatch_x = groups.min(u64::from(max_compute_workgroups_per_dimension));
    let dispatch_y = if dispatch_x == 0 {
        0
    } else {
        groups.div_ceil(dispatch_x)
    };
    if dispatch_x == 0 || dispatch_y > u64::from(max_compute_workgroups_per_dimension) {
        return Err(GpuBoxSelectionError::DispatchLimitExceeded {
            groups,
            limit: max_compute_workgroups_per_dimension,
        });
    }
    // The shader uses a u32 flattened pixel index and reserves zero in the
    // representative table. Include padded dispatch lanes in this check.
    if dispatch_x * dispatch_y * 64 > u64::from(u32::MAX) {
        return Err(GpuBoxSelectionError::LayoutOverflow);
    }
    let max_ids =
        u64::try_from(limits.max_ids).map_err(|_| GpuBoxSelectionError::LayoutOverflow)?;
    let requested_capacity = max_ids
        .max(1)
        .checked_mul(2)
        .ok_or(GpuBoxSelectionError::LayoutOverflow)?;
    let table_capacity = requested_capacity
        .checked_next_power_of_two()
        .ok_or(GpuBoxSelectionError::LayoutOverflow)?;
    let table_capacity =
        u32::try_from(table_capacity).map_err(|_| GpuBoxSelectionError::LayoutOverflow)?;
    let values_bytes = u64::from(table_capacity)
        .checked_mul(16)
        .ok_or(GpuBoxSelectionError::LayoutOverflow)?;
    let representatives_bytes = u64::from(table_capacity)
        .checked_mul(4)
        .ok_or(GpuBoxSelectionError::LayoutOverflow)?;
    let readback_bytes = values_bytes
        .checked_add(8)
        .ok_or(GpuBoxSelectionError::LayoutOverflow)?;
    let staging_limit = limits.max_bytes.min(max_buffer_size).min(usize::MAX as u64);
    if readback_bytes > staging_limit {
        return Err(GpuBoxSelectionError::ByteLimitExceeded {
            bytes: readback_bytes,
            limit: staging_limit,
        });
    }
    let storage_limit = max_storage_buffer_binding_size;
    if values_bytes > storage_limit || representatives_bytes > storage_limit {
        return Err(GpuBoxSelectionError::ByteLimitExceeded {
            bytes: values_bytes.max(representatives_bytes),
            limit: storage_limit,
        });
    }
    Ok(ReductionLayout {
        width,
        height,
        dispatch: [dispatch_x as u32, dispatch_y as u32],
        table_capacity,
        values_bytes,
        readback_bytes,
    })
}

fn parse_reduction(
    bytes: &[u8],
    layout: ReductionLayout,
    max_ids: usize,
) -> Result<Vec<EncodedPickId>, GpuBoxSelectionError> {
    if (bytes.len() as u64) < layout.readback_bytes {
        return Err(GpuPickingError::ReadbackBufferTooSmall {
            actual: bytes.len(),
            required: layout.readback_bytes as usize,
        }
        .into());
    }
    let stats_offset = layout.values_bytes as usize;
    let reported = read_u32(bytes, stats_offset);
    let overflow = read_u32(bytes, stats_offset + 4);
    if overflow != 0 || usize::try_from(reported).unwrap_or(usize::MAX) > max_ids {
        return Err(GpuBoxSelectionError::IdLimitExceeded { limit: max_ids });
    }
    let mut ids = Vec::with_capacity(reported as usize);
    for slot in 0..layout.table_capacity as usize {
        let offset = slot * 16;
        let channels = [
            read_u32(bytes, offset),
            read_u32(bytes, offset + 4),
            read_u32(bytes, offset + 8),
            read_u32(bytes, offset + 12),
        ];
        if channels != [0; 4] {
            ids.push(EncodedPickId::from_channels(channels));
        }
    }
    if ids.len() != reported as usize {
        return Err(GpuBoxSelectionError::ReductionInvariantViolation {
            reported,
            observed: ids.len(),
        });
    }
    // Hash insertion order depends on GPU scheduling. Keep the public result
    // deterministic without scanning or copying the source pixels on the CPU.
    ids.sort_unstable_by_key(|id| id.channels());
    Ok(ids)
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    let mut word = [0_u8; 4];
    word.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(word)
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
    let ids = parse_reduction(&mapped, pending.layout, pending.max_ids);
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
    fn reduction_layout_honors_pixel_table_byte_and_dispatch_limits() {
        let rectangle = PixelRect::new(1, 2, 4, 4).expect("rectangle");
        let config = GpuPickingConfig::new(8, 8);
        let limits = GpuBoxSelectionLimits {
            max_pixels: 6,
            max_bytes: 512,
            max_ids: 1,
        };
        let layout = reduction_layout(config, rectangle, limits, 512, 1024, 65535)
            .expect("reduction layout");
        assert_eq!(
            (
                layout.width,
                layout.height,
                layout.dispatch,
                layout.table_capacity,
                layout.values_bytes,
                layout.readback_bytes
            ),
            (3, 2, [1, 1], 2, 32, 40)
        );
        assert_eq!(
            reduction_layout(
                config,
                rectangle,
                GpuBoxSelectionLimits {
                    max_pixels: 5,
                    ..limits
                },
                512,
                1024,
                65535
            )
            .unwrap_err(),
            GpuBoxSelectionError::PixelLimitExceeded {
                pixels: 6,
                limit: 5
            }
        );
        assert_eq!(
            reduction_layout(config, rectangle, limits, 39, 1024, 65535).unwrap_err(),
            GpuBoxSelectionError::ByteLimitExceeded {
                bytes: 40,
                limit: 39
            }
        );
        assert_eq!(
            reduction_layout(
                GpuPickingConfig::new(3, 8),
                rectangle,
                limits,
                512,
                1024,
                65535
            )
            .unwrap_err(),
            GpuBoxSelectionError::RectangleOutOfBounds
        );
        let empty = PixelRect {
            min_x: 1,
            min_y: 1,
            max_x: 1,
            max_y: 2,
        };
        assert_eq!(
            reduction_layout(config, empty, limits, 512, 1024, 65535).unwrap_err(),
            GpuBoxSelectionError::RectangleOutOfBounds
        );
    }

    #[test]
    fn parser_reads_gpu_unique_slots_and_never_truncates() {
        let layout = ReductionLayout {
            width: 2,
            height: 2,
            dispatch: [1, 1],
            table_capacity: 4,
            values_bytes: 64,
            readback_bytes: 72,
        };
        let mut bytes = vec![0; 72];
        for (offset, id) in [(0, [5, 1, 0, 0]), (16, [9, 2, 0, 0])] {
            for (word, channel) in id.into_iter().enumerate() {
                bytes[offset + word * 4..offset + word * 4 + 4]
                    .copy_from_slice(&u32::to_le_bytes(channel));
            }
        }
        bytes[64..68].copy_from_slice(&2_u32.to_le_bytes());
        let ids = parse_reduction(&bytes, layout, 2).expect("unique IDs");
        assert_eq!(
            ids,
            vec![
                EncodedPickId::from_channels([5, 1, 0, 0]),
                EncodedPickId::from_channels([9, 2, 0, 0]),
            ]
        );
        assert_eq!(
            parse_reduction(&bytes, layout, 1),
            Err(GpuBoxSelectionError::IdLimitExceeded { limit: 1 })
        );
        bytes[68..72].copy_from_slice(&1_u32.to_le_bytes());
        assert_eq!(
            parse_reduction(&bytes, layout, 2),
            Err(GpuBoxSelectionError::IdLimitExceeded { limit: 2 })
        );
        assert!(parse_reduction(&bytes[..71], layout, 2).is_err());
    }

    #[test]
    fn large_rectangles_use_two_dispatch_dimensions_with_checked_padding() {
        let config = GpuPickingConfig::new(4096, 4096);
        let rectangle = PixelRect::new(0, 0, 4096, 4096).expect("4K rectangle");
        let limits = GpuBoxSelectionLimits {
            max_pixels: 4096 * 4096,
            ..Default::default()
        };
        let layout = reduction_layout(config, rectangle, limits, 1 << 30, 1 << 27, 65535)
            .expect("multi-row dispatch");
        assert_eq!(layout.dispatch, [65535, 5]);
        assert_eq!(layout.readback_bytes, 8192 * 16 + 8);
        // The staging size depends on the ID budget rather than the pixel area.
        let small = reduction_layout(
            config,
            PixelRect::new(0, 0, 8, 8).unwrap(),
            limits,
            1 << 30,
            1 << 27,
            65535,
        )
        .unwrap();
        assert_eq!(small.readback_bytes, layout.readback_bytes);
        assert!(matches!(
            reduction_layout(config, rectangle, limits, 1 << 30, 1 << 27, 8),
            Err(GpuBoxSelectionError::DispatchLimitExceeded { .. })
        ));
    }

    #[cfg(feature = "gpu-tests")]
    #[test]
    fn gpu_hash_collisions_and_duplicate_pixels_preserve_all_four_words() {
        let Some((adapter, device, queue)) = crate::test_support::device() else {
            return;
        };
        // Every distinct ID collides in the same initial slot of a 32-slot
        // table. Each appears in 1024 fragments, spanning many workgroups.
        fn hash(id: [u32; 4]) -> u32 {
            let mut h = id[0].wrapping_mul(0x9E3779B1).wrapping_add(id[1]);
            h = (h ^ (h >> 16)).wrapping_mul(0x85EBCA6B);
            h = (h ^ id[2]).wrapping_mul(0xC2B2AE35);
            h = (h ^ id[3]).wrapping_mul(0x27D4EB2F);
            h ^ (h >> 15)
        }
        let mut expected = Vec::new();
        for n in 1..100_000u32 {
            let id = [n, n.wrapping_mul(37), !n, n.rotate_left(13)];
            if hash(id) & 31 == 0 {
                expected.push(id);
            }
            if expected.len() == 16 {
                break;
            }
        }
        assert_eq!(expected.len(), 16);
        let entries = expected
            .iter()
            .map(|v| format!("vec4<u32>({}u,{}u,{}u,{}u)", v[0], v[1], v[2], v[3]))
            .collect::<Vec<_>>()
            .join(",");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("colliding box IDs"),
            source: wgpu::ShaderSource::Wgsl(format!(r"
                const IDS: array<vec4<u32>, 16> = array<vec4<u32>, 16>({entries});
                @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {{
                    let p = array<vec2<f32>, 3>(vec2<f32>(-1.,-1.), vec2<f32>(3.,-1.), vec2<f32>(-1.,3.));
                    return vec4<f32>(p[i], 0., 1.);
                }}
                @fragment fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<u32> {{
                    return IDS[u32(p.x) % 16u];
                }}
            ").into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("colliding box IDs"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: super::super::ID_BUFFER_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let backend =
            GpuPickingBackend::new(&device, &adapter, &queue, GpuPickingConfig::new(128, 128))
                .unwrap();
        let mut viewports = crate::ViewportRegistry::new();
        let viewport = viewports.open("collisions").unwrap();
        let mut targets = crate::RenderTargetRegistry::new();
        let state = targets
            .open(viewport.id(), viewport.generation(), backend)
            .unwrap();
        let scene = crate::SceneIdPass::new();
        let mut boxes = crate::BoxSelectionCoordinator::new(16);
        expected.sort_unstable();
        for max_ids in [16, 16, 16, 15, 0] {
            let request = boxes
                .submit(
                    &viewport,
                    state,
                    &scene,
                    PixelRect::new(0, 0, 128, 128).unwrap(),
                )
                .unwrap();
            let target = targets.get(viewport.id()).unwrap();
            let frame = target.begin_frame(&scene);
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = frame.begin_render_pass(&mut encoder, None);
                pass.set_pipeline(&pipeline);
                pass.draw(0..3, 0..1);
            }
            let pending = frame
                .encode_box_readback(
                    &mut encoder,
                    request,
                    GpuBoxSelectionLimits {
                        max_ids,
                        ..Default::default()
                    },
                )
                .unwrap();
            let result = target.backend().submit_box_and_read_sync(encoder, pending);
            if max_ids == 16 {
                let actual = result
                    .unwrap()
                    .ids
                    .into_iter()
                    .map(EncodedPickId::channels)
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected);
            } else {
                assert_eq!(
                    result,
                    Err(GpuBoxSelectionError::IdLimitExceeded { limit: max_ids })
                );
            }
            boxes.cancel_viewport(viewport.id());
        }
        crate::test_support::diagnostic(format_args!("EXERCISED GPU box hash collisions/full four-word IDs/duplicate workgroups/deterministic ordering/overflow"));
    }
}
