//! Byte-exact GPU-only concatenation of validated resident tile ranges.
//!
//! Buffer copies require word-aligned offsets and lengths. Container entry
//! ranges do not: entries and tile seams may start or end on any byte. This
//! kernel gathers bytes into disjoint destination lanes, atomically merging
//! ranges sharing a word. No decoded bytes visit the CPU.

use alloc::vec::Vec;
use core::{fmt, ops::Range};

use bytemuck::{Pod, Zeroable};

use crate::{buffer, fixed::with_device_errors, FixedDecodeError, GpuContext, ResidentDeflateTile};

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    source_offset: u32,
    destination_offset: u32,
    byte_len: u32,
    word_count: u32,
}

const _: () = assert!(size_of::<Params>() == 16);

/// One borrowed, byte-addressed interval of a validated resident output.
#[derive(Clone, Debug)]
pub struct ResidentByteRange<'a> {
    /// Owned tile providing the source bytes; never read back by assembly.
    pub tile: &'a ResidentDeflateTile,
    /// Tile-relative logical byte range, excluding allocation padding.
    pub range: Range<u64>,
}

/// Explicit per-call bounds for GPU-only resident byte assembly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentAssemblyLimits {
    /// Maximum input slice count, including empty intervals.
    pub max_slices: usize,
    /// Maximum sum of newly allocated GPU buffer descriptor sizes.
    ///
    /// Counts the word-padded output (at least four bytes) and a 16-byte
    /// uniform per nonempty slice. Excludes retained source buffers, the
    /// compiled pipeline and driver bookkeeping; not a live VRAM quota.
    pub max_device_bytes: u64,
}

/// Admission or device failure; no partial output is returned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentAssemblyError {
    /// A reversed or out-of-logical-bounds source interval.
    InvalidRange {
        /// Input slice index.
        index: usize,
    },
    /// The number of supplied intervals exceeds admission.
    TooManySlices {
        /// Supplied interval count.
        requested: usize,
        /// Configured maximum.
        limit: usize,
    },
    /// Required new buffer bytes exceed admission.
    AllocationBudgetExceeded {
        /// Padded output plus per-dispatch uniform bytes.
        required: u64,
        /// Configured maximum.
        limit: u64,
    },
    /// Ownership, device limit, host allocation or device resource failure.
    Device(FixedDecodeError),
}

impl fmt::Display for ResidentAssemblyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRange { index } => write!(formatter, "invalid resident slice {index}"),
            Self::TooManySlices { requested, limit } => {
                write!(
                    formatter,
                    "resident assembly has {requested} slices; limit is {limit}"
                )
            }
            Self::AllocationBudgetExceeded { required, limit } => {
                write!(
                    formatter,
                    "resident assembly needs {required} bytes; limit is {limit}"
                )
            }
            Self::Device(error) => write!(formatter, "resident assembly: {error}"),
        }
    }
}

impl std::error::Error for ResidentAssemblyError {}

/// Reusable byte-gather pipeline compiled for one exact logical device.
pub struct GpuResidentAssembler {
    instance: wgpu::Instance,
    device: wgpu::Device,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl GpuResidentAssembler {
    /// Compiles the gather kernel, reporting scoped device failures.
    pub fn new(ctx: &GpuContext) -> Result<Self, FixedDecodeError> {
        validate_compute_limits(&ctx.device().limits())?;
        with_device_errors(ctx.device(), || {
            let device = ctx.device();
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("prism_resident_assembly"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/resident_assembly.wgsl").into(),
                ),
            });
            let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("prism_resident_assembly_layout"),
                entries: &[
                    entry(0, wgpu::BufferBindingType::Uniform),
                    entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("prism_resident_assembly_pipeline_layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("prism_resident_assembly_pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            Ok(Self {
                instance: ctx.instance().clone(),
                device: device.clone(),
                layout,
                pipeline,
            })
        })
    }

    /// Concatenates input intervals in order into a fresh owned GPU buffer.
    ///
    /// Validates every interval, ownership and allocation budget before any
    /// GPU allocation or submission. Empty intervals are legal, including at
    /// logical EOF. An empty result owns a four-byte zero buffer. Tail padding
    /// is zero. Sources and this assembler may be dropped after return.
    ///
    /// One pass/submission writes the output; there is no payload or status
    /// readback and no completion wait. Downstream work submitted on the same
    /// context queue observes the completed writes through queue ordering.
    /// Device loss after submission remains the renderer's recovery concern.
    pub fn assemble(
        &self,
        ctx: &GpuContext,
        ranges: &[ResidentByteRange<'_>],
        admission: ResidentAssemblyLimits,
    ) -> Result<ResidentDeflateTile, ResidentAssemblyError> {
        use ResidentAssemblyError::Device;
        if ctx.instance() != &self.instance || ctx.device() != &self.device {
            return Err(Device(FixedDecodeError::DeviceMismatch));
        }
        let (byte_len, padded_len, dispatch_count) = self.admit(ctx, ranges, admission)?;
        // Reserve before device allocation. Bind groups retain their buffers
        // until command recording/submission establishes wgpu ownership.
        let mut dispatches = Vec::new();
        dispatches.try_reserve_exact(dispatch_count).map_err(|_| {
            Device(FixedDecodeError::Readback(
                buffer::ReadbackError::Allocation,
            ))
        })?;
        with_device_errors(ctx.device(), || {
            let device = ctx.device();
            let output =
                buffer::storage_rw_zeroed(device, "prism_resident_assembly_output", padded_len);
            let mut destination_offset = 0u32;
            for range in ranges {
                let len = (range.range.end - range.range.start) as u32;
                if len == 0 {
                    continue;
                }
                let params = Params {
                    source_offset: range.range.start as u32,
                    destination_offset,
                    byte_len: len,
                    word_count: ((destination_offset % 4) + len).div_ceil(4),
                };
                let uniform = buffer::uniform(device, "prism_resident_assembly_params", &params);
                let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("prism_resident_assembly_bind"),
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: range.tile.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: output.as_entire_binding(),
                        },
                    ],
                });
                dispatches.push((bind, params.word_count.div_ceil(WORKGROUP_SIZE)));
                destination_offset += len;
            }
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("prism_resident_assembly_encoder"),
            });
            if !dispatches.is_empty() {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("prism_resident_assembly_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                for (bind, workgroups) in &dispatches {
                    pass.set_bind_group(0, bind, &[]);
                    pass.dispatch_workgroups(*workgroups, 1, 1);
                }
            }
            ctx.queue().submit([encoder.finish()]);
            Ok(ResidentDeflateTile::from_buffer(ctx, output, byte_len))
        })
        .map_err(Device)
    }

    fn admit(
        &self,
        ctx: &GpuContext,
        ranges: &[ResidentByteRange<'_>],
        admission: ResidentAssemblyLimits,
    ) -> Result<(u64, u64, usize), ResidentAssemblyError> {
        use ResidentAssemblyError::Device;
        let reject = || Device(FixedDecodeError::DeviceLimitExceeded);
        if ranges.len() > admission.max_slices {
            return Err(ResidentAssemblyError::TooManySlices {
                requested: ranges.len(),
                limit: admission.max_slices,
            });
        }
        let limits = ctx.device().limits();
        let mut total = 0u64;
        let mut nonempty = 0usize;
        for (index, range) in ranges.iter().enumerate() {
            if !range.tile.is_on_context(ctx) {
                return Err(Device(FixedDecodeError::DeviceMismatch));
            }
            if range.range.start > range.range.end || range.range.end > range.tile.byte_len() {
                return Err(ResidentAssemblyError::InvalidRange { index });
            }
            let len = range.range.end - range.range.start;
            if range.range.end > u64::from(u32::MAX)
                || range.tile.buffer().size() > limits.max_storage_buffer_binding_size
            {
                return Err(reject());
            }
            let words = (total % 4).checked_add(len).ok_or_else(reject)?.div_ceil(4);
            if words.div_ceil(u64::from(WORKGROUP_SIZE))
                > u64::from(limits.max_compute_workgroups_per_dimension)
            {
                return Err(reject());
            }
            total = total.checked_add(len).ok_or_else(reject)?;
            if len != 0 {
                nonempty += 1;
            }
        }
        let padded = total.checked_add(3).ok_or_else(reject)? / 4 * 4;
        let padded = padded.max(4);
        if padded > u64::from(u32::MAX)
            || padded > limits.max_buffer_size
            || padded > limits.max_storage_buffer_binding_size
        {
            return Err(reject());
        }
        let uniforms = (nonempty as u64)
            .checked_mul(size_of::<Params>() as u64)
            .ok_or_else(reject)?;
        let required = padded.checked_add(uniforms).ok_or_else(reject)?;
        if required > admission.max_device_bytes {
            return Err(ResidentAssemblyError::AllocationBudgetExceeded {
                required,
                limit: admission.max_device_bytes,
            });
        }
        Ok((total, padded, nonempty))
    }
}

fn validate_compute_limits(limits: &wgpu::Limits) -> Result<(), FixedDecodeError> {
    if limits.max_compute_workgroup_size_x < WORKGROUP_SIZE
        || limits.max_compute_invocations_per_workgroup < WORKGROUP_SIZE
        || limits.max_compute_workgroups_per_dimension == 0
        || limits.max_storage_buffers_per_shader_stage < 2
        || limits.max_uniform_buffers_per_shader_stage < 1
        || limits.max_bind_groups < 1
        || limits.max_bindings_per_bind_group < 3
        || limits.max_uniform_buffer_binding_size < size_of::<Params>() as u64
        || limits.max_buffer_size < size_of::<Params>() as u64
    {
        return Err(FixedDecodeError::DeviceLimitExceeded);
    }
    Ok(())
}
