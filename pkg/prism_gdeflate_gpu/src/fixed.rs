//! Fully device-resident decode of fixed-Huffman raw-DEFLATE `GDeflate` tiles.
//!
//! The kernel reads lane-major `GDeflate` words directly, decodes every fixed
//! block in one tile, and writes the tile bytes into a GPU buffer.  A tile is
//! intentionally decoded by one invocation because literal/LZ77 state is
//! serial. Independent tiles share one compute pass and one submission. Only
//! status records are read back by the resident batch API. This module never
//! accepts a partial result: every status and exact advertised output length
//! is checked before owned output buffers are returned. The unified decoder
//! additionally verifies each tile checksum;
//! this low-level API intentionally does not receive container metadata.
//! The same bounded kernel and host transport are shared with the complete
//! [`crate::GpuDeflate`] API; the uniform mode keeps this API strictly fixed-only.

use alloc::vec::Vec;
use core::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;
use crate::deinterleave::{GROUP, WORD};

const STATUS_OK: u32 = 0;
const STATUS_UNSUPPORTED: u32 = 1;
const STATUS_TRUNCATED: u32 = 2;
const STATUS_INVALID: u32 = 3;
const STATUS_OUTPUT: u32 = 4;
const STATUS_DISTANCE: u32 = 5;
const STATUS_PARAMETERS: u32 = 6;
const STATUS_BYTES: u64 = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    source_words: u32,
    rounds: u32,
    compressed_bytes: u32,
    output_bytes: u32,
    fixed_only: u32,
    padding: [u32; 3],
}

// Eight scalar u32 slots match the WGSL uniform, without vec3 alignment gaps.
const _: () = assert!(size_of::<Params>() == 32);

/// Why a tile cannot be accepted from the GPU DEFLATE path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixedDecodeError {
    /// The tile shape, byte counts, or host ABI cannot be represented safely.
    InvalidInput,
    /// A buffer or binding would exceed the selected adapter's limits.
    DeviceLimitExceeded,
    /// The pipeline belongs to a different logical device than the request.
    DeviceMismatch,
    /// Device resource creation, validation, or submission failed.
    DeviceResource,
    /// Polling, mapping, or copying the submitted result failed.
    Readback(buffer::ReadbackError),
    /// A reserved block type, or a non-fixed block in the fixed-only API.
    UnsupportedBlockType,
    /// The GPU detected a truncated stream, invalid tree, or invalid symbol.
    MalformedStream,
    /// The stream attempted to write more or fewer bytes than its descriptor.
    OutputSizeMismatch,
    /// An LZ77 reference precedes the start of the tile output.
    DistanceTooFar,
}

/// Error shared by complete and fixed-only tile decoding.
pub type DeflateDecodeError = FixedDecodeError;

impl fmt::Display for FixedDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput => write!(formatter, "invalid GPU DEFLATE tile input"),
            Self::DeviceLimitExceeded => {
                write!(formatter, "GPU DEFLATE tile exceeds device limits")
            }
            Self::DeviceMismatch => write!(formatter, "GPU DEFLATE pipeline device mismatch"),
            Self::DeviceResource => write!(formatter, "GPU DEFLATE device resource failure"),
            Self::Readback(error) => write!(formatter, "{error}"),
            Self::UnsupportedBlockType => write!(formatter, "unsupported DEFLATE block type"),
            Self::MalformedStream => write!(formatter, "malformed DEFLATE tile"),
            Self::OutputSizeMismatch => {
                write!(formatter, "DEFLATE output size mismatch")
            }
            Self::DistanceTooFar => {
                write!(formatter, "DEFLATE distance precedes output")
            }
        }
    }
}

impl std::error::Error for FixedDecodeError {}

/// One independent raw-DEFLATE stream in Prism's lane-major tile format.
#[derive(Clone, Copy, Debug)]
pub struct DeflateTile<'a> {
    /// Source bytes including complete lane-group padding.
    pub interleaved: &'a [u8],
    /// Logical compressed byte count, excluding lane-group padding.
    pub compressed_size: usize,
    /// Exact logical output byte count, at most 64 KiB (including zero).
    pub expected_output_size: usize,
}

/// Explicit admission bounds for one resident batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeflateBatchLimits {
    /// Maximum number of independent tiles admitted in one call.
    pub max_tiles: usize,
    /// Maximum sum of buffer descriptor sizes allocated by this call.
    ///
    /// Includes padded sources and outputs, uniforms, per-tile status storage,
    /// and the packed status readback buffer. Driver bookkeeping and retained
    /// buffers from previous calls are outside this per-call budget.
    pub max_device_bytes: u64,
}

/// A batch failure never exposes any of that batch's output buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeflateBatchError {
    /// A tile failed shape, device-size, or shader-status validation.
    Tile {
        /// Index in the submitted request slice.
        index: usize,
        /// Concrete rejection reason.
        error: DeflateDecodeError,
    },
    /// The requested tile count exceeds admission.
    TooManyTiles {
        /// Number of requested tiles.
        requested: usize,
        /// Configured maximum.
        limit: usize,
    },
    /// All buffer allocations together exceed admission.
    AllocationBudgetExceeded {
        /// Required buffer bytes, including padding and status readback.
        required: u64,
        /// Configured maximum.
        limit: u64,
    },
    /// A batch-wide device, host-allocation, or readback failure.
    Device(DeflateDecodeError),
}

impl fmt::Display for DeflateBatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tile { index, error } => write!(formatter, "DEFLATE tile {index}: {error}"),
            Self::TooManyTiles { requested, limit } => {
                write!(
                    formatter,
                    "DEFLATE batch has {requested} tiles; limit is {limit}"
                )
            }
            Self::AllocationBudgetExceeded { required, limit } => {
                write!(
                    formatter,
                    "DEFLATE batch needs {required} device bytes; limit is {limit}"
                )
            }
            Self::Device(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for DeflateBatchError {}

/// Owned, validated GPU output; no payload bytes were read back by the batch.
///
/// The buffer supports STORAGE, COPY_SRC, and COPY_DST. Its allocation is
/// rounded up to a nonzero word; only `byte_len()` bytes are logical output.
/// Buffers remain usable after the decoder, request slices, and sibling
/// results are dropped. Container checksums remain the caller's responsibility.
#[derive(Debug)]
pub struct ResidentDeflateTile {
    buffer: wgpu::Buffer,
    byte_len: u64,
}

impl ResidentDeflateTile {
    /// Buffer for downstream GPU binding or copying on the same device.
    #[must_use]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Exact decoded byte length, excluding word padding.
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }

    /// Transfers the buffer and its exact logical byte length to the consumer.
    #[must_use]
    pub fn into_buffer(self) -> (wgpu::Buffer, u64) {
        (self.buffer, self.byte_len)
    }
}

struct PreparedTile {
    params: Params,
    words: Vec<u32>,
    output_bytes: u64,
}

struct DispatchTile {
    bind: wgpu::BindGroup,
    status: wgpu::Buffer,
    output: wgpu::Buffer,
}

/// Compiled fixed-Huffman `GDeflate` tile decoder.
pub struct GpuFixedGDeflate {
    instance: wgpu::Instance,
    device: wgpu::Device,
    pipeline: ComputePipeline,
    layout: wgpu::BindGroupLayout,
    fixed_only: bool,
}

impl GpuFixedGDeflate {
    /// Compiles the fixed-Huffman tile kernel for `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        Self::with_mode(ctx, true)
    }

    pub(crate) fn with_mode(ctx: &GpuContext, fixed_only: bool) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_gdeflate_deflate_decode"),
            source: ShaderSource::Wgsl(include_str!("shaders/deflate_decode.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_gdeflate_deflate_decode_layout"),
            entries: &[
                buffer_layout(0, BufferBindingType::Uniform),
                buffer_layout(1, BufferBindingType::Storage { read_only: true }),
                buffer_layout(2, BufferBindingType::Storage { read_only: false }),
                buffer_layout(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_gdeflate_deflate_decode_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_gdeflate_deflate_decode_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            instance: ctx.instance().clone(),
            device: device.clone(),
            pipeline,
            layout,
            fixed_only,
        }
    }

    /// Decodes one complete lane-major `GDeflate` tile without CPU inflate.
    ///
    /// `interleaved` includes the mandatory lane-group padding; only its first
    /// `compressed_size` logical bytes belong to the raw-DEFLATE stream.
    /// `expected_output_size` is the tile descriptor's exact byte count, at
    /// most 64 KiB. Checksums must be verified separately by the caller.
    pub fn decompress(
        &self,
        ctx: &GpuContext,
        interleaved: &[u8],
        compressed_size: usize,
        expected_output_size: usize,
    ) -> Result<Vec<u8>, FixedDecodeError> {
        let mut tiles = self
            .decompress_batch_resident(
                ctx,
                &[DeflateTile {
                    interleaved,
                    compressed_size,
                    expected_output_size,
                }],
                DeflateBatchLimits {
                    max_tiles: 1,
                    max_device_bytes: u64::MAX,
                },
            )
            .map_err(|error| match error {
                DeflateBatchError::Tile { error, .. } | DeflateBatchError::Device(error) => error,
                _ => FixedDecodeError::DeviceLimitExceeded,
            })?;
        let tile = tiles.pop().ok_or(FixedDecodeError::MalformedStream)?;
        // Only the compatibility API requests a payload staging allocation.
        let output_words = with_device_errors(ctx.device(), || {
            let stage = buffer::staging(
                ctx.device(),
                "prism_gdeflate_deflate_output_stage",
                tile.buffer.size(),
            );
            let mut encoder = ctx.device().create_command_encoder(&Default::default());
            buffer::copy(&mut encoder, &tile.buffer, &stage, tile.buffer.size());
            let submission = ctx.queue().submit([encoder.finish()]);
            buffer::read_back::<u32>(ctx, &stage, submission).map_err(FixedDecodeError::Readback)
        })?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(expected_output_size)
            .map_err(|_| FixedDecodeError::Readback(buffer::ReadbackError::Allocation))?;
        for word in output_words {
            let bytes = word.to_le_bytes();
            let count = (expected_output_size - output.len()).min(WORD);
            output.extend_from_slice(&bytes[..count]);
        }
        Ok(output)
    }

    /// Decodes independent tiles in one compute pass and one submission.
    ///
    /// All shape, admission, and device-limit checks and host input reserves
    /// complete before GPU resource creation. The only readback is one packed
    /// array of eight-byte status records. Outputs are published together, in
    /// request order, after every status and exact byte count is valid. An
    /// empty request returns an empty vector without GPU work.
    pub fn decompress_batch_resident(
        &self,
        ctx: &GpuContext,
        requests: &[DeflateTile<'_>],
        admission: DeflateBatchLimits,
    ) -> Result<Vec<ResidentDeflateTile>, DeflateBatchError> {
        // Device IDs may alias across independently created wgpu instances.
        // The retained instance also prevents address reuse while we exist.
        if ctx.instance() != &self.instance || ctx.device() != &self.device {
            return Err(DeflateBatchError::Device(FixedDecodeError::DeviceMismatch));
        }
        let status_bytes = admit_batch(requests, admission, &ctx.device().limits())?;
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        // Reserve input, dispatch, and result storage before GPU allocation.
        // The readback helper also reserves its status result fallibly.
        let mut prepared = Vec::new();
        let mut dispatches = Vec::new();
        let mut outputs = Vec::new();
        for result in [
            prepared.try_reserve_exact(requests.len()),
            dispatches.try_reserve_exact(requests.len()),
            outputs.try_reserve_exact(requests.len()),
        ] {
            result.map_err(|_| DeflateBatchError::Device(host_allocation_error()))?;
        }
        for (index, request) in requests.iter().enumerate() {
            let (params, output_bytes) = tile_params(request, self.fixed_only)
                .map_err(|error| DeflateBatchError::Tile { index, error })?;
            let words = to_words_le(request.interleaved)
                .map_err(|error| DeflateBatchError::Tile { index, error })?;
            prepared.push(PreparedTile {
                params,
                words,
                output_bytes,
            });
        }

        let device = ctx.device();
        let statuses = with_device_errors(device, || {
            for tile in &prepared {
                let params = buffer::uniform(device, "prism_gdeflate_deflate_params", &tile.params);
                let source =
                    buffer::storage_read(device, "prism_gdeflate_deflate_source", &tile.words);
                let output = buffer::storage_rw_zeroed(
                    device,
                    "prism_gdeflate_deflate_output",
                    tile.output_bytes,
                );
                // A missing dispatch must fail even for zero-length output.
                let status = buffer::storage_rw_init(
                    device,
                    "prism_gdeflate_deflate_status",
                    &[STATUS_PARAMETERS, 0u32],
                );
                let bind = device.create_bind_group(&BindGroupDescriptor {
                    label: Some("prism_gdeflate_deflate_decode_bind"),
                    layout: &self.layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: params.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: source.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: output.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: status.as_entire_binding(),
                        },
                    ],
                });
                dispatches.push(DispatchTile {
                    bind,
                    status,
                    output,
                });
            }
            let stage =
                buffer::staging(device, "prism_gdeflate_deflate_batch_status", status_bytes);
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("prism_gdeflate_deflate_batch_encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("prism_gdeflate_deflate_batch_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                for tile in &dispatches {
                    pass.set_bind_group(0, &tile.bind, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
            for (index, tile) in dispatches.iter().enumerate() {
                encoder.copy_buffer_to_buffer(
                    &tile.status,
                    0,
                    &stage,
                    index as u64 * STATUS_BYTES,
                    STATUS_BYTES,
                );
            }
            let submission = ctx.queue().submit([encoder.finish()]);
            buffer::read_back::<u32>(ctx, &stage, submission).map_err(FixedDecodeError::Readback)
        })
        .map_err(DeflateBatchError::Device)?;
        validate_statuses(&statuses, requests)?;
        for (tile, request) in dispatches.into_iter().zip(requests) {
            outputs.push(ResidentDeflateTile {
                buffer: tile.output,
                byte_len: request.expected_output_size as u64,
            });
        }
        Ok(outputs)
    }
}

fn host_allocation_error() -> FixedDecodeError {
    FixedDecodeError::Readback(buffer::ReadbackError::Allocation)
}

fn with_device_errors<T>(
    device: &wgpu::Device,
    operation: impl FnOnce() -> Result<T, FixedDecodeError>,
) -> Result<T, FixedDecodeError> {
    let allocation_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal_scope = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result =
        catch_unwind(AssertUnwindSafe(operation)).unwrap_or(Err(FixedDecodeError::DeviceResource));
    // Always consume the actual wgpu 30 scope handles, including on unwind.
    let validation_error = crate::context::block_on(validation_scope.pop());
    let internal_error = crate::context::block_on(internal_scope.pop());
    let allocation_error = crate::context::block_on(allocation_scope.pop());
    if validation_error.is_some() || internal_error.is_some() || allocation_error.is_some() {
        return Err(FixedDecodeError::DeviceResource);
    }
    result
}

fn tile_params(
    request: &DeflateTile<'_>,
    fixed_only: bool,
) -> Result<(Params, u64), FixedDecodeError> {
    if request.expected_output_size > 1 << 16 {
        return Err(FixedDecodeError::InvalidInput);
    }
    let (source_words, rounds) = validate_input(request.interleaved, request.compressed_size)?;
    let params = Params {
        source_words: u32::try_from(source_words).map_err(|_| FixedDecodeError::InvalidInput)?,
        rounds: u32::try_from(rounds).map_err(|_| FixedDecodeError::InvalidInput)?,
        compressed_bytes: u32::try_from(request.compressed_size)
            .map_err(|_| FixedDecodeError::InvalidInput)?,
        output_bytes: u32::try_from(request.expected_output_size)
            .map_err(|_| FixedDecodeError::InvalidInput)?,
        fixed_only: u32::from(fixed_only),
        padding: [0; 3],
    };
    Ok((
        params,
        request.expected_output_size.div_ceil(WORD).max(1) as u64 * WORD as u64,
    ))
}

fn admit_batch(
    requests: &[DeflateTile<'_>],
    admission: DeflateBatchLimits,
    limits: &wgpu::Limits,
) -> Result<u64, DeflateBatchError> {
    if requests.len() > admission.max_tiles {
        return Err(DeflateBatchError::TooManyTiles {
            requested: requests.len(),
            limit: admission.max_tiles,
        });
    }
    if requests.is_empty() {
        return Ok(0);
    }
    let device_limit = || DeflateBatchError::Device(FixedDecodeError::DeviceLimitExceeded);
    let count = u64::try_from(requests.len()).map_err(|_| device_limit())?;
    let status_bytes = count.checked_mul(STATUS_BYTES).ok_or_else(device_limit)?;
    let uniform_bytes = size_of::<Params>() as u64;
    if status_bytes > limits.max_buffer_size
        || uniform_bytes > limits.max_buffer_size
        || uniform_bytes > limits.max_uniform_buffer_binding_size
        || STATUS_BYTES > limits.max_buffer_size
        || STATUS_BYTES > limits.max_storage_buffer_binding_size
        || limits.max_bind_groups < 1
        || limits.max_bindings_per_bind_group < 4
        || limits.max_storage_buffers_per_shader_stage < 3
        || limits.max_uniform_buffers_per_shader_stage < 1
        || limits.max_compute_workgroups_per_dimension < 1
        || limits.max_compute_invocations_per_workgroup < 1
        || limits.max_compute_workgroup_size_x < 1
        || limits.max_compute_workgroup_size_y < 1
        || limits.max_compute_workgroup_size_z < 1
        || count.checked_mul(4).ok_or_else(device_limit)?
            > u64::from(limits.max_non_sampler_bindings)
    {
        return Err(device_limit());
    }
    let mut required = status_bytes;
    for (index, request) in requests.iter().enumerate() {
        let (_, output_bytes) = tile_params(request, false)
            .map_err(|error| DeflateBatchError::Tile { index, error })?;
        let source_bytes = u64::try_from(request.interleaved.len()).map_err(|_| device_limit())?;
        if [source_bytes, output_bytes].iter().any(|&size| {
            size > limits.max_buffer_size || size > limits.max_storage_buffer_binding_size
        }) {
            return Err(DeflateBatchError::Tile {
                index,
                error: FixedDecodeError::DeviceLimitExceeded,
            });
        }
        for size in [source_bytes, output_bytes, uniform_bytes, STATUS_BYTES] {
            required = required.checked_add(size).ok_or_else(device_limit)?;
        }
    }
    if required > admission.max_device_bytes {
        return Err(DeflateBatchError::AllocationBudgetExceeded {
            required,
            limit: admission.max_device_bytes,
        });
    }
    Ok(status_bytes)
}

fn validate_statuses(
    statuses: &[u32],
    requests: &[DeflateTile<'_>],
) -> Result<(), DeflateBatchError> {
    if statuses.len()
        != requests
            .len()
            .checked_mul(2)
            .ok_or(DeflateBatchError::Device(FixedDecodeError::MalformedStream))?
    {
        return Err(DeflateBatchError::Device(FixedDecodeError::MalformedStream));
    }
    for (index, (status, request)) in statuses.chunks_exact(2).zip(requests).enumerate() {
        let error = if status[0] != STATUS_OK {
            Some(status_error(status[0]))
        } else if u64::from(status[1]) != request.expected_output_size as u64 {
            Some(FixedDecodeError::OutputSizeMismatch)
        } else {
            None
        };
        if let Some(error) = error {
            return Err(DeflateBatchError::Tile { index, error });
        }
    }
    Ok(())
}

fn validate_input(
    interleaved: &[u8],
    compressed_size: usize,
) -> Result<(usize, usize), FixedDecodeError> {
    if interleaved.is_empty()
        || interleaved.len() % GROUP != 0
        || compressed_size == 0
        || compressed_size > interleaved.len()
        || interleaved.len() > u32::MAX as usize
    {
        return Err(FixedDecodeError::InvalidInput);
    }
    let source_words = interleaved.len() / WORD;
    let rounds = source_words / 32;
    if rounds == 0 || source_words != rounds * 32 {
        return Err(FixedDecodeError::InvalidInput);
    }
    Ok((source_words, rounds))
}

fn to_words_le(bytes: &[u8]) -> Result<Vec<u32>, FixedDecodeError> {
    let mut words = Vec::new();
    words
        .try_reserve_exact(bytes.len() / WORD)
        .map_err(|_| FixedDecodeError::Readback(buffer::ReadbackError::Allocation))?;
    words.extend(
        bytes
            .chunks_exact(WORD)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
    );
    Ok(words)
}

fn status_error(code: u32) -> FixedDecodeError {
    match code {
        STATUS_UNSUPPORTED => FixedDecodeError::UnsupportedBlockType,
        STATUS_TRUNCATED | STATUS_INVALID | STATUS_PARAMETERS => FixedDecodeError::MalformedStream,
        STATUS_OUTPUT => FixedDecodeError::OutputSizeMismatch,
        STATUS_DISTANCE => FixedDecodeError::DistanceTooFar,
        _ => FixedDecodeError::MalformedStream,
    }
}

fn buffer_layout(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_admission_checks_uniform_storage_readback_and_dispatch_limits() {
        let source = [0; GROUP];
        let tile = DeflateTile {
            interleaved: &source,
            compressed_size: 2,
            expected_output_size: 0,
        };
        let admission = DeflateBatchLimits {
            max_tiles: 32,
            max_device_bytes: u64::MAX,
        };
        let limits = wgpu::Limits::default();
        assert_eq!(admit_batch(&[tile; 2], admission, &limits), Ok(16));
        for restricted in [
            wgpu::Limits {
                max_uniform_buffer_binding_size: 31,
                ..limits.clone()
            },
            wgpu::Limits {
                max_storage_buffer_binding_size: 7,
                ..limits.clone()
            },
            wgpu::Limits {
                max_buffer_size: 31,
                ..limits.clone()
            },
            wgpu::Limits {
                max_bind_groups: 0,
                ..limits.clone()
            },
            wgpu::Limits {
                max_bindings_per_bind_group: 3,
                ..limits.clone()
            },
            wgpu::Limits {
                max_storage_buffers_per_shader_stage: 2,
                ..limits.clone()
            },
            wgpu::Limits {
                max_uniform_buffers_per_shader_stage: 0,
                ..limits.clone()
            },
            wgpu::Limits {
                max_compute_workgroups_per_dimension: 0,
                ..limits.clone()
            },
            wgpu::Limits {
                max_compute_invocations_per_workgroup: 0,
                ..limits.clone()
            },
            wgpu::Limits {
                max_compute_workgroup_size_x: 0,
                ..limits.clone()
            },
            wgpu::Limits {
                max_compute_workgroup_size_y: 0,
                ..limits.clone()
            },
            wgpu::Limits {
                max_compute_workgroup_size_z: 0,
                ..limits.clone()
            },
            wgpu::Limits {
                max_non_sampler_bindings: 7,
                ..limits.clone()
            },
        ] {
            assert_eq!(
                admit_batch(&[tile; 2], admission, &restricted),
                Err(DeflateBatchError::Device(
                    FixedDecodeError::DeviceLimitExceeded
                )),
            );
        }
        for restricted in [
            wgpu::Limits {
                max_buffer_size: GROUP as u64 - 1,
                ..limits.clone()
            },
            wgpu::Limits {
                max_storage_buffer_binding_size: GROUP as u64 - 1,
                ..limits.clone()
            },
        ] {
            assert_eq!(
                admit_batch(&[tile], admission, &restricted),
                Err(DeflateBatchError::Tile {
                    index: 0,
                    error: FixedDecodeError::DeviceLimitExceeded
                }),
            );
        }
        assert_eq!(
            admit_batch(
                &[DeflateTile {
                    expected_output_size: GROUP + 1,
                    ..tile
                }],
                admission,
                &wgpu::Limits {
                    max_storage_buffer_binding_size: GROUP as u64 + 3,
                    ..limits.clone()
                },
            ),
            Err(DeflateBatchError::Tile {
                index: 0,
                error: FixedDecodeError::DeviceLimitExceeded
            }),
        );
        // Readback is one aggregate allocation, even though each status is 8B.
        assert_eq!(
            admit_batch(
                &[tile; 17],
                admission,
                &wgpu::Limits {
                    max_buffer_size: GROUP as u64,
                    ..limits
                }
            ),
            Err(DeflateBatchError::Device(
                FixedDecodeError::DeviceLimitExceeded
            )),
        );
    }

    #[test]
    fn batch_admission_includes_padding_and_checks_late_tiles_before_budget() {
        let source = [0; GROUP];
        let tile = DeflateTile {
            interleaved: &source,
            compressed_size: 2,
            expected_output_size: 5,
        };
        let limits = wgpu::Limits::default();
        let required = 2 * (GROUP as u64 + 8 + 32 + 8 + 8);
        let mut admission = DeflateBatchLimits {
            max_tiles: 2,
            max_device_bytes: required - 1,
        };
        assert_eq!(
            admit_batch(&[tile; 2], admission, &limits),
            Err(DeflateBatchError::AllocationBudgetExceeded {
                required,
                limit: required - 1
            }),
        );
        admission.max_device_bytes = required;
        assert_eq!(admit_batch(&[tile; 2], admission, &limits), Ok(16));
        admission.max_device_bytes = 0;
        assert_eq!(
            admit_batch(
                &[
                    tile,
                    DeflateTile {
                        compressed_size: 0,
                        ..tile
                    }
                ],
                admission,
                &limits
            ),
            Err(DeflateBatchError::Tile {
                index: 1,
                error: FixedDecodeError::InvalidInput
            }),
        );
    }

    #[test]
    fn batch_status_validation_fails_closed_for_missing_or_late_records() {
        let requests = [DeflateTile {
            interleaved: &[],
            compressed_size: 0,
            expected_output_size: 0,
        }; 2];
        assert_eq!(
            validate_statuses(&[STATUS_OK, 0, STATUS_OK, 0], &requests),
            Ok(())
        );
        assert_eq!(
            validate_statuses(&[STATUS_OK, 0], &requests),
            Err(DeflateBatchError::Device(FixedDecodeError::MalformedStream)),
        );
        assert_eq!(
            validate_statuses(&[STATUS_OK, 0, STATUS_PARAMETERS, 0], &requests),
            Err(DeflateBatchError::Tile {
                index: 1,
                error: FixedDecodeError::MalformedStream
            }),
        );
        assert_eq!(
            validate_statuses(&[STATUS_OK, 0, STATUS_OK, 1], &requests),
            Err(DeflateBatchError::Tile {
                index: 1,
                error: FixedDecodeError::OutputSizeMismatch
            }),
        );
    }

    #[test]
    fn input_validation_requires_exact_lane_padding_and_nonempty_stream() {
        assert_eq!(validate_input(&[], 0), Err(FixedDecodeError::InvalidInput));
        assert_eq!(
            validate_input(&[0; GROUP - 1], 1),
            Err(FixedDecodeError::InvalidInput)
        );
        assert_eq!(
            validate_input(&[0; GROUP], 0),
            Err(FixedDecodeError::InvalidInput)
        );
        assert_eq!(
            validate_input(&[0; GROUP], GROUP + 1),
            Err(FixedDecodeError::InvalidInput)
        );
        assert_eq!(validate_input(&[0; GROUP], 17), Ok((32, 1)));
    }

    #[test]
    fn shader_statuses_have_non_ambiguous_host_errors() {
        assert_eq!(
            status_error(STATUS_UNSUPPORTED),
            FixedDecodeError::UnsupportedBlockType
        );
        assert_eq!(
            status_error(STATUS_TRUNCATED),
            FixedDecodeError::MalformedStream
        );
        assert_eq!(
            status_error(STATUS_INVALID),
            FixedDecodeError::MalformedStream
        );
        assert_eq!(
            status_error(STATUS_OUTPUT),
            FixedDecodeError::OutputSizeMismatch
        );
        assert_eq!(
            status_error(STATUS_DISTANCE),
            FixedDecodeError::DistanceTooFar
        );
        assert_eq!(
            status_error(STATUS_PARAMETERS),
            FixedDecodeError::MalformedStream
        );
    }
}
