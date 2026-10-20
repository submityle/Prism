//! Fully device-resident decode of fixed-Huffman raw-DEFLATE `GDeflate` tiles.
//!
//! The kernel reads lane-major `GDeflate` words directly, decodes every fixed
//! block in one tile, and writes the tile bytes into a GPU buffer.  A tile is
//! intentionally decoded by one invocation because literal/LZ77 state is
//! serial. The current host API dispatches one tile at a time; a future batch
//! scheduler can submit independent tiles concurrently. This module never
//! accepts a partial result: shader
//! status and exact advertised output length are checked before bytes are
//! returned. The unified decoder additionally verifies each tile checksum;
//! this low-level API intentionally does not receive container metadata.
//! The same bounded kernel and host transport are shared with the complete
//! [`crate::GpuDeflate`] API; the uniform mode keeps this API strictly fixed-only.

use alloc::vec::Vec;
use core::fmt;

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

/// Compiled fixed-Huffman `GDeflate` tile decoder.
pub struct GpuFixedGDeflate {
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
        if ctx.device() != &self.device {
            return Err(FixedDecodeError::DeviceMismatch);
        }
        if expected_output_size > 1 << 16 {
            return Err(FixedDecodeError::InvalidInput);
        }
        let (source_words, rounds) = validate_input(interleaved, compressed_size)?;
        let output_words = expected_output_size.div_ceil(WORD).max(1);
        let output_bytes = output_words
            .checked_mul(WORD)
            .ok_or(FixedDecodeError::InvalidInput)?;
        let source_bytes = interleaved.len();
        let limits = ctx.device().limits();
        if source_bytes as u64 > limits.max_buffer_size
            || source_bytes as u64 > limits.max_storage_buffer_binding_size
            || output_bytes as u64 > limits.max_buffer_size
            || output_bytes as u64 > limits.max_storage_buffer_binding_size
            || STATUS_BYTES > limits.max_buffer_size
            || STATUS_BYTES > limits.max_storage_buffer_binding_size
        {
            return Err(FixedDecodeError::DeviceLimitExceeded);
        }
        let source_words =
            u32::try_from(source_words).map_err(|_| FixedDecodeError::InvalidInput)?;
        let rounds = u32::try_from(rounds).map_err(|_| FixedDecodeError::InvalidInput)?;
        let compressed_bytes =
            u32::try_from(compressed_size).map_err(|_| FixedDecodeError::InvalidInput)?;
        let output_bytes_u32 =
            u32::try_from(expected_output_size).map_err(|_| FixedDecodeError::InvalidInput)?;

        let device = ctx.device();
        // Capture failures rather than allowing wgpu's uncaptured-error handler
        // to panic. Scopes are thread-local in this native, std-enabled backend.
        let allocation_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal_scope = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = (|| {
        let params = buffer::uniform(
            device,
            "prism_gdeflate_deflate_params",
            &Params {
                source_words,
                rounds,
                compressed_bytes,
                output_bytes: output_bytes_u32,
                fixed_only: u32::from(self.fixed_only),
                padding: [0; 3],
            },
        );
        let source = buffer::storage_read(
            device,
            "prism_gdeflate_deflate_source",
            &to_words_le(interleaved)?,
        );
        let output =
            buffer::storage_rw_zeroed(device, "prism_gdeflate_deflate_output", output_bytes as u64);
        // Fail closed even if the kernel never executes (notably empty output
        // after device loss); zero initialization would look like STATUS_OK.
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
        let output_stage = buffer::staging(
            device,
            "prism_gdeflate_deflate_output_stage",
            output_bytes as u64,
        );
        let status_stage =
            buffer::staging(device, "prism_gdeflate_deflate_status_stage", STATUS_BYTES);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_gdeflate_deflate_decode_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_gdeflate_deflate_decode_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        buffer::copy(&mut encoder, &output, &output_stage, output_bytes as u64);
        buffer::copy(&mut encoder, &status, &status_stage, STATUS_BYTES);
        let submission = ctx.queue().submit([encoder.finish()]);

        let status_values = buffer::read_back::<u32>(ctx, &status_stage, submission.clone())
            .map_err(FixedDecodeError::Readback)?;
        let code = *status_values
            .first()
            .ok_or(FixedDecodeError::MalformedStream)?;
        let written = *status_values
            .get(1)
            .ok_or(FixedDecodeError::MalformedStream)?;
        if code != STATUS_OK {
            return Err(status_error(code));
        }
        if written != output_bytes_u32 {
            return Err(FixedDecodeError::OutputSizeMismatch);
        }
        let output_words = buffer::read_back::<u32>(ctx, &output_stage, submission)
            .map_err(FixedDecodeError::Readback)?;
        let mut output = Vec::new();
        output.try_reserve_exact(expected_output_size)
            .map_err(|_| FixedDecodeError::Readback(buffer::ReadbackError::Allocation))?;
        for word in output_words {
            let bytes = word.to_le_bytes();
            let count = (expected_output_size - output.len()).min(WORD);
            output.extend_from_slice(&bytes[..count]);
        }
        Ok(output)
        })();
        let validation_error = crate::context::block_on(validation_scope.pop());
        let internal_error = crate::context::block_on(internal_scope.pop());
        let allocation_error = crate::context::block_on(allocation_scope.pop());
        if validation_error.is_some() || internal_error.is_some() || allocation_error.is_some() {
            return Err(FixedDecodeError::DeviceResource);
        }
        result
    }
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
    words.try_reserve_exact(bytes.len() / WORD)
        .map_err(|_| FixedDecodeError::Readback(buffer::ReadbackError::Allocation))?;
    words.extend(bytes.chunks_exact(WORD)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])));
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
