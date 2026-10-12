//! GPU fast path for `GDeflate` tiles made entirely of RFC 1951 stored blocks.
//!
//! This is deliberately a narrow production slice, not a claim of complete
//! GPU `GDeflate` support. The host scans the recovered raw-DEFLATE headers and
//! builds a validated `(source_offset, destination_offset, length)` plan. The
//! GPU then performs the actual de-interleave and parallel word assembly in a
//! single command submission. Fixed- and dynamic-Huffman blocks are rejected
//! by [`StoredBlockPlan::from_linear_deflate`] so callers can choose the
//! existing CPU `inflate` fallback instead of silently producing bad bytes.
//!
//! Stored blocks are useful for incompressible asset pages, where a GPU path
//! must not pay Huffman setup for data that has no compression benefit. The
//! parser validates `LEN/NLEN`, exact compressed and output bounds, and the
//! final-block marker before any device work is submitted.

use alloc::vec::Vec;
use core::mem::size_of;

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;
use crate::deinterleave::{GpuDeinterleave, GROUP, WORD};

/// A validated stored-block copy range in the linear DEFLATE stream.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Pod, Zeroable)]
pub struct StoredBlock {
    /// Byte offset of the payload in the recovered raw-DEFLATE stream,
    /// immediately after the block's five-byte stored-block header.
    pub source_offset: u32,
    /// Byte offset in the uncompressed output.
    pub destination_offset: u32,
    /// Number of payload bytes to copy.
    pub length: u32,
}

/// A validated plan for a raw DEFLATE stream containing only stored blocks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoredBlockPlan {
    blocks: Vec<StoredBlock>,
    compressed_size: u32,
    output_size: u32,
}

/// Why a raw DEFLATE stream cannot use the stored-block GPU fast path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StoredPlanError {
    /// The compressed stream is empty or the advertised output is too large
    /// for the GPU ABI.
    InvalidSize,
    /// The stream ended before a stored-block header, length pair, or payload.
    Truncated,
    /// The block uses fixed or dynamic Huffman coding (or the reserved type).
    UnsupportedBlockType(u8),
    /// `NLEN` is not the one's complement of `LEN`.
    LengthCheckMismatch,
    /// The stream did not contain a final block or had bytes after it.
    TrailingOrMissingFinal,
    /// The sum of stored payload lengths did not equal the advertised output.
    OutputSizeMismatch,
    /// The number of blocks cannot be represented by the GPU uniform ABI.
    TooManyBlocks,
    /// A source, block table, output buffer, or dispatch exceeds device limits.
    DeviceLimitExceeded,
}

impl StoredBlockPlan {
    /// Parses and validates a raw stored-block DEFLATE stream.
    pub fn from_linear_deflate(
        linear: &[u8],
        expected_output_size: usize,
    ) -> Result<Self, StoredPlanError> {
        if linear.is_empty()
            || linear.len() > u32::MAX as usize
            || expected_output_size > u32::MAX as usize
        {
            return Err(StoredPlanError::InvalidSize);
        }

        let mut cursor = 0usize;
        let mut destination = 0usize;
        let mut blocks = Vec::new();
        loop {
            let header = *linear.get(cursor).ok_or(StoredPlanError::Truncated)?;
            cursor += 1;
            let final_block = header & 1 != 0;
            let block_type = (header >> 1) & 0b11;
            if block_type != 0 {
                return Err(StoredPlanError::UnsupportedBlockType(block_type));
            }

            let len_bytes = linear
                .get(cursor..cursor + 4)
                .ok_or(StoredPlanError::Truncated)?;
            let length = u16::from_le_bytes([len_bytes[0], len_bytes[1]]) as usize;
            let inverse = u16::from_le_bytes([len_bytes[2], len_bytes[3]]);
            if inverse != !(length as u16) {
                return Err(StoredPlanError::LengthCheckMismatch);
            }
            cursor += 4;

            let end = cursor
                .checked_add(length)
                .ok_or(StoredPlanError::Truncated)?;
            if end > linear.len() {
                return Err(StoredPlanError::Truncated);
            }
            let next_destination = destination
                .checked_add(length)
                .ok_or(StoredPlanError::InvalidSize)?;
            if next_destination > u32::MAX as usize {
                return Err(StoredPlanError::InvalidSize);
            }
            blocks.push(StoredBlock {
                source_offset: cursor as u32,
                destination_offset: destination as u32,
                length: length as u32,
            });
            destination = next_destination;
            cursor = end;

            if final_block {
                if cursor != linear.len() {
                    return Err(StoredPlanError::TrailingOrMissingFinal);
                }
                break;
            }
        }

        if destination != expected_output_size {
            return Err(StoredPlanError::OutputSizeMismatch);
        }
        if blocks.len() > u32::MAX as usize {
            return Err(StoredPlanError::TooManyBlocks);
        }
        Ok(Self {
            blocks,
            compressed_size: linear.len() as u32,
            output_size: expected_output_size as u32,
        })
    }

    /// Returns the validated block ranges.
    #[must_use]
    pub fn blocks(&self) -> &[StoredBlock] {
        &self.blocks
    }

    /// Returns the exact raw-DEFLATE byte length represented by this plan.
    #[must_use]
    pub const fn compressed_size(&self) -> usize {
        self.compressed_size as usize
    }

    /// Returns the exact uncompressed byte length represented by this plan.
    #[must_use]
    pub const fn output_size(&self) -> usize {
        self.output_size as usize
    }

    /// Returns the exact lane-major byte length required by the `GDeflate`
    /// transpose for this plan.
    ///
    /// The padding is part of the layout, not arbitrary trailing capacity:
    /// changing it changes the number of transpose rounds and therefore the
    /// mapping from lane-major words back to the linear DEFLATE stream.
    #[must_use]
    pub const fn interleaved_size(&self) -> usize {
        (self.compressed_size as usize).div_ceil(GROUP) * GROUP
    }
}

/// Parses a raw stored-block stream into a GPU-ready plan.
#[must_use = "the validated plan contains the ranges needed by the GPU path"]
pub fn plan_stored_deflate(
    linear: &[u8],
    expected_output_size: usize,
) -> Result<StoredBlockPlan, StoredPlanError> {
    StoredBlockPlan::from_linear_deflate(linear, expected_output_size)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    block_count: u32,
    output_words: u32,
    output_bytes: u32,
    _pad: u32,
}

/// GPU implementation of the validated stored-block `GDeflate` fast path.
pub struct GpuStoredGDeflate {
    deinterleave: GpuDeinterleave,
    pipeline: ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl GpuStoredGDeflate {
    /// Compiles the de-interleave and stored-block copy kernels.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_gdeflate_stored_copy"),
            source: ShaderSource::Wgsl(include_str!("shaders/stored_copy.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_gdeflate_stored_copy_layout"),
            entries: &[
                buffer_layout(0, BufferBindingType::Uniform),
                buffer_layout(1, BufferBindingType::Storage { read_only: true }),
                buffer_layout(2, BufferBindingType::Storage { read_only: true }),
                buffer_layout(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_gdeflate_stored_copy_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_gdeflate_stored_copy_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            deinterleave: GpuDeinterleave::new(ctx),
            pipeline,
            layout,
        }
    }

    /// Executes the stored-block path and returns the exact output bytes.
    ///
    /// `interleaved` must be the complete lane-major payload, including zero
    /// padding up to a [`GROUP`] boundary. The plan must have been made from
    /// the corresponding linear DEFLATE bytes and its `compressed_size` must
    /// match the bytes before padding.
    pub fn decompress(
        &self,
        ctx: &GpuContext,
        interleaved: &[u8],
        plan: &StoredBlockPlan,
    ) -> Result<Vec<u8>, StoredPlanError> {
        if interleaved.len() != plan.interleaved_size()
            || interleaved.len() > u32::MAX as usize
            || plan.blocks.is_empty()
        {
            return Err(StoredPlanError::InvalidSize);
        }
        if plan.output_size() == 0 {
            return Ok(Vec::new());
        }

        let device = ctx.device();
        let output_bytes = plan.output_size().div_ceil(WORD) * WORD;
        let source_bytes = interleaved.len();
        let block_bytes = plan.blocks.len().saturating_mul(size_of::<StoredBlock>());
        let limits = device.limits();
        let output_words = plan.output_size().div_ceil(WORD) as u32;
        if source_bytes as u64 > limits.max_buffer_size
            || output_bytes as u64 > limits.max_buffer_size
            || source_bytes as u64 > limits.max_storage_buffer_binding_size
            || output_bytes as u64 > limits.max_storage_buffer_binding_size
            || block_bytes as u64 > limits.max_buffer_size
            || block_bytes as u64 > limits.max_storage_buffer_binding_size
            || groups(output_words) > limits.max_compute_workgroups_per_dimension
        {
            return Err(StoredPlanError::DeviceLimitExceeded);
        }
        let job = self.deinterleave.prepare(ctx, interleaved);
        let params = buffer::uniform(
            device,
            "prism_gdeflate_stored_params",
            &Params {
                block_count: plan.blocks.len() as u32,
                output_words,
                output_bytes: plan.output_size() as u32,
                _pad: 0,
            },
        );
        let blocks = buffer::storage_read(device, "prism_gdeflate_stored_blocks", &plan.blocks);
        let output =
            buffer::storage_rw_zeroed(device, "prism_gdeflate_stored_output", output_bytes as u64);
        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_gdeflate_stored_bind"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: job.output.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: blocks.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: output.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_gdeflate_stored_encoder"),
        });
        self.deinterleave.record(&mut encoder, &job);
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_gdeflate_stored_copy_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups(plan.output_size().div_ceil(WORD) as u32), 1, 1);
        }
        let stage = buffer::staging(device, "prism_gdeflate_stored_stage", output_bytes as u64);
        buffer::copy(&mut encoder, &output, &stage, output_bytes as u64);
        ctx.queue().submit([encoder.finish()]);
        let mut result = buffer::read_back::<u32>(ctx, &stage);
        result.truncate(plan.output_size().div_ceil(WORD));
        let bytes = bytemuck::cast_slice::<u32, u8>(&result);
        Ok(bytes[..plan.output_size()].to_vec())
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

fn groups(n: u32) -> u32 {
    n.div_ceil(256).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored_stream(payloads: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for (index, payload) in payloads.iter().enumerate() {
            assert!(payload.len() <= u16::MAX as usize);
            let len = payload.len() as u16;
            out.push(if index + 1 == payloads.len() { 1 } else { 0 });
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&(!len).to_le_bytes());
            out.extend_from_slice(payload);
        }
        out
    }

    #[test]
    fn plan_accepts_multiple_stored_blocks() {
        let stream = stored_stream(&[b"abc", b"012345"]);
        let plan = plan_stored_deflate(&stream, 9).unwrap();
        assert_eq!(plan.blocks().len(), 2);
        assert_eq!(plan.blocks()[0].source_offset, 5);
        assert_eq!(plan.blocks()[1].source_offset, 13);
        assert_eq!(plan.output_size(), 9);
    }

    #[test]
    fn plan_rejects_huffman_and_bad_bounds() {
        assert_eq!(
            plan_stored_deflate(&[0b0000_0011], 0),
            Err(StoredPlanError::UnsupportedBlockType(1))
        );
        let stream = stored_stream(&[b"abc"]);
        assert_eq!(
            plan_stored_deflate(&stream, 2),
            Err(StoredPlanError::OutputSizeMismatch)
        );
        let mut corrupt = stream.clone();
        corrupt[3] ^= 1;
        assert_eq!(
            plan_stored_deflate(&corrupt, 3),
            Err(StoredPlanError::LengthCheckMismatch)
        );
        let mut trailing = stream;
        trailing.extend_from_slice(&[0, 0]);
        assert_eq!(
            plan_stored_deflate(&trailing, 3),
            Err(StoredPlanError::TrailingOrMissingFinal)
        );
    }

    #[test]
    fn plan_accepts_empty_final_stored_block() {
        let stream = stored_stream(&[b""]);
        let plan = plan_stored_deflate(&stream, 0).unwrap();
        assert_eq!(plan.blocks()[0].length, 0);
        assert_eq!(plan.output_size(), 0);
    }

    #[test]
    fn interleaved_size_is_derived_from_compressed_size() {
        let stream = stored_stream(&[b"abc"]);
        let plan = plan_stored_deflate(&stream, 3).unwrap();
        assert_eq!(plan.compressed_size(), stream.len());
        assert_eq!(plan.interleaved_size(), GROUP);

        let payload = vec![0u8; 124];
        let two_group_stream = stored_stream(&[payload.as_slice()]);
        let two_group_plan = plan_stored_deflate(&two_group_stream, payload.len()).unwrap();
        assert_eq!(two_group_plan.interleaved_size(), GROUP * 2);
    }
}
