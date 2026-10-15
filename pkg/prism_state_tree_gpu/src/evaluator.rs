//! `StateTree` score-kernel upload, readback, validation, and parity.

use alloc::vec::Vec;
use core::mem::size_of;

use bytemuck::{Pod, Zeroable};
use prism_state_tree_ecs::{
    compare_parity, evaluate_cpu, GpuBatchContract, GpuBatchError, GpuBatchInput, GpuBatchResult,
    GpuBatchValidationError, GpuFallbackReason, GpuParityError, GpuResultStatus,
    GPU_INPUT_STRIDE_BYTES, GPU_OUTPUT_STRIDE_BYTES,
};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;
use crate::{INPUT_ROW_WORDS, OUTPUT_ROW_WORDS, SCORE_SHADER, WORKGROUP_SIZE};

/// One 40-byte device input row.
///
/// The batch's validated plan revision is carried by the uniform block. The
/// row contains sequence, entity, evaluator, consideration, weight, bias, and
/// reserved words in that order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Pod, Zeroable)]
pub struct GpuBatchInputRow {
    /// Fixed-width little-endian words sent to the compute shader.
    pub words: [u32; INPUT_ROW_WORDS],
}

impl GpuBatchInputRow {
    fn from_input(input: GpuBatchInput<'_>, index: usize) -> Self {
        let entity = input.entity_id[index];
        Self {
            words: [
                input.sequence[index],
                entity as u32,
                (entity >> 32) as u32,
                input.evaluator_id[index],
                input.consideration[index] as u32,
                input.weight[index] as u32,
                input.bias[index] as u32,
                0,
                0,
                0,
            ],
        }
    }
}

/// One 48-byte device output row.
///
/// Words contain sequence, entity low/high, four plan-revision limbs,
/// evaluator, signed score bits, status, error code, and a reserved word.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Pod, Zeroable)]
pub struct GpuBatchOutputRow {
    /// Fixed-width little-endian words returned by the compute shader.
    pub words: [u32; OUTPUT_ROW_WORDS],
}

impl GpuBatchOutputRow {
    /// Decodes this fixed row into the ECS GPU contract result.
    pub fn decode(self) -> Result<GpuBatchResult, GpuOutputError> {
        let status = match self.words[9] {
            0 => GpuResultStatus::Ok,
            1 => GpuResultStatus::Saturated,
            actual => return Err(GpuOutputError::InvalidStatus { actual }),
        };
        if self.words[10] != 0 {
            return Err(GpuOutputError::UnexpectedErrorCode {
                actual: self.words[10],
            });
        }
        let revision_words = [self.words[3], self.words[4], self.words[5], self.words[6]];
        let mut revision_bytes = [0u8; 16];
        for (chunk, word) in revision_bytes.chunks_exact_mut(4).zip(revision_words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        Ok(GpuBatchResult {
            sequence: self.words[0],
            entity_id: u64::from(self.words[1]) | (u64::from(self.words[2]) << 32),
            plan_revision: u128::from_le_bytes(revision_bytes),
            evaluator_id: self.words[7],
            score: self.words[8] as i32,
            status,
            error_code: 0,
        })
    }
}

/// Error found while decoding a device output row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuOutputError {
    /// The shader emitted a status value outside the fixed ABI.
    InvalidStatus {
        /// Raw status word.
        actual: u32,
    },
    /// The fixed score kernel emitted a non-zero error code.
    UnexpectedErrorCode {
        /// Raw error code word.
        actual: u32,
    },
}

/// Host-side validation facts retained with a successful evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuValidationRecord {
    /// Number of rows validated and dispatched.
    pub row_count: usize,
    /// Plan revision accepted for every input row.
    pub plan_revision: u128,
    /// Input bytes represented by the fixed row ABI.
    pub input_bytes: usize,
    /// Output bytes represented by the fixed row ABI.
    pub output_bytes: usize,
}

/// Successful device evaluation and its validation record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuEvaluation {
    rows: Vec<GpuBatchResult>,
    validation: GpuValidationRecord,
    fallback: Option<GpuFallbackReason>,
}

impl GpuEvaluation {
    /// Returns the rows in input sequence order.
    #[must_use]
    pub fn rows(&self) -> &[GpuBatchResult] {
        &self.rows
    }

    /// Returns the validation facts recorded before dispatch.
    #[must_use]
    pub const fn validation(&self) -> GpuValidationRecord {
        self.validation
    }

    /// Returns the fallback reason, if this result was marked for fallback.
    #[must_use]
    pub const fn fallback_reason(&self) -> Option<GpuFallbackReason> {
        self.fallback
    }
}

/// Error returned by device evaluation or output validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuEvaluateError {
    /// The input columns or plan revision failed the shared contract.
    InputValidation {
        /// Shared contract validation detail.
        error: GpuBatchValidationError,
        /// Stable fallback classification.
        fallback: GpuFallbackReason,
    },
    /// The input row count cannot be represented by the shader's dispatch word.
    RowCountOverflow {
        /// Number of rows supplied by the caller.
        rows: usize,
        /// Stable fallback classification.
        fallback: GpuFallbackReason,
    },
    /// A device row did not match its input identity or validated revision.
    OutputValidation {
        /// Zero-based row index.
        index: usize,
        /// Stable mismatch field name.
        field: &'static str,
        /// Stable fallback classification.
        fallback: GpuFallbackReason,
    },
    /// The fixed output status/error fields were malformed.
    OutputEncoding {
        /// Zero-based row index.
        index: usize,
        /// Device row decoding detail.
        error: GpuOutputError,
        /// Stable fallback classification.
        fallback: GpuFallbackReason,
    },
    /// The device output disagreed with the CPU golden reference.
    ParityMismatch {
        /// First mismatch returned by the shared parity helper.
        error: GpuParityError,
        /// Stable fallback classification.
        fallback: GpuFallbackReason,
    },
    /// The CPU golden rejected a batch before comparison.
    GoldenEvaluation {
        /// Shared CPU contract error.
        error: GpuBatchError,
        /// Stable fallback classification.
        fallback: GpuFallbackReason,
    },
}

impl GpuEvaluateError {
    /// Returns the stable fallback classification recorded by this error.
    #[must_use]
    pub const fn fallback_reason(self) -> GpuFallbackReason {
        match self {
            Self::InputValidation { fallback, .. }
            | Self::RowCountOverflow { fallback, .. }
            | Self::OutputValidation { fallback, .. }
            | Self::OutputEncoding { fallback, .. }
            | Self::ParityMismatch { fallback, .. }
            | Self::GoldenEvaluation { fallback, .. } => fallback,
        }
    }
}

/// A parity result after a successful device evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuParityReport {
    /// Number of rows compared against the CPU golden.
    pub row_count: usize,
    /// Validation facts used for the dispatch.
    pub validation: GpuValidationRecord,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    row_count: u32,
    revision_0: u32,
    revision_1: u32,
    revision_2: u32,
    revision_3: u32,
}

/// Compiled `StateTree` score pipeline for one fixed batch contract.
pub struct GpuEvaluator {
    contract: GpuBatchContract,
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuEvaluator {
    /// Compiles the score kernel on the context's logical device.
    #[must_use]
    pub fn new(context: &GpuContext, contract: GpuBatchContract) -> Self {
        let device = context.device();
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_state_tree_score_shader"),
            source: ShaderSource::Wgsl(SCORE_SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_state_tree_score_bind_group_layout"),
            entries: &[
                storage_layout(
                    0,
                    BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                storage_layout(
                    1,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                storage_layout(
                    2,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_state_tree_score_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_state_tree_score_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            contract,
            pipeline,
            layout,
        }
    }

    /// Returns the contract used by this compiled evaluator.
    #[must_use]
    pub const fn contract(&self) -> GpuBatchContract {
        self.contract
    }

    /// Evaluates a validated batch on the device and reads back output rows.
    ///
    /// The caller supplies the expected plan revision explicitly. The shared
    /// ECS contract validates every row against it before any GPU submission.
    pub fn evaluate(
        &self,
        context: &GpuContext,
        input: GpuBatchInput<'_>,
        plan_revision: u128,
    ) -> Result<GpuEvaluation, GpuEvaluateError> {
        self.contract
            .validate_plan(input, plan_revision)
            .map_err(|error| GpuEvaluateError::InputValidation {
                error,
                fallback: GpuFallbackReason::Validation,
            })?;
        let row_count = input.len();
        let row_count_u32 =
            u32::try_from(row_count).map_err(|_| GpuEvaluateError::RowCountOverflow {
                rows: row_count,
                fallback: GpuFallbackReason::BufferLimit,
            })?;
        let validation = GpuValidationRecord {
            row_count,
            plan_revision,
            input_bytes: row_count.saturating_mul(GPU_INPUT_STRIDE_BYTES),
            output_bytes: row_count.saturating_mul(GPU_OUTPUT_STRIDE_BYTES),
        };
        if row_count == 0 {
            return Ok(GpuEvaluation {
                rows: Vec::new(),
                validation,
                fallback: None,
            });
        }

        let input_rows: Vec<GpuBatchInputRow> = (0..row_count)
            .map(|index| GpuBatchInputRow::from_input(input, index))
            .collect();
        let revision = plan_revision.to_le_bytes();
        let params = Params {
            row_count: row_count_u32,
            revision_0: u32::from_le_bytes(revision[0..4].try_into().expect("fixed revision limb")),
            revision_1: u32::from_le_bytes(revision[4..8].try_into().expect("fixed revision limb")),
            revision_2: u32::from_le_bytes(
                revision[8..12].try_into().expect("fixed revision limb"),
            ),
            revision_3: u32::from_le_bytes(
                revision[12..16].try_into().expect("fixed revision limb"),
            ),
        };
        let device = context.device();
        let input_buffer =
            buffer::storage_read(device, "prism_state_tree_score_input", &input_rows);
        let params_buffer = buffer::uniform(device, "prism_state_tree_score_params", &params);
        let output_bytes = u64::try_from(validation.output_bytes)
            .expect("validated output byte count should fit the host address space");
        let output_buffer =
            buffer::storage_rw(device, "prism_state_tree_score_output", output_bytes);
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_state_tree_score_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_state_tree_score_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_state_tree_score_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(row_count_u32.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        let staging = buffer::staging(device, "prism_state_tree_score_readback", output_bytes);
        buffer::copy(&mut encoder, &output_buffer, &staging, output_bytes);
        context.queue().submit([encoder.finish()]);
        let raw_rows = buffer::read_back::<GpuBatchOutputRow>(context, &staging);
        let mut rows = Vec::with_capacity(row_count);
        for (index, raw) in raw_rows.into_iter().take(row_count).enumerate() {
            let row = raw
                .decode()
                .map_err(|error| GpuEvaluateError::OutputEncoding {
                    index,
                    error,
                    fallback: GpuFallbackReason::Validation,
                })?;
            validate_output_row(row, input, index, plan_revision)?;
            rows.push(row);
        }
        if rows.len() != row_count {
            return Err(GpuEvaluateError::OutputValidation {
                index: rows.len(),
                field: "row_count",
                fallback: GpuFallbackReason::Validation,
            });
        }
        Ok(GpuEvaluation {
            rows,
            validation,
            fallback: None,
        })
    }

    /// Evaluates the batch and compares every returned field with the CPU golden.
    pub fn parity(
        &self,
        context: &GpuContext,
        input: GpuBatchInput<'_>,
        plan_revision: u128,
    ) -> Result<GpuParityReport, GpuEvaluateError> {
        let evaluation = self.evaluate(context, input, plan_revision)?;
        let mut expected = vec![zero_result(); input.len()];
        let count = evaluate_cpu(self.contract, input, &mut expected).map_err(|error| {
            GpuEvaluateError::GoldenEvaluation {
                error,
                fallback: GpuFallbackReason::Validation,
            }
        })?;
        expected.truncate(count);
        compare_parity(&expected, evaluation.rows()).map_err(|error| {
            GpuEvaluateError::ParityMismatch {
                error,
                fallback: GpuFallbackReason::ParityMismatch,
            }
        })?;
        Ok(GpuParityReport {
            row_count: count,
            validation: evaluation.validation(),
        })
    }
}

fn validate_output_row(
    actual: GpuBatchResult,
    input: GpuBatchInput<'_>,
    index: usize,
    plan_revision: u128,
) -> Result<(), GpuEvaluateError> {
    let checks = [
        (actual.sequence == input.sequence[index], "sequence"),
        (actual.entity_id == input.entity_id[index], "entity_id"),
        (actual.plan_revision == plan_revision, "plan_revision"),
        (
            actual.evaluator_id == input.evaluator_id[index],
            "evaluator_id",
        ),
    ];
    if let Some((_, field)) = checks.into_iter().find(|(matches, _)| !matches) {
        return Err(GpuEvaluateError::OutputValidation {
            index,
            field,
            fallback: if field == "plan_revision" {
                GpuFallbackReason::PlanRevisionMismatch
            } else {
                GpuFallbackReason::Validation
            },
        });
    }
    Ok(())
}

fn zero_result() -> GpuBatchResult {
    GpuBatchResult {
        sequence: 0,
        entity_id: 0,
        plan_revision: 0,
        evaluator_id: 0,
        score: 0,
        status: GpuResultStatus::Ok,
        error_code: 0,
    }
}

fn storage_layout(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

const _: () = assert!(size_of::<GpuBatchInputRow>() == GPU_INPUT_STRIDE_BYTES);
const _: () = assert!(size_of::<GpuBatchOutputRow>() == GPU_OUTPUT_STRIDE_BYTES);

#[cfg(test)]
mod tests {
    use super::*;
    use prism_state_tree_ecs::GpuBatchInput;

    fn input<'a>(sequence: &'a [u32], entity: &'a [u64], plan: &'a [u128]) -> GpuBatchInput<'a> {
        GpuBatchInput::new(
            sequence,
            entity,
            plan,
            &[3, 4],
            &[3 << 16, 4 << 16],
            &[1 << 16, 2 << 16],
            &[0, 1 << 16],
        )
    }

    #[test]
    fn fixed_rows_match_contract_strides() {
        assert_eq!(size_of::<GpuBatchInputRow>(), GPU_INPUT_STRIDE_BYTES);
        assert_eq!(size_of::<GpuBatchOutputRow>(), GPU_OUTPUT_STRIDE_BYTES);
    }

    #[test]
    fn output_row_round_trips_all_fixed_fields() {
        let row = GpuBatchOutputRow {
            words: [
                9,
                0x1122_3344,
                0x5566_7788,
                0x99aa_bbcc,
                0xddee_ff00,
                0x0123_4567,
                0x89ab_cdef,
                17,
                (-12i32) as u32,
                1,
                0,
                0,
            ],
        };
        let decoded = row.decode().unwrap();
        assert_eq!(decoded.sequence, 9);
        assert_eq!(decoded.entity_id, 0x5566_7788_1122_3344);
        assert_eq!(
            decoded.plan_revision,
            0x89ab_cdef_0123_4567_ddee_ff00_99aa_bbcc
        );
        assert_eq!(decoded.evaluator_id, 17);
        assert_eq!(decoded.score, -12);
        assert_eq!(decoded.status, GpuResultStatus::Saturated);
    }

    #[test]
    fn validation_rejects_a_mixed_plan_before_device_use() {
        let contract = GpuBatchContract::new(2);
        let bad = input(&[0, 1], &[7, 8], &[11, 12]);
        assert!(matches!(
            contract.validate_plan(bad, 11),
            Err(GpuBatchValidationError::PlanRevisionMismatch { index: 1, .. })
        ));
    }

    #[test]
    fn input_row_uses_fixed_columns_and_reserved_words() {
        let values = input(&[7, 8], &[0x1122_3344_5566_7788, 9], &[11, 11]);
        let row = GpuBatchInputRow::from_input(values, 0);
        assert_eq!(row.words[0], 7);
        assert_eq!(row.words[1], 0x5566_7788);
        assert_eq!(row.words[2], 0x1122_3344);
        assert_eq!(row.words[4], 3 << 16);
        assert_eq!(row.words[6], 0);
        assert_eq!(&row.words[7..], &[0, 0, 0]);
    }
}
