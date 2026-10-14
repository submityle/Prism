//! Device-neutral batch contract for pure `StateTree` evaluators.
//!
//! The contract is deliberately independent of a graphics API. A device
//! backend can upload the same fixed columns and return the same result
//! records; the CPU evaluator remains the reference implementation and the
//! authority for deterministic tests.

use alloc::vec::Vec;
use core::fmt;

/// Number of bytes occupied by one logical input row in the packed ABI.
pub const GPU_INPUT_STRIDE_BYTES: usize = 40;

/// Number of bytes occupied by one logical output row in the packed ABI.
pub const GPU_OUTPUT_STRIDE_BYTES: usize = 48;

/// Number of fractional bits used by evaluator scores.
pub const SCORE_FRACTION_BITS: u32 = 16;

/// Default row bound for reusable CPU reference buffers.
pub const DEFAULT_MAX_ROWS: usize = 1024;

/// Stable numeric keys for input columns.
#[repr(u16)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum GpuBatchInputKey {
    /// Caller sequence number.
    Sequence = 1,
    /// Stable host entity identity.
    EntityId = 2,
    /// Semantic plan revision fingerprint.
    PlanRevision = 3,
    /// Stable evaluator identity.
    EvaluatorId = 4,
    /// Quantized consideration value.
    ConsiderationQ16 = 5,
    /// Quantized evaluator weight.
    WeightQ16 = 6,
    /// Quantized score bias.
    BiasQ16 = 7,
}

impl GpuBatchInputKey {
    /// Returns the stable wire identifier.
    pub const fn wire_id(self) -> u16 {
        self as u16
    }

    /// Returns the stable textual key.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sequence => "sequence",
            Self::EntityId => "entity_id",
            Self::PlanRevision => "plan_revision",
            Self::EvaluatorId => "evaluator_id",
            Self::ConsiderationQ16 => "consideration_q16",
            Self::WeightQ16 => "weight_q16",
            Self::BiasQ16 => "bias_q16",
        }
    }

    /// Converts a stable wire identifier to a key.
    pub const fn from_wire_id(id: u16) -> Option<Self> {
        match id {
            1 => Some(Self::Sequence),
            2 => Some(Self::EntityId),
            3 => Some(Self::PlanRevision),
            4 => Some(Self::EvaluatorId),
            5 => Some(Self::ConsiderationQ16),
            6 => Some(Self::WeightQ16),
            7 => Some(Self::BiasQ16),
            _ => None,
        }
    }
}

/// Stable numeric keys for output columns.
#[repr(u16)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum GpuBatchOutputKey {
    /// Caller sequence number.
    Sequence = 1,
    /// Stable host entity identity.
    EntityId = 2,
    /// Semantic plan revision fingerprint.
    PlanRevision = 3,
    /// Stable evaluator identity.
    EvaluatorId = 4,
    /// Quantized score.
    ScoreQ16 = 5,
    /// Result status.
    Status = 6,
    /// Stable error code; zero means success.
    ErrorCode = 7,
}

impl GpuBatchOutputKey {
    /// Returns the stable wire identifier.
    pub const fn wire_id(self) -> u16 {
        self as u16
    }

    /// Returns the stable textual key.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sequence => "sequence",
            Self::EntityId => "entity_id",
            Self::PlanRevision => "plan_revision",
            Self::EvaluatorId => "evaluator_id",
            Self::ScoreQ16 => "score_q16",
            Self::Status => "status",
            Self::ErrorCode => "error_code",
        }
    }

    /// Converts a stable wire identifier to a key.
    pub const fn from_wire_id(id: u16) -> Option<Self> {
        match id {
            1 => Some(Self::Sequence),
            2 => Some(Self::EntityId),
            3 => Some(Self::PlanRevision),
            4 => Some(Self::EvaluatorId),
            5 => Some(Self::ScoreQ16),
            6 => Some(Self::Status),
            7 => Some(Self::ErrorCode),
            _ => None,
        }
    }
}

/// Quantizes a normalized input to signed Q16.
pub fn quantize_unit(value: f32) -> Result<i32, GpuQuantizationError> {
    if !value.is_finite() {
        return Err(GpuQuantizationError::NonFinite);
    }
    if value < 0.0 {
        return Err(GpuQuantizationError::BelowZero);
    }
    if value > 1.0 {
        return Err(GpuQuantizationError::AboveOne);
    }
    Ok((f64::from(value) * 65_536.0 + 0.5) as i32)
}

/// Converts a normalized bias to signed Q16 with deterministic rounding.
pub fn quantize_bias(value: f32) -> Result<i32, GpuQuantizationError> {
    if !value.is_finite() {
        return Err(GpuQuantizationError::NonFinite);
    }
    if value < -1.0 {
        return Err(GpuQuantizationError::BelowMinusOne);
    }
    if value > 1.0 {
        return Err(GpuQuantizationError::AboveOne);
    }
    let scaled = f64::from(value) * 65_536.0;
    Ok(if scaled >= 0.0 {
        (scaled + 0.5) as i32
    } else {
        (scaled - 0.5) as i32
    })
}

/// Converts a signed Q16 score to a normalized host value.
#[must_use]
pub fn dequantize_score(value: i32) -> f32 {
    value as f32 / 65_536.0
}

/// Quantization input error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuQuantizationError {
    /// The value is NaN or infinite.
    NonFinite,
    /// A normalized value is below zero.
    BelowZero,
    /// A normalized value is above one.
    AboveOne,
    /// A bias is below minus one.
    BelowMinusOne,
}

/// A fixed-column, read-only batch view supplied by a host or device backend.
#[derive(Clone, Copy, Debug)]
pub struct GpuBatchInput<'a> {
    /// Stable input sequence number.
    pub sequence: &'a [u32],
    /// Stable host entity identity.
    pub entity_id: &'a [u64],
    /// Semantic plan revision fingerprint.
    pub plan_revision: &'a [u128],
    /// Stable evaluator identity.
    pub evaluator_id: &'a [u32],
    /// Quantized consideration value in Q16 format.
    pub consideration: &'a [i32],
    /// Quantized evaluator weight in Q16 format.
    pub weight: &'a [i32],
    /// Quantized score bias in Q16 format.
    pub bias: &'a [i32],
}

impl<'a> GpuBatchInput<'a> {
    /// Creates a column view without copying any input data.
    pub const fn new(
        sequence: &'a [u32],
        entity_id: &'a [u64],
        plan_revision: &'a [u128],
        evaluator_id: &'a [u32],
        consideration: &'a [i32],
        weight: &'a [i32],
        bias: &'a [i32],
    ) -> Self {
        Self {
            sequence,
            entity_id,
            plan_revision,
            evaluator_id,
            consideration,
            weight,
            bias,
        }
    }

    /// Returns the number of logical rows represented by the first column.
    pub const fn len(self) -> usize {
        self.sequence.len()
    }

    /// Returns whether all columns have the same row count.
    pub const fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Validates equal column lengths and a caller-supplied batch limit.
    pub fn validate(self, max_rows: usize) -> Result<(), GpuBatchError> {
        let expected = self.len();
        let columns = [
            self.entity_id.len(),
            self.plan_revision.len(),
            self.evaluator_id.len(),
            self.consideration.len(),
            self.weight.len(),
            self.bias.len(),
        ];
        for (column, actual) in columns.into_iter().enumerate() {
            if actual != expected {
                return Err(GpuBatchError::ColumnLength {
                    column,
                    expected,
                    actual,
                });
            }
        }
        if expected > max_rows {
            return Err(GpuBatchError::BatchLimit {
                requested: expected,
                maximum: max_rows,
            });
        }
        Ok(())
    }
}

/// Result status emitted for one evaluator row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuResultStatus {
    /// The row was evaluated successfully.
    Ok,
    /// The arithmetic result saturated at the representable score range.
    Saturated,
    /// The row was not produced by a backend and must use CPU fallback.
    Fallback(GpuFallbackReason),
}

/// Explicit reason a device batch cannot be trusted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuFallbackReason {
    /// No compatible device backend is available.
    Unsupported,
    /// The device exceeded the caller's execution deadline.
    Timeout,
    /// The device was lost or reset.
    DeviceLost,
    /// Result validation failed.
    Validation,
    /// The device lacks a required capability.
    CapabilityInsufficient,
    /// The bounded row or byte budget was exceeded.
    BufferLimit,
    /// Device output differs from the CPU reference.
    ParityMismatch,
    /// Device output uses another plan revision.
    PlanRevisionMismatch,
}

impl GpuFallbackReason {
    /// Returns the stable numeric reason code.
    pub const fn code(self) -> u16 {
        match self {
            Self::Unsupported => 1,
            Self::Timeout => 2,
            Self::DeviceLost => 3,
            Self::Validation => 4,
            Self::CapabilityInsufficient => 5,
            Self::BufferLimit => 6,
            Self::ParityMismatch => 7,
            Self::PlanRevisionMismatch => 8,
        }
    }

    /// Returns the stable textual reason.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Unsupported => "unsupported",
            Self::Timeout => "timeout",
            Self::DeviceLost => "device_lost",
            Self::Validation => "validation",
            Self::CapabilityInsufficient => "capability_insufficient",
            Self::BufferLimit => "buffer_limit",
            Self::ParityMismatch => "parity_mismatch",
            Self::PlanRevisionMismatch => "plan_revision_mismatch",
        }
    }
}

/// One fixed-layout evaluator result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuBatchResult {
    /// Stable input sequence number.
    pub sequence: u32,
    /// Stable host entity identity.
    pub entity_id: u64,
    /// Semantic plan revision fingerprint.
    pub plan_revision: u128,
    /// Stable evaluator identity.
    pub evaluator_id: u32,
    /// Quantized score in Q16 format.
    pub score: i32,
    /// Validation/fallback status.
    pub status: GpuResultStatus,
    /// Stable producer error code; zero means success.
    pub error_code: u16,
}

/// Device-neutral limits and ABI metadata for one batch submission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuBatchContract {
    max_rows: usize,
    max_input_bytes: usize,
    max_output_bytes: usize,
}

impl GpuBatchContract {
    /// Creates a contract with a fixed maximum row count.
    pub const fn new(max_rows: usize) -> Self {
        Self {
            max_rows,
            max_input_bytes: max_rows.saturating_mul(GPU_INPUT_STRIDE_BYTES),
            max_output_bytes: max_rows.saturating_mul(GPU_OUTPUT_STRIDE_BYTES),
        }
    }

    /// Returns the maximum row count accepted by the contract.
    pub const fn max_rows(self) -> usize {
        self.max_rows
    }

    /// Returns the packed input stride advertised to a backend.
    pub const fn input_stride_bytes(self) -> usize {
        GPU_INPUT_STRIDE_BYTES
    }

    /// Returns the fixed output stride advertised to a backend.
    pub const fn output_stride_bytes(self) -> usize {
        GPU_OUTPUT_STRIDE_BYTES
    }

    /// Returns the maximum input byte budget.
    pub const fn max_input_bytes(self) -> usize {
        self.max_input_bytes
    }

    /// Returns the maximum output byte budget.
    pub const fn max_output_bytes(self) -> usize {
        self.max_output_bytes
    }

    /// Validates a batch against this contract.
    pub fn validate(self, input: GpuBatchInput<'_>) -> Result<(), GpuBatchError> {
        input.validate(self.max_rows)
    }

    /// Validates a batch against one exact plan revision.
    pub fn validate_plan(
        self,
        input: GpuBatchInput<'_>,
        plan_revision: u128,
    ) -> Result<(), GpuBatchValidationError> {
        input.validate_plan(self, plan_revision)
    }
}

/// Input validation error with stable column identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuBatchValidationError {
    /// A column does not match the sequence column length.
    ColumnLength {
        /// Mismatched column.
        key: GpuBatchInputKey,
        /// Expected row count.
        expected: usize,
        /// Actual row count.
        actual: usize,
    },
    /// The batch exceeds the row limit.
    BatchLimit {
        /// Requested row count.
        requested: usize,
        /// Maximum row count.
        maximum: usize,
    },
    /// A row uses another plan revision.
    PlanRevisionMismatch {
        /// Row index.
        index: usize,
        /// Required revision.
        expected: u128,
        /// Supplied revision.
        actual: u128,
    },
}

impl GpuBatchInput<'_> {
    fn validate_plan(
        self,
        contract: GpuBatchContract,
        expected_revision: u128,
    ) -> Result<(), GpuBatchValidationError> {
        let columns = [
            (GpuBatchInputKey::EntityId, self.entity_id.len()),
            (GpuBatchInputKey::PlanRevision, self.plan_revision.len()),
            (GpuBatchInputKey::EvaluatorId, self.evaluator_id.len()),
            (GpuBatchInputKey::ConsiderationQ16, self.consideration.len()),
            (GpuBatchInputKey::WeightQ16, self.weight.len()),
            (GpuBatchInputKey::BiasQ16, self.bias.len()),
        ];
        for (key, actual) in columns {
            if actual != self.len() {
                return Err(GpuBatchValidationError::ColumnLength {
                    key,
                    expected: self.len(),
                    actual,
                });
            }
        }
        if self.len() > contract.max_rows {
            return Err(GpuBatchValidationError::BatchLimit {
                requested: self.len(),
                maximum: contract.max_rows,
            });
        }
        for (index, &actual) in self.plan_revision.iter().enumerate() {
            if actual != expected_revision {
                return Err(GpuBatchValidationError::PlanRevisionMismatch {
                    index,
                    expected: expected_revision,
                    actual,
                });
            }
        }
        Ok(())
    }
}

/// Errors returned before a batch is evaluated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuBatchError {
    /// Input validation failed with a stable structured reason.
    Validation(GpuBatchValidationError),
    /// One column does not match the row count of the sequence column.
    ColumnLength {
        /// Zero-based column index in the fixed ABI.
        column: usize,
        /// Expected row count.
        expected: usize,
        /// Actual row count.
        actual: usize,
    },
    /// The batch exceeds the contract's bounded capacity.
    BatchLimit {
        /// Requested row count.
        requested: usize,
        /// Maximum accepted row count.
        maximum: usize,
    },
    /// The output slice cannot hold all result rows.
    OutputCapacity {
        /// Requested result count.
        requested: usize,
        /// Available result slots.
        available: usize,
    },
    /// Fixed-point multiplication overflowed before saturation.
    ArithmeticOverflow,
}

/// One row accepted by an owned bounded input buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuBatchInputRow {
    /// Stable input sequence number.
    pub sequence: u32,
    /// Stable host entity identity.
    pub entity_id: u64,
    /// Semantic plan revision fingerprint.
    pub plan_revision: u128,
    /// Stable evaluator identity.
    pub evaluator_id: u32,
    /// Quantized consideration in Q16.
    pub consideration: i32,
    /// Quantized evaluator weight in Q16.
    pub weight: i32,
    /// Quantized score bias in Q16.
    pub bias: i32,
}

/// Error raised by a bounded reusable buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuBatchBufferError {
    /// Host allocation could not reserve the configured capacity.
    AllocationFailed,
    /// A push would exceed the row limit.
    CapacityExceeded {
        /// Maximum accepted rows.
        maximum: usize,
    },
    /// An input row did not use the next sequence number.
    NonCanonicalSequence {
        /// Required sequence number.
        expected: u32,
        /// Supplied sequence number.
        actual: u32,
    },
}

/// Owned, bounded `SoA` input storage for repeated submissions.
pub struct GpuBatchInputBuffer {
    contract: GpuBatchContract,
    sequence: Vec<u32>,
    entity_id: Vec<u64>,
    plan_revision: Vec<u128>,
    evaluator_id: Vec<u32>,
    consideration: Vec<i32>,
    weight: Vec<i32>,
    bias: Vec<i32>,
}

impl GpuBatchInputBuffer {
    /// Creates a buffer and reserves every column to the contract limit.
    pub fn with_contract(contract: GpuBatchContract) -> Result<Self, GpuBatchBufferError> {
        let mut buffer = Self {
            contract,
            sequence: Vec::new(),
            entity_id: Vec::new(),
            plan_revision: Vec::new(),
            evaluator_id: Vec::new(),
            consideration: Vec::new(),
            weight: Vec::new(),
            bias: Vec::new(),
        };
        for result in [
            buffer.sequence.try_reserve_exact(contract.max_rows),
            buffer.entity_id.try_reserve_exact(contract.max_rows),
            buffer.plan_revision.try_reserve_exact(contract.max_rows),
            buffer.evaluator_id.try_reserve_exact(contract.max_rows),
            buffer.consideration.try_reserve_exact(contract.max_rows),
            buffer.weight.try_reserve_exact(contract.max_rows),
            buffer.bias.try_reserve_exact(contract.max_rows),
        ] {
            result.map_err(|_| GpuBatchBufferError::AllocationFailed)?;
        }
        Ok(buffer)
    }

    /// Creates a buffer using [`DEFAULT_MAX_ROWS`].
    pub fn new() -> Result<Self, GpuBatchBufferError> {
        Self::with_contract(GpuBatchContract::new(DEFAULT_MAX_ROWS))
    }

    /// Returns the contract used by this buffer.
    pub const fn contract(&self) -> GpuBatchContract {
        self.contract
    }

    /// Returns the current row count.
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Returns whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    /// Returns retained sequence-column capacity.
    pub fn capacity(&self) -> usize {
        self.sequence.capacity()
    }

    /// Clears rows without releasing allocations.
    pub fn clear(&mut self) {
        self.sequence.clear();
        self.entity_id.clear();
        self.plan_revision.clear();
        self.evaluator_id.clear();
        self.consideration.clear();
        self.weight.clear();
        self.bias.clear();
    }

    /// Appends one row without growing beyond the configured limit.
    pub fn push(&mut self, row: GpuBatchInputRow) -> Result<(), GpuBatchBufferError> {
        if self.len() >= self.contract.max_rows {
            return Err(GpuBatchBufferError::CapacityExceeded {
                maximum: self.contract.max_rows,
            });
        }
        let expected = self.len() as u32;
        if row.sequence != expected {
            return Err(GpuBatchBufferError::NonCanonicalSequence {
                expected,
                actual: row.sequence,
            });
        }
        self.sequence.push(row.sequence);
        self.entity_id.push(row.entity_id);
        self.plan_revision.push(row.plan_revision);
        self.evaluator_id.push(row.evaluator_id);
        self.consideration.push(row.consideration);
        self.weight.push(row.weight);
        self.bias.push(row.bias);
        Ok(())
    }

    /// Borrows this buffer as a fixed-column input view.
    pub fn as_input(&self) -> GpuBatchInput<'_> {
        GpuBatchInput::new(
            &self.sequence,
            &self.entity_id,
            &self.plan_revision,
            &self.evaluator_id,
            &self.consideration,
            &self.weight,
            &self.bias,
        )
    }
}

/// Owned, bounded output storage for repeated submissions.
pub struct GpuBatchOutputBuffer {
    contract: GpuBatchContract,
    rows: Vec<GpuBatchResult>,
}

impl GpuBatchOutputBuffer {
    /// Creates an output buffer reserved to the contract row limit.
    pub fn with_contract(contract: GpuBatchContract) -> Result<Self, GpuBatchBufferError> {
        let mut rows = Vec::new();
        rows.try_reserve_exact(contract.max_rows)
            .map_err(|_| GpuBatchBufferError::AllocationFailed)?;
        Ok(Self { contract, rows })
    }

    /// Creates an output buffer using [`DEFAULT_MAX_ROWS`].
    pub fn new() -> Result<Self, GpuBatchBufferError> {
        Self::with_contract(GpuBatchContract::new(DEFAULT_MAX_ROWS))
    }

    /// Returns the contract used by this buffer.
    pub const fn contract(&self) -> GpuBatchContract {
        self.contract
    }

    /// Returns the number of rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Returns whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Returns retained row capacity.
    pub fn capacity(&self) -> usize {
        self.rows.capacity()
    }

    /// Clears rows without releasing the allocation.
    pub fn clear(&mut self) {
        self.rows.clear();
    }

    /// Returns output rows in their current order.
    pub fn rows(&self) -> &[GpuBatchResult] {
        &self.rows
    }

    /// Returns mutable output rows for a backend adapter.
    pub fn rows_mut(&mut self) -> &mut [GpuBatchResult] {
        &mut self.rows
    }

    /// Appends a row without allocating beyond the configured limit.
    pub fn push(&mut self, row: GpuBatchResult) -> Result<(), GpuBatchBufferError> {
        if self.len() >= self.contract.max_rows {
            return Err(GpuBatchBufferError::CapacityExceeded {
                maximum: self.contract.max_rows,
            });
        }
        self.rows.push(row);
        Ok(())
    }

    /// Sorts rows by stable entity, evaluator, then sequence identity.
    pub fn sort_canonical(&mut self) {
        self.rows
            .sort_unstable_by_key(|row| (row.entity_id, row.evaluator_id, row.sequence));
    }
}

impl fmt::Display for GpuBatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(error) => {
                write!(formatter, "GPU batch input validation failed: {error:?}")
            }
            Self::ColumnLength {
                column,
                expected,
                actual,
            } => write!(
                formatter,
                "column {column} has {actual} rows; expected {expected}"
            ),
            Self::BatchLimit { requested, maximum } => {
                write!(
                    formatter,
                    "batch has {requested} rows; maximum is {maximum}"
                )
            }
            Self::OutputCapacity {
                requested,
                available,
            } => write!(formatter, "output has {available} slots; need {requested}"),
            Self::ArithmeticOverflow => formatter.write_str("fixed-point arithmetic overflow"),
        }
    }
}

/// Evaluates a batch using the deterministic CPU reference implementation.
pub fn evaluate_cpu(
    contract: GpuBatchContract,
    input: GpuBatchInput<'_>,
    output: &mut [GpuBatchResult],
) -> Result<usize, GpuBatchError> {
    contract.validate(input)?;
    if output.len() < input.len() {
        return Err(GpuBatchError::OutputCapacity {
            requested: input.len(),
            available: output.len(),
        });
    }
    for (index, output_row) in output.iter_mut().enumerate().take(input.len()) {
        let product = i64::from(input.consideration[index])
            .checked_mul(i64::from(input.weight[index]))
            .ok_or(GpuBatchError::ArithmeticOverflow)?;
        let scaled = product >> SCORE_FRACTION_BITS;
        let combined = scaled + i64::from(input.bias[index]);
        let score = combined.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        let status = if combined < i64::from(i32::MIN) || combined > i64::from(i32::MAX) {
            GpuResultStatus::Saturated
        } else {
            GpuResultStatus::Ok
        };
        *output_row = GpuBatchResult {
            sequence: input.sequence[index],
            entity_id: input.entity_id[index],
            plan_revision: input.plan_revision[index],
            evaluator_id: input.evaluator_id[index],
            score,
            status,
            error_code: 0,
        };
    }
    Ok(input.len())
}

/// Evaluates directly into a reusable bounded output buffer.
pub fn evaluate_cpu_buffer(
    contract: GpuBatchContract,
    input: GpuBatchInput<'_>,
    output: &mut GpuBatchOutputBuffer,
) -> Result<usize, GpuBatchError> {
    contract.validate(input)?;
    if output.contract.max_rows < input.len() {
        return Err(GpuBatchError::OutputCapacity {
            requested: input.len(),
            available: output.contract.max_rows,
        });
    }
    output.clear();
    for index in 0..input.len() {
        let product = i64::from(input.consideration[index])
            .checked_mul(i64::from(input.weight[index]))
            .ok_or(GpuBatchError::ArithmeticOverflow)?;
        let combined = (product >> SCORE_FRACTION_BITS) + i64::from(input.bias[index]);
        let saturated = combined < i64::from(i32::MIN) || combined > i64::from(i32::MAX);
        output
            .push(GpuBatchResult {
                sequence: input.sequence[index],
                entity_id: input.entity_id[index],
                plan_revision: input.plan_revision[index],
                evaluator_id: input.evaluator_id[index],
                score: combined.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
                status: if saturated {
                    GpuResultStatus::Saturated
                } else {
                    GpuResultStatus::Ok
                },
                error_code: 0,
            })
            .map_err(|_| GpuBatchError::OutputCapacity {
                requested: input.len(),
                available: output.contract.max_rows,
            })?;
    }
    Ok(input.len())
}

/// Compares CPU and backend rows in canonical `(entity, evaluator, sequence)` order.
pub fn compare_parity(
    expected: &[GpuBatchResult],
    actual: &[GpuBatchResult],
) -> Result<(), GpuParityError> {
    if expected.len() != actual.len() {
        return Err(GpuParityError::Length {
            expected: expected.len(),
            actual: actual.len(),
        });
    }
    for (index, (left, right)) in expected.iter().zip(actual).enumerate() {
        if left != right {
            return Err(GpuParityError::Mismatch {
                index,
                expected: *left,
                actual: *right,
            });
        }
    }
    Ok(())
}

/// Error returned when CPU and backend golden results disagree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuParityError {
    /// The backends returned different row counts.
    Length {
        /// CPU row count.
        expected: usize,
        /// Backend row count.
        actual: usize,
    },
    /// A row differs in identity, score, or status.
    Mismatch {
        /// Zero-based row index.
        index: usize,
        /// CPU reference row.
        expected: GpuBatchResult,
        /// Backend row.
        actual: GpuBatchResult,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input<'a>(sequence: &'a [u32], consideration: &'a [i32]) -> GpuBatchInput<'a> {
        GpuBatchInput::new(
            sequence,
            &[7, 8],
            &[11, 11],
            &[3, 4],
            consideration,
            &[1 << 16, 2 << 16],
            &[0, 1 << 16],
        )
    }

    #[test]
    fn stable_input_and_output_keys_round_trip() {
        assert_eq!(GpuBatchInputKey::EntityId.wire_id(), 2);
        assert_eq!(GpuBatchInputKey::EntityId.name(), "entity_id");
        assert_eq!(
            GpuBatchInputKey::from_wire_id(7),
            Some(GpuBatchInputKey::BiasQ16)
        );
        assert_eq!(GpuBatchOutputKey::ScoreQ16.wire_id(), 5);
        assert_eq!(
            GpuBatchOutputKey::from_wire_id(6),
            Some(GpuBatchOutputKey::Status)
        );
        assert_eq!(GpuBatchOutputKey::from_wire_id(99), None);
    }

    #[test]
    fn quantization_is_deterministic_and_bounded() {
        assert_eq!(quantize_unit(0.0), Ok(0));
        assert_eq!(quantize_unit(0.5), Ok(32_768));
        assert_eq!(quantize_unit(1.0), Ok(65_536));
        assert_eq!(quantize_bias(-1.0), Ok(-65_536));
        assert_eq!(quantize_bias(1.0), Ok(65_536));
        assert_eq!(dequantize_score(65_536), 1.0);
        assert_eq!(
            quantize_unit(f32::NAN),
            Err(GpuQuantizationError::NonFinite)
        );
        assert_eq!(
            quantize_bias(-1.1),
            Err(GpuQuantizationError::BelowMinusOne)
        );
    }

    #[test]
    fn cpu_reference_is_fixed_point_and_bounded() {
        let contract = GpuBatchContract::new(2);
        let mut output = [GpuBatchResult {
            sequence: 0,
            entity_id: 0,
            plan_revision: 0,
            evaluator_id: 0,
            score: 0,
            status: GpuResultStatus::Fallback(GpuFallbackReason::Unsupported),
            error_code: GpuFallbackReason::Unsupported.code(),
        }; 2];
        let count =
            evaluate_cpu(contract, input(&[1, 2], &[3 << 16, 4 << 16]), &mut output).unwrap();
        assert_eq!(count, 2);
        assert_eq!(output[0].score, 3 << 16);
        assert_eq!(output[1].score, 9 << 16);
        assert_eq!(output[0].status, GpuResultStatus::Ok);
    }

    #[test]
    fn malformed_columns_and_output_are_rejected() {
        let contract = GpuBatchContract::new(2);
        let mut output = [GpuBatchResult {
            sequence: 0,
            entity_id: 0,
            plan_revision: 0,
            evaluator_id: 0,
            score: 0,
            status: GpuResultStatus::Ok,
            error_code: 0,
        }; 1];
        assert!(matches!(
            evaluate_cpu(contract, input(&[1], &[1]), &mut output),
            Err(GpuBatchError::ColumnLength { .. })
        ));
        assert!(matches!(
            evaluate_cpu(contract, input(&[1, 2], &[1, 1]), &mut output),
            Err(GpuBatchError::OutputCapacity { .. })
        ));
    }

    #[test]
    fn strict_plan_validation_and_bounded_buffers_work() {
        let contract = GpuBatchContract::new(2);
        assert!(matches!(
            contract.validate_plan(input(&[1, 2], &[1, 1]), 12),
            Err(GpuBatchValidationError::PlanRevisionMismatch { .. })
        ));

        let mut input_buffer =
            GpuBatchInputBuffer::with_contract(GpuBatchContract::new(1)).unwrap();
        let capacity = input_buffer.capacity();
        input_buffer
            .push(GpuBatchInputRow {
                sequence: 0,
                entity_id: 1,
                plan_revision: 11,
                evaluator_id: 1,
                consideration: 65_536,
                weight: 65_536,
                bias: 0,
            })
            .unwrap();
        assert_eq!(input_buffer.capacity(), capacity);
        assert!(matches!(
            input_buffer.push(GpuBatchInputRow {
                sequence: 1,
                entity_id: 2,
                plan_revision: 11,
                evaluator_id: 1,
                consideration: 0,
                weight: 0,
                bias: 0,
            }),
            Err(GpuBatchBufferError::CapacityExceeded { maximum: 1 })
        ));

        let mut output = GpuBatchOutputBuffer::with_contract(GpuBatchContract::new(1)).unwrap();
        let output_capacity = output.capacity();
        evaluate_cpu_buffer(contract, input_buffer.as_input(), &mut output).unwrap();
        assert_eq!(output.capacity(), output_capacity);
        assert_eq!(output.rows()[0].error_code, 0);
    }

    #[test]
    fn parity_reports_first_mismatch() {
        let row = GpuBatchResult {
            sequence: 1,
            entity_id: 7,
            plan_revision: 11,
            evaluator_id: 3,
            score: 1,
            status: GpuResultStatus::Ok,
            error_code: 0,
        };
        let mut changed = row;
        changed.score = 2;
        assert!(matches!(
            compare_parity(&[row], &[changed]),
            Err(GpuParityError::Mismatch { index: 0, .. })
        ));
    }
}
