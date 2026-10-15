//! Real-device `wgpu` compute twin for the `StateTree` GPU batch contract.
//!
//! The device kernel consumes the fixed-width rows described by
//! [`prism_state_tree_ecs`], performs the Q16 operation
//! `((consideration * weight) >> 16) + bias`, and writes a fixed-width result
//! row. The ECS crate remains the CPU golden authority; [`GpuEvaluator::parity`]
//! compares the complete device readback with that reference.
//!
//! Device acquisition is best-effort. [`GpuContext::try_headless`] returns
//! `None` when no native adapter is available, which lets CI run the
//! device-neutral unit tests on hosts without a GPU while a Metal/Vulkan/DX12
//! host executes the same integration test against a real device.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

extern crate alloc;

mod buffer;
pub mod context;
mod evaluator;

pub use context::GpuContext;
pub use evaluator::{
    GpuBatchInputRow, GpuBatchOutputRow, GpuEvaluateError, GpuEvaluation, GpuEvaluator,
    GpuOutputError, GpuParityReport, GpuValidationRecord,
};

/// The fixed number of 32-bit words in one device input row.
pub const INPUT_ROW_WORDS: usize = 10;

/// The fixed number of 32-bit words in one device output row.
pub const OUTPUT_ROW_WORDS: usize = 12;

/// The workgroup width used by the score kernel.
pub const WORKGROUP_SIZE: u32 = 64;

/// The WGSL source used by [`GpuEvaluator`].
pub const SCORE_SHADER: &str = include_str!("shaders/state_tree_score.wgsl");
