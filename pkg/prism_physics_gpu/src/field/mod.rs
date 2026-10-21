//! Batched analytic field sampling using the core M11 public field types.
//!
//! [`cpu_sample_fields`] is an independent golden twin of the compute kernel,
//! checked against [`FieldRegistry::sample`]. [`GpuFieldSampler`] uploads the
//! registry in its existing order and evaluates one body per GPU invocation.
//! It never calls the CPU twin or core sampler. Force and torque are returned
//! in body order; analytic point samples currently produce zero torque in core.
//!
//! All shapes, effects, masks, and falloffs use core's semantics, including
//! global fields ignoring falloff, inclusive volume boundaries, clamped drag
//! mass, and normalized-distance inverse-square attenuation.

mod cpu;
mod gpu;
mod packed;
mod runtime;
mod workspace;

pub use cpu::cpu_sample_fields;
pub use gpu::{FieldBufferStats, FieldGpuError, GpuFieldSampler};
pub use prism_physics_core::dynamics::field::{
    Field, FieldBody, FieldEffect, FieldFalloff, FieldId, FieldRegistry, FieldSample, FieldShape,
    FieldTargetMask,
};
pub use runtime::GpuFieldRuntime;

#[cfg(test)]
mod tests;
