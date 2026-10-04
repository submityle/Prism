//! Optional `wgpu` compute twin for Prism's §24.7 GPU-side hierarchy
//! propagation.
//!
//! The portable, device-free half of the contract lives in
//! [`prism_transform::compute_hierarchy`]: the per-level dispatch plan
//! [`LevelSchedule`](prism_transform::compute_hierarchy::LevelSchedule), the
//! device-free upload payload
//! [`ComputeHierarchyInput`](prism_transform::compute_hierarchy::ComputeHierarchyInput),
//! and the CPU reference / parity oracle
//! [`propagate_by_levels`](prism_transform::compute_hierarchy::propagate_by_levels).
//! This crate is the *real-device* half: it uploads the parent-index topology,
//! the row-major local matrices, and the flattened dispatch order produced by
//! `ComputeHierarchyInput::pack`, then dispatches one compute pass per depth
//! level (shallow to deep) so every child reads an already-finalized parent
//! world matrix, and reads the world matrices back. The parity tests diff the
//! GPU readback against `propagate_by_levels`.
//!
//! # Why a tolerance, not bit-exactness
//!
//! The kernel composes affine transforms (3x3 matrix products plus a
//! translation), which is floating-point multiply-accumulate. Metal compiles
//! WGSL under fast-math, so the shader compiler may contract `a + b * c` into a
//! single `fma` and reassociate, rounding differently from the CPU's
//! separately-rounded operations. The §24.7 twin is therefore validated as a
//! *tolerance* round-trip: the two sides must agree within a small
//! absolute+relative epsilon, which rejects a genuine
//! algorithm/operand-order/layout drift while tolerating last-ULP FMA rounding.
//!
//! # Single-sourced layout
//!
//! The kernel decodes exactly the row-major 3x4 affine
//! [`MatrixLayout::RowMajor3x4`](prism_transform::gpu_upload::MatrixLayout) that
//! `ComputeHierarchyInput::pack` emits and that the CPU output/instancing path
//! in `prism_transform::gpu_upload` writes, so the device and host share one
//! matrix encoding and cannot silently disagree on byte layout.
//!
//! # Graceful skip
//!
//! [`GpuContext::try_headless`] returns [`None`] on a host with no usable
//! adapter so the parity suite skips rather than fails on a device-less CI
//! image, while running the full dispatch on any real `GPU`.
//!
//! Provenance: standard `wgpu` compute orchestration; the affine composition is
//! the classic column-major `world = parent ∘ local`. No neural, learned, or
//! data-driven components. No Unreal Engine or Unity source or derived code.

extern crate alloc;

pub mod buffer;
pub mod context;
pub mod hierarchy;

pub use context::{GpuContext, block_on};
pub use hierarchy::GpuHierarchyPropagate;
