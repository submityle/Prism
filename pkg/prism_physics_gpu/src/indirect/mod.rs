//! `GPU`-driven indirect dispatch: letting the device size its own launches.
//!
//! A classic compute pipeline reads every stage's work-item count back to the
//! host to pick a dispatch size, stalling on a round-trip between stages. A
//! `GPU`-driven pipeline keeps the counts resident and derives each launch from
//! them on device, so a whole chain — broad phase, constraint solve, integration
//! — submits once and runs without the host learning any intermediate size.
//! This module is the primitive that makes that possible: it turns a device-side
//! count into the `[x, y, z]` argument block that `dispatch_workgroups_indirect`
//! consumes.
//!
//! # Layout
//!
//! - [`args`] — the [`DispatchArgs`] block and its [`cpu_dispatch_args`] golden
//!   twin, including the `y`-tiling that keeps a launch under the device's
//!   per-dimension cap.
//! - [`gpu`] — the real-device [`GpuDispatchArgs`] pipeline that fills the block
//!   from a device-side count, either fused into a caller's encoder or standalone
//!   for parity testing.
//!
//! # Correctness model
//!
//! The kernel is integer arithmetic — a ceil-division and a cap comparison — so
//! its result is exact, and the parity test asserts the device block equals the
//! [`cpu_dispatch_args`] twin bit-for-bit. A companion end-to-end test builds a
//! block from a device-side count and drives a real `dispatch_workgroups_indirect`
//! from it, proving the indirect launch covers exactly the intended work with no
//! host readback of the count.
//!
//! # Provenance
//!
//! Indirect compute dispatch and ceil-division grid sizing are standard,
//! openly published `GPU`-driven techniques. No Unreal Engine source or derived
//! code.

pub mod args;
pub mod gpu;

pub use args::{cpu_dispatch_args, DispatchArgs, DEFAULT_MAX_WORKGROUPS_PER_DIM};
pub use gpu::GpuDispatchArgs;
