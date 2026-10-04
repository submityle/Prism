//! # `prism_platform_os`
//!
//! Real-OS probe **twin** for the portable data models in [`prism_platform`]
//! (design doc §24.3).
//!
//! The core [`prism_platform`] crate is `#![forbid(unsafe_code)]` + `no_std`
//! and deliberately *probes nothing it cannot prove*: its
//! [`prism_platform::topology::CpuTopology::detect`] reports the logical-core
//! count but leaves [`is_probed`](prism_platform::topology::CpuTopology::is_probed)
//! `false`, because a genuine hybrid-core / `SMT` / `NUMA` / cache read needs
//! per-OS syscalls that do not belong in a portable, unsafe-free kernel.
//!
//! This sibling crate is where that real read lives. It is `std`, links the
//! host C library, and performs the actual OS queries, then hands the result
//! back through [`prism_platform`]'s validated [`TopologyBuilder`] so the rest
//! of the engine consumes one topology type regardless of how it was obtained.
//! The split mirrors the GPU twins elsewhere in the workspace (a pure-`core`
//! model plus a device/OS sibling that fills it in and is validated against the
//! real hardware).
//!
//! ## What is verified
//!
//! [`probe_topology`] is implemented and tested on **Apple Silicon macOS**,
//! where the result is an exact, `SMT`-free `P`/`E` + shared-`L2` description.
//! On other targets it returns [`ProbeError::Unsupported`] rather than
//! fabricating a layout it cannot confirm; callers fall back to
//! [`prism_platform::topology::CpuTopology::detect`].
//!
//! [`TopologyBuilder`]: prism_platform::topology::TopologyBuilder

extern crate alloc;

pub mod sysctl;
pub mod topology;

pub use topology::{probe_topology, ProbeError};

#[cfg(test)]
mod tests_topology;
