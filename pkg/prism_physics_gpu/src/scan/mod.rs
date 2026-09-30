//! `GPU` work-efficient exclusive scan and flag-driven stream compaction.
//!
//! An exclusive prefix sum turns a stream of per-element counts into the
//! offsets at which each element's output begins, the workhorse behind
//! allocation-free parallel output: histogram-to-bucket layout, radix-sort
//! digit placement, and active-cell list construction all reduce to a scan.
//! Stream compaction — gathering the elements that pass a predicate into a
//! dense array — is the same scan followed by a scatter to the scanned offsets.
//!
//! [`GpuScan`] compiles the kernels once and runs both primitives over an
//! arbitrarily long `u32` stream by walking a pyramid of blocks; each stage is
//! mirrored by a `CPU` golden twin ([`cpu_exclusive_scan`], [`cpu_compact`])
//! that folds with the same wrapping `u32` addition, so a passing real-device
//! parity test is bit-for-bit evidence of a faithful port.
//!
//! # Provenance
//!
//! The Blelloch work-efficient scan and flag-driven stream compaction are
//! classical, openly published parallel primitives (Blelloch 1990; Harris,
//! Sengupta, Owens, GPU Gems 3, 2007). This module contains no Unreal Engine
//! source or derived code.

pub mod config;
pub mod cpu;
pub mod gpu;
pub mod layout;

pub use config::{BLOCK, WORKGROUP};
pub use cpu::{cpu_compact, cpu_exclusive_scan};
pub use gpu::GpuScan;
