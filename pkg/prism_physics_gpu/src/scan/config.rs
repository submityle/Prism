//! Compile-time tunables shared by the `GPU` scan host and its `WGSL` kernels.
//!
//! A work-efficient scan is organised around a fixed *block*: each workgroup
//! exclusively scans one block of elements in shared memory, emits the block's
//! total, and a second stage scans those totals and folds them back. The block
//! width and the workgroup width must agree between the host dispatch maths and
//! the kernel, so both live here as a single source of truth.
//!
//! # Block geometry
//!
//! The Blelloch work-efficient scan processes two elements per thread, so a
//! block spans [`WORKGROUP`] `* 2 =` [`BLOCK`] elements. A dispatch over `n`
//! elements therefore launches `ceil(n / BLOCK)` workgroups, and the per-block
//! totals form the next, `BLOCK`-times smaller level of the scan pyramid.
//!
//! # Provenance
//!
//! The block/level decomposition of a parallel prefix sum is a classical,
//! openly published `GPU` technique (Blelloch 1990; Harris, Sengupta, Owens,
//! GPU Gems 3, 2007). This module contains no Unreal Engine source or derived
//! code.

/// Threads per workgroup; must match `@workgroup_size` in `shaders/scan.wgsl`.
pub const WORKGROUP: u32 = 256;

/// Elements scanned by one workgroup (two per thread), the width of one level
/// of the scan pyramid. Must match the `BLOCK` constant in `shaders/scan.wgsl`.
pub const BLOCK: u32 = WORKGROUP * 2;
