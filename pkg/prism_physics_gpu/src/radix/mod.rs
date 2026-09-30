//! `GPU` stable least-significant-digit (`LSD`) radix sort of 32-bit keys.
//!
//! Sorting is the backbone of spatial data structures: building a
//! broad-phase grid, ordering particles by cell for coherent gather, and
//! compacting sparse structures all reduce to sorting keys (often with a
//! payload index) and reading them back in order. This module sorts `u32` keys
//! — and an optional `u32` payload — ascending and stably in [`PASSES`] passes
//! over [`RADIX_BITS`]-bit digits.
//!
//! Each pass tiles the keys into [`TILE`]-wide blocks and runs three device
//! steps: a per-block digit histogram, an exclusive scan of that histogram (via
//! the sibling [`GpuScan`](crate::scan::GpuScan)) into output offsets, and a
//! stable scatter to those offsets. Every step is mirrored by a `CPU` golden
//! twin ([`cpu_radix_sort_keys`], [`cpu_radix_sort_pairs`]); because sorting
//! integers is a pure permutation, a passing real-device parity test is
//! bit-for-bit evidence of a faithful port.
//!
//! # Provenance
//!
//! The `LSD` radix sort with a per-block, digit-major histogram scanned to
//! output offsets is a classical, openly published `GPU` technique (Blelloch
//! 1990; Satish, Harris, Garland, "Designing Efficient Sorting Algorithms for
//! Manycore GPUs", 2009). This module contains no Unreal Engine source or
//! derived code.

pub mod config;
pub mod cpu;
pub mod gpu;
pub mod layout;

pub use config::{PASSES, RADIX, RADIX_BITS, TILE};
pub use cpu::{cpu_radix_sort_keys, cpu_radix_sort_pairs};
pub use gpu::GpuRadixSort;
