//! Compile-time tunables shared by the `GPU` 64-bit radix sort host and its
//! `WGSL` kernels.
//!
//! A least-significant-digit (`LSD`) radix sort orders keys by processing
//! [`RADIX_BITS`] bits at a time, from the low digit up, so a 64-bit
//! [`CoherenceKey`](prism_render_architecture::ray_scene::reorder::CoherenceKey)
//! completes in [`PASSES`] stable counting-sort passes. Each pass tiles the
//! keys into [`TILE`]-wide blocks, one workgroup per block, so the block width
//! and the digit width must agree between the host dispatch maths and the
//! kernels and therefore live here as a single source of truth.
//!
//! # Digit geometry
//!
//! With an 8-bit digit there are [`RADIX`] `= 256` buckets, and a 64-bit key is
//! consumed in `64 / 8 =` [`PASSES`] `= 8` passes. `WGSL` has no native 64-bit
//! integer, so the key is carried as two `u32` words (`lo`, `hi`); passes `0..4`
//! read digits from `lo` and passes `4..8` read digits from `hi`, which is
//! identical in ordering to one logical 64-bit LSD sort. The workgroup width
//! equals the bucket count so each lane can clear and later export exactly one
//! shared histogram bin.
//!
//! # Provenance
//!
//! The `LSD` radix sort and its per-block histogram decomposition are classical,
//! openly published `GPU` techniques (Blelloch 1990; Satish, Harris, Garland,
//! "Designing Efficient Sorting Algorithms for Manycore GPUs", 2009). This
//! module contains no Unreal Engine source or derived code.

/// Bits consumed per sorting pass.
pub const RADIX_BITS: u32 = 8;

/// Buckets per pass, `2^RADIX_BITS`. Must match `RADIX` in the radix shaders.
pub const RADIX: u32 = 1 << RADIX_BITS;

/// Bit mask selecting one digit, `RADIX - 1`.
pub const RADIX_MASK: u32 = RADIX - 1;

/// Passes needed to consume a 64-bit key, `64 / RADIX_BITS`.
pub const PASSES: u32 = u64::BITS / RADIX_BITS;

/// Passes whose digit is read from the low 32-bit word (the remaining passes
/// read from the high word).
pub const LOW_WORD_PASSES: u32 = u32::BITS / RADIX_BITS;

/// Keys handled by one workgroup, one per lane; also the workgroup width. Must
/// match `TILE` and `@workgroup_size` in the radix shaders, and equals [`RADIX`]
/// so each lane owns one histogram bin.
pub const TILE: u32 = RADIX;
