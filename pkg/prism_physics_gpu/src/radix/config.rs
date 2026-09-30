//! Compile-time tunables shared by the `GPU` radix sort host and its `WGSL`
//! kernels.
//!
//! A least-significant-digit (`LSD`) radix sort orders 32-bit keys by
//! processing [`RADIX_BITS`] bits at a time, from the low digit up, so it
//! completes in [`PASSES`] stable counting-sort passes. Each pass tiles the
//! keys into [`TILE`]-wide blocks, one workgroup per block, so the block width
//! and the digit width must agree between the host dispatch maths and the
//! kernels and therefore live here as a single source of truth.
//!
//! # Digit geometry
//!
//! With an 8-bit digit there are [`RADIX`] `= 256` buckets and a 32-bit key is
//! consumed in `32 / 8 =` [`PASSES`] `= 4` passes. The workgroup width equals
//! the bucket count so each lane can clear and later export exactly one shared
//! histogram bin.
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

/// Passes needed to consume a 32-bit key, `32 / RADIX_BITS`.
pub const PASSES: u32 = u32::BITS / RADIX_BITS;

/// Keys handled by one workgroup, one per lane; also the workgroup width. Must
/// match `TILE` and `@workgroup_size` in the radix shaders, and equals [`RADIX`]
/// so each lane owns one histogram bin.
pub const TILE: u32 = RADIX;
