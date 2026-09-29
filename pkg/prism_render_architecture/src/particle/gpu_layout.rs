//! Device-free `std430` byte-layout primitives shared by the particle
//! per-pass `GPU` buffer contracts (design §5, §9).
//!
//! `hair/` publishes one `*_buffers.rs` contract per compute pass (see
//! [`crate::hair::gpu_buffers`] and its siblings): each names the storage
//! buffers a `WESL` kernel binds at `@group(0)`, their element stride, access
//! mode, element count and total byte size, so the render graph binds against a
//! stable `ABI` instead of hand-computing strides next to the pipeline. The
//! particle subsystem's §9 pipeline (`EmitterUpdate`, `Spawn`,
//! `SimulationStages`, `Event Scatter`, `Compaction`, `Bounds`, `Cull`, `Sort`,
//! `Fill Draw Args`, `Render Draw`) needs the same treatment; this module owns
//! the small pieces every per-pass file reuses so the access enum and stride
//! arithmetic are defined exactly once.
//!
//! [`super::attributes`] already owns the *Structure-of-Arrays attribute pool*
//! layout (which per-particle channels exist and how wide each is). This module
//! is orthogonal: it supplies the `std430` stride constants and the
//! clamp-to-one byte-size rule the *per-pass bind-group* contracts share, and it
//! never re-derives attribute widths.
//!
//! Everything is pure integer arithmetic: an empty pool still yields a valid,
//! non-empty `WebGPU` storage binding (one element), and nothing panics or
//! divides by zero.

/// Byte stride of a scalar `u32` / `f32` `std430` storage element.
pub const U32_STRIDE: usize = 4;

/// Byte stride of a `vec2<f32>` / `vec2<u32>` `std430` storage element.
pub const VEC2_STRIDE: usize = 8;

/// Byte stride of a `vec4<f32>` / `vec4<u32>` `std430` storage element (also the
/// natural stride an aligned `vec3` is padded up to).
pub const VEC4_STRIDE: usize = 16;

/// How a storage buffer is accessed by the compute kernel that binds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParticleBufferAccess {
    /// `var<storage, read>` — read-only input for this pass.
    Read,
    /// `var<storage, read_write>` — mutated in place by this pass.
    ReadWrite,
}

impl ParticleBufferAccess {
    /// Whether the pass may write to a buffer with this access.
    #[must_use]
    pub fn is_writable(self) -> bool {
        matches!(self, Self::ReadWrite)
    }
}

/// Total byte size of a storage buffer holding `count` elements of `stride`
/// bytes each, clamped up to a single element.
///
/// A `WebGPU` storage binding may not be zero-sized, so an empty pool still
/// reserves one element. The multiplication saturates rather than overflowing,
/// so a degenerate `count` can never wrap to a small allocation.
#[must_use]
pub fn storage_bytes(stride: usize, count: usize) -> usize {
    stride.saturating_mul(count.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_follow_std430() {
        assert_eq!(U32_STRIDE, 4);
        assert_eq!(VEC2_STRIDE, 8);
        assert_eq!(VEC4_STRIDE, 16);
    }

    #[test]
    fn access_writability() {
        assert!(ParticleBufferAccess::ReadWrite.is_writable());
        assert!(!ParticleBufferAccess::Read.is_writable());
    }

    #[test]
    fn empty_pool_reserves_one_element() {
        assert_eq!(storage_bytes(VEC4_STRIDE, 0), VEC4_STRIDE);
        assert_eq!(storage_bytes(U32_STRIDE, 0), U32_STRIDE);
    }

    #[test]
    fn byte_size_scales_with_count() {
        assert_eq!(storage_bytes(VEC4_STRIDE, 10), 160);
        assert_eq!(storage_bytes(U32_STRIDE, 256), 1024);
    }

    #[test]
    fn byte_size_saturates_instead_of_overflowing() {
        assert_eq!(storage_bytes(VEC4_STRIDE, usize::MAX), usize::MAX);
    }
}
