//! Device-free `std430` byte-layout contract for the particle pipeline's two
//! *draw-preparation* `GPU` passes at the tail of §9: `FillDrawArgs` and
//! `RenderDraw`.
//!
//! [`super::gpu_layout`] owns the shared `std430` stride primitives
//! ([`U32_STRIDE`], [`VEC2_STRIDE`], [`VEC4_STRIDE`]), the
//! [`ParticleBufferAccess`] read/read-write mode, and the clamp-to-one
//! [`storage_bytes`] rule. This module reuses all of them and only adds the
//! two draw-stage bind-group descriptions, mirroring how `hair/` publishes one
//! `*_pass_buffers.rs` per compute pass: each enum names the storage buffers a
//! `WESL` kernel binds at `@group(0)`, in `@binding` order, and reports the
//! element stride, access mode, element count and total byte size so the render
//! graph binds against a stable `ABI` instead of hand-computing strides next to
//! the pipeline.
//!
//! The two passes close the §9 pipeline once culling and sorting have produced
//! a compact list of survivors:
//! - `FillDrawArgs` is a tiny compute pass (a single workgroup) that reads the
//!   post-cull/sort live particle count and the sorted index list and writes the
//!   indirect draw-args buffer the rasterizer is dispatched from. The draw-args
//!   record follows `WebGPU`'s `DrawIndexedIndirectArgs` layout (five `u32`s,
//!   see [`DRAW_INDEXED_INDIRECT_STRIDE`]); [`DRAW_INDIRECT_STRIDE`] models the
//!   non-indexed four-`u32` variant for reference.
//! - `RenderDraw` is the raster draw itself: the vertex/instance stages read a
//!   per-instance packed particle descriptor buffer (see [`INSTANCE_STRIDE`]),
//!   the same sorted index list, and the draw-args buffer as an indirect source.
//!
//! `RenderDraw`'s [`RenderDrawBuffer::DrawArgs`] *aliases* the very buffer
//! [`FillDrawArgsBuffer::DrawArgs`] wrote — it is bound `Read` as the indirect
//! source and allocates no fresh storage, exactly as `hair`'s `Wind` and
//! `SdfCollision` passes alias the persistent positions buffer rather than
//! reallocating it. Likewise `RenderDraw` and `FillDrawArgs` share the one
//! sorted-index buffer produced by the sort pass.
//!
//! The pool `capacity` and the runtime `live_count` are not dispatch domains, so
//! they travel together in [`ParticleDrawExtent`], mirroring how `hair`'s
//! per-pass files carry their non-domain counts in a dedicated extent struct.
//!
//! Everything is pure integer arithmetic: byte sizes clamp up to one element so
//! an empty pool still yields a valid non-empty `WebGPU` storage binding, and
//! the multiplication saturates rather than wrapping.

use crate::particle::gpu_layout::{
    storage_bytes, ParticleBufferAccess, U32_STRIDE, VEC2_STRIDE, VEC4_STRIDE,
};

/// `std430` byte stride of a `WebGPU` `DrawIndirectArgs` record: four `u32`s —
/// `vertex_count`, `instance_count`, `first_vertex`, `first_instance` — so
/// `4 * 4 = 16` bytes. Modelled for reference; the particle path draws indexed
/// quads and uses [`DRAW_INDEXED_INDIRECT_STRIDE`].
const DRAW_INDIRECT_STRIDE: usize = 4 * U32_STRIDE;

/// `std430` byte stride of a `WebGPU` `DrawIndexedIndirectArgs` record:
/// `index_count`, `instance_count`, `first_index`, `base_vertex`,
/// `first_instance` — the four `DrawIndirectArgs` fields plus one extra
/// `base_vertex` `u32`, so `16 + 4 = 20` bytes. This is the layout
/// [`FillDrawArgsBuffer::DrawArgs`] writes for the indexed quad draw.
const DRAW_INDEXED_INDIRECT_STRIDE: usize = DRAW_INDIRECT_STRIDE + U32_STRIDE;

/// `std430` array stride of a per-instance particle descriptor read by the
/// `RenderDraw` vertex/instance stages: `world_position: vec4<f32>` (16) +
/// `color: vec4<f32>` (16) + `uv_frame: vec4<f32>` (16, packed `uv` rect
/// `min.xy` / `max.xy` for the current flipbook cell) + `size: vec2<f32>` (8) +
/// `rotation: vec2<f32>` (8, `cos`/`sin`) = 64 bytes. All members respect their
/// `std430` alignment and the total is already a multiple of the 16-byte `vec4`
/// alignment, so the array stride is exactly 64.
const INSTANCE_STRIDE: usize = 3 * VEC4_STRIDE + 2 * VEC2_STRIDE;

/// The non-domain extents the two draw-preparation passes size their buffers
/// against.
///
/// `capacity` is a genuine buffer-sizing domain (the sorted-index and instance
/// arrays are allocated at the pool's worst case). `live_count` is the runtime
/// survivor scalar `FillDrawArgs` reads from the `LiveCount` buffer and writes
/// into the indirect args' `instance_count`; it is not itself a buffer-sizing
/// domain but travels here so the passes' full non-dispatch input set is
/// described in one place.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ParticleDrawExtent {
    /// Particle pool capacity: the worst-case element count of the sorted-index
    /// list and the per-instance descriptor buffer.
    pub capacity: u32,
    /// Post-cull/sort surviving particle count that becomes the indirect draw's
    /// `instance_count`. Not a buffer-sizing domain (see the struct docs).
    pub live_count: u32,
}

/// One storage buffer bound by the fill-draw-args kernel
/// (`particle_fill_draw_args.wesl` `@group(0)`), in binding order `0..3`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FillDrawArgsBuffer {
    /// `@binding(0)` post-cull/sort survivor count `array<u32>` (a single-`u32`
    /// counter written by the cull/compaction pass), read to fill the draw's
    /// `instance_count`.
    LiveCount,
    /// `@binding(1)` sorted survivor indices `array<u32>`, read to know how many
    /// instances the args must cover; shared with `RenderDraw`.
    SortedIndices,
    /// `@binding(2)` indirect draw-args record (`DrawIndexedIndirectArgs`),
    /// written by this pass and later aliased read-only by `RenderDraw`.
    DrawArgs,
}

impl FillDrawArgsBuffer {
    /// Every fill-draw-args buffer in `@binding` order.
    pub const ALL: [FillDrawArgsBuffer; 3] = [Self::LiveCount, Self::SortedIndices, Self::DrawArgs];

    /// The `@group(0)` binding index in `particle_fill_draw_args.wesl`.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::LiveCount => 0,
            Self::SortedIndices => 1,
            Self::DrawArgs => 2,
        }
    }

    /// Byte stride of one element, matching the `WESL` scalar / record layout.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::LiveCount | Self::SortedIndices => U32_STRIDE,
            Self::DrawArgs => DRAW_INDEXED_INDIRECT_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer.
    #[must_use]
    pub fn access(self) -> ParticleBufferAccess {
        match self {
            Self::LiveCount | Self::SortedIndices => ParticleBufferAccess::Read,
            Self::DrawArgs => ParticleBufferAccess::ReadWrite,
        }
    }

    /// Whether this pass writes the buffer rather than only reading it.
    #[must_use]
    pub fn is_output(self) -> bool {
        self.access().is_writable()
    }

    /// Element count for a pool with the given `extent`: the survivor count and
    /// the draw-args record are single elements, while the sorted-index list
    /// spans the whole pool `capacity`.
    #[must_use]
    pub fn element_count(self, extent: &ParticleDrawExtent) -> u32 {
        match self {
            Self::LiveCount | Self::DrawArgs => 1,
            Self::SortedIndices => extent.capacity,
        }
    }

    /// Total byte size, clamped up to one element so an empty pool still yields
    /// a valid non-empty `WebGPU` storage binding.
    #[must_use]
    pub fn byte_size(self, extent: &ParticleDrawExtent) -> usize {
        storage_bytes(self.stride(), self.element_count(extent) as usize)
    }
}

/// One storage buffer bound by the raster draw (`particle_render_draw.wesl`
/// `@group(0)`), in binding order `0..3`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderDrawBuffer {
    /// `@binding(0)` per-instance packed particle descriptors
    /// `array<ParticleInstance>` (position/color/`uv`-frame/size/rotation), read
    /// by the vertex/instance stages. See [`INSTANCE_STRIDE`].
    Instances,
    /// `@binding(1)` sorted survivor indices `array<u32>`, read to fetch the
    /// draw order; the same buffer `FillDrawArgs` reads.
    SortedIndices,
    /// `@binding(2)` indirect draw-args record, bound `Read` as the indirect
    /// source. This *aliases* [`FillDrawArgsBuffer::DrawArgs`] — it is never
    /// reallocated, only consumed.
    DrawArgs,
}

impl RenderDrawBuffer {
    /// Every render-draw buffer in `@binding` order.
    pub const ALL: [RenderDrawBuffer; 3] = [Self::Instances, Self::SortedIndices, Self::DrawArgs];

    /// The `@group(0)` binding index in `particle_render_draw.wesl`.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::Instances => 0,
            Self::SortedIndices => 1,
            Self::DrawArgs => 2,
        }
    }

    /// Byte stride of one element, matching the `WESL` struct / scalar / record
    /// layout.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::Instances => INSTANCE_STRIDE,
            Self::SortedIndices => U32_STRIDE,
            Self::DrawArgs => DRAW_INDEXED_INDIRECT_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer. Every buffer this
    /// raster pass binds is read-only (`DrawArgs` is the indirect source it
    /// aliases from `FillDrawArgs`).
    #[must_use]
    pub fn access(self) -> ParticleBufferAccess {
        match self {
            Self::Instances | Self::SortedIndices | Self::DrawArgs => ParticleBufferAccess::Read,
        }
    }

    /// Whether this pass writes the buffer rather than only reading it. Always
    /// `false`: `RenderDraw` only consumes storage buffers.
    #[must_use]
    pub fn is_output(self) -> bool {
        self.access().is_writable()
    }

    /// Element count for a pool with the given `extent`: the instance and
    /// sorted-index arrays span the whole pool `capacity`, and the draw-args
    /// record is a single element.
    #[must_use]
    pub fn element_count(self, extent: &ParticleDrawExtent) -> u32 {
        match self {
            Self::Instances | Self::SortedIndices => extent.capacity,
            Self::DrawArgs => 1,
        }
    }

    /// Total byte size, clamped up to one element so an empty pool still yields
    /// a valid non-empty `WebGPU` storage binding.
    #[must_use]
    pub fn byte_size(self, extent: &ParticleDrawExtent) -> usize {
        storage_bytes(self.stride(), self.element_count(extent) as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pool with a distinct capacity and a smaller survivor count.
    fn sample_extent() -> ParticleDrawExtent {
        ParticleDrawExtent {
            capacity: 4096,
            live_count: 1500,
        }
    }

    #[test]
    fn draw_args_record_strides_follow_webgpu_layout() {
        assert_eq!(DRAW_INDEXED_INDIRECT_STRIDE, 20);
        assert_eq!(DRAW_INDIRECT_STRIDE, 16);
        assert_eq!(INSTANCE_STRIDE, 64);
    }

    #[test]
    fn all_lengths_match_the_binding_counts() {
        assert_eq!(FillDrawArgsBuffer::ALL.len(), 3);
        assert_eq!(RenderDrawBuffer::ALL.len(), 3);
    }

    #[test]
    fn fill_draw_args_bindings_are_unique_contiguous_from_zero() {
        for (index, buffer) in FillDrawArgsBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
    }

    #[test]
    fn render_draw_bindings_are_unique_contiguous_from_zero() {
        for (index, buffer) in RenderDrawBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
    }

    #[test]
    fn strides_match_the_wesl_layout() {
        assert_eq!(FillDrawArgsBuffer::LiveCount.stride(), 4);
        assert_eq!(FillDrawArgsBuffer::SortedIndices.stride(), 4);
        assert_eq!(FillDrawArgsBuffer::DrawArgs.stride(), 20);
        assert_eq!(RenderDrawBuffer::Instances.stride(), 64);
        assert_eq!(RenderDrawBuffer::SortedIndices.stride(), 4);
        assert_eq!(RenderDrawBuffer::DrawArgs.stride(), 20);
    }

    #[test]
    fn access_modes_match_the_kernels() {
        assert_eq!(
            FillDrawArgsBuffer::LiveCount.access(),
            ParticleBufferAccess::Read
        );
        assert_eq!(
            FillDrawArgsBuffer::SortedIndices.access(),
            ParticleBufferAccess::Read
        );
        assert_eq!(
            FillDrawArgsBuffer::DrawArgs.access(),
            ParticleBufferAccess::ReadWrite
        );
        for buffer in RenderDrawBuffer::ALL {
            assert_eq!(buffer.access(), ParticleBufferAccess::Read);
        }
    }

    #[test]
    fn only_fill_draw_args_output_is_the_draw_args_buffer() {
        assert!(!FillDrawArgsBuffer::LiveCount.is_output());
        assert!(!FillDrawArgsBuffer::SortedIndices.is_output());
        assert!(FillDrawArgsBuffer::DrawArgs.is_output());
        for buffer in RenderDrawBuffer::ALL {
            assert!(!buffer.is_output());
        }
    }

    #[test]
    fn element_counts_follow_the_extent() {
        let extent = sample_extent();
        assert_eq!(FillDrawArgsBuffer::LiveCount.element_count(&extent), 1);
        assert_eq!(
            FillDrawArgsBuffer::SortedIndices.element_count(&extent),
            4096
        );
        assert_eq!(FillDrawArgsBuffer::DrawArgs.element_count(&extent), 1);
        assert_eq!(RenderDrawBuffer::Instances.element_count(&extent), 4096);
        assert_eq!(RenderDrawBuffer::SortedIndices.element_count(&extent), 4096);
        assert_eq!(RenderDrawBuffer::DrawArgs.element_count(&extent), 1);
    }

    #[test]
    fn empty_pool_clamps_every_buffer_to_one_element() {
        let extent = ParticleDrawExtent::default();
        for buffer in FillDrawArgsBuffer::ALL {
            assert_eq!(buffer.byte_size(&extent), buffer.stride());
        }
        for buffer in RenderDrawBuffer::ALL {
            assert_eq!(buffer.byte_size(&extent), buffer.stride());
        }
    }

    #[test]
    fn byte_size_matches_storage_bytes_of_stride_and_count() {
        let extent = sample_extent();
        for buffer in FillDrawArgsBuffer::ALL {
            assert_eq!(
                buffer.byte_size(&extent),
                storage_bytes(buffer.stride(), buffer.element_count(&extent) as usize)
            );
        }
        for buffer in RenderDrawBuffer::ALL {
            assert_eq!(
                buffer.byte_size(&extent),
                storage_bytes(buffer.stride(), buffer.element_count(&extent) as usize)
            );
        }
    }

    #[test]
    fn byte_size_scales_with_capacity() {
        let small = ParticleDrawExtent {
            capacity: 1000,
            live_count: 0,
        };
        let large = ParticleDrawExtent {
            capacity: 2000,
            live_count: 0,
        };
        assert_eq!(RenderDrawBuffer::Instances.byte_size(&small), 1000 * 64);
        assert_eq!(
            RenderDrawBuffer::Instances.byte_size(&large),
            2 * RenderDrawBuffer::Instances.byte_size(&small)
        );
    }

    #[test]
    fn byte_size_saturates_instead_of_overflowing() {
        let extent = ParticleDrawExtent {
            capacity: u32::MAX,
            live_count: u32::MAX,
        };
        // Must equal the saturating primitive, never a wrapped small value.
        assert_eq!(
            RenderDrawBuffer::Instances.byte_size(&extent),
            storage_bytes(INSTANCE_STRIDE, u32::MAX as usize)
        );
    }
}
