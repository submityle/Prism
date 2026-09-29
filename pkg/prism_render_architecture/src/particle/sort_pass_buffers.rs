//! Device-free `@group(0)` bind-group and `std430` byte-layout contract for the
//! particle §9 `GPU` pipeline's post-simulation passes: `Compaction`,
//! `Bounds`, `Cull` and `Sort` (design §11 prefix-sum, §13 cull/bounds, §12
//! sort/`OIT`).
//!
//! Like the `hair/` per-pass files (see
//! [`super::gpu_layout`]'s module doc), each pass here names the storage
//! buffers its `WESL` kernel binds, in `@binding` order, together with the
//! element stride, access mode, element count and total byte size — so the
//! render graph binds against a stable `ABI` instead of hand-computing strides
//! next to the pipeline. The layout mirrors production `GPU` particle stacks
//! (`Niagara`'s compute sort/cull and `Frostbite`'s `radix` depth sort) at the
//! buffer-contract level.
//!
//! This module is deliberately orthogonal to [`super::sort_cull`]: that module
//! owns the *decisions and math* — depth quantization, sort-key packing,
//! frustum/`HZB` cull tests and `AABB` reduction on the `CPU` reference. Here we
//! own only *where the bytes live* for the `GPU` build of those same passes: the
//! bind indices, strides and sizes. We never re-quantize a depth or re-run a
//! cull test, and we never import `sort_cull` (avoiding a cycle and keeping the
//! layout contract free of policy). Stride primitives and the clamp-to-one
//! byte-size rule are reused from [`super::gpu_layout`] rather than redefined.
//!
//! Everything is pure integer arithmetic: an empty pool still yields a valid,
//! non-empty `WebGPU` storage binding (one element), degenerate `radix` bucket
//! or workgroup counts clamp up to one, and products saturate rather than
//! overflowing, so nothing panics or wraps.

use super::gpu_layout::{
    storage_bytes, ParticleBufferAccess, U32_STRIDE, VEC2_STRIDE, VEC4_STRIDE,
};

/// `std430` size of the `Cull` parameter uniform: six frustum planes
/// (`vec4<f32>` each, 96 bytes) + a distance-range `vec4<f32>` (near, far,
/// max distance, pad) + an `HZB` `vec4<f32>` (mip count, width, height, pad).
const CULL_PARAMS_STRIDE: usize = 6 * VEC4_STRIDE + VEC4_STRIDE + VEC4_STRIDE;

/// The four post-simulation compute passes this module lays out (design §9).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SortCullPass {
    /// Prefix-sum stream compaction of the alive set (design §11).
    Compaction,
    /// Parallel `AABB` reduction over live positions (design §13).
    Bounds,
    /// Frustum / distance / `HZB` visibility culling (design §13).
    Cull,
    /// `radix` depth sort of the visible set for `OIT` (design §12).
    Sort,
}

impl SortCullPass {
    /// Every pass in pipeline order.
    pub const ALL: [SortCullPass; 4] = [Self::Compaction, Self::Bounds, Self::Cull, Self::Sort];

    /// Number of `@group(0)` bindings the pass's kernel declares.
    #[must_use]
    pub fn binding_count(self) -> u32 {
        match self {
            Self::Compaction => CompactionBuffer::ALL.len() as u32,
            Self::Bounds => BoundsBuffer::ALL.len() as u32,
            Self::Cull => CullBuffer::ALL.len() as u32,
            Self::Sort => SortBuffer::ALL.len() as u32,
        }
    }
}

/// Non-domain extents the four passes size their buffers against. Dispatch
/// domains (particle / candidate counts) and tuning knobs (`radix` bucket count,
/// workgroup count) travel together, mirroring `hair`'s per-pass extent structs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SortCullExtent {
    /// Live particle pool size (compaction, bounds and sort domains).
    pub particle_count: u32,
    /// Candidate count entering `Cull` (compacted survivors); the visible
    /// output can be at most this many.
    pub candidate_count: u32,
    /// `radix` histogram bucket count (for example 256 for an 8-bit digit).
    pub radix_buckets: u32,
    /// Number of workgroups a pass dispatches; sizes per-workgroup partials.
    pub workgroup_count: u32,
}

impl SortCullExtent {
    /// `radix` bucket count clamped to at least one, so a zero knob never sizes
    /// a histogram to nothing.
    #[must_use]
    pub fn effective_radix_buckets(self) -> u32 {
        self.radix_buckets.max(1)
    }

    /// Workgroup count clamped to at least one, so a zero dispatch still
    /// reserves one partial slot.
    #[must_use]
    pub fn effective_workgroup_count(self) -> u32 {
        self.workgroup_count.max(1)
    }
}

/// One storage buffer bound by the compaction kernel (`@group(0)`), in binding
/// order `0..3` (design §11 prefix-sum stream compaction).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompactionBuffer {
    /// `@binding(0)` per-particle alive flags `array<u32>` (`0`/`1`), read.
    AliveFlags,
    /// `@binding(1)` prefix-sum scan scratch `array<u32>`, read-write.
    ScanScratch,
    /// `@binding(2)` compacted live-index output `array<u32>`, read-write.
    CompactedIndices,
}

impl CompactionBuffer {
    /// Every compaction buffer in `@binding` order.
    pub const ALL: [CompactionBuffer; 3] =
        [Self::AliveFlags, Self::ScanScratch, Self::CompactedIndices];

    /// The pass this buffer belongs to.
    #[must_use]
    pub fn pass(self) -> SortCullPass {
        SortCullPass::Compaction
    }

    /// The `@group(0)` binding index.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::AliveFlags => 0,
            Self::ScanScratch => 1,
            Self::CompactedIndices => 2,
        }
    }

    /// Byte stride of one element (all `u32`).
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::AliveFlags | Self::ScanScratch | Self::CompactedIndices => U32_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer.
    #[must_use]
    pub fn access(self) -> ParticleBufferAccess {
        match self {
            Self::AliveFlags => ParticleBufferAccess::Read,
            Self::ScanScratch | Self::CompactedIndices => ParticleBufferAccess::ReadWrite,
        }
    }

    /// Whether this pass writes the buffer rather than only reading it.
    #[must_use]
    pub fn is_output(self) -> bool {
        self.access().is_writable()
    }

    /// Element count: one slot per particle across all three buffers.
    #[must_use]
    pub fn element_count(self, extent: SortCullExtent) -> u32 {
        match self {
            Self::AliveFlags | Self::ScanScratch | Self::CompactedIndices => extent.particle_count,
        }
    }

    /// Total byte size, clamped up to one element for a valid `WebGPU` binding.
    #[must_use]
    pub fn byte_size(self, extent: SortCullExtent) -> usize {
        storage_bytes(self.stride(), self.element_count(extent) as usize)
    }
}

/// One storage buffer bound by the bounds-reduction kernel (`@group(0)`), in
/// binding order `0..2` (design §13 `AABB` reduction).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BoundsBuffer {
    /// `@binding(0)` live particle positions `array<vec4<f32>>`, read.
    Positions,
    /// `@binding(1)` per-workgroup partial `AABB`s `array<vec4<f32>>`,
    /// read-write; two entries per workgroup (packed min then max).
    PartialBounds,
}

impl BoundsBuffer {
    /// Every bounds buffer in `@binding` order.
    pub const ALL: [BoundsBuffer; 2] = [Self::Positions, Self::PartialBounds];

    /// The pass this buffer belongs to.
    #[must_use]
    pub fn pass(self) -> SortCullPass {
        SortCullPass::Bounds
    }

    /// The `@group(0)` binding index.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::Positions => 0,
            Self::PartialBounds => 1,
        }
    }

    /// Byte stride of one element (both `vec4<f32>`).
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::Positions | Self::PartialBounds => VEC4_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer.
    #[must_use]
    pub fn access(self) -> ParticleBufferAccess {
        match self {
            Self::Positions => ParticleBufferAccess::Read,
            Self::PartialBounds => ParticleBufferAccess::ReadWrite,
        }
    }

    /// Whether this pass writes the buffer rather than only reading it.
    #[must_use]
    pub fn is_output(self) -> bool {
        self.access().is_writable()
    }

    /// Element count: one entry per particle for positions; two `vec4`s (min
    /// and max) per workgroup for the partial-bounds scratch.
    #[must_use]
    pub fn element_count(self, extent: SortCullExtent) -> u32 {
        match self {
            Self::Positions => extent.particle_count,
            Self::PartialBounds => extent.effective_workgroup_count().saturating_mul(2),
        }
    }

    /// Total byte size, clamped up to one element for a valid `WebGPU` binding.
    #[must_use]
    pub fn byte_size(self, extent: SortCullExtent) -> usize {
        storage_bytes(self.stride(), self.element_count(extent) as usize)
    }
}

/// One binding of the cull kernel (`@group(0)`), in binding order `0..4`
/// (design §13 frustum / distance / `HZB` culling).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CullBuffer {
    /// `@binding(0)` candidate indices `array<u32>` (compaction output), read.
    Candidates,
    /// `@binding(1)` cull parameters `var<uniform>`: six frustum planes, the
    /// distance range and the `HZB` descriptor. Read-only uniform.
    CullParams,
    /// `@binding(2)` visible-index output `array<u32>`, read-write.
    VisibleIndices,
    /// `@binding(3)` visible counter `array<vec2<u32>>` of length one:
    /// `(atomic visible count, overflow flag)`, read-write.
    VisibleCounter,
}

impl CullBuffer {
    /// Every cull binding in `@binding` order.
    pub const ALL: [CullBuffer; 4] = [
        Self::Candidates,
        Self::CullParams,
        Self::VisibleIndices,
        Self::VisibleCounter,
    ];

    /// The pass this buffer belongs to.
    #[must_use]
    pub fn pass(self) -> SortCullPass {
        SortCullPass::Cull
    }

    /// The `@group(0)` binding index.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::Candidates => 0,
            Self::CullParams => 1,
            Self::VisibleIndices => 2,
            Self::VisibleCounter => 3,
        }
    }

    /// Byte stride of one element, matching the `WESL` layout.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::Candidates | Self::VisibleIndices => U32_STRIDE,
            Self::CullParams => CULL_PARAMS_STRIDE,
            Self::VisibleCounter => VEC2_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer. The uniform
    /// parameters are read-only.
    #[must_use]
    pub fn access(self) -> ParticleBufferAccess {
        match self {
            Self::Candidates | Self::CullParams => ParticleBufferAccess::Read,
            Self::VisibleIndices | Self::VisibleCounter => ParticleBufferAccess::ReadWrite,
        }
    }

    /// Whether this pass writes the buffer rather than only reading it.
    #[must_use]
    pub fn is_output(self) -> bool {
        self.access().is_writable()
    }

    /// Element count: candidates and their visible subset are sized to the
    /// candidate count; the uniform and the counter are single elements.
    #[must_use]
    pub fn element_count(self, extent: SortCullExtent) -> u32 {
        match self {
            Self::Candidates | Self::VisibleIndices => extent.candidate_count,
            Self::CullParams | Self::VisibleCounter => 1,
        }
    }

    /// Total byte size, clamped up to one element for a valid `WebGPU` binding.
    #[must_use]
    pub fn byte_size(self, extent: SortCullExtent) -> usize {
        storage_bytes(self.stride(), self.element_count(extent) as usize)
    }
}

/// One storage buffer bound by the `radix` sort kernel (`@group(0)`), in binding
/// order `0..4` (design §12 depth sort for `OIT`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SortBuffer {
    /// `@binding(0)` sort keys `array<u32>` (quantized depth), read-write.
    Keys,
    /// `@binding(1)` payload / particle indices `array<u32>`, read-write.
    Payload,
    /// `@binding(2)` `radix` digit histograms `array<u32>`
    /// (`radix_buckets * workgroup_count`), read-write.
    Histogram,
    /// `@binding(3)` histogram prefix-sum scan scratch `array<u32>`, read-write.
    ScanScratch,
}

impl SortBuffer {
    /// Every sort buffer in `@binding` order.
    pub const ALL: [SortBuffer; 4] = [
        Self::Keys,
        Self::Payload,
        Self::Histogram,
        Self::ScanScratch,
    ];

    /// The pass this buffer belongs to.
    #[must_use]
    pub fn pass(self) -> SortCullPass {
        SortCullPass::Sort
    }

    /// The `@group(0)` binding index.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::Keys => 0,
            Self::Payload => 1,
            Self::Histogram => 2,
            Self::ScanScratch => 3,
        }
    }

    /// Byte stride of one element (all `u32`).
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::Keys | Self::Payload | Self::Histogram | Self::ScanScratch => U32_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer. Every sort buffer is
    /// mutated in place across the multi-digit passes.
    #[must_use]
    pub fn access(self) -> ParticleBufferAccess {
        match self {
            Self::Keys | Self::Payload | Self::Histogram | Self::ScanScratch => {
                ParticleBufferAccess::ReadWrite
            }
        }
    }

    /// Whether this pass writes the buffer rather than only reading it.
    #[must_use]
    pub fn is_output(self) -> bool {
        self.access().is_writable()
    }

    /// Element count: keys and payload are per-particle; the histogram and its
    /// scan scratch hold one slot per (`radix` bucket, workgroup) pair.
    #[must_use]
    pub fn element_count(self, extent: SortCullExtent) -> u32 {
        match self {
            Self::Keys | Self::Payload => extent.particle_count,
            Self::Histogram | Self::ScanScratch => extent
                .effective_radix_buckets()
                .saturating_mul(extent.effective_workgroup_count()),
        }
    }

    /// Total byte size, clamped up to one element for a valid `WebGPU` binding.
    #[must_use]
    pub fn byte_size(self, extent: SortCullExtent) -> usize {
        storage_bytes(self.stride(), self.element_count(extent) as usize)
    }
}

/// Total bytes a pass allocates for the buffers it *writes* (its outputs and
/// scratch), clamped per buffer to a valid non-empty `WebGPU` binding.
#[must_use]
pub fn pass_output_bytes(pass: SortCullPass, extent: SortCullExtent) -> usize {
    match pass {
        SortCullPass::Compaction => CompactionBuffer::ALL
            .into_iter()
            .filter(|buffer| buffer.is_output())
            .map(|buffer| buffer.byte_size(extent))
            .sum(),
        SortCullPass::Bounds => BoundsBuffer::ALL
            .into_iter()
            .filter(|buffer| buffer.is_output())
            .map(|buffer| buffer.byte_size(extent))
            .sum(),
        SortCullPass::Cull => CullBuffer::ALL
            .into_iter()
            .filter(|buffer| buffer.is_output())
            .map(|buffer| buffer.byte_size(extent))
            .sum(),
        SortCullPass::Sort => SortBuffer::ALL
            .into_iter()
            .filter(|buffer| buffer.is_output())
            .map(|buffer| buffer.byte_size(extent))
            .sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_extent() -> SortCullExtent {
        SortCullExtent {
            particle_count: 4096,
            candidate_count: 3000,
            radix_buckets: 256,
            workgroup_count: 16,
        }
    }

    #[test]
    fn compaction_bindings_are_dense_and_ordered() {
        for (index, buffer) in CompactionBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
            assert_eq!(buffer.pass(), SortCullPass::Compaction);
        }
    }

    #[test]
    fn bounds_bindings_are_dense_and_ordered() {
        for (index, buffer) in BoundsBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
            assert_eq!(buffer.pass(), SortCullPass::Bounds);
        }
    }

    #[test]
    fn cull_bindings_are_dense_and_ordered() {
        for (index, buffer) in CullBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
            assert_eq!(buffer.pass(), SortCullPass::Cull);
        }
    }

    #[test]
    fn sort_bindings_are_dense_and_ordered() {
        for (index, buffer) in SortBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
            assert_eq!(buffer.pass(), SortCullPass::Sort);
        }
    }

    #[test]
    fn binding_counts_match_the_binding_sets() {
        assert_eq!(SortCullPass::Compaction.binding_count(), 3);
        assert_eq!(SortCullPass::Bounds.binding_count(), 2);
        assert_eq!(SortCullPass::Cull.binding_count(), 4);
        assert_eq!(SortCullPass::Sort.binding_count(), 4);
    }

    #[test]
    fn empty_extent_still_reserves_one_element() {
        let empty = SortCullExtent::default();
        // Every buffer in every pass clamps to at least one element.
        for buffer in CompactionBuffer::ALL {
            assert_eq!(buffer.byte_size(empty), buffer.stride());
        }
        for buffer in BoundsBuffer::ALL {
            assert!(buffer.byte_size(empty) >= buffer.stride());
        }
        for buffer in CullBuffer::ALL {
            assert!(buffer.byte_size(empty) >= buffer.stride());
        }
        for buffer in SortBuffer::ALL {
            assert!(buffer.byte_size(empty) >= buffer.stride());
        }
    }

    #[test]
    fn degenerate_radix_and_workgroups_clamp_to_one() {
        let zeroed = SortCullExtent {
            particle_count: 10,
            candidate_count: 10,
            radix_buckets: 0,
            workgroup_count: 0,
        };
        assert_eq!(zeroed.effective_radix_buckets(), 1);
        assert_eq!(zeroed.effective_workgroup_count(), 1);
        // Histogram = 1 bucket * 1 workgroup = 1 element.
        assert_eq!(SortBuffer::Histogram.element_count(zeroed), 1);
        // Partial bounds = 1 workgroup * 2 (min/max) = 2 elements.
        assert_eq!(BoundsBuffer::PartialBounds.element_count(zeroed), 2);
    }

    #[test]
    fn radix_histogram_sizes_by_buckets_times_workgroups() {
        let extent = sample_extent();
        assert_eq!(SortBuffer::Histogram.element_count(extent), 256 * 16);
        assert_eq!(SortBuffer::ScanScratch.element_count(extent), 256 * 16);
        assert_eq!(
            SortBuffer::Histogram.byte_size(extent),
            256 * 16 * U32_STRIDE
        );
    }

    #[test]
    fn element_counts_map_to_the_right_domain() {
        let extent = sample_extent();
        assert_eq!(CompactionBuffer::AliveFlags.element_count(extent), 4096);
        assert_eq!(BoundsBuffer::Positions.element_count(extent), 4096);
        assert_eq!(BoundsBuffer::PartialBounds.element_count(extent), 32);
        assert_eq!(CullBuffer::Candidates.element_count(extent), 3000);
        assert_eq!(CullBuffer::VisibleIndices.element_count(extent), 3000);
        assert_eq!(CullBuffer::CullParams.element_count(extent), 1);
        assert_eq!(CullBuffer::VisibleCounter.element_count(extent), 1);
        assert_eq!(SortBuffer::Keys.element_count(extent), 4096);
    }

    #[test]
    fn strides_follow_std430() {
        assert_eq!(CompactionBuffer::AliveFlags.stride(), 4);
        assert_eq!(BoundsBuffer::Positions.stride(), 16);
        assert_eq!(CullBuffer::VisibleCounter.stride(), 8);
        assert_eq!(CullBuffer::CullParams.stride(), 128);
        assert_eq!(SortBuffer::Keys.stride(), 4);
    }

    #[test]
    fn access_modes_match_the_kernels() {
        assert_eq!(
            CompactionBuffer::AliveFlags.access(),
            ParticleBufferAccess::Read
        );
        assert!(CompactionBuffer::CompactedIndices.is_output());
        assert_eq!(BoundsBuffer::Positions.access(), ParticleBufferAccess::Read);
        assert!(BoundsBuffer::PartialBounds.is_output());
        assert_eq!(CullBuffer::CullParams.access(), ParticleBufferAccess::Read);
        assert!(!CullBuffer::CullParams.is_output());
        assert!(CullBuffer::VisibleCounter.is_output());
        assert!(SortBuffer::Payload.is_output());
    }

    #[test]
    fn pass_output_bytes_sums_only_writable_buffers() {
        let extent = sample_extent();
        // Compaction writes scan scratch + compacted indices (2 * particle u32).
        let expected_compaction = CompactionBuffer::ScanScratch.byte_size(extent)
            + CompactionBuffer::CompactedIndices.byte_size(extent);
        assert_eq!(
            pass_output_bytes(SortCullPass::Compaction, extent),
            expected_compaction
        );
        // Cull's read-only candidates and uniform are excluded.
        let expected_cull = CullBuffer::VisibleIndices.byte_size(extent)
            + CullBuffer::VisibleCounter.byte_size(extent);
        assert_eq!(pass_output_bytes(SortCullPass::Cull, extent), expected_cull);
    }
}
