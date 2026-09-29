//! Device-free byte-layout contract for the hair one-shot import (`Import`-stage)
//! `GPU` buffers.
//!
//! [`gpu_dispatch`](super::gpu_dispatch) publishes *how many* workgroups the
//! [`RootBind`](super::gpu_dispatch::HairComputePass::RootBind) and
//! [`Resample`](super::gpu_dispatch::HairComputePass::Resample) passes dispatch;
//! this module publishes *what those two passes bind* — the authoritative
//! element stride, access mode, element count and total byte size of every
//! storage buffer in `hair_root_bind.wesl`'s and `hair_resample.wesl`'s
//! `@group(0)`. Exactly as [`gpu_buffers`](super::gpu_buffers) (sim),
//! [`interp_buffers`](super::interp_buffers) (resolve) and
//! [`shadow_buffers`](super::shadow_buffers) (self-shadow) do, the sizing lives
//! once here in the zero-dependency crate so the render graph binds against a
//! stable ABI instead of hand-computing strides next to the pipeline.
//!
//! These are the *import* head of the pipeline (design §3 阶段 1): `RootBind`
//! projects rest-pose strand roots onto the rest-pose scalp mesh (reading roots,
//! scalp vertices and the triangle index list, writing the per-root
//! `HairMeshBinding` that `hair_root_skinning.wesl` later consumes each frame),
//! and `Resample` rewrites the ragged authored control points to the uniform
//! per-strand stride (reading the raw point pool and per-strand ranges, writing
//! the fixed-stride guide particles the sim then integrates). Both are one-shot
//! passes run when a groom is loaded, so their outputs seed the persistent state
//! rather than being rebuilt per frame.
//!
//! The scalp vertex / index counts and the raw authored point count are not
//! dispatch domains, so they travel together in [`HairImportExtent`], mirroring
//! how [`gpu_buffers`](super::gpu_buffers) takes `collider_count`,
//! [`interp_buffers`](super::interp_buffers) takes `render_points` and
//! [`shadow_buffers`](super::shadow_buffers) takes `HairShadowExtent`.
//!
//! Everything is pure integer arithmetic: byte sizes are clamped up to one
//! element so an empty groom still yields a valid non-empty `WebGPU` storage
//! binding, and nothing panics or divides by zero.

use crate::hair::gpu_buffers::HairBufferAccess;
use crate::hair::gpu_dispatch::HairGpuCounts;

/// Byte stride of a `vec4<f32>` storage element: four 4-byte scalars, 16 bytes.
const VEC4_STRIDE: usize = 16;

/// Byte stride of a scalar `u32` storage element (flat triangle index list).
const U32_STRIDE: usize = 4;

/// `std430` array stride of `HairMeshBinding`: `bary_height: vec4<f32>` (16) +
/// `triangle: u32` + three `u32` pads (16) = 32 bytes, the clean stride
/// `hair_root_skinning.wesl` consumes.
const MESH_BINDING_STRIDE: usize = 32;

/// `std430` array stride of `HairRawStrandRange` (`start: u32`, `len: u32`): two
/// tightly packed 4-byte scalars, 8 bytes.
const RAW_RANGE_STRIDE: usize = 8;

/// The non-domain extents the import passes size their buffers against: the
/// scalp mesh's vertex and flat-index counts (for `RootBind`) and the raw
/// authored control-point pool length (for `Resample`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HairImportExtent {
    /// Rest-pose scalp vertex count (`RootBind` `vertices`).
    pub scalp_vertex_count: u32,
    /// Flat scalp triangle index count (three per triangle; `RootBind`
    /// `indices`).
    pub scalp_index_count: u32,
    /// Total raw authored control points across every guide (`Resample`
    /// `raw_points`).
    pub raw_point_count: u32,
}

/// One storage buffer bound by the root-projection kernel
/// (`hair_root_bind.wesl` `@group(0)`), in binding order `0..4`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairRootBindBuffer {
    /// `@binding(0)` rest-pose strand root positions `array<vec4<f32>>`
    /// (`xyz` used).
    Roots,
    /// `@binding(1)` rest-pose scalp vertex positions `array<vec4<f32>>`
    /// (`xyz` used).
    Vertices,
    /// `@binding(2)` flat scalp triangle index list `array<u32>` (three per
    /// triangle).
    Indices,
    /// `@binding(3)` resolved per-root attachments `array<HairMeshBinding>`,
    /// written by this pass.
    OutBindings,
}

impl HairRootBindBuffer {
    /// Every root-bind buffer in `@binding` order. Its length matches
    /// [`HairComputePass::RootBind`](super::gpu_dispatch::HairComputePass)'s
    /// binding count.
    pub const ALL: [HairRootBindBuffer; 4] = [
        Self::Roots,
        Self::Vertices,
        Self::Indices,
        Self::OutBindings,
    ];

    /// The `@group(0)` binding index in `hair_root_bind.wesl`.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::Roots => 0,
            Self::Vertices => 1,
            Self::Indices => 2,
            Self::OutBindings => 3,
        }
    }

    /// Byte stride of one element, matching the `WESL` struct / scalar layout.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::Roots | Self::Vertices => VEC4_STRIDE,
            Self::Indices => U32_STRIDE,
            Self::OutBindings => MESH_BINDING_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer.
    #[must_use]
    pub fn access(self) -> HairBufferAccess {
        match self {
            Self::Roots | Self::Vertices | Self::Indices => HairBufferAccess::Read,
            Self::OutBindings => HairBufferAccess::ReadWrite,
        }
    }

    /// Whether this pass writes the buffer rather than only reading it.
    #[must_use]
    pub fn is_output(self) -> bool {
        matches!(self.access(), HairBufferAccess::ReadWrite)
    }

    /// Element count for a groom with `counts` domain totals and `extent` import
    /// extents: one entry per root (roots / `out_bindings`), per scalp vertex, or
    /// per flat index.
    #[must_use]
    pub fn element_count(self, counts: &HairGpuCounts, extent: HairImportExtent) -> u32 {
        match self {
            Self::Roots | Self::OutBindings => counts.roots,
            Self::Vertices => extent.scalp_vertex_count,
            Self::Indices => extent.scalp_index_count,
        }
    }

    /// Total byte size, clamped up to one element so an empty groom still yields
    /// a valid non-empty `WebGPU` storage binding.
    #[must_use]
    pub fn byte_size(self, counts: &HairGpuCounts, extent: HairImportExtent) -> usize {
        let elements = self.element_count(counts, extent).max(1) as usize;
        elements * self.stride()
    }
}

/// One storage buffer bound by the arc-length resample kernel
/// (`hair_resample.wesl` `@group(0)`), in binding order `0..3`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairResampleBuffer {
    /// `@binding(0)` shared flat raw authored control points `array<vec4<f32>>`
    /// (`xyz` = position).
    RawPoints,
    /// `@binding(1)` per-strand compacted slice descriptor
    /// `array<HairRawStrandRange>` (`start`, `len` as `u32`) into `raw_points`.
    StrandRanges,
    /// `@binding(2)` fixed-stride resampled guide particles `array<vec4<f32>>`
    /// (`vec4(pos, 0)`), written by this pass.
    OutPoints,
}

impl HairResampleBuffer {
    /// Every resample buffer in `@binding` order. Its length matches
    /// [`HairComputePass::Resample`](super::gpu_dispatch::HairComputePass)'s
    /// binding count.
    pub const ALL: [HairResampleBuffer; 3] = [Self::RawPoints, Self::StrandRanges, Self::OutPoints];

    /// The `@group(0)` binding index in `hair_resample.wesl`.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::RawPoints => 0,
            Self::StrandRanges => 1,
            Self::OutPoints => 2,
        }
    }

    /// Byte stride of one element, matching the `WESL` struct / scalar layout.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::RawPoints | Self::OutPoints => VEC4_STRIDE,
            Self::StrandRanges => RAW_RANGE_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer.
    #[must_use]
    pub fn access(self) -> HairBufferAccess {
        match self {
            Self::RawPoints | Self::StrandRanges => HairBufferAccess::Read,
            Self::OutPoints => HairBufferAccess::ReadWrite,
        }
    }

    /// Whether this pass writes the buffer rather than only reading it.
    #[must_use]
    pub fn is_output(self) -> bool {
        matches!(self.access(), HairBufferAccess::ReadWrite)
    }

    /// Element count for a groom with `counts` domain totals and `extent` import
    /// extents: `raw_points` is the authored pool, `strand_ranges` one per guide
    /// strand, and `out_points` the uniform-stride guide particles (`strand *
    /// points_per_strand`, i.e. `guide_particles`).
    #[must_use]
    pub fn element_count(self, counts: &HairGpuCounts, extent: HairImportExtent) -> u32 {
        match self {
            Self::RawPoints => extent.raw_point_count,
            Self::StrandRanges => counts.guide_strands,
            Self::OutPoints => counts.guide_particles,
        }
    }

    /// Total byte size, clamped up to one element so an empty groom still yields
    /// a valid non-empty `WebGPU` storage binding.
    #[must_use]
    pub fn byte_size(self, counts: &HairGpuCounts, extent: HairImportExtent) -> usize {
        let elements = self.element_count(counts, extent).max(1) as usize;
        elements * self.stride()
    }
}

/// Total bytes the import stage seeds into persistent state: the per-root
/// `HairMeshBinding` table (`RootBind` output) plus the resampled guide
/// particles (`Resample` output), which downstream per-frame passes then consume.
#[must_use]
pub fn import_output_bytes(counts: &HairGpuCounts, extent: HairImportExtent) -> usize {
    let bindings: usize = HairRootBindBuffer::ALL
        .into_iter()
        .filter(|buffer| buffer.is_output())
        .map(|buffer| buffer.byte_size(counts, extent))
        .sum();
    let points: usize = HairResampleBuffer::ALL
        .into_iter()
        .filter(|buffer| buffer.is_output())
        .map(|buffer| buffer.byte_size(counts, extent))
        .sum();
    bindings + points
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::gpu_dispatch::HairComputePass;

    fn sample_counts() -> HairGpuCounts {
        HairGpuCounts {
            roots: 100,
            guide_strands: 100,
            guide_particles: 3200,
            render_strands: 50_000,
            light_texels: 0,
        }
    }

    fn sample_extent() -> HairImportExtent {
        HairImportExtent {
            scalp_vertex_count: 2048,
            scalp_index_count: 6000,
            raw_point_count: 2500,
        }
    }

    #[test]
    fn root_bind_bindings_are_dense_and_ordered() {
        for (index, buffer) in HairRootBindBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
    }

    #[test]
    fn resample_bindings_are_dense_and_ordered() {
        for (index, buffer) in HairResampleBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
    }

    #[test]
    fn buffer_sets_match_the_dispatch_binding_counts() {
        assert_eq!(
            HairRootBindBuffer::ALL.len() as u32,
            HairComputePass::RootBind.binding_count()
        );
        assert_eq!(
            HairResampleBuffer::ALL.len() as u32,
            HairComputePass::Resample.binding_count()
        );
    }

    #[test]
    fn strides_match_the_wesl_struct_layout() {
        assert_eq!(HairRootBindBuffer::Roots.stride(), 16);
        assert_eq!(HairRootBindBuffer::Vertices.stride(), 16);
        assert_eq!(HairRootBindBuffer::Indices.stride(), 4);
        assert_eq!(HairRootBindBuffer::OutBindings.stride(), 32);
        assert_eq!(HairResampleBuffer::RawPoints.stride(), 16);
        assert_eq!(HairResampleBuffer::StrandRanges.stride(), 8);
        assert_eq!(HairResampleBuffer::OutPoints.stride(), 16);
    }

    #[test]
    fn access_modes_match_the_kernels() {
        assert_eq!(HairRootBindBuffer::Roots.access(), HairBufferAccess::Read);
        assert_eq!(
            HairRootBindBuffer::Vertices.access(),
            HairBufferAccess::Read
        );
        assert_eq!(HairRootBindBuffer::Indices.access(), HairBufferAccess::Read);
        assert_eq!(
            HairRootBindBuffer::OutBindings.access(),
            HairBufferAccess::ReadWrite
        );
        assert_eq!(
            HairResampleBuffer::RawPoints.access(),
            HairBufferAccess::Read
        );
        assert_eq!(
            HairResampleBuffer::StrandRanges.access(),
            HairBufferAccess::Read
        );
        assert_eq!(
            HairResampleBuffer::OutPoints.access(),
            HairBufferAccess::ReadWrite
        );
    }

    #[test]
    fn outputs_are_the_seeded_state() {
        assert!(!HairRootBindBuffer::Roots.is_output());
        assert!(!HairRootBindBuffer::Vertices.is_output());
        assert!(!HairRootBindBuffer::Indices.is_output());
        assert!(HairRootBindBuffer::OutBindings.is_output());
        assert!(!HairResampleBuffer::RawPoints.is_output());
        assert!(!HairResampleBuffer::StrandRanges.is_output());
        assert!(HairResampleBuffer::OutPoints.is_output());
    }

    #[test]
    fn root_bind_element_counts_follow_domains_and_extent() {
        let counts = sample_counts();
        let extent = sample_extent();
        assert_eq!(
            HairRootBindBuffer::Roots.element_count(&counts, extent),
            100
        );
        assert_eq!(
            HairRootBindBuffer::Vertices.element_count(&counts, extent),
            2048
        );
        assert_eq!(
            HairRootBindBuffer::Indices.element_count(&counts, extent),
            6000
        );
        assert_eq!(
            HairRootBindBuffer::OutBindings.element_count(&counts, extent),
            100
        );
    }

    #[test]
    fn resample_element_counts_follow_domains_and_extent() {
        let counts = sample_counts();
        let extent = sample_extent();
        assert_eq!(
            HairResampleBuffer::RawPoints.element_count(&counts, extent),
            2500
        );
        assert_eq!(
            HairResampleBuffer::StrandRanges.element_count(&counts, extent),
            100
        );
        // Uniform resample output == guide particle pool.
        assert_eq!(
            HairResampleBuffer::OutPoints.element_count(&counts, extent),
            3200
        );
    }

    #[test]
    fn byte_sizes_multiply_count_by_stride() {
        let counts = sample_counts();
        let extent = sample_extent();
        assert_eq!(
            HairRootBindBuffer::OutBindings.byte_size(&counts, extent),
            100 * 32
        );
        assert_eq!(
            HairRootBindBuffer::Indices.byte_size(&counts, extent),
            6000 * 4
        );
        assert_eq!(
            HairResampleBuffer::OutPoints.byte_size(&counts, extent),
            3200 * 16
        );
    }

    #[test]
    fn empty_groom_clamps_every_buffer_to_one_element() {
        let counts = HairGpuCounts::default();
        let extent = HairImportExtent::default();
        for buffer in HairRootBindBuffer::ALL {
            assert_eq!(buffer.byte_size(&counts, extent), buffer.stride());
        }
        for buffer in HairResampleBuffer::ALL {
            assert_eq!(buffer.byte_size(&counts, extent), buffer.stride());
        }
    }

    #[test]
    fn import_output_bytes_sums_the_seeded_state() {
        let counts = sample_counts();
        let extent = sample_extent();
        // out_bindings (100 * 32) + out_points (3200 * 16).
        let expected = (100 * 32) + (3200 * 16);
        assert_eq!(import_output_bytes(&counts, extent), expected);
    }
}
