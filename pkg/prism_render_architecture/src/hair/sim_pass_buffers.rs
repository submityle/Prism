//! Device-free byte-layout contract for the hair per-frame `Simulate`-stage
//! auxiliary `GPU` passes: root skinning, wind and `SDF` collision.
//!
//! [`gpu_buffers`](super::gpu_buffers) already owns the persistent guide-`XPBD`
//! integrator (`GuideSim`) state. This module publishes *what the other three
//! `Simulate`-stage passes bind* — the authoritative element stride, access
//! mode, element count and total byte size of every storage buffer in
//! `hair_root_skinning.wesl`, `hair_wind.wesl` and `hair_sdf_collision.wesl`'s
//! `@group(0)`. As with [`gpu_buffers`](super::gpu_buffers) (sim),
//! [`interp_buffers`](super::interp_buffers) (resolve),
//! [`shadow_buffers`](super::shadow_buffers) (self-shadow) and
//! [`import_buffers`](super::import_buffers) (import), the sizing lives once
//! here in the zero-dependency crate so the render graph binds against a stable
//! ABI instead of hand-computing strides next to the pipeline.
//!
//! The three passes (design §3 阶段 2..4) run every frame before the integrator:
//! - `RootSkinning` re-skins each strand root against the deformed scalp mesh,
//!   reading the `HairMeshBinding` table (seeded once by
//!   [`import_buffers`](super::import_buffers)'s `RootBind`), the deformed scalp
//!   vertices and the flat index list, and writing a fresh per-root
//!   `HairRootFrame` (origin + orthonormal basis) the sim anchors pinned roots
//!   to.
//! - `Wind` perturbs the persistent guide particle positions in place.
//! - `SdfCollision` resolves the guide particles against a union of analytic
//!   `SDF` primitives, again mutating the persistent positions in place.
//!
//! `Wind` and `SdfCollision` therefore *alias* the persistent
//! [`HairSimBuffer::Positions`](super::gpu_buffers::HairSimBuffer::Positions)
//! buffer rather than allocating fresh storage; only `RootSkinning`'s frame
//! table is a genuinely new per-frame allocation, so
//! [`root_skinning_output_bytes`] sums only that.
//!
//! The scalp vertex / index counts and the `SDF` primitive count are not
//! dispatch domains, so they travel together in [`HairSimPassExtent`],
//! mirroring how [`import_buffers`](super::import_buffers) uses
//! `HairImportExtent`.
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
/// `triangle: u32` + three `u32` pads (16) = 32 bytes.
const MESH_BINDING_STRIDE: usize = 32;

/// `std430` array stride of `HairRootFrame`: four `vec4<f32>` (position,
/// tangent, normal, bitangent) = 64 bytes.
const ROOT_FRAME_STRIDE: usize = 64;

/// `std430` array stride of `HairSdfPrimitive`: `p0: vec4<f32>` (16) +
/// `p1: vec4<f32>` (16) + `kind: u32` + three `u32` pads (16) = 48 bytes.
const SDF_PRIMITIVE_STRIDE: usize = 48;

/// The non-domain extents the per-frame `Simulate` passes size their buffers
/// against: the deformed scalp mesh's vertex and flat-index counts (for
/// `RootSkinning`) and the analytic `SDF` primitive count (for `SdfCollision`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HairSimPassExtent {
    /// Deformed scalp vertex count (`RootSkinning` `vertices`).
    pub scalp_vertex_count: u32,
    /// Flat scalp triangle index count (three per triangle; `RootSkinning`
    /// `indices`).
    pub scalp_index_count: u32,
    /// Analytic `SDF` collision primitive count (`SdfCollision` `primitives`).
    pub sdf_primitive_count: u32,
}

/// One storage buffer bound by the root-skinning kernel
/// (`hair_root_skinning.wesl` `@group(0)`), in binding order `0..4`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairRootSkinningBuffer {
    /// `@binding(0)` per-root attachments `array<HairMeshBinding>` (seeded by
    /// `RootBind`).
    Bindings,
    /// `@binding(1)` deformed scalp vertex positions `array<vec4<f32>>`.
    Vertices,
    /// `@binding(2)` flat scalp triangle index list `array<u32>`.
    Indices,
    /// `@binding(3)` resolved per-root world frames `array<HairRootFrame>`,
    /// written by this pass.
    Frames,
}

impl HairRootSkinningBuffer {
    /// Every root-skinning buffer in `@binding` order. Its length matches
    /// [`HairComputePass::RootSkinning`](super::gpu_dispatch::HairComputePass)'s
    /// binding count.
    pub const ALL: [HairRootSkinningBuffer; 4] =
        [Self::Bindings, Self::Vertices, Self::Indices, Self::Frames];

    /// The `@group(0)` binding index in `hair_root_skinning.wesl`.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::Bindings => 0,
            Self::Vertices => 1,
            Self::Indices => 2,
            Self::Frames => 3,
        }
    }

    /// Byte stride of one element, matching the `WESL` struct / scalar layout.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::Bindings => MESH_BINDING_STRIDE,
            Self::Vertices => VEC4_STRIDE,
            Self::Indices => U32_STRIDE,
            Self::Frames => ROOT_FRAME_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer.
    #[must_use]
    pub fn access(self) -> HairBufferAccess {
        match self {
            Self::Bindings | Self::Vertices | Self::Indices => HairBufferAccess::Read,
            Self::Frames => HairBufferAccess::ReadWrite,
        }
    }

    /// Whether this pass writes the buffer rather than only reading it.
    #[must_use]
    pub fn is_output(self) -> bool {
        matches!(self.access(), HairBufferAccess::ReadWrite)
    }

    /// Element count for a groom with `counts` domain totals and `extent`
    /// per-frame extents: one entry per root (`bindings` / `frames`), per scalp
    /// vertex, or per flat index.
    #[must_use]
    pub fn element_count(self, counts: &HairGpuCounts, extent: HairSimPassExtent) -> u32 {
        match self {
            Self::Bindings | Self::Frames => counts.roots,
            Self::Vertices => extent.scalp_vertex_count,
            Self::Indices => extent.scalp_index_count,
        }
    }

    /// Total byte size, clamped up to one element so an empty groom still yields
    /// a valid non-empty `WebGPU` storage binding.
    #[must_use]
    pub fn byte_size(self, counts: &HairGpuCounts, extent: HairSimPassExtent) -> usize {
        let elements = self.element_count(counts, extent).max(1) as usize;
        elements * self.stride()
    }
}

/// The single storage buffer bound by the wind kernel (`hair_wind.wesl`
/// `@group(0)`): the persistent guide particle positions, perturbed in place.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairWindBuffer {
    /// `@binding(0)` guide particle positions `array<vec4<f32>>` (`xyz` =
    /// position, `w` = inverse mass), the persistent
    /// [`HairSimBuffer::Positions`](super::gpu_buffers::HairSimBuffer::Positions)
    /// buffer mutated in place.
    Positions,
}

impl HairWindBuffer {
    /// Every wind buffer in `@binding` order. Its length matches
    /// [`HairComputePass::Wind`](super::gpu_dispatch::HairComputePass)'s
    /// binding count.
    pub const ALL: [HairWindBuffer; 1] = [Self::Positions];

    /// The `@group(0)` binding index in `hair_wind.wesl`.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::Positions => 0,
        }
    }

    /// Byte stride of one element (`vec4<f32>`).
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::Positions => VEC4_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer.
    #[must_use]
    pub fn access(self) -> HairBufferAccess {
        match self {
            Self::Positions => HairBufferAccess::ReadWrite,
        }
    }

    /// Whether this pass writes the buffer rather than only reading it. `true`,
    /// but note the write aliases the persistent
    /// [`HairSimBuffer::Positions`](super::gpu_buffers::HairSimBuffer::Positions)
    /// buffer rather than a fresh allocation.
    #[must_use]
    pub fn is_output(self) -> bool {
        matches!(self.access(), HairBufferAccess::ReadWrite)
    }

    /// Element count: one per guide particle.
    #[must_use]
    pub fn element_count(self, counts: &HairGpuCounts) -> u32 {
        match self {
            Self::Positions => counts.guide_particles,
        }
    }

    /// Total byte size, clamped up to one element so an empty groom still yields
    /// a valid non-empty `WebGPU` storage binding.
    #[must_use]
    pub fn byte_size(self, counts: &HairGpuCounts) -> usize {
        let elements = self.element_count(counts).max(1) as usize;
        elements * self.stride()
    }
}

/// One storage buffer bound by the `SDF`-collision kernel
/// (`hair_sdf_collision.wesl` `@group(0)`), in binding order `0..2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairSdfCollisionBuffer {
    /// `@binding(0)` guide particle positions `array<vec4<f32>>`, the persistent
    /// [`HairSimBuffer::Positions`](super::gpu_buffers::HairSimBuffer::Positions)
    /// buffer pushed out of the primitives in place.
    Positions,
    /// `@binding(1)` analytic collision primitives `array<HairSdfPrimitive>`
    /// (`p0`, `p1`, `kind`).
    Primitives,
}

impl HairSdfCollisionBuffer {
    /// Every `SDF`-collision buffer in `@binding` order. Its length matches
    /// [`HairComputePass::SdfCollision`](super::gpu_dispatch::HairComputePass)'s
    /// binding count.
    pub const ALL: [HairSdfCollisionBuffer; 2] = [Self::Positions, Self::Primitives];

    /// The `@group(0)` binding index in `hair_sdf_collision.wesl`.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::Positions => 0,
            Self::Primitives => 1,
        }
    }

    /// Byte stride of one element, matching the `WESL` struct / scalar layout.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::Positions => VEC4_STRIDE,
            Self::Primitives => SDF_PRIMITIVE_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer.
    #[must_use]
    pub fn access(self) -> HairBufferAccess {
        match self {
            Self::Positions => HairBufferAccess::ReadWrite,
            Self::Primitives => HairBufferAccess::Read,
        }
    }

    /// Whether this pass writes the buffer rather than only reading it.
    #[must_use]
    pub fn is_output(self) -> bool {
        matches!(self.access(), HairBufferAccess::ReadWrite)
    }

    /// Element count for a groom with `counts` domain totals and `extent`
    /// per-frame extents: `positions` is the guide particle pool and
    /// `primitives` the analytic `SDF` primitive count.
    #[must_use]
    pub fn element_count(self, counts: &HairGpuCounts, extent: HairSimPassExtent) -> u32 {
        match self {
            Self::Positions => counts.guide_particles,
            Self::Primitives => extent.sdf_primitive_count,
        }
    }

    /// Total byte size, clamped up to one element so an empty groom still yields
    /// a valid non-empty `WebGPU` storage binding.
    #[must_use]
    pub fn byte_size(self, counts: &HairGpuCounts, extent: HairSimPassExtent) -> usize {
        let elements = self.element_count(counts, extent).max(1) as usize;
        elements * self.stride()
    }
}

/// Bytes the per-frame `Simulate` passes allocate afresh each frame: only
/// `RootSkinning`'s `HairRootFrame` table, since `Wind` and `SdfCollision`
/// mutate the persistent guide positions in place and allocate nothing new.
#[must_use]
pub fn root_skinning_output_bytes(counts: &HairGpuCounts, extent: HairSimPassExtent) -> usize {
    HairRootSkinningBuffer::ALL
        .into_iter()
        .filter(|buffer| buffer.is_output())
        .map(|buffer| buffer.byte_size(counts, extent))
        .sum()
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

    fn sample_extent() -> HairSimPassExtent {
        HairSimPassExtent {
            scalp_vertex_count: 2048,
            scalp_index_count: 6000,
            sdf_primitive_count: 12,
        }
    }

    #[test]
    fn bindings_are_dense_and_ordered() {
        for (index, buffer) in HairRootSkinningBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
        for (index, buffer) in HairWindBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
        for (index, buffer) in HairSdfCollisionBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
    }

    #[test]
    fn buffer_sets_match_the_dispatch_binding_counts() {
        assert_eq!(
            HairRootSkinningBuffer::ALL.len() as u32,
            HairComputePass::RootSkinning.binding_count()
        );
        assert_eq!(
            HairWindBuffer::ALL.len() as u32,
            HairComputePass::Wind.binding_count()
        );
        assert_eq!(
            HairSdfCollisionBuffer::ALL.len() as u32,
            HairComputePass::SdfCollision.binding_count()
        );
    }

    #[test]
    fn strides_match_the_wesl_struct_layout() {
        assert_eq!(HairRootSkinningBuffer::Bindings.stride(), 32);
        assert_eq!(HairRootSkinningBuffer::Vertices.stride(), 16);
        assert_eq!(HairRootSkinningBuffer::Indices.stride(), 4);
        assert_eq!(HairRootSkinningBuffer::Frames.stride(), 64);
        assert_eq!(HairWindBuffer::Positions.stride(), 16);
        assert_eq!(HairSdfCollisionBuffer::Positions.stride(), 16);
        assert_eq!(HairSdfCollisionBuffer::Primitives.stride(), 48);
    }

    #[test]
    fn access_modes_match_the_kernels() {
        assert_eq!(
            HairRootSkinningBuffer::Bindings.access(),
            HairBufferAccess::Read
        );
        assert_eq!(
            HairRootSkinningBuffer::Vertices.access(),
            HairBufferAccess::Read
        );
        assert_eq!(
            HairRootSkinningBuffer::Indices.access(),
            HairBufferAccess::Read
        );
        assert_eq!(
            HairRootSkinningBuffer::Frames.access(),
            HairBufferAccess::ReadWrite
        );
        assert_eq!(
            HairWindBuffer::Positions.access(),
            HairBufferAccess::ReadWrite
        );
        assert_eq!(
            HairSdfCollisionBuffer::Positions.access(),
            HairBufferAccess::ReadWrite
        );
        assert_eq!(
            HairSdfCollisionBuffer::Primitives.access(),
            HairBufferAccess::Read
        );
    }

    #[test]
    fn outputs_are_the_written_buffers() {
        assert!(!HairRootSkinningBuffer::Bindings.is_output());
        assert!(!HairRootSkinningBuffer::Vertices.is_output());
        assert!(!HairRootSkinningBuffer::Indices.is_output());
        assert!(HairRootSkinningBuffer::Frames.is_output());
        assert!(HairWindBuffer::Positions.is_output());
        assert!(HairSdfCollisionBuffer::Positions.is_output());
        assert!(!HairSdfCollisionBuffer::Primitives.is_output());
    }

    #[test]
    fn root_skinning_element_counts_follow_domains_and_extent() {
        let counts = sample_counts();
        let extent = sample_extent();
        assert_eq!(
            HairRootSkinningBuffer::Bindings.element_count(&counts, extent),
            100
        );
        assert_eq!(
            HairRootSkinningBuffer::Vertices.element_count(&counts, extent),
            2048
        );
        assert_eq!(
            HairRootSkinningBuffer::Indices.element_count(&counts, extent),
            6000
        );
        assert_eq!(
            HairRootSkinningBuffer::Frames.element_count(&counts, extent),
            100
        );
    }

    #[test]
    fn wind_and_sdf_element_counts_follow_domains_and_extent() {
        let counts = sample_counts();
        let extent = sample_extent();
        assert_eq!(HairWindBuffer::Positions.element_count(&counts), 3200);
        assert_eq!(
            HairSdfCollisionBuffer::Positions.element_count(&counts, extent),
            3200
        );
        assert_eq!(
            HairSdfCollisionBuffer::Primitives.element_count(&counts, extent),
            12
        );
    }

    #[test]
    fn byte_sizes_multiply_count_by_stride() {
        let counts = sample_counts();
        let extent = sample_extent();
        assert_eq!(
            HairRootSkinningBuffer::Frames.byte_size(&counts, extent),
            100 * 64
        );
        assert_eq!(HairWindBuffer::Positions.byte_size(&counts), 3200 * 16);
        assert_eq!(
            HairSdfCollisionBuffer::Primitives.byte_size(&counts, extent),
            12 * 48
        );
    }

    #[test]
    fn empty_groom_clamps_every_buffer_to_one_element() {
        let counts = HairGpuCounts::default();
        let extent = HairSimPassExtent::default();
        for buffer in HairRootSkinningBuffer::ALL {
            assert_eq!(buffer.byte_size(&counts, extent), buffer.stride());
        }
        for buffer in HairWindBuffer::ALL {
            assert_eq!(buffer.byte_size(&counts), buffer.stride());
        }
        for buffer in HairSdfCollisionBuffer::ALL {
            assert_eq!(buffer.byte_size(&counts, extent), buffer.stride());
        }
    }

    #[test]
    fn root_skinning_output_bytes_sums_only_the_frame_table() {
        let counts = sample_counts();
        let extent = sample_extent();
        assert_eq!(root_skinning_output_bytes(&counts, extent), 100 * 64);
    }
}
