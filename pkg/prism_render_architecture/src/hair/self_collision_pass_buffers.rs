//! Device-free byte-layout contract for the hair self-collision `GPU` twin
//! buffers (`hair_self_collision.wesl`).
//!
//! [`gpu_buffers`](super::gpu_buffers) and
//! [`vbd_pass_buffers`](super::vbd_pass_buffers) publish the ABI for the two
//! guide solver twins; this module is their peer for the self-collision resolve
//! pass, whose two `@compute @workgroup_size(64)` entry points
//! (`accumulate_self_collision` then `apply_self_collision`) share one
//! `@group(0)`. It consumes the `CSR` acceleration structure built by
//! [`self_collision_grid::build_csr`](super::self_collision_grid) and the
//! forward-Jacobi golden owned by
//! [`self_collision_jacobi`](super::self_collision_jacobi).
//!
//! As with the sibling contracts, the scene/render crate owns the actual wgpu
//! `Buffer`s, but sizing them (and building the bind-group layout) must agree
//! byte-for-byte with the `WESL` `@group(0)` declarations, so — like water's
//! device-free `WaterBufferPlan` — that sizing lives once here in the
//! zero-dependency crate and the render graph binds against a stable ABI.
//!
//! Residency is three-way: `positions` is the read-write guide state the pass
//! reads and the paired sim/`VBD` twins persist across frames (shared
//! byte-for-byte, `vec4<f32>` with `w` = inverse mass); `corrections` is
//! transient `GPU`-only scratch the accumulate entry sums into and the apply
//! entry drains each resolve (cleared per pass, neither persisted nor uploaded);
//! the three `CSR` arrays (`cell_keys` / `cell_starts` / `cell_indices`) are
//! read-only inputs the host re-uploads each frame from the freshly rebuilt
//! [`GridCsr`](super::self_collision_grid::GridCsr).
//!
//! Everything is pure integer arithmetic: byte sizes clamp up to one element so
//! an empty groom still yields a valid non-empty `WebGPU` storage binding, the
//! prefix-sum length saturates so it never underflows, and nothing panics or
//! divides by zero.

use crate::hair::gpu_dispatch::HairGpuCounts;

/// Byte stride of a `vec4<f32>` / `vec4<i32>` storage element: four 4-byte
/// scalars, the natural 16-byte stride.
const VEC4_STRIDE: usize = 16;

/// Byte stride of a scalar `u32` storage element.
const U32_STRIDE: usize = 4;

/// Byte size of the `HairSelfCollisionParams` immediate (push-constant) block
/// declared by `hair_self_collision.wesl`: five 4-byte scalars
/// (`particle_radius`, `stiffness`, `cell_size`, `particle_count`,
/// `cell_count`) laid out contiguously, struct alignment 4 — 20 bytes.
pub const PARAMS_IMMEDIATE_BYTES: usize = 20;

/// How a self-collision buffer is accessed by the `hair_self_collision.wesl`
/// kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairBufferAccess {
    /// `var<storage, read>` — read-only input.
    Read,
    /// `var<storage, read_write>` — mutated in place by the pass.
    ReadWrite,
}

/// Where a self-collision buffer's contents live across the frame boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairBufferResidency {
    /// Read-write guide state persisted across frames (shared with the sim /
    /// `VBD` twins' double-buffered deformed guide state).
    Persistent,
    /// Transient `GPU`-only scratch cleared per resolve; never persisted or
    /// host-uploaded.
    Scratch,
    /// Read-only input the host re-uploads each frame (the rebuilt `CSR` grid).
    Upload,
}

/// One storage buffer bound by the self-collision resolve pass
/// (`hair_self_collision.wesl` `@group(0)`), in binding order `0..5`. Both the
/// `accumulate_self_collision` and `apply_self_collision` entry points bind the
/// identical set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairSelfCollisionBuffer {
    /// `@binding(0)` particle state `array<vec4<f32>>` (`xyz` = position,
    /// `w` = inverse mass, `0` = pinned). Byte-identical to the sim/`VBD`
    /// `Positions`, so the same persistent allocation feeds every twin.
    Positions,
    /// `@binding(1)` per-particle accumulated correction `array<vec4<f32>>`
    /// (`xyz` = delta position, `w` unused).
    Corrections,
    /// `@binding(2)` `CSR` occupied cell coordinates `array<vec4<i32>>`,
    /// ascending lexicographic order.
    CellKeys,
    /// `@binding(3)` `CSR` prefix-sum offsets `array<u32>` (length
    /// `cell_count + 1`).
    CellStarts,
    /// `@binding(4)` `CSR` flat buckets `array<u32>`, ascending particle
    /// indices grouped by cell.
    CellIndices,
}

impl HairSelfCollisionBuffer {
    /// Every self-collision buffer in `@binding` order, matching the five
    /// `@group(0)` bindings shared by both kernel entry points.
    pub const ALL: [HairSelfCollisionBuffer; 5] = [
        Self::Positions,
        Self::Corrections,
        Self::CellKeys,
        Self::CellStarts,
        Self::CellIndices,
    ];

    /// The `@group(0)` binding index this buffer occupies.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::Positions => 0,
            Self::Corrections => 1,
            Self::CellKeys => 2,
            Self::CellStarts => 3,
            Self::CellIndices => 4,
        }
    }

    /// Byte stride of one element, matching the `WESL` scalar / vector layout.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            // vec4<f32> positions / corrections and vec4<i32> cell keys.
            Self::Positions | Self::Corrections | Self::CellKeys => VEC4_STRIDE,
            Self::CellStarts | Self::CellIndices => U32_STRIDE,
        }
    }

    /// Whether the kernels read or read-write this buffer.
    #[must_use]
    pub fn access(self) -> HairBufferAccess {
        match self {
            Self::Positions | Self::Corrections => HairBufferAccess::ReadWrite,
            Self::CellKeys | Self::CellStarts | Self::CellIndices => HairBufferAccess::Read,
        }
    }

    /// Where this buffer's contents live across the frame boundary.
    #[must_use]
    pub fn residency(self) -> HairBufferResidency {
        match self {
            Self::Positions => HairBufferResidency::Persistent,
            Self::Corrections => HairBufferResidency::Scratch,
            Self::CellKeys | Self::CellStarts | Self::CellIndices => HairBufferResidency::Upload,
        }
    }

    /// Number of elements this buffer holds for the given groom counts.
    ///
    /// `cell_count` is the number of occupied grid cells
    /// ([`GridCsr::cell_count`](super::self_collision_grid::GridCsr::cell_count)),
    /// independent of the [`HairGpuCounts`] element domains. `cell_starts` is a
    /// prefix-sum table with one more entry than there are cells, and
    /// `cell_indices` holds at most one entry per guide particle (each finite
    /// particle lands in exactly one bucket), so it is sized to the particle
    /// count worst case.
    #[must_use]
    pub fn element_count(self, counts: &HairGpuCounts, cell_count: u32) -> u32 {
        match self {
            Self::Positions | Self::Corrections | Self::CellIndices => counts.guide_particles,
            Self::CellKeys => cell_count,
            Self::CellStarts => cell_count.saturating_add(1),
        }
    }

    /// Total byte size to allocate for this buffer, clamped up to one element so
    /// an empty groom still yields a valid non-empty `WebGPU` storage binding.
    #[must_use]
    pub fn byte_size(self, counts: &HairGpuCounts, cell_count: u32) -> usize {
        let elements = self.element_count(counts, cell_count).max(1) as usize;
        elements * self.stride()
    }
}

/// Total bytes of read-write guide state that must persist across frames
/// (`positions`), shared byte-for-byte with the sim / `VBD` persistent
/// allocation.
#[must_use]
pub fn persistent_state_bytes(counts: &HairGpuCounts, cell_count: u32) -> usize {
    residency_bytes(counts, cell_count, HairBufferResidency::Persistent)
}

/// Total bytes of transient `GPU`-only scratch (`corrections`) cleared each
/// resolve — allocated on device but never persisted or host-uploaded.
#[must_use]
pub fn scratch_bytes(counts: &HairGpuCounts, cell_count: u32) -> usize {
    residency_bytes(counts, cell_count, HairBufferResidency::Scratch)
}

/// Total bytes of the read-only `CSR` inputs (`cell_keys`, `cell_starts`,
/// `cell_indices`) the host re-uploads each frame from the rebuilt grid.
#[must_use]
pub fn upload_input_bytes(counts: &HairGpuCounts, cell_count: u32) -> usize {
    residency_bytes(counts, cell_count, HairBufferResidency::Upload)
}

fn residency_bytes(
    counts: &HairGpuCounts,
    cell_count: u32,
    residency: HairBufferResidency,
) -> usize {
    HairSelfCollisionBuffer::ALL
        .into_iter()
        .filter(|buffer| buffer.residency() == residency)
        .map(|buffer| buffer.byte_size(counts, cell_count))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_counts() -> HairGpuCounts {
        HairGpuCounts {
            roots: 10,
            guide_strands: 100,
            guide_particles: 3200,
            render_strands: 50_000,
            light_texels: 0,
        }
    }

    #[test]
    fn bindings_are_dense_and_ordered() {
        for (index, buffer) in HairSelfCollisionBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
    }

    #[test]
    fn strides_match_the_wesl_layout() {
        assert_eq!(HairSelfCollisionBuffer::Positions.stride(), 16);
        assert_eq!(HairSelfCollisionBuffer::Corrections.stride(), 16);
        assert_eq!(HairSelfCollisionBuffer::CellKeys.stride(), 16);
        assert_eq!(HairSelfCollisionBuffer::CellStarts.stride(), 4);
        assert_eq!(HairSelfCollisionBuffer::CellIndices.stride(), 4);
    }

    #[test]
    fn access_modes_match_the_kernels() {
        assert_eq!(
            HairSelfCollisionBuffer::Positions.access(),
            HairBufferAccess::ReadWrite
        );
        assert_eq!(
            HairSelfCollisionBuffer::Corrections.access(),
            HairBufferAccess::ReadWrite
        );
        assert_eq!(
            HairSelfCollisionBuffer::CellKeys.access(),
            HairBufferAccess::Read
        );
        assert_eq!(
            HairSelfCollisionBuffer::CellStarts.access(),
            HairBufferAccess::Read
        );
        assert_eq!(
            HairSelfCollisionBuffer::CellIndices.access(),
            HairBufferAccess::Read
        );
    }

    #[test]
    fn residency_splits_persist_scratch_and_upload() {
        assert_eq!(
            HairSelfCollisionBuffer::Positions.residency(),
            HairBufferResidency::Persistent
        );
        assert_eq!(
            HairSelfCollisionBuffer::Corrections.residency(),
            HairBufferResidency::Scratch
        );
        for buffer in [
            HairSelfCollisionBuffer::CellKeys,
            HairSelfCollisionBuffer::CellStarts,
            HairSelfCollisionBuffer::CellIndices,
        ] {
            assert_eq!(buffer.residency(), HairBufferResidency::Upload);
        }
    }

    #[test]
    fn positions_layout_matches_the_sim_twin() {
        use crate::hair::gpu_buffers::HairSimBuffer;
        assert_eq!(
            HairSelfCollisionBuffer::Positions.stride(),
            HairSimBuffer::Positions.stride()
        );
    }

    #[test]
    fn element_counts_follow_groom_and_grid() {
        let counts = sample_counts();
        assert_eq!(
            HairSelfCollisionBuffer::Positions.element_count(&counts, 512),
            3200
        );
        assert_eq!(
            HairSelfCollisionBuffer::Corrections.element_count(&counts, 512),
            3200
        );
        // At most one bucket entry per particle.
        assert_eq!(
            HairSelfCollisionBuffer::CellIndices.element_count(&counts, 512),
            3200
        );
        assert_eq!(
            HairSelfCollisionBuffer::CellKeys.element_count(&counts, 512),
            512
        );
        // Prefix sum has one more entry than there are cells.
        assert_eq!(
            HairSelfCollisionBuffer::CellStarts.element_count(&counts, 512),
            513
        );
    }

    #[test]
    fn empty_grid_still_gives_one_prefix_entry() {
        let counts = sample_counts();
        assert_eq!(
            HairSelfCollisionBuffer::CellKeys.element_count(&counts, 0),
            0
        );
        assert_eq!(
            HairSelfCollisionBuffer::CellStarts.element_count(&counts, 0),
            1
        );
    }

    #[test]
    fn byte_sizes_multiply_count_by_stride() {
        let counts = sample_counts();
        assert_eq!(
            HairSelfCollisionBuffer::Positions.byte_size(&counts, 512),
            3200 * 16
        );
        assert_eq!(
            HairSelfCollisionBuffer::CellKeys.byte_size(&counts, 512),
            512 * 16
        );
        assert_eq!(
            HairSelfCollisionBuffer::CellStarts.byte_size(&counts, 512),
            513 * 4
        );
    }

    #[test]
    fn empty_groom_clamps_every_buffer_to_one_element() {
        let counts = HairGpuCounts::default();
        for buffer in HairSelfCollisionBuffer::ALL {
            assert_eq!(buffer.byte_size(&counts, 0), buffer.stride());
        }
    }

    #[test]
    fn residency_buckets_partition_the_buffers() {
        let counts = sample_counts();
        // positions alone.
        assert_eq!(persistent_state_bytes(&counts, 512), 3200 * 16);
        // corrections scratch alone.
        assert_eq!(scratch_bytes(&counts, 512), 3200 * 16);
        // cell_keys + cell_starts + cell_indices.
        let expected = 512 * 16 + 513 * 4 + 3200 * 4;
        assert_eq!(upload_input_bytes(&counts, 512), expected);
    }

    #[test]
    fn params_immediate_block_is_twenty_bytes() {
        assert_eq!(PARAMS_IMMEDIATE_BYTES, 20);
    }
}
