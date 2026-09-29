//! Device-free byte-layout contract for the hair guide-`XPBD` `GPU` sim buffers.
//!
//! [`gpu_dispatch`](super::gpu_dispatch) publishes *how many* workgroups each
//! hair compute pass dispatches; this module publishes *what the per-frame guide
//! simulation binds* — the authoritative element stride, access mode, element
//! count and total byte size of every storage buffer in `hair_sim.wesl`'s
//! `@group(0)`. The scene/render crate owns the actual wgpu `Buffer`s, but
//! sizing them (and building the bind-group layout) must agree byte-for-byte
//! with the `WESL` struct declarations, so — exactly as water's device-free
//! `WaterBufferPlan` does — that sizing lives once here in the zero-dependency
//! crate rather than being hand-computed next to the pipeline, and the render
//! graph binds against a stable ABI instead of duplicating the derivation.
//!
//! This is the persistent-state half of the §8 GPU-driven persistence boundary:
//! `positions`/`prev_positions` are read-write state the solver integrates in
//! place and that must therefore survive across frames (the double-buffered
//! deformed guide state), whereas `goals`, `rest_lengths`, `strands` and
//! `colliders` are read-only inputs refreshed by the host or earlier passes.
//!
//! Everything is pure integer arithmetic: byte sizes are clamped up to one
//! element so an empty groom still yields a valid non-empty `WebGPU` storage
//! binding, per-segment counts saturate so degenerate inputs never underflow,
//! and nothing panics or divides by zero.

use crate::hair::gpu_dispatch::HairGpuCounts;

/// Byte stride of a `vec4<f32>` storage element (also a `vec4<u32>` and the
/// 4-`u32` `HairStrand`): four 4-byte scalars, the natural 16-byte stride.
const VEC4_STRIDE: usize = 16;

/// Byte stride of a scalar `f32` storage element.
const F32_STRIDE: usize = 4;

/// How a hair sim buffer is accessed by the `hair_sim.wesl` kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairBufferAccess {
    /// `var<storage, read>` — read-only input.
    Read,
    /// `var<storage, read_write>` — mutated in place by the solver.
    ReadWrite,
}

/// One storage buffer bound by the guide-`XPBD` sim kernel (`hair_sim.wesl`
/// `@group(0)`), in binding order `0..6`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairSimBuffer {
    /// `@binding(0)` particle state `array<vec4<f32>>` (`xyz` = position,
    /// `w` = inverse mass, `0` = pinned).
    Positions,
    /// `@binding(1)` previous positions `array<vec4<f32>>` (implicit velocity).
    PrevPositions,
    /// `@binding(2)` per-particle global goal pose `array<vec4<f32>>`.
    Goals,
    /// `@binding(3)` per-segment rest lengths `array<f32>`, flat across strands.
    RestLengths,
    /// `@binding(4)` per-strand flat offset descriptors `array<HairStrand>`
    /// (4 `u32`).
    Strands,
    /// `@binding(5)` analytic body colliders `array<HairCollider>` (2 `vec4<f32>`).
    Colliders,
}

impl HairSimBuffer {
    /// Every sim buffer in `@binding` order. Its length matches
    /// [`HairComputePass::GuideSim`](super::gpu_dispatch::HairComputePass)'s
    /// binding count, keeping this layout in lock-step with the dispatch ABI.
    pub const ALL: [HairSimBuffer; 6] = [
        Self::Positions,
        Self::PrevPositions,
        Self::Goals,
        Self::RestLengths,
        Self::Strands,
        Self::Colliders,
    ];

    /// The `@group(0)` binding index this buffer occupies in `hair_sim.wesl`.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::Positions => 0,
            Self::PrevPositions => 1,
            Self::Goals => 2,
            Self::RestLengths => 3,
            Self::Strands => 4,
            Self::Colliders => 5,
        }
    }

    /// Byte stride of one element, matching the `WESL` struct / scalar layout.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            // vec4<f32> particle state / goals, and HairStrand = 4 * u32.
            Self::Positions | Self::PrevPositions | Self::Goals | Self::Strands => VEC4_STRIDE,
            Self::RestLengths => F32_STRIDE,
            // HairCollider = 2 * vec4<f32>.
            Self::Colliders => 2 * VEC4_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer.
    #[must_use]
    pub fn access(self) -> HairBufferAccess {
        match self {
            Self::Positions | Self::PrevPositions => HairBufferAccess::ReadWrite,
            Self::Goals | Self::RestLengths | Self::Strands | Self::Colliders => {
                HairBufferAccess::Read
            }
        }
    }

    /// Whether this buffer's contents must persist across frames (the solver
    /// integrates the particle state in place) rather than being re-supplied
    /// each frame. These are the buffers the render graph double-buffers when it
    /// hands the previous frame's deformed guides to downstream passes while the
    /// next solve runs.
    #[must_use]
    pub fn persists_across_frames(self) -> bool {
        match self {
            Self::Positions | Self::PrevPositions => true,
            Self::Goals | Self::RestLengths | Self::Strands | Self::Colliders => false,
        }
    }

    /// Number of elements this buffer holds for the given groom counts.
    ///
    /// `collider_count` is the analytic collider count
    /// (`HairXpbdParams.collider_count`), independent of the [`HairGpuCounts`]
    /// element domains. Per-segment rest lengths total
    /// `guide_particles - guide_strands` (each strand of `n` particles has
    /// `n - 1` segments), saturating so degenerate inputs never underflow.
    #[must_use]
    pub fn element_count(self, counts: &HairGpuCounts, collider_count: u32) -> u32 {
        match self {
            Self::Positions | Self::PrevPositions | Self::Goals => counts.guide_particles,
            Self::RestLengths => counts.guide_particles.saturating_sub(counts.guide_strands),
            Self::Strands => counts.guide_strands,
            Self::Colliders => collider_count,
        }
    }

    /// Total byte size to allocate for this buffer, clamped up to one element so
    /// an empty groom still yields a valid non-empty `WebGPU` storage binding.
    #[must_use]
    pub fn byte_size(self, counts: &HairGpuCounts, collider_count: u32) -> usize {
        let elements = self.element_count(counts, collider_count).max(1) as usize;
        elements * self.stride()
    }
}

/// Total bytes of read-write guide state that must persist across frames (the
/// double-buffered `positions` + `prev_positions`), for capacity planning of
/// the persistent sim allocation.
#[must_use]
pub fn persistent_state_bytes(counts: &HairGpuCounts, collider_count: u32) -> usize {
    HairSimBuffer::ALL
        .into_iter()
        .filter(|buffer| buffer.persists_across_frames())
        .map(|buffer| buffer.byte_size(counts, collider_count))
        .sum()
}

/// Total bytes of the read-only per-frame inputs (`goals`, `rest_lengths`,
/// `strands`, `colliders`) the host / earlier passes refresh each frame.
#[must_use]
pub fn upload_input_bytes(counts: &HairGpuCounts, collider_count: u32) -> usize {
    HairSimBuffer::ALL
        .into_iter()
        .filter(|buffer| !buffer.persists_across_frames())
        .map(|buffer| buffer.byte_size(counts, collider_count))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::gpu_dispatch::HairComputePass;

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
        for (index, buffer) in HairSimBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
    }

    #[test]
    fn buffer_set_matches_the_dispatch_binding_count() {
        assert_eq!(
            HairSimBuffer::ALL.len() as u32,
            HairComputePass::GuideSim.binding_count()
        );
    }

    #[test]
    fn strides_match_the_wesl_struct_layout() {
        assert_eq!(HairSimBuffer::Positions.stride(), 16);
        assert_eq!(HairSimBuffer::PrevPositions.stride(), 16);
        assert_eq!(HairSimBuffer::Goals.stride(), 16);
        assert_eq!(HairSimBuffer::RestLengths.stride(), 4);
        assert_eq!(HairSimBuffer::Strands.stride(), 16);
        assert_eq!(HairSimBuffer::Colliders.stride(), 32);
    }

    #[test]
    fn access_modes_match_the_kernel() {
        assert_eq!(
            HairSimBuffer::Positions.access(),
            HairBufferAccess::ReadWrite
        );
        assert_eq!(
            HairSimBuffer::PrevPositions.access(),
            HairBufferAccess::ReadWrite
        );
        assert_eq!(HairSimBuffer::Goals.access(), HairBufferAccess::Read);
        assert_eq!(HairSimBuffer::RestLengths.access(), HairBufferAccess::Read);
        assert_eq!(HairSimBuffer::Strands.access(), HairBufferAccess::Read);
        assert_eq!(HairSimBuffer::Colliders.access(), HairBufferAccess::Read);
    }

    #[test]
    fn only_particle_state_persists_across_frames() {
        assert!(HairSimBuffer::Positions.persists_across_frames());
        assert!(HairSimBuffer::PrevPositions.persists_across_frames());
        assert!(!HairSimBuffer::Goals.persists_across_frames());
        assert!(!HairSimBuffer::RestLengths.persists_across_frames());
        assert!(!HairSimBuffer::Strands.persists_across_frames());
        assert!(!HairSimBuffer::Colliders.persists_across_frames());
    }

    #[test]
    fn element_counts_follow_the_groom_domains() {
        let counts = sample_counts();
        assert_eq!(HairSimBuffer::Positions.element_count(&counts, 8), 3200);
        assert_eq!(HairSimBuffer::Goals.element_count(&counts, 8), 3200);
        // 3200 particles across 100 strands -> 3100 segments.
        assert_eq!(HairSimBuffer::RestLengths.element_count(&counts, 8), 3100);
        assert_eq!(HairSimBuffer::Strands.element_count(&counts, 8), 100);
        assert_eq!(HairSimBuffer::Colliders.element_count(&counts, 8), 8);
    }

    #[test]
    fn rest_length_count_saturates_on_degenerate_input() {
        let counts = HairGpuCounts {
            roots: 0,
            guide_strands: 10,
            guide_particles: 4,
            render_strands: 0,
            light_texels: 0,
        };
        assert_eq!(HairSimBuffer::RestLengths.element_count(&counts, 0), 0);
    }

    #[test]
    fn byte_sizes_multiply_count_by_stride() {
        let counts = sample_counts();
        assert_eq!(HairSimBuffer::Positions.byte_size(&counts, 8), 3200 * 16);
        assert_eq!(HairSimBuffer::RestLengths.byte_size(&counts, 8), 3100 * 4);
        assert_eq!(HairSimBuffer::Colliders.byte_size(&counts, 8), 8 * 32);
    }

    #[test]
    fn empty_groom_clamps_every_buffer_to_one_element() {
        let counts = HairGpuCounts::default();
        for buffer in HairSimBuffer::ALL {
            assert_eq!(buffer.byte_size(&counts, 0), buffer.stride());
        }
    }

    #[test]
    fn persistent_and_upload_bytes_partition_the_buffers() {
        let counts = sample_counts();
        // positions + prev_positions.
        assert_eq!(persistent_state_bytes(&counts, 8), 2 * 3200 * 16);
        // goals + rest_lengths + strands + colliders.
        let expected = 3200 * 16 + 3100 * 4 + 100 * 16 + 8 * 32;
        assert_eq!(upload_input_bytes(&counts, 8), expected);
    }
}
