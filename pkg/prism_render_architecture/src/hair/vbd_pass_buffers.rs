//! Device-free byte-layout contract for the hair guide-`VBD` `GPU` solver
//! buffers.
//!
//! [`gpu_buffers`](super::gpu_buffers) publishes the ABI for the guide-`XPBD`
//! sim (`hair_sim.wesl`); this module is its twin for the high-fidelity
//! Vertex-Block-Descent solver (`hair_vbd.wesl`). Both kernels bind the exact
//! same `positions` / `prev_positions` particle state and the byte-identical
//! four-`u32` `HairStrand` descriptor, so the [`SolverSelection`] slot swap
//! (`solver::HairSolverKind::Xpbd` vs `Vbd`) rebinds only the compute pipeline —
//! the persistent deformed-guide allocation and the strand descriptor upload are
//! shared verbatim. The two differences are: `hair_vbd.wesl` needs a per-frame
//! `targets` scratch buffer (the `GPU` equivalent of the CPU `targets` `Vec`
//! rebuilt every substep) in place of the read-only `goals` pose array, and it
//! never reads `goal_offset`.
//!
//! As in [`gpu_buffers`], the scene/render crate owns the actual wgpu `Buffer`s,
//! but sizing them (and building the bind-group layout) must agree byte-for-byte
//! with the `WESL` `@group(0)` declarations, so — exactly like water's
//! device-free `WaterBufferPlan` — that sizing lives once here in the
//! zero-dependency crate and the render graph binds against a stable ABI.
//!
//! Residency is three-way, sharper than the sim's binary persist/upload split:
//! `positions` / `prev_positions` are the read-write state the solver integrates
//! in place and must survive across frames; `targets` is transient `GPU`-only
//! scratch fully overwritten each substep (neither persisted nor host-uploaded);
//! `rest_lengths` / `strands` / `colliders` are read-only inputs the host or
//! earlier passes refresh each frame.
//!
//! Everything is pure integer arithmetic: byte sizes clamp up to one element so
//! an empty groom still yields a valid non-empty `WebGPU` storage binding,
//! per-segment counts saturate so degenerate inputs never underflow, and nothing
//! panics or divides by zero.

use crate::hair::gpu_dispatch::HairGpuCounts;

/// Byte stride of a `vec4<f32>` storage element and of the four-`u32`
/// `HairStrand` descriptor: four 4-byte scalars, the natural 16-byte stride.
const VEC4_STRIDE: usize = 16;

/// Byte stride of a scalar `f32` storage element.
const F32_STRIDE: usize = 4;

/// Byte size of the `HairVbdParams` immediate (push-constant) block declared by
/// `hair_vbd.wesl`.
///
/// std430/immediate layout: `gravity` is a `vec3<f32>` (align 16, bytes 0..12),
/// `dt` packs into the trailing scalar slot at byte 12, then five `f32` /`u32`
/// scalars (`stretch_stiffness`, `bending_stiffness`, `damping`, `substeps`,
/// `iterations`, `strand_count`, `collider_count`) run contiguously from byte
/// 16, and the struct size rounds up to the `vec3` alignment of 16 — 48 bytes.
pub const PARAMS_IMMEDIATE_BYTES: usize = 48;

/// How a hair `VBD` buffer is accessed by the `hair_vbd.wesl` kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairBufferAccess {
    /// `var<storage, read>` — read-only input.
    Read,
    /// `var<storage, read_write>` — mutated in place by the solver.
    ReadWrite,
}

/// Where a hair `VBD` buffer's contents live across the frame boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairBufferResidency {
    /// Read-write guide state the solver integrates in place; must survive
    /// across frames (the double-buffered deformed guide state).
    Persistent,
    /// Transient `GPU`-only scratch fully overwritten each substep; never
    /// persisted and never host-uploaded.
    Scratch,
    /// Read-only input refreshed by the host or an earlier pass each frame.
    Upload,
}

/// One storage buffer bound by the guide-`VBD` solver kernel (`hair_vbd.wesl`
/// `@group(0)`), in binding order `0..6`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairVbdBuffer {
    /// `@binding(0)` particle state `array<vec4<f32>>` (`xyz` = position,
    /// `w` = inverse mass, `0` = pinned). Byte-identical to the sim's
    /// `Positions`, so the same persistent allocation feeds either solver.
    Positions,
    /// `@binding(1)` previous positions `array<vec4<f32>>` (implicit velocity).
    PrevPositions,
    /// `@binding(2)` per-particle inertial-target scratch `array<vec4<f32>>`,
    /// the `GPU` twin of the CPU `targets` `Vec` rebuilt every substep.
    Targets,
    /// `@binding(3)` per-segment rest lengths `array<f32>`, flat across strands.
    RestLengths,
    /// `@binding(4)` per-strand flat offset descriptors `array<HairStrand>`
    /// (four `u32`, byte-identical to the sim descriptor).
    Strands,
    /// `@binding(5)` analytic body colliders `array<HairCollider>`
    /// (2 `vec4<f32>`).
    Colliders,
}

impl HairVbdBuffer {
    /// Every `VBD` buffer in `@binding` order, matching the six `@group(0)`
    /// bindings declared by `hair_vbd.wesl`.
    pub const ALL: [HairVbdBuffer; 6] = [
        Self::Positions,
        Self::PrevPositions,
        Self::Targets,
        Self::RestLengths,
        Self::Strands,
        Self::Colliders,
    ];

    /// The `@group(0)` binding index this buffer occupies in `hair_vbd.wesl`.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::Positions => 0,
            Self::PrevPositions => 1,
            Self::Targets => 2,
            Self::RestLengths => 3,
            Self::Strands => 4,
            Self::Colliders => 5,
        }
    }

    /// Byte stride of one element, matching the `WESL` struct / scalar layout.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::Positions | Self::PrevPositions | Self::Targets | Self::Strands => VEC4_STRIDE,
            Self::RestLengths => F32_STRIDE,
            // HairCollider = 2 * vec4<f32>.
            Self::Colliders => 2 * VEC4_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer.
    #[must_use]
    pub fn access(self) -> HairBufferAccess {
        match self {
            Self::Positions | Self::PrevPositions | Self::Targets => HairBufferAccess::ReadWrite,
            Self::RestLengths | Self::Strands | Self::Colliders => HairBufferAccess::Read,
        }
    }

    /// Where this buffer's contents live across the frame boundary.
    #[must_use]
    pub fn residency(self) -> HairBufferResidency {
        match self {
            Self::Positions | Self::PrevPositions => HairBufferResidency::Persistent,
            Self::Targets => HairBufferResidency::Scratch,
            Self::RestLengths | Self::Strands | Self::Colliders => HairBufferResidency::Upload,
        }
    }

    /// Number of elements this buffer holds for the given groom counts.
    ///
    /// `collider_count` is the analytic collider count
    /// (`HairVbdParams.collider_count`), independent of the [`HairGpuCounts`]
    /// element domains. Per-segment rest lengths total
    /// `guide_particles - guide_strands` (each strand of `n` particles has
    /// `n - 1` segments), saturating so degenerate inputs never underflow.
    #[must_use]
    pub fn element_count(self, counts: &HairGpuCounts, collider_count: u32) -> u32 {
        match self {
            Self::Positions | Self::PrevPositions | Self::Targets => counts.guide_particles,
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
/// double-buffered `positions` + `prev_positions`), shared byte-for-byte with
/// the `XPBD` sim's persistent allocation.
#[must_use]
pub fn persistent_state_bytes(counts: &HairGpuCounts, collider_count: u32) -> usize {
    residency_bytes(counts, collider_count, HairBufferResidency::Persistent)
}

/// Total bytes of transient `GPU`-only scratch (`targets`) the solver rewrites
/// every substep — allocated on device but never persisted or host-uploaded.
#[must_use]
pub fn scratch_bytes(counts: &HairGpuCounts, collider_count: u32) -> usize {
    residency_bytes(counts, collider_count, HairBufferResidency::Scratch)
}

/// Total bytes of the read-only per-frame inputs (`rest_lengths`, `strands`,
/// `colliders`) the host / earlier passes refresh each frame.
#[must_use]
pub fn upload_input_bytes(counts: &HairGpuCounts, collider_count: u32) -> usize {
    residency_bytes(counts, collider_count, HairBufferResidency::Upload)
}

fn residency_bytes(
    counts: &HairGpuCounts,
    collider_count: u32,
    residency: HairBufferResidency,
) -> usize {
    HairVbdBuffer::ALL
        .into_iter()
        .filter(|buffer| buffer.residency() == residency)
        .map(|buffer| buffer.byte_size(counts, collider_count))
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
        for (index, buffer) in HairVbdBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
    }

    #[test]
    fn strides_match_the_wesl_struct_layout() {
        assert_eq!(HairVbdBuffer::Positions.stride(), 16);
        assert_eq!(HairVbdBuffer::PrevPositions.stride(), 16);
        assert_eq!(HairVbdBuffer::Targets.stride(), 16);
        assert_eq!(HairVbdBuffer::RestLengths.stride(), 4);
        // Four-u32 HairStrand, byte-identical to the sim descriptor.
        assert_eq!(HairVbdBuffer::Strands.stride(), 16);
        assert_eq!(HairVbdBuffer::Colliders.stride(), 32);
    }

    #[test]
    fn access_modes_match_the_kernel() {
        assert_eq!(
            HairVbdBuffer::Positions.access(),
            HairBufferAccess::ReadWrite
        );
        assert_eq!(
            HairVbdBuffer::PrevPositions.access(),
            HairBufferAccess::ReadWrite
        );
        assert_eq!(HairVbdBuffer::Targets.access(), HairBufferAccess::ReadWrite);
        assert_eq!(HairVbdBuffer::RestLengths.access(), HairBufferAccess::Read);
        assert_eq!(HairVbdBuffer::Strands.access(), HairBufferAccess::Read);
        assert_eq!(HairVbdBuffer::Colliders.access(), HairBufferAccess::Read);
    }

    #[test]
    fn residency_splits_persist_scratch_and_upload() {
        assert_eq!(
            HairVbdBuffer::Positions.residency(),
            HairBufferResidency::Persistent
        );
        assert_eq!(
            HairVbdBuffer::PrevPositions.residency(),
            HairBufferResidency::Persistent
        );
        assert_eq!(
            HairVbdBuffer::Targets.residency(),
            HairBufferResidency::Scratch
        );
        assert_eq!(
            HairVbdBuffer::RestLengths.residency(),
            HairBufferResidency::Upload
        );
        assert_eq!(
            HairVbdBuffer::Strands.residency(),
            HairBufferResidency::Upload
        );
        assert_eq!(
            HairVbdBuffer::Colliders.residency(),
            HairBufferResidency::Upload
        );
    }

    #[test]
    fn positions_layout_matches_the_sim_twin() {
        // Seamless solver-slot swap: the persistent particle state is
        // byte-identical to hair_sim.wesl so the same allocation feeds either.
        use crate::hair::gpu_buffers::HairSimBuffer;
        let counts = sample_counts();
        assert_eq!(
            HairVbdBuffer::Positions.stride(),
            HairSimBuffer::Positions.stride()
        );
        assert_eq!(
            HairVbdBuffer::PrevPositions.stride(),
            HairSimBuffer::PrevPositions.stride()
        );
        assert_eq!(
            HairVbdBuffer::Strands.stride(),
            HairSimBuffer::Strands.stride()
        );
        assert_eq!(persistent_state_bytes(&counts, 8), 2 * 3200 * 16);
    }

    #[test]
    fn element_counts_follow_the_groom_domains() {
        let counts = sample_counts();
        assert_eq!(HairVbdBuffer::Positions.element_count(&counts, 8), 3200);
        assert_eq!(HairVbdBuffer::Targets.element_count(&counts, 8), 3200);
        // 3200 particles across 100 strands -> 3100 segments.
        assert_eq!(HairVbdBuffer::RestLengths.element_count(&counts, 8), 3100);
        assert_eq!(HairVbdBuffer::Strands.element_count(&counts, 8), 100);
        assert_eq!(HairVbdBuffer::Colliders.element_count(&counts, 8), 8);
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
        assert_eq!(HairVbdBuffer::RestLengths.element_count(&counts, 0), 0);
    }

    #[test]
    fn byte_sizes_multiply_count_by_stride() {
        let counts = sample_counts();
        assert_eq!(HairVbdBuffer::Positions.byte_size(&counts, 8), 3200 * 16);
        assert_eq!(HairVbdBuffer::RestLengths.byte_size(&counts, 8), 3100 * 4);
        assert_eq!(HairVbdBuffer::Colliders.byte_size(&counts, 8), 8 * 32);
    }

    #[test]
    fn empty_groom_clamps_every_buffer_to_one_element() {
        let counts = HairGpuCounts::default();
        for buffer in HairVbdBuffer::ALL {
            assert_eq!(buffer.byte_size(&counts, 0), buffer.stride());
        }
    }

    #[test]
    fn residency_buckets_partition_the_buffers() {
        let counts = sample_counts();
        // positions + prev_positions.
        assert_eq!(persistent_state_bytes(&counts, 8), 2 * 3200 * 16);
        // targets scratch alone.
        assert_eq!(scratch_bytes(&counts, 8), 3200 * 16);
        // rest_lengths + strands + colliders.
        let expected = 3100 * 4 + 100 * 16 + 8 * 32;
        assert_eq!(upload_input_bytes(&counts, 8), expected);
    }

    #[test]
    fn params_immediate_block_is_forty_eight_bytes() {
        assert_eq!(PARAMS_IMMEDIATE_BYTES, 48);
    }
}
