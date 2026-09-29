//! Deterministic `GPU` compute-dispatch contract for the particle §9 pipeline.
//!
//! Each particle compute stage has a `WESL` twin whose `global_invocation_id.x`
//! indexes one element of a domain (an emitter, a particle slot, an emitted
//! event, ...). Turning a frame into actual dispatches means, per pass: take the
//! element count of its domain and divide by that pass's `@workgroup_size` to
//! get a 1-D workgroup count, then issue the passes in the fixed §9 order.
//!
//! Some counts are known on the `CPU` (emitter count, pool capacity, the single
//! group of the draw-args fill), so those passes dispatch directly. Others
//! depend on a count only the `GPU` knows at that point in the frame (how many
//! particles survived, how many events were emitted), so those passes read an
//! on-device `DispatchIndirectArgs` record and issue `dispatchWorkgroupsIndirect`
//! instead. [`ParticleComputePass::is_indirect`] draws that line and
//! [`DISPATCH_INDIRECT_STRIDE`] fixes the `std430` byte layout of the indirect
//! record the fill pass writes.
//!
//! The scene/render crate owns the `wgpu` buffers, bind groups and queue; this
//! zero-dependency module owns only the *contract* — the authoritative pass
//! order, per-pass workgroup size, direct/indirect classification, and the
//! ceil-division from element count to workgroup count — so the render graph
//! binds against a stable `ABI` instead of duplicating the derivation next to
//! the pipeline. It is the particle sibling of `hair`'s dispatch contract and is
//! orthogonal to the per-pass `SoA` buffer layouts published by the
//! `*_pass_buffers.rs` files.
//!
//! Everything is pure integer arithmetic and panic-free: an empty domain yields
//! a zero workgroup count, and a degenerate zero workgroup size yields zero
//! rather than dividing by zero.

use crate::particle::gpu_layout::U32_STRIDE;

/// `std430` byte stride of a `WebGPU` `DispatchIndirectArgs` record: the three
/// `u32` workgroup counts `x`, `y`, `z`, so `3 * 4 = 12` bytes. This is the
/// layout the fill pass writes and the indirect passes dispatch from.
pub const DISPATCH_INDIRECT_STRIDE: usize = 3 * U32_STRIDE;

/// The default 1-D `@workgroup_size` shared by the per-element particle kernels
/// (`emitter update`, `spawn`, `simulate`, `event scatter`, `compaction`,
/// `bounds`, `cull`, and the tiny draw-args fill). One thread per domain
/// element along `x`.
pub const WORKGROUP_SIZE_1D: u32 = 64;

/// The wider `@workgroup_size` used by the `sort` pass, whose bitonic-style
/// compare/exchange network shares more work within a group than the simple
/// per-element kernels.
pub const WORKGROUP_SIZE_SORT: u32 = 256;

/// One particle compute pass of the §9 pipeline: a `WESL` kernel plus the
/// metadata needed to order and dispatch it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParticleComputePass {
    /// Advance per-emitter state and accumulate the spawn credit for this frame.
    /// Domain is the emitter set, known on the `CPU`.
    EmitterUpdate,
    /// Allocate and initialize newly spawned particles from the emitter credit.
    /// Dispatched over the emitter set (`CPU`-known).
    Spawn,
    /// Integrate forces and advance every live particle for one step.
    /// Domain is the live particle set, decided on the `GPU`.
    Simulate,
    /// Scatter simulation-emitted events (collisions, deaths, sub-emitter
    /// triggers) into their queues. Domain is the emitted-event count, decided
    /// on the `GPU`.
    EventScatter,
    /// Compact the live particles to the front of the pool, reclaiming dead
    /// slots. Scans the whole pool `capacity` (`CPU`-known).
    Compaction,
    /// Reduce the live particles into a bounding volume for culling/sorting.
    /// Domain is the live particle set (`GPU`-decided).
    Bounds,
    /// Frustum/visibility cull the live particles, producing the survivor set.
    /// Domain is the live particle set (`GPU`-decided).
    Cull,
    /// Sort the surviving particles (e.g. back-to-front for blending). Domain is
    /// the survivor set (`GPU`-decided).
    Sort,
    /// Fill the indirect draw-args buffer from the survivor count. A single
    /// workgroup; its dispatch dimension is the `CPU`-known constant `1`.
    FillDrawArgs,
}

impl ParticleComputePass {
    /// Every particle compute pass, in canonical §9 pipeline order. Its length
    /// equals the number of `WESL` twins and lets callers enumerate the full
    /// pass set without hand-listing variants.
    pub const ALL: [ParticleComputePass; 9] = [
        Self::EmitterUpdate,
        Self::Spawn,
        Self::Simulate,
        Self::EventScatter,
        Self::Compaction,
        Self::Bounds,
        Self::Cull,
        Self::Sort,
        Self::FillDrawArgs,
    ];

    /// The pass's position in the fixed pipeline order, `0..9`, increasing and
    /// contiguous. Matches this pass's index in [`ParticleComputePass::ALL`].
    #[must_use]
    pub fn order(self) -> u32 {
        match self {
            Self::EmitterUpdate => 0,
            Self::Spawn => 1,
            Self::Simulate => 2,
            Self::EventScatter => 3,
            Self::Compaction => 4,
            Self::Bounds => 5,
            Self::Cull => 6,
            Self::Sort => 7,
            Self::FillDrawArgs => 8,
        }
    }

    /// The 1-D `@workgroup_size` this pass's `WESL` kernel declares. Every pass
    /// uses [`WORKGROUP_SIZE_1D`] except `Sort`, which uses the wider
    /// [`WORKGROUP_SIZE_SORT`]. Always non-zero.
    #[must_use]
    pub fn workgroup_size(self) -> u32 {
        match self {
            Self::Sort => WORKGROUP_SIZE_SORT,
            Self::EmitterUpdate
            | Self::Spawn
            | Self::Simulate
            | Self::EventScatter
            | Self::Compaction
            | Self::Bounds
            | Self::Cull
            | Self::FillDrawArgs => WORKGROUP_SIZE_1D,
        }
    }

    /// Whether this pass dispatches indirectly from a `GPU`-computed count.
    ///
    /// `true` for the passes whose dispatch dimension is only known on-device at
    /// that point in the frame — the live particle set (`Simulate`, `Bounds`,
    /// `Cull`), the survivor set (`Sort`), and the emitted-event count
    /// (`EventScatter`). `false` for the passes sized by a `CPU`-known count:
    /// the emitter set (`EmitterUpdate`, `Spawn`), the pool `capacity`
    /// (`Compaction`), and the single-group `FillDrawArgs`.
    #[must_use]
    pub fn is_indirect(self) -> bool {
        matches!(
            self,
            Self::Simulate | Self::EventScatter | Self::Bounds | Self::Cull | Self::Sort
        )
    }
}

/// Workgroups needed to cover `element_count` elements at `workgroup_size` per
/// group: `ceil(element_count / workgroup_size)`.
///
/// Follows the `hair` dispatch semantics: an empty domain needs `0` groups, and
/// a degenerate zero `workgroup_size` returns `0` rather than dividing by zero
/// (it is treated as "no valid group size, nothing to dispatch"). Never panics
/// and never overflows — `u32::div_ceil` computes the ceiling without the
/// `(n + d - 1)` addition that could wrap.
#[must_use]
pub fn workgroup_count(element_count: u32, workgroup_size: u32) -> u32 {
    if workgroup_size == 0 {
        return 0;
    }
    element_count.div_ceil(workgroup_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_has_every_pass_in_order() {
        assert_eq!(ParticleComputePass::ALL.len(), 9);
    }

    #[test]
    fn order_is_unique_contiguous_from_zero() {
        for (index, pass) in ParticleComputePass::ALL.into_iter().enumerate() {
            assert_eq!(pass.order() as usize, index);
        }
    }

    #[test]
    fn workgroup_sizes_are_non_zero() {
        for pass in ParticleComputePass::ALL {
            assert!(pass.workgroup_size() > 0);
        }
    }

    #[test]
    fn sort_uses_the_wider_workgroup_size() {
        assert_eq!(ParticleComputePass::Sort.workgroup_size(), 256);
        assert_eq!(ParticleComputePass::Simulate.workgroup_size(), 64);
        assert_eq!(ParticleComputePass::FillDrawArgs.workgroup_size(), 64);
    }

    #[test]
    fn is_indirect_classifies_gpu_and_cpu_sized_passes() {
        // GPU-decided counts dispatch indirectly.
        assert!(ParticleComputePass::Simulate.is_indirect());
        assert!(ParticleComputePass::EventScatter.is_indirect());
        assert!(ParticleComputePass::Bounds.is_indirect());
        assert!(ParticleComputePass::Cull.is_indirect());
        assert!(ParticleComputePass::Sort.is_indirect());
        // CPU-known counts dispatch directly.
        assert!(!ParticleComputePass::EmitterUpdate.is_indirect());
        assert!(!ParticleComputePass::Spawn.is_indirect());
        assert!(!ParticleComputePass::Compaction.is_indirect());
        assert!(!ParticleComputePass::FillDrawArgs.is_indirect());
    }

    #[test]
    fn dispatch_indirect_stride_is_three_u32() {
        assert_eq!(DISPATCH_INDIRECT_STRIDE, 12);
    }

    #[test]
    fn workgroup_count_is_ceiling_division() {
        // Empty domain -> zero groups.
        assert_eq!(workgroup_count(0, 64), 0);
        // Exact multiples divide evenly.
        assert_eq!(workgroup_count(64, 64), 1);
        assert_eq!(workgroup_count(128, 64), 2);
        // Remainders round up.
        assert_eq!(workgroup_count(1, 64), 1);
        assert_eq!(workgroup_count(65, 64), 2);
        assert_eq!(workgroup_count(129, 64), 3);
        // Wider sort group size.
        assert_eq!(workgroup_count(256, 256), 1);
        assert_eq!(workgroup_count(257, 256), 2);
    }

    #[test]
    fn workgroup_count_zero_size_is_safe() {
        // Zero workgroup size must not panic or divide by zero.
        assert_eq!(workgroup_count(1000, 0), 0);
        assert_eq!(workgroup_count(0, 0), 0);
    }

    #[test]
    fn workgroup_count_does_not_overflow_near_u32_max() {
        // ceil(u32::MAX / 64) without the (n + d - 1) wrap.
        assert_eq!(workgroup_count(u32::MAX, 64), (u32::MAX / 64) + 1);
    }
}
