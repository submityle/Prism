//! Cross-pass memory-barrier plan for the persistent guide-position state.
//!
//! [`frame_schedule`](super::frame_schedule) orders the per-frame compute
//! passes and [`frame_pass_layout`](super::frame_pass_layout) sizes each pass's
//! bindings, but neither answers *where the render graph must insert a
//! compute-to-compute storage memory barrier* so one pass's writes are visible
//! to the next. Missing such a barrier between two passes that touch the same
//! `var<storage, read_write>` buffer is a data race that silently corrupts the
//! solve; inserting one everywhere is correct but needlessly serializes work.
//! This module computes exactly the required set.
//!
//! Only buffers shared across more than one per-frame pass can create a
//! cross-pass hazard. Auditing the buffer-contract modules shows the persistent
//! guide particle positions is the sole such buffer in the per-frame schedule:
//! - `Wind` ([`HairWindBuffer::Positions`](super::sim_pass_buffers)) and
//!   `SdfCollision` ([`HairSdfCollisionBuffer`](super::sim_pass_buffers)) alias
//!   the persistent
//!   [`HairSimBuffer::Positions`](super::gpu_buffers::HairSimBuffer::Positions)
//!   in place;
//! - the guide solver `GuideSim` ([`HairSimBuffer`](super::gpu_buffers)) or its
//!   `VbdSolve` alternative
//!   ([`HairVbdBuffer::Positions`](super::vbd_pass_buffers)) read-writes the
//!   same allocation;
//! - the self-collision `SelfCollisionAccumulate` / `SelfCollisionApply` pair
//!   ([`HairSelfCollisionBuffer::Positions`](super::self_collision_pass_buffers))
//!   likewise read-writes it in place;
//! - `Interpolate` reads it as its guide-point pool
//!   ([`HairInterpBuffer::GuidePoints`](super::interp_buffers)).
//!
//! Every other buffer each pass binds is private to that pass within the frame
//! (root frames for `RootSkinning`, the `SDF` primitive list for
//! `SdfCollision`, the self-collision grid, `Interpolate`'s output points,
//! `LodDither`'s mask, the two self-shadow passes' independent sample pools),
//! so none of them adds a cross-pass hazard. The plan is therefore driven
//! entirely by the ordered sequence of accesses to the guide-position resource.
//!
//! The plan is structural and count-independent: it is computed from the
//! schedule shape alone, so the render graph builds the barrier list once per
//! [`HairFrameConfig`](super::frame_schedule::HairFrameConfig) and reuses it
//! every frame. Everything is pure and deterministic and nothing panics.

use alloc::vec::Vec;

use crate::hair::frame_schedule::{per_frame_schedule, HairFrameConfig, HairScheduledPass};
use crate::hair::gpu_buffers::HairBufferAccess;
use crate::hair::gpu_dispatch::HairComputePass;
use crate::hair::optional_pass_dispatch::HairOptionalPass;

/// A logical `GPU` resource that flows across more than one per-frame hair pass,
/// so a compute memory barrier may be required between the passes that touch
/// it. Pass-private buffers are deliberately excluded — they never create a
/// cross-pass hazard.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairFrameResource {
    /// The persistent guide particle positions buffer
    /// ([`HairSimBuffer::Positions`](super::gpu_buffers::HairSimBuffer::Positions)),
    /// aliased in place by `Wind`, the guide solver (`GuideSim` / `VbdSolve`),
    /// `SdfCollision` and the self-collision pair, and read by `Interpolate` as
    /// its guide-point pool.
    GuidePositions,
}

/// The ordering hazard a barrier resolves between two consecutive passes that
/// touch the same resource. Because [`HairBufferAccess`] is binary
/// (`Read` / `ReadWrite`, i.e. no write-only access), only these two hazards
/// arise; a read-write-after-read-write pair carries both a read-after-write
/// and a write-after-write dependency, resolved by the same storage barrier and
/// labelled [`ReadAfterWrite`](Self::ReadAfterWrite) since the stale read is the
/// dependency that drives correctness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairBarrierHazard {
    /// The next pass reads (or reads and writes) a buffer the previous pass
    /// wrote: its reads must observe those writes.
    ReadAfterWrite,
    /// The next pass writes a buffer the previous pass only read: the write must
    /// not race the previous pass's reads.
    WriteAfterRead,
}

/// One required compute memory barrier between two passes of the frame
/// schedule. The render graph issues the barrier for `resource` immediately
/// before dispatching the pass at `before_pass_index`, ordering it after the
/// pass at `after_pass_index` (the most recent prior pass that touched the
/// resource). Indices are into the ordered schedule
/// [`per_frame_schedule`](super::frame_schedule::per_frame_schedule) produces
/// for the same [`HairFrameConfig`](super::frame_schedule::HairFrameConfig).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HairPassBarrier {
    /// Schedule index of the pass that must observe the prior access; the
    /// barrier is issued just before this pass's dispatch.
    pub before_pass_index: usize,
    /// Schedule index of the most recent prior pass that touched `resource`.
    pub after_pass_index: usize,
    /// The shared resource whose accesses this barrier orders.
    pub resource: HairFrameResource,
    /// Which ordering hazard the barrier resolves.
    pub hazard: HairBarrierHazard,
}

/// How a scheduled pass accesses the shared [`HairFrameResource::GuidePositions`]
/// resource, or `None` when the pass does not touch it. Grounded in the
/// buffer-contract modules: the wind, guide-solver (`GuideSim` / `VbdSolve`),
/// `SDF` collision and self-collision passes read-write the persistent
/// positions in place, `Interpolate` reads them, and every other pass leaves
/// them untouched.
#[must_use]
pub fn guide_positions_access(pass: HairScheduledPass) -> Option<HairBufferAccess> {
    match pass {
        HairScheduledPass::Main(HairComputePass::Wind)
        | HairScheduledPass::Main(HairComputePass::GuideSim)
        | HairScheduledPass::Main(HairComputePass::SdfCollision)
        | HairScheduledPass::Optional(HairOptionalPass::VbdSolve)
        | HairScheduledPass::Optional(HairOptionalPass::SelfCollisionAccumulate)
        | HairScheduledPass::Optional(HairOptionalPass::SelfCollisionApply) => {
            Some(HairBufferAccess::ReadWrite)
        }
        HairScheduledPass::Main(HairComputePass::Interpolate) => Some(HairBufferAccess::Read),
        HairScheduledPass::Main(
            HairComputePass::RootBind
            | HairComputePass::Resample
            | HairComputePass::RootSkinning
            | HairComputePass::LodDither
            | HairComputePass::Transmittance
            | HairComputePass::DeepOpacity,
        ) => None,
    }
}

/// Classifies the barrier hazard between a previous access and the next access
/// to the same resource, or `None` when both are read-only (no barrier needed).
#[must_use]
fn hazard_between(previous: HairBufferAccess, next: HairBufferAccess) -> Option<HairBarrierHazard> {
    match (previous, next) {
        (HairBufferAccess::Read, HairBufferAccess::Read) => None,
        (HairBufferAccess::Read, HairBufferAccess::ReadWrite) => {
            Some(HairBarrierHazard::WriteAfterRead)
        }
        (HairBufferAccess::ReadWrite, _) => Some(HairBarrierHazard::ReadAfterWrite),
    }
}

/// Computes the ordered set of compute memory barriers required on
/// [`HairFrameResource::GuidePositions`] for the per-frame schedule of `config`,
/// appending to `out` (which is cleared first). Walks the structural schedule,
/// tracking the most recent pass that touched the resource, and emits a barrier
/// before each subsequent touching pass whose access forms a hazard with it.
/// Read-after-read transitions emit nothing. Never panics.
pub fn plan_frame_barriers(config: HairFrameConfig, out: &mut Vec<HairPassBarrier>) {
    out.clear();
    let mut schedule = Vec::new();
    per_frame_schedule(config, &mut schedule);
    let mut previous: Option<(usize, HairBufferAccess)> = None;
    for (index, pass) in schedule.iter().enumerate() {
        let Some(access) = guide_positions_access(*pass) else {
            continue;
        };
        if let Some((previous_index, previous_access)) = previous
            && let Some(hazard) = hazard_between(previous_access, access)
        {
            out.push(HairPassBarrier {
                before_pass_index: index,
                after_pass_index: previous_index,
                resource: HairFrameResource::GuidePositions,
                hazard,
            });
        }
        previous = Some((index, access));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::solver::HairSolverKind;

    fn plan(config: HairFrameConfig) -> Vec<HairPassBarrier> {
        let mut out = Vec::new();
        plan_frame_barriers(config, &mut out);
        out
    }

    #[test]
    fn default_schedule_barriers_chain_the_three_position_writers_to_interpolate() {
        // Schedule: RootSkinning(0), Wind(1), GuideSim(2), SdfCollision(3),
        // Interpolate(4), LodDither(5). Touchers: 1,2,3 (RW) then 4 (R).
        let barriers = plan(HairFrameConfig::default());
        assert_eq!(barriers.len(), 3);
        let expected = [
            (2usize, 1usize, HairBarrierHazard::ReadAfterWrite),
            (3, 2, HairBarrierHazard::ReadAfterWrite),
            (4, 3, HairBarrierHazard::ReadAfterWrite),
        ];
        for (barrier, &(before, after, hazard)) in barriers.iter().zip(expected.iter()) {
            assert_eq!(barrier.before_pass_index, before);
            assert_eq!(barrier.after_pass_index, after);
            assert_eq!(barrier.hazard, hazard);
            assert_eq!(barrier.resource, HairFrameResource::GuidePositions);
        }
    }

    #[test]
    fn vbd_solver_barrier_count_matches_the_xpbd_spine() {
        // VbdSolve replaces GuideSim in the same slot and read-writes the same
        // persistent positions, so the barrier structure is identical.
        let config = HairFrameConfig {
            solver: HairSolverKind::Vbd,
            self_collision: false,
        };
        assert_eq!(plan(config).len(), 3);
    }

    #[test]
    fn self_collision_inserts_two_more_barriers() {
        // Schedule gains SelfCollisionAccumulate + SelfCollisionApply after
        // SdfCollision, both RW on positions, extending the chain to five.
        let config = HairFrameConfig {
            solver: HairSolverKind::Xpbd,
            self_collision: true,
        };
        let barriers = plan(config);
        assert_eq!(barriers.len(), 5);
        // Every hazard in the all-read-write chain except the final read is a
        // read-after-write; the last (into Interpolate) is too.
        for barrier in &barriers {
            assert_eq!(barrier.hazard, HairBarrierHazard::ReadAfterWrite);
        }
        // Indices are strictly increasing and each barrier orders adjacent
        // touchers (after_pass_index precedes before_pass_index).
        for barrier in &barriers {
            assert!(barrier.after_pass_index < barrier.before_pass_index);
        }
        let mut last_before = 0usize;
        for barrier in &barriers {
            assert!(barrier.before_pass_index > last_before);
            last_before = barrier.before_pass_index;
        }
    }

    #[test]
    fn read_after_read_needs_no_barrier() {
        assert_eq!(
            hazard_between(HairBufferAccess::Read, HairBufferAccess::Read),
            None
        );
    }

    #[test]
    fn write_after_read_is_classified() {
        assert_eq!(
            hazard_between(HairBufferAccess::Read, HairBufferAccess::ReadWrite),
            Some(HairBarrierHazard::WriteAfterRead)
        );
    }

    #[test]
    fn any_prior_write_is_a_read_after_write() {
        assert_eq!(
            hazard_between(HairBufferAccess::ReadWrite, HairBufferAccess::Read),
            Some(HairBarrierHazard::ReadAfterWrite)
        );
        assert_eq!(
            hazard_between(HairBufferAccess::ReadWrite, HairBufferAccess::ReadWrite),
            Some(HairBarrierHazard::ReadAfterWrite)
        );
    }

    #[test]
    fn non_touching_passes_report_no_access() {
        assert_eq!(
            guide_positions_access(HairScheduledPass::Main(HairComputePass::RootSkinning)),
            None
        );
        assert_eq!(
            guide_positions_access(HairScheduledPass::Main(HairComputePass::LodDither)),
            None
        );
        assert_eq!(
            guide_positions_access(HairScheduledPass::Main(HairComputePass::Transmittance)),
            None
        );
    }

    #[test]
    fn interpolate_reads_and_solvers_read_write() {
        assert_eq!(
            guide_positions_access(HairScheduledPass::Main(HairComputePass::Interpolate)),
            Some(HairBufferAccess::Read)
        );
        assert_eq!(
            guide_positions_access(HairScheduledPass::Main(HairComputePass::GuideSim)),
            Some(HairBufferAccess::ReadWrite)
        );
        assert_eq!(
            guide_positions_access(HairScheduledPass::Optional(HairOptionalPass::VbdSolve)),
            Some(HairBufferAccess::ReadWrite)
        );
    }

    #[test]
    fn barrier_before_indices_are_valid_schedule_positions() {
        for config in [
            HairFrameConfig::default(),
            HairFrameConfig {
                solver: HairSolverKind::Vbd,
                self_collision: true,
            },
        ] {
            let mut schedule = Vec::new();
            per_frame_schedule(config, &mut schedule);
            for barrier in plan(config) {
                assert!(barrier.before_pass_index < schedule.len());
                assert!(barrier.after_pass_index < schedule.len());
            }
        }
    }
}
