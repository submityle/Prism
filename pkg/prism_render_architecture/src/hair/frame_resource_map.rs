//! Frame-level logical resource identity and residency classification: the
//! allocation-correctness contract the render graph consults when it turns the
//! per-frame schedule ([`per_frame_schedule`]) into concrete physical buffers.
//!
//! The single-pass buffer ABIs (`sim_pass_buffers`, `interp_buffers`,
//! `vbd_pass_buffers`, `self_collision_pass_buffers`, ...) each describe one
//! kernel's `@group(0)` bindings in isolation. But several of those per-pass
//! bindings are *the same physical buffer viewed from different passes*: the
//! persistent guide-position state is bound (and mutated in place) by `Wind`,
//! `GuideSim`/`VbdSolve`, `SdfCollision`, the self-collision pair, and then read
//! by `Interpolate`. If the render graph allocated one buffer per
//! (pass, binding) it would give each of those views its own storage and the
//! guide solve would silently lose its integration between passes.
//!
//! This module is the authority on *which logical allocation each per-frame
//! binding maps to*, so the graph can de-alias: bindings that resolve to the
//! same [`HairFrameAllocation`] must share one physical buffer. It also
//! classifies every allocation by [`HairAllocationResidency`] so the graph
//! knows what must persist across frames (guide state, render output), what is
//! transient per-frame scratch, and what is host-uploaded input.
//!
//! Scope is exactly the *per-frame* schedule (the guide simulate + interpolate
//! spine plus its optional twins). The one-time import passes (`RootBind`,
//! `Resample`) and the shadow passes (`Transmittance`, `DeepOpacity`) are not
//! part of a frame's persistent-state aliasing and resolve to [`None`] here.
//!
//! Everything is deterministic, pure-integer, allocation-free per query, and
//! never panics: out-of-range bindings and out-of-scope passes yield [`None`].

use alloc::vec::Vec;

use crate::hair::frame_schedule::{per_frame_schedule, HairFrameConfig, HairScheduledPass};
use crate::hair::gpu_dispatch::HairComputePass;
use crate::hair::optional_pass_dispatch::HairOptionalPass;

/// How a [`HairFrameAllocation`] lives across the frame timeline, telling the
/// render graph how to back it with memory.
///
/// This is the *frame-allocation* view. It is intentionally distinct from the
/// per-pass `HairBufferResidency` enums declared locally by the optional buffer
/// modules (which classify one kernel's bindings): those stay decoupled and are
/// normalised onto the canonical access enum elsewhere. This enum instead
/// classifies the deduplicated whole-frame allocation set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairAllocationResidency {
    /// Read-write state that must survive between frames (and be ring-buffered
    /// when frames overlap): the guide particle state the solver integrates in
    /// place and the render control points a later pass consumes.
    Persistent,
    /// Transient scratch produced and consumed within a single frame; its
    /// contents need not survive to the next frame.
    Scratch,
    /// Read-only input the host (or a one-time import) uploads; the frame reads
    /// it but never writes it.
    Upload,
}

/// A distinct logical buffer the per-frame schedule needs, deduplicated across
/// every pass and binding that aliases it.
///
/// Two per-frame `(pass, binding)` views that resolve to the same variant *must*
/// be backed by one physical buffer. In particular [`GuidePositions`] is the
/// persistent guide state shared by wind, the guide solve, collision, the
/// self-collision pair, and interpolation.
///
/// [`GuidePositions`]: HairFrameAllocation::GuidePositions
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairFrameAllocation {
    // ---- Persistent guide/render state ----
    /// Persistent guide particle state (`xyz` = position, `w` = inverse mass).
    /// Aliased by wind, the guide solve, `SDF` collision, self-collision, and
    /// read by interpolation.
    GuidePositions,
    /// Persistent previous guide positions (implicit velocity), integrated by
    /// the guide solve.
    GuidePrevPositions,
    /// Persistent render-strand control points produced by interpolation and
    /// consumed downstream (raster / shadows).
    RenderOutPoints,

    // ---- Per-frame scratch ----
    /// Per-root resolved world frames produced by root skinning.
    RootFrames,
    /// Per-particle inertial-target scratch rebuilt every `VBD` substep.
    VbdTargets,
    /// Per-particle accumulated self-collision correction deltas.
    SelfCollisionCorrections,
    /// Per-render-strand keep mask produced by the `LOD` dither pass.
    LodKeepMask,

    // ---- Host-uploaded input ----
    /// Per-root scalp attachment descriptors seeding root skinning.
    RootBindings,
    /// Deformed scalp vertex positions.
    ScalpVertices,
    /// Flat scalp triangle index list.
    ScalpIndices,
    /// Per-particle global goal pose read by the `XPBD` guide solve.
    GuideGoals,
    /// Per-segment rest lengths, flat across strands.
    GuideRestLengths,
    /// Per-strand flat offset descriptors.
    GuideStrands,
    /// Analytic body colliders.
    BodyColliders,
    /// `SDF` collider primitives.
    SdfPrimitives,
    /// Per-guide slice descriptors consumed by interpolation.
    GuideRanges,
    /// Per-render-strand binding table consumed by interpolation.
    RenderBindings,
    /// `CSR` occupied cell coordinates for self-collision.
    SelfCollisionCellKeys,
    /// `CSR` prefix-sum offsets for self-collision.
    SelfCollisionCellStarts,
    /// `CSR` flat bucket particle indices for self-collision.
    SelfCollisionCellIndices,
}

impl HairFrameAllocation {
    /// How this allocation must be backed across the frame timeline.
    #[must_use]
    pub fn residency(self) -> HairAllocationResidency {
        match self {
            Self::GuidePositions | Self::GuidePrevPositions | Self::RenderOutPoints => {
                HairAllocationResidency::Persistent
            }
            Self::RootFrames
            | Self::VbdTargets
            | Self::SelfCollisionCorrections
            | Self::LodKeepMask => HairAllocationResidency::Scratch,
            Self::RootBindings
            | Self::ScalpVertices
            | Self::ScalpIndices
            | Self::GuideGoals
            | Self::GuideRestLengths
            | Self::GuideStrands
            | Self::BodyColliders
            | Self::SdfPrimitives
            | Self::GuideRanges
            | Self::RenderBindings
            | Self::SelfCollisionCellKeys
            | Self::SelfCollisionCellStarts
            | Self::SelfCollisionCellIndices => HairAllocationResidency::Upload,
        }
    }
}

/// Resolves one per-frame `(pass, binding)` view to the logical allocation it
/// aliases, or [`None`] when the binding is out of range for the pass or the
/// pass is outside the per-frame schedule's scope (import / shadow passes).
///
/// The mapping mirrors each pass's single-pass buffer ABI in `@binding` order.
/// It never panics.
#[must_use]
pub fn allocation_of(pass: HairScheduledPass, binding: u32) -> Option<HairFrameAllocation> {
    match pass {
        HairScheduledPass::Main(main) => main_allocation(main, binding),
        HairScheduledPass::Optional(optional) => optional_allocation(optional, binding),
    }
}

/// Per-frame fixed-spine binding map. Out-of-scope import/shadow passes and
/// out-of-range bindings resolve to [`None`].
fn main_allocation(pass: HairComputePass, binding: u32) -> Option<HairFrameAllocation> {
    match pass {
        HairComputePass::RootSkinning => match binding {
            0 => Some(HairFrameAllocation::RootBindings),
            1 => Some(HairFrameAllocation::ScalpVertices),
            2 => Some(HairFrameAllocation::ScalpIndices),
            3 => Some(HairFrameAllocation::RootFrames),
            _ => None,
        },
        HairComputePass::Wind => match binding {
            0 => Some(HairFrameAllocation::GuidePositions),
            _ => None,
        },
        HairComputePass::GuideSim => match binding {
            0 => Some(HairFrameAllocation::GuidePositions),
            1 => Some(HairFrameAllocation::GuidePrevPositions),
            2 => Some(HairFrameAllocation::GuideGoals),
            3 => Some(HairFrameAllocation::GuideRestLengths),
            4 => Some(HairFrameAllocation::GuideStrands),
            5 => Some(HairFrameAllocation::BodyColliders),
            _ => None,
        },
        HairComputePass::SdfCollision => match binding {
            0 => Some(HairFrameAllocation::GuidePositions),
            1 => Some(HairFrameAllocation::SdfPrimitives),
            _ => None,
        },
        HairComputePass::Interpolate => match binding {
            0 => Some(HairFrameAllocation::GuidePositions),
            1 => Some(HairFrameAllocation::GuideRanges),
            2 => Some(HairFrameAllocation::RenderBindings),
            3 => Some(HairFrameAllocation::RenderOutPoints),
            _ => None,
        },
        HairComputePass::LodDither => match binding {
            0 => Some(HairFrameAllocation::LodKeepMask),
            _ => None,
        },
        // Import (one-time) and shadow passes are outside the per-frame
        // persistent-state aliasing scope.
        HairComputePass::RootBind
        | HairComputePass::Resample
        | HairComputePass::Transmittance
        | HairComputePass::DeepOpacity => None,
    }
}

/// Per-frame optional binding map (alternative solver + self-collision twins).
/// Out-of-range bindings resolve to [`None`].
fn optional_allocation(pass: HairOptionalPass, binding: u32) -> Option<HairFrameAllocation> {
    match pass {
        HairOptionalPass::VbdSolve => match binding {
            0 => Some(HairFrameAllocation::GuidePositions),
            1 => Some(HairFrameAllocation::GuidePrevPositions),
            2 => Some(HairFrameAllocation::VbdTargets),
            3 => Some(HairFrameAllocation::GuideRestLengths),
            4 => Some(HairFrameAllocation::GuideStrands),
            5 => Some(HairFrameAllocation::BodyColliders),
            _ => None,
        },
        HairOptionalPass::SelfCollisionAccumulate | HairOptionalPass::SelfCollisionApply => {
            match binding {
                0 => Some(HairFrameAllocation::GuidePositions),
                1 => Some(HairFrameAllocation::SelfCollisionCorrections),
                2 => Some(HairFrameAllocation::SelfCollisionCellKeys),
                3 => Some(HairFrameAllocation::SelfCollisionCellStarts),
                4 => Some(HairFrameAllocation::SelfCollisionCellIndices),
                _ => None,
            }
        }
    }
}

/// Walks the per-frame schedule for `config` and collects the deduplicated set
/// of logical allocations it needs, appending to `out` (cleared first).
///
/// Every `(pass, binding)` in the schedule is resolved through
/// [`allocation_of`]; the first appearance of each distinct allocation is kept
/// in schedule order, so aliased views (e.g. guide positions across many
/// passes) collapse to a single entry. The result is the exact set of physical
/// buffers the render graph must allocate for the frame. Deterministic; never
/// panics.
pub fn plan_frame_allocations(config: HairFrameConfig, out: &mut Vec<HairFrameAllocation>) {
    out.clear();
    let mut schedule = Vec::new();
    per_frame_schedule(config, &mut schedule);
    for pass in schedule {
        for binding in 0..pass.binding_count() {
            if let Some(alloc) = allocation_of(pass, binding)
                && !out.contains(&alloc)
            {
                out.push(alloc);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::solver::HairSolverKind;

    #[test]
    fn guide_positions_alias_resolves_identically_across_passes() {
        // Every per-frame view of the persistent guide state must resolve to the
        // one shared allocation, or the render graph would split it.
        let views = [
            HairScheduledPass::Main(HairComputePass::Wind),
            HairScheduledPass::Main(HairComputePass::GuideSim),
            HairScheduledPass::Main(HairComputePass::SdfCollision),
            HairScheduledPass::Main(HairComputePass::Interpolate),
            HairScheduledPass::Optional(HairOptionalPass::VbdSolve),
            HairScheduledPass::Optional(HairOptionalPass::SelfCollisionAccumulate),
            HairScheduledPass::Optional(HairOptionalPass::SelfCollisionApply),
        ];
        for pass in views {
            assert_eq!(
                allocation_of(pass, 0),
                Some(HairFrameAllocation::GuidePositions),
                "{pass:?} binding 0 should alias guide positions",
            );
        }
    }

    #[test]
    fn default_schedule_allocation_set_is_expected_and_deduplicated() {
        let mut out = Vec::new();
        plan_frame_allocations(HairFrameConfig::default(), &mut out);
        // First-seen order across RootSkinning, Wind, GuideSim, SdfCollision,
        // Interpolate, LodDither.
        assert_eq!(
            out,
            [
                HairFrameAllocation::RootBindings,
                HairFrameAllocation::ScalpVertices,
                HairFrameAllocation::ScalpIndices,
                HairFrameAllocation::RootFrames,
                HairFrameAllocation::GuidePositions,
                HairFrameAllocation::GuidePrevPositions,
                HairFrameAllocation::GuideGoals,
                HairFrameAllocation::GuideRestLengths,
                HairFrameAllocation::GuideStrands,
                HairFrameAllocation::BodyColliders,
                HairFrameAllocation::SdfPrimitives,
                HairFrameAllocation::GuideRanges,
                HairFrameAllocation::RenderBindings,
                HairFrameAllocation::RenderOutPoints,
                HairFrameAllocation::LodKeepMask,
            ]
        );
        // Guide positions is bound by four spine passes but appears once.
        let guide_positions = out
            .iter()
            .filter(|&&a| a == HairFrameAllocation::GuidePositions)
            .count();
        assert_eq!(guide_positions, 1);
    }

    #[test]
    fn vbd_config_adds_vbd_targets_and_keeps_guide_positions_single() {
        let config = HairFrameConfig {
            solver: HairSolverKind::Vbd,
            self_collision: false,
        };
        let mut out = Vec::new();
        plan_frame_allocations(config, &mut out);
        assert!(out.contains(&HairFrameAllocation::VbdTargets));
        // XPBD-only inputs the VBD path does not bind must be absent.
        assert!(!out.contains(&HairFrameAllocation::GuideGoals));
        let guide_positions = out
            .iter()
            .filter(|&&a| a == HairFrameAllocation::GuidePositions)
            .count();
        assert_eq!(guide_positions, 1);
    }

    #[test]
    fn self_collision_config_adds_all_three_csr_buffers_and_corrections() {
        let config = HairFrameConfig {
            solver: HairSolverKind::Xpbd,
            self_collision: true,
        };
        let mut out = Vec::new();
        plan_frame_allocations(config, &mut out);
        assert!(out.contains(&HairFrameAllocation::SelfCollisionCorrections));
        assert!(out.contains(&HairFrameAllocation::SelfCollisionCellKeys));
        assert!(out.contains(&HairFrameAllocation::SelfCollisionCellStarts));
        assert!(out.contains(&HairFrameAllocation::SelfCollisionCellIndices));
        // The accumulate/apply pair share the same buffers, so each appears once.
        let corrections = out
            .iter()
            .filter(|&&a| a == HairFrameAllocation::SelfCollisionCorrections)
            .count();
        assert_eq!(corrections, 1);
    }

    #[test]
    fn residency_classifies_each_family() {
        assert_eq!(
            HairFrameAllocation::GuidePositions.residency(),
            HairAllocationResidency::Persistent
        );
        assert_eq!(
            HairFrameAllocation::RenderOutPoints.residency(),
            HairAllocationResidency::Persistent
        );
        assert_eq!(
            HairFrameAllocation::VbdTargets.residency(),
            HairAllocationResidency::Scratch
        );
        assert_eq!(
            HairFrameAllocation::LodKeepMask.residency(),
            HairAllocationResidency::Scratch
        );
        assert_eq!(
            HairFrameAllocation::SdfPrimitives.residency(),
            HairAllocationResidency::Upload
        );
    }

    #[test]
    fn out_of_range_binding_is_none() {
        assert_eq!(
            allocation_of(HairScheduledPass::Main(HairComputePass::Wind), 1),
            None
        );
        assert_eq!(
            allocation_of(HairScheduledPass::Main(HairComputePass::GuideSim), 6),
            None
        );
        assert_eq!(
            allocation_of(HairScheduledPass::Optional(HairOptionalPass::VbdSolve), 99),
            None
        );
    }

    #[test]
    fn out_of_scope_passes_resolve_to_none() {
        for pass in [
            HairComputePass::RootBind,
            HairComputePass::Resample,
            HairComputePass::Transmittance,
            HairComputePass::DeepOpacity,
        ] {
            for binding in 0..8 {
                assert_eq!(
                    allocation_of(HairScheduledPass::Main(pass), binding),
                    None,
                    "{pass:?} binding {binding} should be out of per-frame scope",
                );
            }
        }
    }
}
