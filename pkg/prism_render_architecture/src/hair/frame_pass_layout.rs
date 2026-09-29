//! Whole-frame ordered bind-table confluence for the hair compute schedule.
//!
//! [`pass_layout`](super::pass_layout) and
//! [`optional_pass_layout`](super::optional_pass_layout) each answer *what one
//! pass binds and dispatches* — the former for the fixed main spine
//! ([`HairComputePass`](super::gpu_dispatch::HairComputePass)), the latter for
//! the optional / alternative-solver passes
//! ([`HairOptionalPass`](super::optional_pass_dispatch::HairOptionalPass)).
//! [`frame_schedule`](super::frame_schedule) answers *in what order* the two
//! families combine into one frame ([`per_frame_schedule`]), including the
//! guide-`VBD` in-place replacement of `GuideSim` and the self-collision
//! `accumulate`→`apply` insertion after `SdfCollision`. What was still missing is
//! the join of those two axes: one ordered list where every entry already
//! carries its dispatch dimension, its dense `@group(0)` binding table, and its
//! `var<immediate>` push-constant size. That is exactly what the render graph
//! consumes to walk a frame — bind group by bind group, dispatch by dispatch —
//! so this module produces it in a single call.
//!
//! [`HairScheduledPassPlan`] is the unified row (the schedule analogue of
//! [`HairPassPlan`](super::pass_layout::HairPassPlan) /
//! [`HairOptionalPassPlan`](super::optional_pass_layout::HairOptionalPassPlan)),
//! tagged with the schedule's own [`HairScheduledPass`] so a caller need not know
//! whether a row came from the main or optional confluence. The immediate size
//! is sourced from [`pass_params`](super::pass_params) for main passes and from
//! the optional plan (which already carries it) for optional passes.
//!
//! [`plan_frame_passes`] skips empty-domain passes exactly like
//! [`plan_frame_dispatches`](super::frame_schedule::plan_frame_dispatches), so
//! the bind-table list matches the dispatch list one-for-one (same passes, same
//! order, same dimensions). Everything is pure, integer and deterministic: an
//! empty groom yields an empty plan list without panicking.

use alloc::vec::Vec;

use crate::hair::frame_schedule::{per_frame_schedule, HairFrameConfig, HairScheduledPass};
use crate::hair::gpu_dispatch::HairGpuCounts;
use crate::hair::optional_pass_layout::{plan_optional_pass, HairOptionalExtents};
use crate::hair::pass_layout::{plan_pass, HairBindingLayout, HairGpuExtents};
use crate::hair::pass_params::params_immediate_bytes;

/// One resolved entry of the per-frame hair compute schedule: a scheduled pass
/// joined with its dispatch dimension, its dense `@group(0)` binding table, and
/// the byte size of its `var<immediate>` push-constant block.
///
/// This is the schedule-level analogue of
/// [`HairPassPlan`](super::pass_layout::HairPassPlan) and
/// [`HairOptionalPassPlan`](super::optional_pass_layout::HairOptionalPassPlan),
/// unified under the schedule's [`HairScheduledPass`] tag so the render graph
/// consumes main and optional rows uniformly. Contains a [`Vec`], so it is not
/// `Copy`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HairScheduledPassPlan {
    /// The scheduled pass (main-spine or optional) this row binds and dispatches.
    pub pass: HairScheduledPass,
    /// Number of domain elements this dispatch covers.
    pub domain_count: u32,
    /// Workgroups to dispatch along X (`ceil(domain_count / 64)`).
    pub workgroup_count: u32,
    /// The pass's `@group(0)` bindings, in dense binding order `0..len`.
    pub bindings: Vec<HairBindingLayout>,
    /// Sum of every binding's `byte_size` — the pass's total storage footprint.
    pub total_binding_bytes: usize,
    /// Byte size of the pass's `var<immediate>` params block (push constants).
    pub params_immediate_bytes: usize,
}

/// Resolves the whole per-frame schedule for `config` into ordered
/// [`HairScheduledPassPlan`] rows, appending to `out` (cleared first).
///
/// The main-spine extents feed [`plan_pass`] and the optional extents feed
/// [`plan_optional_pass`]; each row's immediate size comes from
/// [`params_immediate_bytes`] for main passes and from the optional plan for
/// optional passes. Empty-domain passes (`workgroup_count == 0`) are skipped so
/// the list matches
/// [`plan_frame_dispatches`](super::frame_schedule::plan_frame_dispatches)
/// one-for-one; order is otherwise the schedule order. Never panics.
pub fn plan_frame_passes(
    config: HairFrameConfig,
    counts: &HairGpuCounts,
    extents: HairGpuExtents,
    optional_extents: HairOptionalExtents,
    out: &mut Vec<HairScheduledPassPlan>,
) {
    out.clear();
    let mut schedule = Vec::new();
    per_frame_schedule(config, &mut schedule);
    for scheduled in schedule {
        let plan = match scheduled {
            HairScheduledPass::Main(pass) => {
                let plan = plan_pass(pass, counts, extents);
                HairScheduledPassPlan {
                    pass: scheduled,
                    domain_count: plan.domain_count,
                    workgroup_count: plan.workgroup_count,
                    bindings: plan.bindings,
                    total_binding_bytes: plan.total_binding_bytes,
                    params_immediate_bytes: params_immediate_bytes(pass),
                }
            }
            HairScheduledPass::Optional(pass) => {
                let plan = plan_optional_pass(pass, counts, optional_extents);
                HairScheduledPassPlan {
                    pass: scheduled,
                    domain_count: plan.domain_count,
                    workgroup_count: plan.workgroup_count,
                    bindings: plan.bindings,
                    total_binding_bytes: plan.total_binding_bytes,
                    params_immediate_bytes: plan.params_immediate_bytes,
                }
            }
        };
        if plan.workgroup_count == 0 {
            continue;
        }
        out.push(plan);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::frame_schedule::plan_frame_dispatches;
    use crate::hair::gpu_dispatch::HairComputePass;
    use crate::hair::optional_pass_dispatch::HairOptionalPass;
    use crate::hair::solver::HairSolverKind;

    fn counts() -> HairGpuCounts {
        HairGpuCounts {
            roots: 100,
            guide_strands: 100,
            guide_particles: 3200,
            render_strands: 5000,
            light_texels: 4096,
        }
    }

    fn extents() -> HairGpuExtents {
        HairGpuExtents {
            collider_count: 4,
            render_points: 60_000,
            ..HairGpuExtents::default()
        }
    }

    fn optional_extents() -> HairOptionalExtents {
        HairOptionalExtents {
            collider_count: 4,
            cell_count: 512,
        }
    }

    #[test]
    fn default_frame_is_all_main_passes_in_schedule_order() {
        let mut out = Vec::new();
        plan_frame_passes(
            HairFrameConfig::default(),
            &counts(),
            extents(),
            optional_extents(),
            &mut out,
        );
        assert!(!out.is_empty());
        for plan in &out {
            assert!(
                matches!(plan.pass, HairScheduledPass::Main(_)),
                "default config must not schedule optional passes",
            );
        }
    }

    #[test]
    fn every_row_binding_table_and_immediate_are_self_consistent() {
        let mut out = Vec::new();
        plan_frame_passes(
            HairFrameConfig {
                solver: HairSolverKind::Vbd,
                self_collision: true,
            },
            &counts(),
            extents(),
            optional_extents(),
            &mut out,
        );
        for plan in &out {
            assert_eq!(
                plan.bindings.len() as u32,
                plan.pass.binding_count(),
                "{:?} binding table width must match binding_count",
                plan.pass,
            );
            let summed: usize = plan.bindings.iter().map(|b| b.byte_size).sum();
            assert_eq!(plan.total_binding_bytes, summed);
            assert!(plan.params_immediate_bytes > 0);
            assert_eq!(plan.params_immediate_bytes % 4, 0);
            assert!(
                plan.workgroup_count > 0,
                "empty-domain passes must be skipped"
            );
        }
    }

    #[test]
    fn matches_dispatch_planner_one_for_one() {
        let config = HairFrameConfig {
            solver: HairSolverKind::Vbd,
            self_collision: true,
        };
        let mut layouts = Vec::new();
        plan_frame_passes(
            config,
            &counts(),
            extents(),
            optional_extents(),
            &mut layouts,
        );
        let mut dispatches = Vec::new();
        plan_frame_dispatches(config, &counts(), &mut dispatches);
        assert_eq!(layouts.len(), dispatches.len());
        for (layout, dispatch) in layouts.iter().zip(dispatches.iter()) {
            assert_eq!(layout.pass, dispatch.pass);
            assert_eq!(layout.domain_count, dispatch.domain_count);
            assert_eq!(layout.workgroup_count, dispatch.workgroup_count);
        }
    }

    #[test]
    fn vbd_config_replaces_guide_sim_with_a_forty_eight_byte_block() {
        let mut out = Vec::new();
        plan_frame_passes(
            HairFrameConfig {
                solver: HairSolverKind::Vbd,
                self_collision: false,
            },
            &counts(),
            extents(),
            optional_extents(),
            &mut out,
        );
        assert!(
            !out.iter()
                .any(|p| p.pass == HairScheduledPass::Main(HairComputePass::GuideSim)),
            "VBD solver must displace the GuideSim slot",
        );
        let vbd = out
            .iter()
            .find(|p| p.pass == HairScheduledPass::Optional(HairOptionalPass::VbdSolve))
            .expect("VBD solve pass must be scheduled");
        assert_eq!(vbd.params_immediate_bytes, 48);
        assert_eq!(
            vbd.bindings.len() as u32,
            HairOptionalPass::VbdSolve.binding_count()
        );
    }

    #[test]
    fn self_collision_pair_sits_between_sdf_and_interpolate() {
        let mut out = Vec::new();
        plan_frame_passes(
            HairFrameConfig {
                solver: HairSolverKind::Xpbd,
                self_collision: true,
            },
            &counts(),
            extents(),
            optional_extents(),
            &mut out,
        );
        let index = |target: HairScheduledPass| {
            out.iter()
                .position(|p| p.pass == target)
                .expect("pass scheduled")
        };
        let sdf = index(HairScheduledPass::Main(HairComputePass::SdfCollision));
        let accumulate = index(HairScheduledPass::Optional(
            HairOptionalPass::SelfCollisionAccumulate,
        ));
        let apply = index(HairScheduledPass::Optional(
            HairOptionalPass::SelfCollisionApply,
        ));
        let interpolate = index(HairScheduledPass::Main(HairComputePass::Interpolate));
        assert!(sdf < accumulate);
        assert!(accumulate < apply);
        assert!(apply < interpolate);
    }

    #[test]
    fn empty_groom_yields_no_rows() {
        let mut out = Vec::new();
        plan_frame_passes(
            HairFrameConfig::default(),
            &HairGpuCounts::default(),
            HairGpuExtents::default(),
            HairOptionalExtents::default(),
            &mut out,
        );
        assert!(out.is_empty(), "an empty groom schedules no work");
    }
}
