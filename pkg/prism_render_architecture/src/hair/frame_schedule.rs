//! Per-frame hair compute schedule: the authoritative ordering that composes
//! the fixed spine ([`gpu_dispatch`](super::gpu_dispatch)) with the optional
//! passes ([`optional_pass_dispatch`](super::optional_pass_dispatch)) for one
//! frame, given the solver choice and whether self-collision is enabled.
//!
//! [`per_frame_passes`](super::gpu_dispatch::per_frame_passes) fixes the
//! always-on order and [`HairOptionalPass`](super::optional_pass_dispatch::HairOptionalPass)
//! describes the two twins outside it, but *where* the optional passes slot into
//! the frame is itself a contract the render graph must not re-derive by hand:
//!
//! * When [`SolverSelection`](super::solver::SolverSelection) picks
//!   [`HairSolverKind::Vbd`](super::solver::HairSolverKind), the `VbdSolve` pass
//!   *substitutes* for the `GuideSim` slot in place — same position, same
//!   domain, same buffers — so the guide solve happens exactly once.
//! * When self-collision is enabled, its accumulate-then-apply pair runs after
//!   the body/`SDF` collision has settled the guides against the character but
//!   before `Interpolate` derives render strands, so render strands see the
//!   self-separated guide state. It is a guide-domain post-pass, so it lands at
//!   the end of the simulate phase, not inside the resolve phase.
//!
//! The result is one ordered [`HairScheduledPass`] list unifying both pass
//! families, plus a plan that resolves it to workgroup counts for the frame's
//! groom counts. Everything is deterministic, pure-integer, and never panics.

use alloc::vec::Vec;

use crate::hair::gpu_dispatch::{
    dispatch_groups, per_frame_passes, HairComputePass, HairDispatchDomain, HairGpuCounts,
    HAIR_WORKGROUP_SIZE,
};
use crate::hair::optional_pass_dispatch::{self_collision_passes, HairOptionalPass};
use crate::hair::solver::HairSolverKind;

/// One entry in the per-frame schedule: either a fixed-spine pass or an
/// optional pass, unified so the render graph can walk a single ordered list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairScheduledPass {
    /// A fixed per-frame spine pass (see [`HairComputePass`]).
    Main(HairComputePass),
    /// An optional / alternative pass (see [`HairOptionalPass`]).
    Optional(HairOptionalPass),
}

impl HairScheduledPass {
    /// The `WESL` shader file base name backing this scheduled pass.
    #[must_use]
    pub fn kernel(self) -> &'static str {
        match self {
            Self::Main(pass) => pass.kernel(),
            Self::Optional(pass) => pass.kernel(),
        }
    }

    /// The compute entry-point function name inside the shader file. For fixed
    /// spine passes this equals the file base name (one kernel per file); for
    /// optional passes it can differ (self-collision packs two entries in one
    /// file).
    #[must_use]
    pub fn entry_point(self) -> &'static str {
        match self {
            Self::Main(pass) => pass.kernel(),
            Self::Optional(pass) => pass.entry_point(),
        }
    }

    /// The element domain this pass dispatches over.
    #[must_use]
    pub fn domain(self) -> HairDispatchDomain {
        match self {
            Self::Main(pass) => pass.domain(),
            Self::Optional(pass) => pass.domain(),
        }
    }

    /// Number of `@group(0)` storage bindings the kernel declares.
    #[must_use]
    pub fn binding_count(self) -> u32 {
        match self {
            Self::Main(pass) => pass.binding_count(),
            Self::Optional(pass) => pass.binding_count(),
        }
    }

    /// The 1-D workgroup count for `counts`: `ceil(domain_count / 64)`.
    #[must_use]
    pub fn workgroup_count(self, counts: &HairGpuCounts) -> u32 {
        dispatch_groups(counts.domain_count(self.domain()), HAIR_WORKGROUP_SIZE)
    }
}

/// Which optional behaviour is active for a frame, driving how the schedule is
/// composed from the fixed spine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HairFrameConfig {
    /// The guide solver chosen for this groom this frame.
    /// [`HairSolverKind::Vbd`] substitutes `VbdSolve` for the `GuideSim` slot.
    pub solver: HairSolverKind,
    /// Whether the opt-in self-collision post-pass runs this frame.
    pub self_collision: bool,
}

impl Default for HairFrameConfig {
    /// The default groom: `XPBD` guide solve, no self-collision — matching the
    /// fixed spine with no substitution or insertion.
    fn default() -> Self {
        Self {
            solver: HairSolverKind::Xpbd,
            self_collision: false,
        }
    }
}

/// One resolved scheduled dispatch: the pass, its domain element count, and the
/// 1-D workgroup count to issue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HairScheduledDispatch {
    /// The scheduled pass to bind and run.
    pub pass: HairScheduledPass,
    /// Number of domain elements this dispatch covers.
    pub domain_count: u32,
    /// Number of workgroups to dispatch along X (`ceil(domain_count / 64)`).
    pub workgroup_count: u32,
}

/// Composes the ordered per-frame pass schedule for `config`, appending to
/// `out` (which is cleared first).
///
/// Starts from [`per_frame_passes`], swaps `GuideSim` for `VbdSolve` when the
/// solver is [`HairSolverKind::Vbd`], and inserts the self-collision
/// accumulate/apply pair immediately after `SdfCollision` when enabled. Order
/// is otherwise the fixed spine order. Never panics.
pub fn per_frame_schedule(config: HairFrameConfig, out: &mut Vec<HairScheduledPass>) {
    out.clear();
    for pass in per_frame_passes() {
        match (pass, config.solver) {
            (HairComputePass::GuideSim, HairSolverKind::Vbd) => {
                out.push(HairScheduledPass::Optional(HairOptionalPass::VbdSolve));
            }
            _ => out.push(HairScheduledPass::Main(pass)),
        }
        if pass == HairComputePass::SdfCollision && config.self_collision {
            for sc in self_collision_passes() {
                out.push(HairScheduledPass::Optional(sc));
            }
        }
    }
}

/// Resolves the per-frame schedule for `config` and `counts` into ordered
/// [`HairScheduledDispatch`] entries, appending to `out` (which is cleared
/// first). A pass whose domain is empty (zero workgroups) is skipped so the
/// plan only contains work that will run; order is otherwise preserved. Never
/// panics.
pub fn plan_frame_dispatches(
    config: HairFrameConfig,
    counts: &HairGpuCounts,
    out: &mut Vec<HairScheduledDispatch>,
) {
    out.clear();
    let mut schedule = Vec::new();
    per_frame_schedule(config, &mut schedule);
    for pass in schedule {
        let domain_count = counts.domain_count(pass.domain());
        let workgroup_count = dispatch_groups(domain_count, HAIR_WORKGROUP_SIZE);
        if workgroup_count == 0 {
            continue;
        }
        out.push(HairScheduledDispatch {
            pass,
            domain_count,
            workgroup_count,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_counts() -> HairGpuCounts {
        HairGpuCounts {
            roots: 100,
            guide_strands: 100,
            guide_particles: 3200,
            render_strands: 50_000,
            light_texels: 4096,
        }
    }

    fn kernels(schedule: &[HairScheduledPass]) -> Vec<&'static str> {
        schedule.iter().map(|p| p.kernel()).collect()
    }

    #[test]
    fn default_config_is_the_untouched_spine() {
        let mut out = Vec::new();
        per_frame_schedule(HairFrameConfig::default(), &mut out);
        let expected: Vec<HairScheduledPass> = per_frame_passes()
            .into_iter()
            .map(HairScheduledPass::Main)
            .collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn vbd_substitutes_the_guide_sim_slot_in_place() {
        let config = HairFrameConfig {
            solver: HairSolverKind::Vbd,
            self_collision: false,
        };
        let mut out = Vec::new();
        per_frame_schedule(config, &mut out);
        assert_eq!(
            kernels(&out),
            [
                "hair_root_skinning",
                "hair_wind",
                "hair_vbd",
                "hair_sdf_collision",
                "hair_interp",
                "hair_lod_dither",
            ]
        );
        // No hair_sim (XPBD) survives when VBD is chosen — the guide solve runs
        // exactly once.
        assert!(!kernels(&out).contains(&"hair_sim"));
    }

    #[test]
    fn self_collision_inserts_after_sdf_and_before_interp() {
        let config = HairFrameConfig {
            solver: HairSolverKind::Xpbd,
            self_collision: true,
        };
        let mut out = Vec::new();
        per_frame_schedule(config, &mut out);
        assert_eq!(
            out,
            [
                HairScheduledPass::Main(HairComputePass::RootSkinning),
                HairScheduledPass::Main(HairComputePass::Wind),
                HairScheduledPass::Main(HairComputePass::GuideSim),
                HairScheduledPass::Main(HairComputePass::SdfCollision),
                HairScheduledPass::Optional(HairOptionalPass::SelfCollisionAccumulate),
                HairScheduledPass::Optional(HairOptionalPass::SelfCollisionApply),
                HairScheduledPass::Main(HairComputePass::Interpolate),
                HairScheduledPass::Main(HairComputePass::LodDither),
            ]
        );
    }

    #[test]
    fn vbd_and_self_collision_compose() {
        let config = HairFrameConfig {
            solver: HairSolverKind::Vbd,
            self_collision: true,
        };
        let mut out = Vec::new();
        per_frame_schedule(config, &mut out);
        assert_eq!(
            kernels(&out),
            [
                "hair_root_skinning",
                "hair_wind",
                "hair_vbd",
                "hair_sdf_collision",
                "hair_self_collision",
                "hair_self_collision",
                "hair_interp",
                "hair_lod_dither",
            ]
        );
    }

    #[test]
    fn scheduled_entry_points_distinguish_the_two_self_collision_passes() {
        let config = HairFrameConfig {
            solver: HairSolverKind::Xpbd,
            self_collision: true,
        };
        let mut out = Vec::new();
        per_frame_schedule(config, &mut out);
        let entries: Vec<&'static str> = out.iter().map(|p| p.entry_point()).collect();
        // Both self-collision entries share the file but not the entry point.
        assert!(entries.contains(&"accumulate_self_collision"));
        assert!(entries.contains(&"apply_self_collision"));
        // Fixed-spine passes report their file base name as the entry point.
        assert!(entries.contains(&"hair_sdf_collision"));
    }

    #[test]
    fn vbd_scheduled_pass_matches_the_slot_it_replaces() {
        // The substituted VBD pass must keep GuideSim's domain and binding count
        // so the swap is drop-in.
        let vbd = HairScheduledPass::Optional(HairOptionalPass::VbdSolve);
        let guide_sim = HairScheduledPass::Main(HairComputePass::GuideSim);
        assert_eq!(vbd.domain(), guide_sim.domain());
        assert_eq!(vbd.binding_count(), guide_sim.binding_count());
    }

    #[test]
    fn plan_resolves_workgroup_counts_in_order() {
        let config = HairFrameConfig {
            solver: HairSolverKind::Vbd,
            self_collision: true,
        };
        let counts = sample_counts();
        let mut out = Vec::new();
        plan_frame_dispatches(config, &counts, &mut out);
        // All domains non-empty, so all eight passes survive.
        assert_eq!(out.len(), 8);
        // VBD dispatches over guide strands: ceil(100 / 64) = 2.
        assert_eq!(
            out[2].pass,
            HairScheduledPass::Optional(HairOptionalPass::VbdSolve)
        );
        assert_eq!(out[2].workgroup_count, 2);
        // Self-collision dispatches over guide particles: ceil(3200 / 64) = 50.
        assert_eq!(
            out[4].pass,
            HairScheduledPass::Optional(HairOptionalPass::SelfCollisionAccumulate)
        );
        assert_eq!(out[4].workgroup_count, 50);
    }

    #[test]
    fn plan_skips_empty_domains() {
        let config = HairFrameConfig {
            solver: HairSolverKind::Xpbd,
            self_collision: true,
        };
        // No render strands: Interpolate and LodDither drop out; guide passes
        // and self-collision stay.
        let counts = HairGpuCounts {
            roots: 10,
            guide_strands: 10,
            guide_particles: 320,
            render_strands: 0,
            light_texels: 0,
        };
        let mut out = Vec::new();
        plan_frame_dispatches(config, &counts, &mut out);
        let kernels: Vec<&'static str> = out.iter().map(|d| d.pass.kernel()).collect();
        assert!(!kernels.contains(&"hair_interp"));
        assert!(!kernels.contains(&"hair_lod_dither"));
        assert!(kernels.contains(&"hair_self_collision"));
    }

    #[test]
    fn plan_clears_prior_contents() {
        let counts = sample_counts();
        let mut out = Vec::new();
        out.push(HairScheduledDispatch {
            pass: HairScheduledPass::Main(HairComputePass::Wind),
            domain_count: 1,
            workgroup_count: 1,
        });
        plan_frame_dispatches(HairFrameConfig::default(), &counts, &mut out);
        // Default spine has six passes, all non-empty for sample counts.
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn empty_counts_produce_no_dispatches() {
        let mut out = Vec::new();
        plan_frame_dispatches(
            HairFrameConfig {
                solver: HairSolverKind::Vbd,
                self_collision: true,
            },
            &HairGpuCounts::default(),
            &mut out,
        );
        assert!(out.is_empty());
    }
}
