//! Dispatch descriptors for the two *optional* hair compute passes that are
//! deliberately absent from the fixed per-frame spine in
//! [`gpu_dispatch`](super::gpu_dispatch).
//!
//! [`HairComputePass`](super::gpu_dispatch::HairComputePass) enumerates the ten
//! always-on twins whose order is fixed by
//! [`per_frame_passes`](super::gpu_dispatch::per_frame_passes). Two further
//! `WESL` twins exist but must **not** be spliced into that spine, because they
//! have different activation semantics:
//!
//! * `hair_vbd.wesl` (entry `simulate_strand_vbd`) is an *alternative solver*:
//!   when [`SolverSelection`](super::solver::SolverSelection) picks
//!   [`HairSolverKind::Vbd`](super::solver::HairSolverKind) for a stiff groom it
//!   *replaces* the `GuideSim` (`XPBD`) slot rather than running alongside it.
//!   Because it shares the sim buffer layout byte-for-byte
//!   ([`vbd_pass_buffers`](super::vbd_pass_buffers)) the render graph swaps the
//!   pipeline in that one slot without reallocating.
//! * `hair_self_collision.wesl` (entries `accumulate_self_collision` then
//!   `apply_self_collision`) is an *opt-in post-pass*: production grooms enable
//!   a spatial-hash self-collision resolve after the guide solve, but many do
//!   not, so it is appended only when requested. Its two entry points share one
//!   `@group(0)` ([`self_collision_pass_buffers`](super::self_collision_pass_buffers))
//!   and must run strictly accumulate-then-apply.
//!
//! Folding either into the mandatory spine would misrepresent the pipeline
//! (VBD would double-solve, self-collision would run for grooms that never
//! asked for it), so their kernel identity, entry-point name, dispatch domain,
//! binding count and workgroup derivation live here as a peer contract. Unlike
//! the main passes — where one file maps to one kernel *and* one entry point of
//! the same base name — these carry an explicit `entry_point` distinct from the
//! shader file's base name (self-collision has two entries in one file), which
//! the render crate needs to select the right pipeline entry.
//!
//! Everything reuses the shared workgroup size and domain-count derivation from
//! [`gpu_dispatch`](super::gpu_dispatch), stays pure-integer and deterministic,
//! and never panics.

use alloc::vec::Vec;

use crate::hair::gpu_dispatch::{
    dispatch_groups, HairComputePass, HairDispatchDomain, HairGpuCounts, HAIR_WORKGROUP_SIZE,
};

/// One optional hair compute pass outside the fixed per-frame spine: an
/// alternative guide solver or an opt-in post-pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairOptionalPass {
    /// High-stiffness `VBD` guide solve (`hair_vbd.wesl`), the alternative that
    /// takes over the `GuideSim` slot when the solver selector prefers `VBD`.
    VbdSolve,
    /// Self-collision correction accumulation (`hair_self_collision.wesl`,
    /// entry `accumulate_self_collision`): sums each particle's separation push
    /// from the read-only position snapshot into the scratch corrections.
    SelfCollisionAccumulate,
    /// Self-collision correction application (`hair_self_collision.wesl`, entry
    /// `apply_self_collision`): adds each accumulated correction in place. Runs
    /// strictly after [`SelfCollisionAccumulate`](Self::SelfCollisionAccumulate).
    SelfCollisionApply,
}

impl HairOptionalPass {
    /// Every optional pass, grouped by twin and in intra-twin run order
    /// (self-collision accumulate before apply).
    pub const ALL: [HairOptionalPass; 3] = [
        Self::VbdSolve,
        Self::SelfCollisionAccumulate,
        Self::SelfCollisionApply,
    ];

    /// The `WESL` shader file base name (without the `.wesl` extension), the
    /// same base name as its file in `prism_render_scene/src/shaders/`.
    #[must_use]
    pub fn kernel(self) -> &'static str {
        match self {
            Self::VbdSolve => "hair_vbd",
            Self::SelfCollisionAccumulate | Self::SelfCollisionApply => "hair_self_collision",
        }
    }

    /// The compute entry-point function name inside [`kernel`](Self::kernel).
    /// Distinct from the file base name because self-collision packs two entry
    /// points into one shader file.
    #[must_use]
    pub fn entry_point(self) -> &'static str {
        match self {
            Self::VbdSolve => "simulate_strand_vbd",
            Self::SelfCollisionAccumulate => "accumulate_self_collision",
            Self::SelfCollisionApply => "apply_self_collision",
        }
    }

    /// The element domain this pass dispatches over. `VBD` owns one guide strand
    /// per invocation (mirroring `GuideSim`); both self-collision entries own
    /// one guide particle per invocation.
    #[must_use]
    pub fn domain(self) -> HairDispatchDomain {
        match self {
            Self::VbdSolve => HairDispatchDomain::GuideStrands,
            Self::SelfCollisionAccumulate | Self::SelfCollisionApply => {
                HairDispatchDomain::GuideParticles
            }
        }
    }

    /// Number of `@group(0)` storage bindings the kernel declares, so the render
    /// crate can size its bind-group layout against this contract. `VBD` binds
    /// the six sim buffers; both self-collision entries bind the same five.
    #[must_use]
    pub fn binding_count(self) -> u32 {
        match self {
            Self::VbdSolve => 6,
            Self::SelfCollisionAccumulate | Self::SelfCollisionApply => 5,
        }
    }

    /// The mandatory-spine pass this optional pass *replaces* when active, or
    /// [`None`] if it is additive. `VBD` substitutes for the `GuideSim` slot;
    /// self-collision is appended and replaces nothing.
    #[must_use]
    pub fn replaces(self) -> Option<HairComputePass> {
        match self {
            Self::VbdSolve => Some(HairComputePass::GuideSim),
            Self::SelfCollisionAccumulate | Self::SelfCollisionApply => None,
        }
    }

    /// Whether this pass is an alternative solver occupying an existing spine
    /// slot (mutually exclusive with the pass it [`replaces`](Self::replaces)).
    #[must_use]
    pub fn is_alternative_solver(self) -> bool {
        self.replaces().is_some()
    }

    /// Whether this pass is an additive opt-in post-pass appended to the spine.
    #[must_use]
    pub fn is_post_pass(self) -> bool {
        self.replaces().is_none()
    }

    /// The 1-D workgroup count to dispatch for `counts`:
    /// `ceil(domain_count / 64)`. An empty domain yields `0`.
    #[must_use]
    pub fn workgroup_count(self, counts: &HairGpuCounts) -> u32 {
        dispatch_groups(counts.domain_count(self.domain()), HAIR_WORKGROUP_SIZE)
    }
}

/// One resolved optional dispatch: the pass, its domain element count, and the
/// 1-D workgroup count to issue. Mirrors
/// [`HairDispatch`](super::gpu_dispatch::HairDispatch) for the optional set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HairOptionalDispatch {
    /// The optional compute pass to bind and run.
    pub pass: HairOptionalPass,
    /// Number of domain elements this dispatch covers.
    pub domain_count: u32,
    /// Number of workgroups to dispatch along X (`ceil(domain_count / 64)`).
    pub workgroup_count: u32,
}

/// The self-collision post-pass in strict run order: accumulate every
/// particle's correction from the read-only snapshot, then apply the
/// corrections in place.
#[must_use]
pub fn self_collision_passes() -> [HairOptionalPass; 2] {
    [
        HairOptionalPass::SelfCollisionAccumulate,
        HairOptionalPass::SelfCollisionApply,
    ]
}

/// Resolves `passes` into ordered [`HairOptionalDispatch`] entries for `counts`,
/// appending to `out` (which is cleared first). A pass whose domain is empty
/// (zero workgroups) is skipped so the plan only contains work that will run;
/// input order is otherwise preserved. Never panics.
pub fn plan_optional_dispatches(
    passes: &[HairOptionalPass],
    counts: &HairGpuCounts,
    out: &mut Vec<HairOptionalDispatch>,
) {
    out.clear();
    for &pass in passes {
        let domain_count = counts.domain_count(pass.domain());
        let workgroup_count = dispatch_groups(domain_count, HAIR_WORKGROUP_SIZE);
        if workgroup_count == 0 {
            continue;
        }
        out.push(HairOptionalDispatch {
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
            light_texels: 0,
        }
    }

    #[test]
    fn kernels_match_the_wesl_files() {
        assert_eq!(HairOptionalPass::VbdSolve.kernel(), "hair_vbd");
        assert_eq!(
            HairOptionalPass::SelfCollisionAccumulate.kernel(),
            "hair_self_collision"
        );
        assert_eq!(
            HairOptionalPass::SelfCollisionApply.kernel(),
            "hair_self_collision"
        );
    }

    #[test]
    fn entry_points_are_distinct_within_a_shared_file() {
        assert_eq!(
            HairOptionalPass::VbdSolve.entry_point(),
            "simulate_strand_vbd"
        );
        assert_eq!(
            HairOptionalPass::SelfCollisionAccumulate.entry_point(),
            "accumulate_self_collision"
        );
        assert_eq!(
            HairOptionalPass::SelfCollisionApply.entry_point(),
            "apply_self_collision"
        );
        // The two self-collision entries share a file but not an entry point.
        assert_ne!(
            HairOptionalPass::SelfCollisionAccumulate.entry_point(),
            HairOptionalPass::SelfCollisionApply.entry_point()
        );
    }

    #[test]
    fn domains_match_the_kernels() {
        assert_eq!(
            HairOptionalPass::VbdSolve.domain(),
            HairDispatchDomain::GuideStrands
        );
        assert_eq!(
            HairOptionalPass::SelfCollisionAccumulate.domain(),
            HairDispatchDomain::GuideParticles
        );
        assert_eq!(
            HairOptionalPass::SelfCollisionApply.domain(),
            HairDispatchDomain::GuideParticles
        );
    }

    #[test]
    fn binding_counts_match_the_buffer_contracts() {
        assert_eq!(HairOptionalPass::VbdSolve.binding_count(), 6);
        assert_eq!(HairOptionalPass::SelfCollisionAccumulate.binding_count(), 5);
        assert_eq!(HairOptionalPass::SelfCollisionApply.binding_count(), 5);
    }

    #[test]
    fn vbd_substitutes_guide_sim_and_self_collision_is_additive() {
        assert_eq!(
            HairOptionalPass::VbdSolve.replaces(),
            Some(HairComputePass::GuideSim)
        );
        assert!(HairOptionalPass::VbdSolve.is_alternative_solver());
        assert!(!HairOptionalPass::VbdSolve.is_post_pass());
        for pass in self_collision_passes() {
            assert_eq!(pass.replaces(), None);
            assert!(pass.is_post_pass());
            assert!(!pass.is_alternative_solver());
        }
    }

    #[test]
    fn vbd_shares_the_guide_sim_domain_it_replaces() {
        // Because VBD takes over the GuideSim slot, it must dispatch over the
        // same domain so the swap needs no re-derivation.
        assert_eq!(
            HairOptionalPass::VbdSolve.domain(),
            HairComputePass::GuideSim.domain()
        );
    }

    #[test]
    fn vbd_shares_the_sim_binding_count_it_replaces() {
        assert_eq!(
            HairOptionalPass::VbdSolve.binding_count(),
            HairComputePass::GuideSim.binding_count()
        );
    }

    #[test]
    fn self_collision_passes_run_accumulate_then_apply() {
        assert_eq!(
            self_collision_passes(),
            [
                HairOptionalPass::SelfCollisionAccumulate,
                HairOptionalPass::SelfCollisionApply,
            ]
        );
    }

    #[test]
    fn workgroup_counts_are_ceiling_division() {
        let counts = sample_counts();
        // 100 guide strands -> ceil(100 / 64) = 2.
        assert_eq!(HairOptionalPass::VbdSolve.workgroup_count(&counts), 2);
        // 3200 guide particles -> ceil(3200 / 64) = 50.
        assert_eq!(
            HairOptionalPass::SelfCollisionAccumulate.workgroup_count(&counts),
            50
        );
        assert_eq!(
            HairOptionalPass::SelfCollisionApply.workgroup_count(&counts),
            50
        );
    }

    #[test]
    fn plan_skips_empty_domains_and_preserves_order() {
        let counts = HairGpuCounts {
            guide_strands: 0,
            guide_particles: 128,
            ..HairGpuCounts::default()
        };
        let mut out = Vec::new();
        let mut passes = Vec::new();
        passes.push(HairOptionalPass::VbdSolve);
        passes.extend_from_slice(&self_collision_passes());
        plan_optional_dispatches(&passes, &counts, &mut out);
        // VBD skipped (no guide strands); both self-collision passes kept.
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].pass, HairOptionalPass::SelfCollisionAccumulate);
        assert_eq!(out[0].workgroup_count, 2);
        assert_eq!(out[1].pass, HairOptionalPass::SelfCollisionApply);
    }

    #[test]
    fn plan_clears_prior_contents() {
        let counts = sample_counts();
        let mut out = Vec::new();
        out.push(HairOptionalDispatch {
            pass: HairOptionalPass::VbdSolve,
            domain_count: 999,
            workgroup_count: 999,
        });
        plan_optional_dispatches(&[HairOptionalPass::VbdSolve], &counts, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].domain_count, 100);
    }

    #[test]
    fn empty_counts_produce_no_dispatches() {
        let counts = HairGpuCounts::default();
        let mut out = Vec::new();
        plan_optional_dispatches(&HairOptionalPass::ALL, &counts, &mut out);
        assert!(out.is_empty());
    }
}
