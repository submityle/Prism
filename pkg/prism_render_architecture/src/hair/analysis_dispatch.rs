//! Deterministic GPU compute-dispatch contract for the hair *analysis* passes.
//!
//! Alongside the ten per-frame/import/shadow passes owned by
//! [`super::gpu_dispatch`], the groom runs a handful of *analysis* kernels whose
//! `WESL` twins also live in the scene crate but which sit off the fixed
//! per-frame render pipeline: they feed host-side decisions rather than the
//! displayed geometry. Two host decisions consume them:
//!
//! - Density-LOD ranking (design §7 连续 LOD): `hair_guide_metrics` measures
//!   each guide's arc length / curvature / authored thickness, `hair_binding_metrics`
//!   propagates those to every render strand through its guide weights, and
//!   `hair_importance` folds the blended triple into a single normalized `[0, 1]`
//!   importance. The host then reduces the groom-global maxima and runs the
//!   CPU-only ranking sort (`decimation::build_decimation_order`) that these
//!   metrics feed — the same host/other-kernel split the twins document.
//! - Sleep gating (design §6.9, §8 休眠不占预算): `hair_motion_energy` maps each
//!   particle's implicit velocity to its squared length; the host sums it into a
//!   groom motion energy for the hysteretic sleep gate (`sleep::update_sleep`).
//!
//! These passes run on the LOD-reevaluation / sleep cadence, not every frame in
//! the render pipeline, so they are a *separate* ordered contract from the main
//! [`HairComputePass`](super::gpu_dispatch::HairComputePass) sequence — mixing
//! them into the per-frame order would misrepresent when they run. They still
//! share the same dispatch primitives: the `@workgroup_size(64)` constant, the
//! [`HairDispatchDomain`] element domains, the [`HairGpuCounts`] element counts
//! and the [`dispatch_groups`] ceiling division, so the render graph binds them
//! against the same stable ABI.
//!
//! Everything is pure, integer, deterministic and panic-free: an empty domain
//! yields a zero workgroup count (skipped by [`plan_analysis_dispatches`]).

use alloc::vec::Vec;

use super::gpu_dispatch::{
    dispatch_groups, HairDispatchDomain, HairGpuCounts, HAIR_WORKGROUP_SIZE,
};

/// Which host decision an analysis pass feeds. Fixes the two independent chains
/// these passes form so callers can schedule (or skip) a whole decision at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairAnalysisKind {
    /// Feeds the density-LOD ranking sort (`hair_guide_metrics` →
    /// `hair_binding_metrics` → `hair_importance`, then the CPU-only
    /// `decimation::build_decimation_order`).
    DensityLod,
    /// Feeds the hysteretic sleep gate (`hair_motion_energy`, then the host
    /// reduction and `sleep::update_sleep`).
    Sleep,
}

/// One hair analysis compute pass: a `WESL` kernel plus the metadata needed to
/// bind and dispatch it. Distinct from
/// [`HairComputePass`](super::gpu_dispatch::HairComputePass): these feed host
/// decisions on the LOD/sleep cadence rather than the per-frame render pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairAnalysisPass {
    /// Per-guide arc length / curvature / authored-thickness measurement
    /// (`hair_guide_metrics.wesl`).
    GuideMetrics,
    /// Per-render-strand weight blend of the guide metrics
    /// (`hair_binding_metrics.wesl`).
    BindingMetrics,
    /// Per-render-strand importance fold, max-normalized to `[0, 1]`
    /// (`hair_importance.wesl`).
    Importance,
    /// Per-particle squared implicit velocity for the sleep gate
    /// (`hair_motion_energy.wesl`).
    MotionEnergy,
}

impl HairAnalysisPass {
    /// Every hair analysis pass, in canonical order: the density-LOD metric
    /// chain (guide metrics → binding metrics → importance) followed by the
    /// sleep-gate motion energy. Its length equals the number of analysis `WESL`
    /// twins and lets callers enumerate the full set without hand-listing
    /// variants.
    pub const ALL: [HairAnalysisPass; 4] = [
        Self::GuideMetrics,
        Self::BindingMetrics,
        Self::Importance,
        Self::MotionEnergy,
    ];

    /// The `WESL` kernel entry-point name (without the `.wesl` extension), the
    /// same base name as its file in `prism_render_scene/src/shaders/`.
    #[must_use]
    pub fn kernel(self) -> &'static str {
        match self {
            Self::GuideMetrics => "hair_guide_metrics",
            Self::BindingMetrics => "hair_binding_metrics",
            Self::Importance => "hair_importance",
            Self::MotionEnergy => "hair_motion_energy",
        }
    }

    /// The element domain this pass dispatches over. Guide metrics run one
    /// invocation per guide strand; binding metrics and importance run one per
    /// render strand; motion energy runs one per guide particle.
    #[must_use]
    pub fn domain(self) -> HairDispatchDomain {
        match self {
            Self::GuideMetrics => HairDispatchDomain::GuideStrands,
            Self::BindingMetrics | Self::Importance => HairDispatchDomain::RenderStrands,
            Self::MotionEnergy => HairDispatchDomain::GuideParticles,
        }
    }

    /// Number of `@group(0)` storage bindings the kernel declares, so the render
    /// crate can size its bind-group layout against this contract.
    #[must_use]
    pub fn binding_count(self) -> u32 {
        match self {
            Self::Importance => 2,
            Self::BindingMetrics | Self::MotionEnergy => 3,
            Self::GuideMetrics => 4,
        }
    }

    /// The host decision this pass feeds.
    #[must_use]
    pub fn kind(self) -> HairAnalysisKind {
        match self {
            Self::GuideMetrics | Self::BindingMetrics | Self::Importance => {
                HairAnalysisKind::DensityLod
            }
            Self::MotionEnergy => HairAnalysisKind::Sleep,
        }
    }
}

/// One resolved analysis dispatch: the pass, its domain element count, and the
/// 1-D workgroup count to issue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HairAnalysisDispatch {
    /// The analysis pass to bind and run.
    pub pass: HairAnalysisPass,
    /// Number of domain elements this dispatch covers.
    pub domain_count: u32,
    /// Number of workgroups to dispatch along X (`ceil(domain_count / 64)`).
    pub workgroup_count: u32,
}

/// The density-LOD metric chain, in dependency order: measure guides, propagate
/// to render strands, then fold to importance. The host reduces the maxima and
/// runs the CPU-only ranking sort these feed.
#[must_use]
pub fn density_lod_passes() -> [HairAnalysisPass; 3] {
    [
        HairAnalysisPass::GuideMetrics,
        HairAnalysisPass::BindingMetrics,
        HairAnalysisPass::Importance,
    ]
}

/// The sleep-gate analysis passes: the per-particle motion-energy map the host
/// sums for the hysteretic sleep gate.
#[must_use]
pub fn sleep_passes() -> [HairAnalysisPass; 1] {
    [HairAnalysisPass::MotionEnergy]
}

/// Resolves `passes` into ordered [`HairAnalysisDispatch`] entries for `counts`,
/// appending to `out` (which is cleared first). A pass whose domain is empty
/// (zero workgroups) is skipped, so the plan only contains work that will
/// actually run; input order is otherwise preserved. Never panics.
pub fn plan_analysis_dispatches(
    passes: &[HairAnalysisPass],
    counts: &HairGpuCounts,
    out: &mut Vec<HairAnalysisDispatch>,
) {
    out.clear();
    for &pass in passes {
        let domain_count = counts.domain_count(pass.domain());
        let workgroup_count = dispatch_groups(domain_count, HAIR_WORKGROUP_SIZE);
        if workgroup_count == 0 {
            continue;
        }
        out.push(HairAnalysisDispatch {
            pass,
            domain_count,
            workgroup_count,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_lists_every_pass_in_canonical_order() {
        assert_eq!(
            HairAnalysisPass::ALL,
            [
                HairAnalysisPass::GuideMetrics,
                HairAnalysisPass::BindingMetrics,
                HairAnalysisPass::Importance,
                HairAnalysisPass::MotionEnergy,
            ]
        );
        // ALL is exactly the density-LOD chain followed by the sleep passes.
        let mut chained = Vec::new();
        chained.extend_from_slice(&density_lod_passes());
        chained.extend_from_slice(&sleep_passes());
        assert_eq!(chained.as_slice(), &HairAnalysisPass::ALL);
    }

    #[test]
    fn kernel_names_match_wesl_twins() {
        assert_eq!(
            HairAnalysisPass::GuideMetrics.kernel(),
            "hair_guide_metrics"
        );
        assert_eq!(
            HairAnalysisPass::BindingMetrics.kernel(),
            "hair_binding_metrics"
        );
        assert_eq!(HairAnalysisPass::Importance.kernel(), "hair_importance");
        assert_eq!(
            HairAnalysisPass::MotionEnergy.kernel(),
            "hair_motion_energy"
        );
    }

    #[test]
    fn domains_and_kinds_group_the_two_chains() {
        assert_eq!(
            HairAnalysisPass::GuideMetrics.domain(),
            HairDispatchDomain::GuideStrands
        );
        assert_eq!(
            HairAnalysisPass::BindingMetrics.domain(),
            HairDispatchDomain::RenderStrands
        );
        assert_eq!(
            HairAnalysisPass::Importance.domain(),
            HairDispatchDomain::RenderStrands
        );
        assert_eq!(
            HairAnalysisPass::MotionEnergy.domain(),
            HairDispatchDomain::GuideParticles
        );
        for pass in density_lod_passes() {
            assert_eq!(pass.kind(), HairAnalysisKind::DensityLod);
        }
        for pass in sleep_passes() {
            assert_eq!(pass.kind(), HairAnalysisKind::Sleep);
        }
    }

    #[test]
    fn binding_counts_match_the_wesl_group0_bindings() {
        assert_eq!(HairAnalysisPass::GuideMetrics.binding_count(), 4);
        assert_eq!(HairAnalysisPass::BindingMetrics.binding_count(), 3);
        assert_eq!(HairAnalysisPass::Importance.binding_count(), 2);
        assert_eq!(HairAnalysisPass::MotionEnergy.binding_count(), 3);
    }

    #[test]
    fn plan_clears_out_and_skips_empty_domains() {
        let counts = HairGpuCounts {
            roots: 10,
            guide_strands: 10,
            guide_particles: 320,
            render_strands: 0,
            light_texels: 0,
        };
        let mut out = Vec::new();
        out.push(HairAnalysisDispatch {
            pass: HairAnalysisPass::Importance,
            domain_count: 999,
            workgroup_count: 999,
        });
        plan_analysis_dispatches(&HairAnalysisPass::ALL, &counts, &mut out);
        // render_strands == 0 drops BindingMetrics + Importance; guides remain.
        assert_eq!(
            out,
            [
                HairAnalysisDispatch {
                    pass: HairAnalysisPass::GuideMetrics,
                    domain_count: 10,
                    workgroup_count: 1,
                },
                HairAnalysisDispatch {
                    pass: HairAnalysisPass::MotionEnergy,
                    domain_count: 320,
                    workgroup_count: 5,
                },
            ]
        );
    }

    #[test]
    fn plan_is_empty_for_an_empty_groom() {
        let counts = HairGpuCounts::default();
        let mut out = Vec::new();
        plan_analysis_dispatches(&HairAnalysisPass::ALL, &counts, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn density_lod_chain_dispatches_in_dependency_order() {
        let counts = HairGpuCounts {
            roots: 4,
            guide_strands: 4,
            guide_particles: 128,
            render_strands: 200,
            light_texels: 0,
        };
        let mut out = Vec::new();
        plan_analysis_dispatches(&density_lod_passes(), &counts, &mut out);
        let passes: Vec<HairAnalysisPass> = out.iter().map(|d| d.pass).collect();
        assert_eq!(
            passes,
            [
                HairAnalysisPass::GuideMetrics,
                HairAnalysisPass::BindingMetrics,
                HairAnalysisPass::Importance,
            ]
        );
        // 200 render strands -> ceil(200 / 64) == 4 workgroups.
        assert_eq!(out[1].workgroup_count, 4);
        assert_eq!(out[2].workgroup_count, 4);
    }
}
