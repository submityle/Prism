//! Per-pass bind-group layout confluence for the hair *analysis* passes.
//!
//! This is the analysis-cadence twin of [`super::pass_layout`]. Where that
//! module joins the ten per-frame [`HairComputePass`] halves, this one joins the
//! four [`HairAnalysisPass`] halves: [`analysis_dispatch`] answers *how many
//! workgroups* each analysis kernel dispatches, and [`analysis_buffers`] answers
//! *which `@group(0)` storage buffers* each one binds together with their
//! `std430` stride, access and element count. Each source module owns exactly
//! one concern, but the render graph, when it comes to bind an analysis pass,
//! needs both halves at once: the workgroup count *and* the fully-sized binding
//! table.
//!
//! This module is that confluence: given a [`HairAnalysisPass`] and the shared
//! [`HairGpuCounts`], it produces — in a single call — a [`HairAnalysisPassPlan`]
//! carrying the dispatch dimension and a dense `0..binding_count` list of
//! [`HairBindingLayout`] entries. The analysis buffer contracts size purely from
//! [`HairGpuCounts`] (unlike the main passes there is no [`HairGpuExtents`] to
//! thread through), so the join here is even tighter: one `&counts` plans any
//! analysis pass. It re-exports no state and allocates no device resources; it
//! only *joins* the existing contracts, so the two analysis modules stay the
//! single source of truth for their own strides and workgroup counts and the
//! join here can never drift from them (the tests assert
//! `bindings.len() == pass.binding_count()` for every pass).
//!
//! The reused [`HairBindingLayout`] row carries no pass identity, so it is
//! shared verbatim with [`super::pass_layout`]; only the plan wrapper differs,
//! because [`HairAnalysisPassPlan::pass`] is a [`HairAnalysisPass`] rather than a
//! [`HairComputePass`].
//!
//! Everything is pure, integer and deterministic: an empty groom
//! ([`HairGpuCounts::default`]) yields zero workgroups and clamped one-element
//! byte sizes without panicking, matching the empty-domain behaviour the
//! analysis contracts already guarantee.

use alloc::vec::Vec;

use crate::hair::analysis_buffers::{
    HairBindingMetricsBuffer, HairGuideMetricsBuffer, HairImportanceBuffer, HairMotionEnergyBuffer,
};
use crate::hair::analysis_dispatch::HairAnalysisPass;
use crate::hair::gpu_dispatch::{dispatch_groups, HairGpuCounts, HAIR_WORKGROUP_SIZE};
use crate::hair::pass_layout::HairBindingLayout;

/// A fully-planned hair analysis compute pass: the pass identity, its 1-D
/// dispatch dimension (domain element count and derived workgroup count) and the
/// dense list of `@group(0)` bindings with their sizes, plus the summed
/// binding-table byte footprint. This is everything the render graph needs to
/// build a bind group and issue the dispatch for one analysis pass. Contains a
/// [`Vec`], so it is not `Copy`.
///
/// Distinct from [`HairPassPlan`](super::pass_layout::HairPassPlan): its `pass`
/// field is a [`HairAnalysisPass`] running on the LOD-reevaluation / sleep
/// cadence, not a per-frame [`HairComputePass`](super::gpu_dispatch::HairComputePass).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HairAnalysisPassPlan {
    /// The analysis pass this plan binds and dispatches.
    pub pass: HairAnalysisPass,
    /// Number of domain elements this pass covers (from [`HairGpuCounts`]).
    pub domain_count: u32,
    /// Workgroups to dispatch along X (`ceil(domain_count / 64)`); `0` skips.
    pub workgroup_count: u32,
    /// The pass's `@group(0)` bindings, in dense binding order `0..len`.
    pub bindings: Vec<HairBindingLayout>,
    /// Sum of every binding's `byte_size` — the pass's total storage footprint.
    pub total_binding_bytes: usize,
}

/// Appends the `@group(0)` binding layouts of `pass` (in dense binding order) to
/// `out`, which is cleared first. Dispatches to the pass's owning analysis
/// buffer-contract enum for every per-binding value, so the strides and access
/// modes here are exactly those [`analysis_buffers`] publishes. Never panics; an
/// empty groom yields clamped one-element byte sizes.
pub fn analysis_pass_bindings(
    pass: HairAnalysisPass,
    counts: &HairGpuCounts,
    out: &mut Vec<HairBindingLayout>,
) {
    out.clear();
    match pass {
        HairAnalysisPass::GuideMetrics => {
            for buffer in HairGuideMetricsBuffer::ALL {
                out.push(HairBindingLayout {
                    binding: buffer.binding(),
                    access: buffer.access(),
                    stride: buffer.stride(),
                    element_count: buffer.element_count(counts),
                    byte_size: buffer.byte_size(counts),
                    is_output: buffer.is_output(),
                });
            }
        }
        HairAnalysisPass::BindingMetrics => {
            for buffer in HairBindingMetricsBuffer::ALL {
                out.push(HairBindingLayout {
                    binding: buffer.binding(),
                    access: buffer.access(),
                    stride: buffer.stride(),
                    element_count: buffer.element_count(counts),
                    byte_size: buffer.byte_size(counts),
                    is_output: buffer.is_output(),
                });
            }
        }
        HairAnalysisPass::Importance => {
            for buffer in HairImportanceBuffer::ALL {
                out.push(HairBindingLayout {
                    binding: buffer.binding(),
                    access: buffer.access(),
                    stride: buffer.stride(),
                    element_count: buffer.element_count(counts),
                    byte_size: buffer.byte_size(counts),
                    is_output: buffer.is_output(),
                });
            }
        }
        HairAnalysisPass::MotionEnergy => {
            for buffer in HairMotionEnergyBuffer::ALL {
                out.push(HairBindingLayout {
                    binding: buffer.binding(),
                    access: buffer.access(),
                    stride: buffer.stride(),
                    element_count: buffer.element_count(counts),
                    byte_size: buffer.byte_size(counts),
                    is_output: buffer.is_output(),
                });
            }
        }
    }
}

/// Plans a single analysis `pass` into a [`HairAnalysisPassPlan`]: joins its
/// dispatch dimension (domain element count → workgroup count) with its
/// fully-sized `@group(0)` binding table in one call. Never panics.
#[must_use]
pub fn plan_analysis_pass(pass: HairAnalysisPass, counts: &HairGpuCounts) -> HairAnalysisPassPlan {
    let mut bindings = Vec::new();
    analysis_pass_bindings(pass, counts, &mut bindings);
    let total_binding_bytes = bindings.iter().map(|b| b.byte_size).sum();
    let domain_count = counts.domain_count(pass.domain());
    let workgroup_count = dispatch_groups(domain_count, HAIR_WORKGROUP_SIZE);
    HairAnalysisPassPlan {
        pass,
        domain_count,
        workgroup_count,
        bindings,
        total_binding_bytes,
    }
}

/// Plans each analysis pass in `passes` into `out` (cleared first), preserving
/// input order. Unlike [`plan_analysis_dispatches`](super::analysis_dispatch::plan_analysis_dispatches),
/// empty-domain passes are *not* dropped — a bind-group layout is still
/// meaningful for a pass that happens to have zero work this cadence — so
/// callers gate on `workgroup_count == 0` themselves. Never panics.
pub fn plan_analysis_passes(
    passes: &[HairAnalysisPass],
    counts: &HairGpuCounts,
    out: &mut Vec<HairAnalysisPassPlan>,
) {
    out.clear();
    for &pass in passes {
        out.push(plan_analysis_pass(pass, counts));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::analysis_dispatch::density_lod_passes;
    use crate::hair::gpu_buffers::HairBufferAccess;

    /// A non-trivial groom exercising every analysis domain.
    fn sample_counts() -> HairGpuCounts {
        HairGpuCounts {
            roots: 100,
            guide_strands: 100,
            guide_particles: 3200,
            render_strands: 5000,
            light_texels: 4096,
        }
    }

    #[test]
    fn binding_count_matches_dispatch_contract_for_every_pass() {
        // The confluence invariant: the join here binds exactly as many buffers
        // as `analysis_dispatch` promises the kernel declares, for all four
        // analysis passes.
        let counts = sample_counts();
        let mut out = Vec::new();
        for pass in HairAnalysisPass::ALL {
            analysis_pass_bindings(pass, &counts, &mut out);
            assert_eq!(
                out.len() as u32,
                pass.binding_count(),
                "binding count mismatch for {pass:?}"
            );
        }
    }

    #[test]
    fn bindings_are_dense_from_zero() {
        let counts = sample_counts();
        let mut out = Vec::new();
        for pass in HairAnalysisPass::ALL {
            analysis_pass_bindings(pass, &counts, &mut out);
            for (i, binding) in out.iter().enumerate() {
                assert_eq!(binding.binding, i as u32, "sparse binding in {pass:?}");
            }
        }
    }

    #[test]
    fn byte_size_is_stride_times_count_clamped() {
        let counts = sample_counts();
        let mut out = Vec::new();
        for pass in HairAnalysisPass::ALL {
            analysis_pass_bindings(pass, &counts, &mut out);
            for binding in &out {
                let raw = binding
                    .stride
                    .saturating_mul(binding.element_count as usize);
                let expected = raw.max(binding.stride);
                assert_eq!(binding.byte_size, expected, "byte size in {pass:?}");
            }
        }
    }

    #[test]
    fn empty_groom_clamps_to_one_element_and_never_panics() {
        let counts = HairGpuCounts::default();
        let mut out = Vec::new();
        for pass in HairAnalysisPass::ALL {
            analysis_pass_bindings(pass, &counts, &mut out);
            assert_eq!(out.len() as u32, pass.binding_count());
            for binding in &out {
                // Empty groom → the contracts clamp element allocation to one.
                assert_eq!(binding.byte_size, binding.stride);
            }
        }
    }

    #[test]
    fn plan_pass_matches_dispatch_and_binding_join() {
        let counts = sample_counts();
        for pass in HairAnalysisPass::ALL {
            let plan = plan_analysis_pass(pass, &counts);
            assert_eq!(plan.pass, pass);
            // Dispatch half matches analysis_dispatch exactly.
            let domain = counts.domain_count(pass.domain());
            assert_eq!(plan.domain_count, domain);
            assert_eq!(
                plan.workgroup_count,
                dispatch_groups(domain, HAIR_WORKGROUP_SIZE)
            );
            // Binding half matches analysis_pass_bindings exactly.
            let mut expected = Vec::new();
            analysis_pass_bindings(pass, &counts, &mut expected);
            assert_eq!(plan.bindings, expected);
            let sum: usize = expected.iter().map(|b| b.byte_size).sum();
            assert_eq!(plan.total_binding_bytes, sum);
        }
    }

    #[test]
    fn access_and_output_agree_with_source_enums() {
        let counts = sample_counts();
        // GuideMetrics: OutMetrics is the read_write output, Points is read-only.
        let metrics = plan_analysis_pass(HairAnalysisPass::GuideMetrics, &counts);
        let out_metrics = metrics
            .bindings
            .iter()
            .find(|b| b.binding == HairGuideMetricsBuffer::OutMetrics.binding())
            .unwrap();
        assert_eq!(out_metrics.access, HairBufferAccess::ReadWrite);
        assert!(out_metrics.is_output);
        let points = metrics
            .bindings
            .iter()
            .find(|b| b.binding == HairGuideMetricsBuffer::Points.binding())
            .unwrap();
        assert_eq!(points.access, HairBufferAccess::Read);
        assert!(!points.is_output);
    }

    #[test]
    fn each_analysis_pass_has_exactly_one_output() {
        // Every analysis kernel writes a single storage buffer (its metric /
        // importance / energy result); the rest are read-only inputs.
        let counts = sample_counts();
        for pass in HairAnalysisPass::ALL {
            let plan = plan_analysis_pass(pass, &counts);
            let outputs = plan.bindings.iter().filter(|b| b.is_output).count();
            assert_eq!(outputs, 1, "exactly one output for {pass:?}");
        }
    }

    #[test]
    fn plan_passes_preserves_order_and_keeps_empty_domains() {
        // guide_strands drive GuideMetrics; render_strands are zero here so
        // BindingMetrics + Importance carry a layout but no work.
        let counts = HairGpuCounts {
            roots: 0,
            guide_strands: 128,
            guide_particles: 0,
            render_strands: 0,
            light_texels: 0,
        };
        let mut plans = Vec::new();
        plan_analysis_passes(&HairAnalysisPass::ALL, &counts, &mut plans);
        // Every pass is retained (unlike plan_analysis_dispatches), in ALL order.
        assert_eq!(plans.len(), HairAnalysisPass::ALL.len());
        for (plan, &pass) in plans.iter().zip(HairAnalysisPass::ALL.iter()) {
            assert_eq!(plan.pass, pass);
        }
        // GuideMetrics (guide_strands = 128) has work; Importance (render_strands = 0) does not.
        let guide = plans
            .iter()
            .find(|p| p.pass == HairAnalysisPass::GuideMetrics)
            .unwrap();
        assert_eq!(guide.workgroup_count, 2); // ceil(128 / 64)
        let importance = plans
            .iter()
            .find(|p| p.pass == HairAnalysisPass::Importance)
            .unwrap();
        assert_eq!(importance.workgroup_count, 0);
        // Even a zero-work pass still carries its full binding table.
        assert_eq!(
            importance.bindings.len() as u32,
            HairAnalysisPass::Importance.binding_count()
        );
    }

    #[test]
    fn density_lod_chain_plans_in_dependency_order() {
        let counts = sample_counts();
        let mut plans = Vec::new();
        plan_analysis_passes(&density_lod_passes(), &counts, &mut plans);
        let passes: Vec<HairAnalysisPass> = plans.iter().map(|p| p.pass).collect();
        assert_eq!(
            passes,
            [
                HairAnalysisPass::GuideMetrics,
                HairAnalysisPass::BindingMetrics,
                HairAnalysisPass::Importance,
            ]
        );
    }
}
