//! Deterministic GPU compute-dispatch contract for the hair pipeline.
//!
//! Every hair compute stage has a `WESL` twin in the scene crate
//! (`prism_render_scene/src/shaders/hair_*.wesl`), each a `@compute
//! @workgroup_size(64)` kernel whose `global_invocation_id.x` indexes one
//! element of a specific domain (a guide strand, a particle, a scalp root, a
//! render strand, or a light texel). Turning a groom into an actual dispatch
//! means: for each kernel, count its domain elements and divide by the shared
//! workgroup size to get a 1-D workgroup count, then issue the kernels in the
//! fixed per-frame order the solver demands.
//!
//! The scene/render crate owns the wgpu buffers, bind groups and queue; this
//! zero-dependency module owns only the *contract* it consumes — the authoritative,
//! deterministic mapping from groom element counts to an ordered list of
//! `(pass, workgroup_count)`. Keeping it here (rather than hard-coded next to
//! the pipeline) means the kernel identity, dispatch domain, binding count,
//! shared `@workgroup_size(64)` and per-frame pass order live in one tested
//! place, in lock-step with the `WESL` twins, and the render crate binds against
//! a stable ABI instead of duplicating the derivation. This is the hair side of
//! the §8 GPU-driven persistence boundary: the architecture crate publishes the
//! dispatch plan, the render graph executes it (design §3 管线 / §8).
//!
//! Everything is pure, integer, deterministic and panic-free: empty domains
//! yield a zero workgroup count (skipped by [`plan_dispatches`]), and a
//! degenerate zero workgroup size yields zero rather than dividing by zero.

use alloc::vec::Vec;

/// The `@workgroup_size` shared by every hair compute kernel. All `hair_*.wesl`
/// twins declare `@compute @workgroup_size(64)` and dispatch 1-D over
/// `global_invocation_id.x`, so one constant governs every workgroup count.
pub const HAIR_WORKGROUP_SIZE: u32 = 64;

/// What a kernel's `global_invocation_id.x` indexes — the element domain whose
/// count drives that kernel's dispatch dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairDispatchDomain {
    /// One invocation per scalp root binding (`hair_root_bind`,
    /// `hair_root_skinning`).
    Roots,
    /// One invocation per guide strand (`hair_sim`).
    GuideStrands,
    /// One invocation per guide particle / control point (`hair_wind`,
    /// `hair_sdf_collision`).
    GuideParticles,
    /// One invocation per render strand (`hair_interp`, `hair_lod_dither`).
    RenderStrands,
    /// One invocation per light-space texel (`hair_transmittance`,
    /// `hair_deep_opacity`).
    LightTexels,
}

/// The coarse pipeline phase a pass belongs to, fixing the order phases run in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairPipelineStage {
    /// One-shot passes run once when a groom is imported/loaded.
    Import,
    /// Per-frame physics passes advancing the guide simulation.
    Simulate,
    /// Per-frame passes deriving render strands from the simulated guides.
    Resolve,
    /// Self-shadow / transmittance build passes.
    Shadow,
}

/// One hair compute pass: a `WESL` kernel plus the metadata needed to bind and
/// dispatch it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairComputePass {
    /// Import-time root projection onto the scalp (`hair_root_bind.wesl`).
    RootBind,
    /// Import-time arc-length resampling to uniform stride (`hair_resample.wesl`).
    Resample,
    /// Per-frame root re-solve from the skinned scalp (`hair_root_skinning.wesl`).
    RootSkinning,
    /// Per-frame wind force pre-pass (`hair_wind.wesl`).
    Wind,
    /// Per-frame guide `XPBD` solve incl. analytic collision (`hair_sim.wesl`).
    GuideSim,
    /// Per-frame heavier `SDF` body collision pass (`hair_sdf_collision.wesl`).
    SdfCollision,
    /// Per-frame guide-to-render interpolation (`hair_interp.wesl`).
    Interpolate,
    /// Per-frame cross-tier LOD dither keep mask (`hair_lod_dither.wesl`).
    LodDither,
    /// Self-shadow voxel transmittance accumulation (`hair_transmittance.wesl`).
    Transmittance,
    /// Deep opacity map slab packing (`hair_deep_opacity.wesl`).
    DeepOpacity,
}

impl HairComputePass {
    /// Every hair compute pass, in canonical pipeline order (import, then the
    /// per-frame simulate/resolve passes, then the self-shadow build). Its
    /// length equals the number of `WESL` twins and lets callers enumerate the
    /// full pass set — e.g. to build a bind-group layout per pass — without
    /// hand-listing variants.
    pub const ALL: [HairComputePass; 10] = [
        Self::RootBind,
        Self::Resample,
        Self::RootSkinning,
        Self::Wind,
        Self::GuideSim,
        Self::SdfCollision,
        Self::Interpolate,
        Self::LodDither,
        Self::Transmittance,
        Self::DeepOpacity,
    ];

    /// The `WESL` kernel entry-point name (without the `.wesl` extension), the
    /// same base name as its file in `prism_render_scene/src/shaders/`.
    #[must_use]
    pub fn kernel(self) -> &'static str {
        match self {
            Self::RootBind => "hair_root_bind",
            Self::Resample => "hair_resample",
            Self::RootSkinning => "hair_root_skinning",
            Self::Wind => "hair_wind",
            Self::GuideSim => "hair_sim",
            Self::SdfCollision => "hair_sdf_collision",
            Self::Interpolate => "hair_interp",
            Self::LodDither => "hair_lod_dither",
            Self::Transmittance => "hair_transmittance",
            Self::DeepOpacity => "hair_deep_opacity",
        }
    }

    /// The element domain this pass dispatches over.
    #[must_use]
    pub fn domain(self) -> HairDispatchDomain {
        match self {
            Self::RootBind | Self::RootSkinning => HairDispatchDomain::Roots,
            Self::Resample | Self::GuideSim => HairDispatchDomain::GuideStrands,
            Self::Wind | Self::SdfCollision => HairDispatchDomain::GuideParticles,
            Self::Interpolate | Self::LodDither => HairDispatchDomain::RenderStrands,
            Self::Transmittance | Self::DeepOpacity => HairDispatchDomain::LightTexels,
        }
    }

    /// Number of `@group(0)` storage bindings the kernel declares, so the render
    /// crate can size its bind-group layout against this contract.
    #[must_use]
    pub fn binding_count(self) -> u32 {
        match self {
            Self::LodDither | Self::Wind => 1,
            Self::SdfCollision => 2,
            Self::Resample | Self::Transmittance => 3,
            Self::RootBind | Self::RootSkinning | Self::Interpolate => 4,
            Self::DeepOpacity => 5,
            Self::GuideSim => 6,
        }
    }

    /// The pipeline phase this pass belongs to.
    #[must_use]
    pub fn stage(self) -> HairPipelineStage {
        match self {
            Self::RootBind | Self::Resample => HairPipelineStage::Import,
            Self::RootSkinning | Self::Wind | Self::GuideSim | Self::SdfCollision => {
                HairPipelineStage::Simulate
            }
            Self::Interpolate | Self::LodDither => HairPipelineStage::Resolve,
            Self::Transmittance | Self::DeepOpacity => HairPipelineStage::Shadow,
        }
    }
}

/// Per-groom element counts feeding dispatch derivation. Each field is the total
/// number of elements of one domain across the whole groom.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HairGpuCounts {
    /// Scalp root bindings (guide strand count; one root per guide).
    pub roots: u32,
    /// Guide strands simulated by `XPBD`.
    pub guide_strands: u32,
    /// Guide particles / control points summed across every guide.
    pub guide_particles: u32,
    /// Render strands produced by interpolation.
    pub render_strands: u32,
    /// Light-space texels in the self-shadow map.
    pub light_texels: u32,
}

impl HairGpuCounts {
    /// The element count for `domain`.
    #[must_use]
    pub fn domain_count(&self, domain: HairDispatchDomain) -> u32 {
        match domain {
            HairDispatchDomain::Roots => self.roots,
            HairDispatchDomain::GuideStrands => self.guide_strands,
            HairDispatchDomain::GuideParticles => self.guide_particles,
            HairDispatchDomain::RenderStrands => self.render_strands,
            HairDispatchDomain::LightTexels => self.light_texels,
        }
    }
}

/// One resolved dispatch: the pass, its domain element count, and the 1-D
/// workgroup count to issue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HairDispatch {
    /// The compute pass to bind and run.
    pub pass: HairComputePass,
    /// Number of domain elements this dispatch covers.
    pub domain_count: u32,
    /// Number of workgroups to dispatch along X (`ceil(domain_count / 64)`).
    pub workgroup_count: u32,
}

/// Workgroups needed to cover `element_count` elements at `workgroup_size` per
/// group: `ceil(element_count / workgroup_size)`. An empty domain needs `0`
/// groups, and a degenerate zero workgroup size returns `0` rather than
/// dividing by zero.
#[must_use]
pub fn dispatch_groups(element_count: u32, workgroup_size: u32) -> u32 {
    if workgroup_size == 0 {
        return 0;
    }
    element_count.div_ceil(workgroup_size)
}

/// The one-shot import passes, in run order: bind roots to the scalp, then
/// resample guides to uniform stride.
#[must_use]
pub fn import_passes() -> [HairComputePass; 2] {
    [HairComputePass::RootBind, HairComputePass::Resample]
}

/// The fixed per-frame pass order of the persistent GPU-driven pipeline
/// (design §3/§8): re-solve roots from the skinned scalp, apply wind, solve the
/// guide `XPBD` (analytic collision projected inside), project the heavier `SDF`
/// body collision, then interpolate render strands and compute the cross-tier
/// LOD dither mask. `self_collision` has no bit-faithful GPU twin (its
/// Gauss-Seidel order does not parallelize) and stays a CPU pass, so it is not
/// listed here.
#[must_use]
pub fn per_frame_passes() -> [HairComputePass; 6] {
    [
        HairComputePass::RootSkinning,
        HairComputePass::Wind,
        HairComputePass::GuideSim,
        HairComputePass::SdfCollision,
        HairComputePass::Interpolate,
        HairComputePass::LodDither,
    ]
}

/// The self-shadow build passes, in run order: accumulate voxel transmittance,
/// then pack the deep opacity slab.
#[must_use]
pub fn shadow_passes() -> [HairComputePass; 2] {
    [HairComputePass::Transmittance, HairComputePass::DeepOpacity]
}

/// Resolves `passes` into ordered [`HairDispatch`] entries for `counts`,
/// appending to `out` (which is cleared first). A pass whose domain is empty
/// (zero workgroups) is skipped, so the plan only contains work that will
/// actually run; input order is otherwise preserved. Never panics.
pub fn plan_dispatches(
    passes: &[HairComputePass],
    counts: &HairGpuCounts,
    out: &mut Vec<HairDispatch>,
) {
    out.clear();
    for &pass in passes {
        let domain_count = counts.domain_count(pass.domain());
        let workgroup_count = dispatch_groups(domain_count, HAIR_WORKGROUP_SIZE);
        if workgroup_count == 0 {
            continue;
        }
        out.push(HairDispatch {
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
    fn dispatch_groups_is_ceiling_division() {
        assert_eq!(dispatch_groups(0, 64), 0);
        assert_eq!(dispatch_groups(1, 64), 1);
        assert_eq!(dispatch_groups(64, 64), 1);
        assert_eq!(dispatch_groups(65, 64), 2);
        assert_eq!(dispatch_groups(128, 64), 2);
        assert_eq!(dispatch_groups(129, 64), 3);
    }

    #[test]
    fn dispatch_groups_zero_workgroup_size_is_safe() {
        assert_eq!(dispatch_groups(1000, 0), 0);
    }

    #[test]
    fn every_pass_maps_to_its_wesl_twin_and_domain() {
        // Kernel name matches the shader file base name.
        assert_eq!(HairComputePass::GuideSim.kernel(), "hair_sim");
        assert_eq!(HairComputePass::DeepOpacity.kernel(), "hair_deep_opacity");
        assert_eq!(HairComputePass::Resample.kernel(), "hair_resample");
        // Domains match each kernel's global_invocation_id guard.
        assert_eq!(
            HairComputePass::GuideSim.domain(),
            HairDispatchDomain::GuideStrands
        );
        assert_eq!(
            HairComputePass::Wind.domain(),
            HairDispatchDomain::GuideParticles
        );
        assert_eq!(
            HairComputePass::RootSkinning.domain(),
            HairDispatchDomain::Roots
        );
        assert_eq!(
            HairComputePass::LodDither.domain(),
            HairDispatchDomain::RenderStrands
        );
        assert_eq!(
            HairComputePass::Transmittance.domain(),
            HairDispatchDomain::LightTexels
        );
    }

    #[test]
    fn binding_counts_match_the_wesl_layouts() {
        assert_eq!(HairComputePass::LodDither.binding_count(), 1);
        assert_eq!(HairComputePass::Wind.binding_count(), 1);
        assert_eq!(HairComputePass::SdfCollision.binding_count(), 2);
        assert_eq!(HairComputePass::Resample.binding_count(), 3);
        assert_eq!(HairComputePass::Transmittance.binding_count(), 3);
        assert_eq!(HairComputePass::Interpolate.binding_count(), 4);
        assert_eq!(HairComputePass::RootBind.binding_count(), 4);
        assert_eq!(HairComputePass::RootSkinning.binding_count(), 4);
        assert_eq!(HairComputePass::DeepOpacity.binding_count(), 5);
        assert_eq!(HairComputePass::GuideSim.binding_count(), 6);
    }

    #[test]
    fn per_frame_order_is_skin_wind_sim_sdf_interp_dither() {
        assert_eq!(
            per_frame_passes(),
            [
                HairComputePass::RootSkinning,
                HairComputePass::Wind,
                HairComputePass::GuideSim,
                HairComputePass::SdfCollision,
                HairComputePass::Interpolate,
                HairComputePass::LodDither,
            ]
        );
    }

    #[test]
    fn stages_group_passes_by_phase() {
        for pass in import_passes() {
            assert_eq!(pass.stage(), HairPipelineStage::Import);
        }
        for pass in [
            HairComputePass::RootSkinning,
            HairComputePass::Wind,
            HairComputePass::GuideSim,
            HairComputePass::SdfCollision,
        ] {
            assert_eq!(pass.stage(), HairPipelineStage::Simulate);
        }
        for pass in [HairComputePass::Interpolate, HairComputePass::LodDither] {
            assert_eq!(pass.stage(), HairPipelineStage::Resolve);
        }
        for pass in shadow_passes() {
            assert_eq!(pass.stage(), HairPipelineStage::Shadow);
        }
    }

    #[test]
    fn all_covers_every_pass_in_phase_order() {
        // `ALL` is exactly the import passes, then the per-frame passes, then the
        // shadow passes, concatenated in run order — a single canonical spine the
        // bind-group planner can walk without re-deriving phase order.
        let mut spine = Vec::new();
        spine.extend_from_slice(&import_passes());
        spine.extend_from_slice(&per_frame_passes());
        spine.extend_from_slice(&shadow_passes());
        assert_eq!(HairComputePass::ALL.to_vec(), spine);

        // Every variant appears exactly once (no dupes, no omissions). We assert
        // presence of each named variant so adding a pass without extending `ALL`
        // fails the build's exhaustiveness here too.
        assert_eq!(HairComputePass::ALL.len(), 10);
        for pass in [
            HairComputePass::RootBind,
            HairComputePass::Resample,
            HairComputePass::RootSkinning,
            HairComputePass::Wind,
            HairComputePass::GuideSim,
            HairComputePass::SdfCollision,
            HairComputePass::Interpolate,
            HairComputePass::LodDither,
            HairComputePass::Transmittance,
            HairComputePass::DeepOpacity,
        ] {
            assert_eq!(
                HairComputePass::ALL.iter().filter(|&&p| p == pass).count(),
                1
            );
        }
    }

    #[test]
    fn plan_derives_ordered_workgroup_counts() {
        let counts = HairGpuCounts {
            roots: 100,
            guide_strands: 100,
            guide_particles: 3200,
            render_strands: 5000,
            light_texels: 0,
        };
        let mut plan = Vec::new();
        plan_dispatches(&per_frame_passes(), &counts, &mut plan);

        // LodDither's domain (render_strands) is non-empty; light_texels is 0 but
        // no per-frame pass uses it, so all six passes survive.
        assert_eq!(plan.len(), 6);
        assert_eq!(plan[0].pass, HairComputePass::RootSkinning);
        assert_eq!(plan[0].domain_count, 100);
        assert_eq!(plan[0].workgroup_count, 2); // ceil(100 / 64)
        assert_eq!(plan[1].pass, HairComputePass::Wind);
        assert_eq!(plan[1].workgroup_count, 50); // ceil(3200 / 64)
        assert_eq!(plan[2].pass, HairComputePass::GuideSim);
        assert_eq!(plan[2].workgroup_count, 2);
        assert_eq!(plan[4].pass, HairComputePass::Interpolate);
        assert_eq!(plan[4].workgroup_count, 79); // ceil(5000 / 64)
    }

    #[test]
    fn plan_skips_empty_domains_and_clears_out() {
        let counts = HairGpuCounts {
            roots: 0,
            guide_strands: 0,
            guide_particles: 0,
            render_strands: 64,
            light_texels: 0,
        };
        // Pre-fill `out` to prove it is cleared first.
        let mut plan = Vec::from([HairDispatch {
            pass: HairComputePass::GuideSim,
            domain_count: 999,
            workgroup_count: 999,
        }]);
        plan_dispatches(&per_frame_passes(), &counts, &mut plan);

        // Only Interpolate and LodDither (render_strands = 64) survive, in order.
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].pass, HairComputePass::Interpolate);
        assert_eq!(plan[0].workgroup_count, 1);
        assert_eq!(plan[1].pass, HairComputePass::LodDither);
        assert_eq!(plan[1].workgroup_count, 1);
    }

    #[test]
    fn empty_counts_yield_empty_plan() {
        let counts = HairGpuCounts::default();
        let mut plan = Vec::new();
        plan_dispatches(&per_frame_passes(), &counts, &mut plan);
        assert!(plan.is_empty());
        plan_dispatches(&import_passes(), &counts, &mut plan);
        assert!(plan.is_empty());
        plan_dispatches(&shadow_passes(), &counts, &mut plan);
        assert!(plan.is_empty());
    }
}
