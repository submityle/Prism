//! Screen-coverage particle LOD, the quality/platform matrix, and the
//! deformation request that charges simulation against the shared budget.
//!
//! A dense emitter cannot simulate and draw every particle at every distance: a
//! fireball filling the screen needs full particle counts and substeps, while
//! the same effect across the map should collapse to a decimated set or a flat
//! impostor. This module maps an emitter's screen coverage to a
//! [`ParticleLodTier`], resolves how many particles and substeps that tier
//! keeps, applies a quality/platform ladder with a deterministic degradation
//! staircase, and — only for simulated tiers — emits the [`DeformationRequest`]
//! that charges the update against the shared deformation budget.
//!
//! Like the sibling cloth and hair subsystems this layer only *emits* requests
//! through [`crate::deformation::schedule`]; it does not own the budget. Every
//! function is a pure, deterministic classification: coverage is supplied by the
//! caller (a screen fraction in `0..=1`), particle counts decimate by fixed
//! integer factors, and no transcendental math is used, so results are exactly
//! reproducible frame to frame.

use alloc::vec::Vec;

use super::EmitterHandle;
use crate::deformation::schedule::DeformationRequest;
use crate::deformation::DeformationHandle;
use crate::deformation::DeformationKind;

/// The rendering form an emitter takes at a given screen coverage.
///
/// Ordered finest-first: [`ParticleLodTier::Full`] is the most detailed and
/// [`ParticleLodTier::Culled`] the coarsest. Only [`ParticleLodTier::Full`] and
/// [`ParticleLodTier::Reduced`] run per-particle simulation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ParticleLodTier {
    /// Full particle count and substeps, per-particle simulation.
    Full,
    /// Decimated particle count and substeps, still simulated.
    Reduced,
    /// A merged billboard/volume impostor: no per-particle simulation.
    Impostor,
    /// Not drawn or simulated this frame.
    Culled,
}

impl ParticleLodTier {
    /// Coarseness rank (`0` finest). Used to clamp against a native form.
    #[must_use]
    pub fn rank(self) -> u8 {
        match self {
            ParticleLodTier::Full => 0,
            ParticleLodTier::Reduced => 1,
            ParticleLodTier::Impostor => 2,
            ParticleLodTier::Culled => 3,
        }
    }

    /// Returns the coarser (higher-rank) of two tiers.
    ///
    /// Clamps a coverage-selected tier so it is never finer than an emitter's
    /// authored native form (an impostor-only smoke card is never promoted to
    /// full simulation, no matter how close the camera).
    #[must_use]
    pub fn coarser_of(self, other: Self) -> Self {
        if self.rank() >= other.rank() {
            self
        } else {
            other
        }
    }

    /// Returns `true` when the tier runs per-particle simulation.
    #[must_use]
    pub fn is_simulated(self) -> bool {
        matches!(self, ParticleLodTier::Full | ParticleLodTier::Reduced)
    }
}

/// Coverage boundaries at which an emitter drops to the next coarser tier.
///
/// Coverage is a screen fraction in `0..=1`. The invariant
/// `reduced_below >= impostor_below >= cull_below` is expected; if it is
/// violated the classification still terminates deterministically by testing
/// boundaries in order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleLodThresholds {
    /// Below this coverage, drop from full to reduced simulation.
    pub reduced_below: f32,
    /// Below this coverage, drop from reduced simulation to an impostor.
    pub impostor_below: f32,
    /// Below this coverage, cull the emitter entirely.
    pub cull_below: f32,
}

/// Classifies a screen coverage into a particle LOD tier.
#[must_use]
pub fn select_particle_lod_tier(
    coverage: f32,
    thresholds: ParticleLodThresholds,
) -> ParticleLodTier {
    if coverage >= thresholds.reduced_below {
        ParticleLodTier::Full
    } else if coverage >= thresholds.impostor_below {
        ParticleLodTier::Reduced
    } else if coverage >= thresholds.cull_below {
        ParticleLodTier::Impostor
    } else {
        ParticleLodTier::Culled
    }
}

/// The authored per-emitter budget LOD decimates from.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EmitterLodInput {
    /// Emitter this input describes.
    pub handle: EmitterHandle,
    /// Deformation handle charged when the emitter simulates.
    pub deformation: DeformationHandle,
    /// Authored maximum live particle count at full detail.
    pub max_particles: u32,
    /// Authored simulation substeps at full detail.
    pub sim_substeps: u32,
    /// Whether the emitter contributes ray-traced geometry (mesh particles),
    /// which needs its acceleration structure refit after simulation.
    pub ray_traced: bool,
    /// The coarsest form the emitter is authored for; the coverage-selected
    /// tier is clamped to be no finer than this.
    pub native_form: ParticleLodTier,
}

/// The resolved LOD for one emitter this frame.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ParticleLodDecision {
    /// Emitter this decision applies to.
    pub handle: EmitterHandle,
    /// Selected tier.
    pub tier: ParticleLodTier,
    /// Live particles kept at this tier (`0` for impostor/culled).
    pub active_particles: u32,
    /// Simulation substeps at this tier (`0` for impostor/culled).
    pub sim_substeps: u32,
}

/// Resolves the particle and substep budget for an emitter at a coverage.
///
/// Full detail keeps the authored counts; reduced detail decimates to a quarter
/// of the particles with half the substeps (never below one); impostor and
/// culled forms keep no simulated particles. The coverage-selected tier is
/// clamped against [`EmitterLodInput::native_form`], so an impostor-authored
/// emitter is never promoted to simulation.
#[must_use]
pub fn resolve_particle_lod(
    input: EmitterLodInput,
    coverage: f32,
    thresholds: ParticleLodThresholds,
) -> ParticleLodDecision {
    let tier = select_particle_lod_tier(coverage, thresholds).coarser_of(input.native_form);
    let (active_particles, sim_substeps) = match tier {
        ParticleLodTier::Full => (input.max_particles, input.sim_substeps),
        ParticleLodTier::Reduced => (
            (input.max_particles / 4).max(1),
            (input.sim_substeps / 2).max(1),
        ),
        ParticleLodTier::Impostor | ParticleLodTier::Culled => (0, 0),
    };
    ParticleLodDecision {
        handle: input.handle,
        tier,
        active_particles,
        sim_substeps,
    }
}

/// Builds the simulation deformation request for a resolved LOD.
///
/// Only simulated tiers charge the budget; impostor and culled emitters return
/// [`None`]. The charged vertex count is the active particle count (the
/// per-frame simulation unit). Ray-traced emitters need their acceleration
/// structure refit after the update, so `needs_blas_refit` follows
/// [`EmitterLodInput::ray_traced`].
#[must_use]
pub fn particle_deformation_request(
    input: EmitterLodInput,
    decision: ParticleLodDecision,
    priority: u32,
) -> Option<DeformationRequest> {
    if !decision.tier.is_simulated() {
        return None;
    }
    Some(DeformationRequest {
        handle: input.deformation,
        kind: DeformationKind::Particle,
        vertex_count: decision.active_particles,
        priority,
        needs_blas_refit: input.ray_traced,
    })
}

/// A rendering quality tier (design §28).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ParticleQuality {
    /// Lowest fidelity: aggressive decimation, impostors kick in early.
    Low,
    /// Medium fidelity.
    Medium,
    /// High fidelity.
    High,
    /// Highest fidelity: full counts held to the smallest coverage.
    Ultra,
}

impl ParticleQuality {
    /// Fidelity rank (`0` lowest).
    #[must_use]
    pub fn rank(self) -> u8 {
        match self {
            ParticleQuality::Low => 0,
            ParticleQuality::Medium => 1,
            ParticleQuality::High => 2,
            ParticleQuality::Ultra => 3,
        }
    }

    /// The next lower quality, or [`None`] at [`ParticleQuality::Low`].
    ///
    /// One rung of the degradation staircase used when the frame budget is
    /// exceeded.
    #[must_use]
    pub fn degrade(self) -> Option<Self> {
        match self {
            ParticleQuality::Ultra => Some(ParticleQuality::High),
            ParticleQuality::High => Some(ParticleQuality::Medium),
            ParticleQuality::Medium => Some(ParticleQuality::Low),
            ParticleQuality::Low => None,
        }
    }

    /// Steps `steps` rungs down the staircase, saturating at
    /// [`ParticleQuality::Low`].
    #[must_use]
    pub fn degrade_steps(self, steps: u32) -> Self {
        let mut q = self;
        for _ in 0..steps {
            match q.degrade() {
                Some(next) => q = next,
                None => break,
            }
        }
        q
    }

    /// Integer divisor applied to a particle budget at this quality.
    ///
    /// Ultra keeps the full budget; each lower rung halves it, so the ladder is
    /// exact and reproducible (no floating-point budget drift).
    #[must_use]
    pub fn particle_divisor(self) -> u32 {
        match self {
            ParticleQuality::Ultra => 1,
            ParticleQuality::High => 2,
            ParticleQuality::Medium => 4,
            ParticleQuality::Low => 8,
        }
    }
}

/// A hardware class with a fidelity ceiling (design §28 platform matrix).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PlatformTier {
    /// Mobile / handheld: capped at [`ParticleQuality::Low`].
    Mobile,
    /// Last-gen or entry console: capped at [`ParticleQuality::Medium`].
    Console,
    /// Desktop GPU: capped at [`ParticleQuality::High`].
    Desktop,
    /// High-end workstation: uncapped ([`ParticleQuality::Ultra`]).
    HighEnd,
}

impl PlatformTier {
    /// The highest quality this platform runs.
    #[must_use]
    pub fn max_quality(self) -> ParticleQuality {
        match self {
            PlatformTier::Mobile => ParticleQuality::Low,
            PlatformTier::Console => ParticleQuality::Medium,
            PlatformTier::Desktop => ParticleQuality::High,
            PlatformTier::HighEnd => ParticleQuality::Ultra,
        }
    }
}

/// Clamps a requested quality to what a platform supports (design §28).
#[must_use]
pub fn resolve_quality(requested: ParticleQuality, platform: PlatformTier) -> ParticleQuality {
    let cap = platform.max_quality();
    if requested.rank() <= cap.rank() {
        requested
    } else {
        cap
    }
}

/// Applies a quality tier's divisor to a particle budget.
///
/// A zero budget stays zero; any positive budget keeps at least one particle so
/// a live emitter never silently vanishes from a quality drop alone.
#[must_use]
pub fn budget_for_quality(max_particles: u32, quality: ParticleQuality) -> u32 {
    if max_particles == 0 {
        return 0;
    }
    (max_particles / quality.particle_divisor()).max(1)
}

/// Emitters partitioned by the LOD tier selected for them this frame.
///
/// The render loop walks the buckets in [`PARTICLE_LOD_ORDER`] to build its
/// indirect simulate/draw passes.
#[derive(Clone, Debug, Default)]
pub struct ParticleLodPlan {
    /// Emitters simulated at full detail.
    pub full: Vec<ParticleLodDecision>,
    /// Emitters simulated at reduced detail.
    pub reduced: Vec<ParticleLodDecision>,
    /// Emitters drawn as merged impostors.
    pub impostor: Vec<ParticleLodDecision>,
    /// Emitters culled this frame.
    pub culled: Vec<ParticleLodDecision>,
}

/// Tier iteration order for building indirect passes.
pub const PARTICLE_LOD_ORDER: [ParticleLodTier; 4] = [
    ParticleLodTier::Full,
    ParticleLodTier::Reduced,
    ParticleLodTier::Impostor,
    ParticleLodTier::Culled,
];

impl ParticleLodPlan {
    /// Total emitters across every bucket.
    #[must_use]
    pub fn total(&self) -> usize {
        self.full.len() + self.reduced.len() + self.impostor.len() + self.culled.len()
    }

    /// Returns `true` when no emitter landed in any bucket.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.full.is_empty()
            && self.reduced.is_empty()
            && self.impostor.is_empty()
            && self.culled.is_empty()
    }

    /// The bucket backing a given tier.
    #[must_use]
    pub fn bucket(&self, tier: ParticleLodTier) -> &[ParticleLodDecision] {
        match tier {
            ParticleLodTier::Full => &self.full,
            ParticleLodTier::Reduced => &self.reduced,
            ParticleLodTier::Impostor => &self.impostor,
            ParticleLodTier::Culled => &self.culled,
        }
    }

    /// Number of emitters routed to a given tier.
    #[must_use]
    pub fn count_of_tier(&self, tier: ParticleLodTier) -> usize {
        self.bucket(tier).len()
    }

    /// Appends a decision to the bucket for its tier.
    pub fn push(&mut self, decision: ParticleLodDecision) {
        match decision.tier {
            ParticleLodTier::Full => self.full.push(decision),
            ParticleLodTier::Reduced => self.reduced.push(decision),
            ParticleLodTier::Impostor => self.impostor.push(decision),
            ParticleLodTier::Culled => self.culled.push(decision),
        }
    }
}

/// Resolves and bins a set of emitters by their per-emitter screen coverage.
///
/// `coverage[i]` is the screen fraction for `inputs[i]`. An emitter with no
/// matching coverage entry is skipped rather than panicking, so a stale or short
/// coverage slice cannot crash LOD selection. Input order is preserved within
/// each bucket.
#[must_use]
pub fn bin_particle_lod(
    inputs: &[EmitterLodInput],
    coverage: &[f32],
    thresholds: ParticleLodThresholds,
) -> ParticleLodPlan {
    let mut plan = ParticleLodPlan::default();
    for (index, &input) in inputs.iter().enumerate() {
        let Some(&cov) = coverage.get(index) else {
            continue;
        };
        plan.push(resolve_particle_lod(input, cov, thresholds));
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    const THRESHOLDS: ParticleLodThresholds = ParticleLodThresholds {
        reduced_below: 0.5,
        impostor_below: 0.2,
        cull_below: 0.05,
    };

    fn input(handle: u32) -> EmitterLodInput {
        EmitterLodInput {
            handle: EmitterHandle(handle),
            deformation: DeformationHandle(handle),
            max_particles: 40_000,
            sim_substeps: 8,
            ray_traced: false,
            native_form: ParticleLodTier::Full,
        }
    }

    fn impostor_authored(handle: u32) -> EmitterLodInput {
        EmitterLodInput {
            native_form: ParticleLodTier::Impostor,
            ..input(handle)
        }
    }

    #[test]
    fn tier_thresholds_are_ordered() {
        assert_eq!(
            select_particle_lod_tier(0.9, THRESHOLDS),
            ParticleLodTier::Full
        );
        assert_eq!(
            select_particle_lod_tier(0.3, THRESHOLDS),
            ParticleLodTier::Reduced
        );
        assert_eq!(
            select_particle_lod_tier(0.1, THRESHOLDS),
            ParticleLodTier::Impostor
        );
        assert_eq!(
            select_particle_lod_tier(0.01, THRESHOLDS),
            ParticleLodTier::Culled
        );
    }

    #[test]
    fn full_detail_keeps_authored_counts() {
        let decision = resolve_particle_lod(input(0), 0.8, THRESHOLDS);
        assert_eq!(decision.tier, ParticleLodTier::Full);
        assert_eq!(decision.active_particles, 40_000);
        assert_eq!(decision.sim_substeps, 8);
    }

    #[test]
    fn reduced_detail_decimates_by_fixed_factors() {
        let decision = resolve_particle_lod(input(0), 0.3, THRESHOLDS);
        assert_eq!(decision.tier, ParticleLodTier::Reduced);
        assert_eq!(decision.active_particles, 10_000);
        assert_eq!(decision.sim_substeps, 4);
    }

    #[test]
    fn impostor_and_culled_drop_simulation() {
        let impostor = resolve_particle_lod(input(0), 0.1, THRESHOLDS);
        assert_eq!(impostor.active_particles, 0);
        assert_eq!(impostor.sim_substeps, 0);
        let culled = resolve_particle_lod(input(0), 0.0, THRESHOLDS);
        assert_eq!(culled.tier, ParticleLodTier::Culled);
        assert_eq!(culled.active_particles, 0);
    }

    #[test]
    fn simulated_tiers_emit_deformation_request() {
        let decision = resolve_particle_lod(input(7), 0.8, THRESHOLDS);
        let request =
            particle_deformation_request(input(7), decision, 3).expect("full tier simulates");
        assert_eq!(request.handle, DeformationHandle(7));
        assert_eq!(request.kind, DeformationKind::Particle);
        assert_eq!(request.vertex_count, 40_000);
        assert_eq!(request.priority, 3);
        assert!(!request.needs_blas_refit);
    }

    #[test]
    fn ray_traced_emitter_requests_blas_refit() {
        let mut rt = input(1);
        rt.ray_traced = true;
        let decision = resolve_particle_lod(rt, 0.8, THRESHOLDS);
        let request = particle_deformation_request(rt, decision, 1).expect("simulates");
        assert!(request.needs_blas_refit);
    }

    #[test]
    fn impostor_tiers_emit_no_deformation_request() {
        let decision = resolve_particle_lod(input(0), 0.1, THRESHOLDS);
        assert!(particle_deformation_request(input(0), decision, 1).is_none());
    }

    #[test]
    fn impostor_authored_emitter_is_never_promoted() {
        // Full-screen coverage would select Full, but an impostor-authored
        // emitter has no per-particle geometry and stays an impostor.
        let decision = resolve_particle_lod(impostor_authored(0), 0.99, THRESHOLDS);
        assert_eq!(decision.tier, ParticleLodTier::Impostor);
        assert_eq!(decision.active_particles, 0);
        assert!(particle_deformation_request(impostor_authored(0), decision, 1).is_none());
    }

    #[test]
    fn impostor_authored_emitter_still_culls_with_distance() {
        let decision = resolve_particle_lod(impostor_authored(0), 0.0, THRESHOLDS);
        assert_eq!(decision.tier, ParticleLodTier::Culled);
    }

    #[test]
    fn quality_clamps_to_platform_ceiling() {
        assert_eq!(
            resolve_quality(ParticleQuality::Ultra, PlatformTier::Mobile),
            ParticleQuality::Low
        );
        assert_eq!(
            resolve_quality(ParticleQuality::Ultra, PlatformTier::Console),
            ParticleQuality::Medium
        );
        assert_eq!(
            resolve_quality(ParticleQuality::Medium, PlatformTier::HighEnd),
            ParticleQuality::Medium
        );
        assert_eq!(
            resolve_quality(ParticleQuality::High, PlatformTier::Desktop),
            ParticleQuality::High
        );
    }

    #[test]
    fn degradation_staircase_saturates_at_low() {
        assert_eq!(
            ParticleQuality::Ultra.degrade(),
            Some(ParticleQuality::High)
        );
        assert_eq!(ParticleQuality::Low.degrade(), None);
        assert_eq!(
            ParticleQuality::Ultra.degrade_steps(2),
            ParticleQuality::Medium
        );
        // Over-stepping clamps at Low rather than wrapping.
        assert_eq!(
            ParticleQuality::Ultra.degrade_steps(10),
            ParticleQuality::Low
        );
        assert_eq!(
            ParticleQuality::High.degrade_steps(0),
            ParticleQuality::High
        );
    }

    #[test]
    fn quality_divisor_scales_budget_but_keeps_one() {
        assert_eq!(budget_for_quality(40_000, ParticleQuality::Ultra), 40_000);
        assert_eq!(budget_for_quality(40_000, ParticleQuality::High), 20_000);
        assert_eq!(budget_for_quality(40_000, ParticleQuality::Medium), 10_000);
        assert_eq!(budget_for_quality(40_000, ParticleQuality::Low), 5_000);
        // A tiny live emitter never vanishes from a quality drop alone.
        assert_eq!(budget_for_quality(3, ParticleQuality::Low), 1);
        // An empty emitter stays empty.
        assert_eq!(budget_for_quality(0, ParticleQuality::Low), 0);
    }

    #[test]
    fn binning_routes_and_preserves_order() {
        let inputs = [input(2), input(0), input(1)];
        let coverage = [0.9, 0.3, 0.9];
        let plan = bin_particle_lod(&inputs, &coverage, THRESHOLDS);
        assert_eq!(plan.total(), 3);
        assert_eq!(plan.count_of_tier(ParticleLodTier::Full), 2);
        assert_eq!(plan.full[0].handle, EmitterHandle(2));
        assert_eq!(plan.full[1].handle, EmitterHandle(1));
        assert_eq!(plan.reduced[0].handle, EmitterHandle(0));
    }

    #[test]
    fn short_coverage_slice_skips_extra_emitters() {
        let inputs = [input(0), input(1)];
        let coverage = [0.9];
        let plan = bin_particle_lod(&inputs, &coverage, THRESHOLDS);
        assert_eq!(plan.total(), 1);
        assert_eq!(plan.full[0].handle, EmitterHandle(0));
    }

    #[test]
    fn empty_input_is_empty_plan() {
        let plan = bin_particle_lod(&[], &[], THRESHOLDS);
        assert!(plan.is_empty());
        assert_eq!(plan.total(), 0);
    }

    #[test]
    fn coarser_of_returns_the_higher_rank_tier() {
        assert_eq!(
            ParticleLodTier::Full.coarser_of(ParticleLodTier::Impostor),
            ParticleLodTier::Impostor
        );
        assert_eq!(
            ParticleLodTier::Impostor.coarser_of(ParticleLodTier::Full),
            ParticleLodTier::Impostor
        );
        assert_eq!(
            ParticleLodTier::Culled.coarser_of(ParticleLodTier::Full),
            ParticleLodTier::Culled
        );
    }
}
