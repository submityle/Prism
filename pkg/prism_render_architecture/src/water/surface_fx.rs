//! Unified surface-effects planner: one call that turns a breaking-wave sample
//! into the full set of downstream water-surface effects.
//!
//! The sibling modules each own one slice of the breaking/foam pipeline:
//! [`super::breaking`] classifies a crest and plans its crest spray, while
//! [`super::foam`] owns the dynamic foam coverage field and its flow-aware
//! decay. Per frame the renderer needs all of those answers together for the
//! same surface sample — the breaking class, a normalized intensity, the foam
//! source strength to inject, the local foam decay rate, and the crest-spray
//! burst handed to the `Ember` particle engine. This module bundles them into
//! a single pure, deterministic [`plan_surface_fx`] so callers assemble one
//! [`SurfaceFxPlan`] instead of threading five separate calls and their shared
//! inputs by hand.
//!
//! Nothing here is a new model: it is a thin, allocation-free aggregator over
//! the existing breaking and foam contracts, so the same crest metrics drive
//! ocean whitecaps, `SWE` shoreline foam, and `PBF` splash spume identically.
//! There is no `GPU` state, no hidden buffers, and no randomness — the plan is
//! a pure function of its inputs and reproduces bit-for-bit frame to frame. The
//! `semi-Lagrangian` foam advection that consumes the returned foam source and
//! decay lives entirely in [`super::foam`]; this module only computes the
//! per-sample rates that feed it.

use super::breaking::{
    breaking_intensity, classify_breaking, foam_source_strength, plan_spray, BreakingClass,
    BreakingCriteria, BreakingSample, SprayEmission,
};
use super::foam::{foam_decay_rate, FoamConfig};
use super::Vec3;

/// Immutable tuning shared by every surface-effect plan on one water body.
///
/// It gathers the breaking [`BreakingCriteria`], the foam-field [`FoamConfig`],
/// and the scalar limits (`max_foam_rate`, `jet_speed`, `max_spray_count`) that
/// bound how much foam and spray a single sample may emit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceFxProfile {
    /// Thresholds classifying and scoring a breaking sample.
    pub criteria: BreakingCriteria,
    /// Grid layout and decay tuning of the target foam field.
    pub foam: FoamConfig,
    /// Peak foam-source rate (per second) a fully breaking sample injects.
    pub max_foam_rate: f32,
    /// Launch-speed scale for the crest-spray jet.
    pub jet_speed: f32,
    /// Upper bound on crest-spray particle count for one sample.
    pub max_spray_count: u32,
}

/// The per-sample surface state a single plan is computed from.
///
/// It carries the breaking [`BreakingSample`] metrics, the surface `tangent`
/// (crest flow direction) and `normal` (upward jet direction) used to aim the
/// spray, and the local `flow_speed` driving foam decay.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceFxInputs {
    /// Breaking metrics (steepness, Jacobian, curvature) for this sample.
    pub sample: BreakingSample,
    /// Surface tangent (crest flow direction) for the spray jet.
    pub tangent: Vec3,
    /// Surface normal (upward jet direction) for the spray jet.
    pub normal: Vec3,
    /// Local surface flow speed driving foam decay.
    pub flow_speed: f32,
}

/// The bundled result: every surface effect for one sample in one struct.
///
/// `class` and `intensity` describe the breaking state, `foam_source` is the
/// rate to inject into the foam field, `foam_decay` is the local decay rate,
/// and `spray` is the crest-spray burst (empty unless the sample is actually
/// breaking).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceFxPlan {
    /// Qualitative breaking state of the sample.
    pub class: BreakingClass,
    /// Normalized breaking intensity in `0..=1`.
    pub intensity: f32,
    /// Foam-source rate (per second) to inject at this sample.
    pub foam_source: f32,
    /// Local foam decay rate (per second) at this sample's flow speed.
    pub foam_decay: f32,
    /// Crest-spray burst, empty unless the sample is breaking.
    pub spray: SprayEmission,
}

/// Plans every surface effect for one breaking sample in a single pass.
///
/// Evaluates the breaking intensity and class, the foam source and decay
/// rates, and the crest-spray burst against the shared `profile`, then
/// assembles them into a [`SurfaceFxPlan`]. The function is pure and
/// deterministic: identical inputs always yield an identical plan, foam and
/// intensity stay in their documented ranges, and spray is emitted only when
/// the sample classifies as [`BreakingClass::Breaking`].
#[must_use]
pub fn plan_surface_fx(profile: SurfaceFxProfile, inputs: SurfaceFxInputs) -> SurfaceFxPlan {
    let intensity = breaking_intensity(inputs.sample, profile.criteria);
    let class = classify_breaking(inputs.sample, profile.criteria);
    let foam_source = foam_source_strength(inputs.sample, profile.criteria, profile.max_foam_rate);
    let foam_decay = foam_decay_rate(inputs.flow_speed, profile.foam);
    let spray = plan_spray(
        inputs.sample,
        profile.criteria,
        inputs.tangent,
        inputs.normal,
        profile.jet_speed,
        profile.max_spray_count,
    );
    SurfaceFxPlan {
        class,
        intensity,
        foam_source,
        foam_decay,
        spray,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROFILE: SurfaceFxProfile = SurfaceFxProfile {
        criteria: BreakingCriteria {
            steepness_threshold: 0.6,
            jacobian_fold_threshold: 0.0,
            curvature_threshold: 1.0,
            breaking_intensity: 0.5,
        },
        foam: FoamConfig {
            nx: 64,
            nz: 64,
            dx: 0.5,
            base_decay: 0.2,
            persistence_floor: 0.02,
            reference_speed: 3.0,
        },
        max_foam_rate: 5.0,
        jet_speed: 4.0,
        max_spray_count: 256,
    };

    fn inputs(steepness: f32, jacobian: f32, curvature: f32, flow_speed: f32) -> SurfaceFxInputs {
        SurfaceFxInputs {
            sample: BreakingSample {
                steepness,
                jacobian,
                curvature,
            },
            tangent: Vec3::new(1.0, 0.0, 0.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
            flow_speed,
        }
    }

    #[test]
    fn calm_sample_produces_no_foam_and_no_spray() {
        let plan = plan_surface_fx(PROFILE, inputs(0.1, 1.0, 0.1, 0.5));
        assert_eq!(plan.class, BreakingClass::Calm);
        assert_eq!(plan.spray, SprayEmission::NONE);
        // Calm foam source is exactly zero; the field is non-negative.
        assert!(plan.foam_source < 1e-6);
        assert!(plan.foam_source <= 0.0);
    }

    #[test]
    fn strongly_breaking_sample_emits_foam_and_spray() {
        let plan = plan_surface_fx(PROFILE, inputs(3.0, -0.5, 8.0, 1.0));
        assert_eq!(plan.class, BreakingClass::Breaking);
        assert!(plan.spray.count > 0);
        assert!(plan.foam_source > 0.0);
        assert!(plan.spray.velocity.length() > 0.0);
    }

    #[test]
    fn foam_decay_is_positive_and_rises_with_flow_speed() {
        let still = plan_surface_fx(PROFILE, inputs(0.1, 1.0, 0.1, 0.0));
        assert!(still.foam_decay > 0.0);
        let mut prev = still.foam_decay;
        let mut speed = 0.0;
        while speed <= 6.0 {
            let plan = plan_surface_fx(PROFILE, inputs(0.1, 1.0, 0.1, speed));
            assert!(plan.foam_decay + 1e-6 >= prev, "foam decay must not fall");
            prev = plan.foam_decay;
            speed += 0.25;
        }
        // Fast water decays strictly faster than perfectly still water.
        let fast = plan_surface_fx(PROFILE, inputs(0.1, 1.0, 0.1, 6.0));
        assert!(fast.foam_decay > still.foam_decay);
    }

    #[test]
    fn intensity_stays_in_unit_range() {
        for &st in &[0.0, 0.3, 0.6, 2.0, 9.0] {
            for &jac in &[-1.0, 0.0, 0.1, 1.0] {
                for &cur in &[0.0, 0.5, 1.0, 6.0] {
                    let plan = plan_surface_fx(PROFILE, inputs(st, jac, cur, 1.0));
                    assert!(
                        (0.0..=1.0).contains(&plan.intensity),
                        "intensity out of range: {}",
                        plan.intensity
                    );
                }
            }
        }
    }

    #[test]
    fn intensity_is_monotonic_in_each_metric() {
        let base = plan_surface_fx(PROFILE, inputs(0.6, 0.0, 1.0, 1.0)).intensity;
        let steeper = plan_surface_fx(PROFILE, inputs(1.2, 0.0, 1.0, 1.0)).intensity;
        let folded = plan_surface_fx(PROFILE, inputs(0.6, -0.5, 1.0, 1.0)).intensity;
        let sharper = plan_surface_fx(PROFILE, inputs(0.6, 0.0, 3.0, 1.0)).intensity;
        assert!(steeper >= base);
        assert!(folded >= base);
        assert!(sharper >= base);
    }

    #[test]
    fn spray_only_fires_for_breaking_samples() {
        // Sweep from calm to violently breaking: any non-empty burst must be
        // paired with a Breaking classification, and a Breaking sample carries
        // no more particles than the configured maximum.
        for &st in &[0.0, 0.5, 1.0, 4.0] {
            for &jac in &[-1.0, 0.0, 0.5, 1.0] {
                for &cur in &[0.0, 1.0, 5.0] {
                    let plan = plan_surface_fx(PROFILE, inputs(st, jac, cur, 1.0));
                    if plan.spray.count > 0 {
                        assert_eq!(plan.class, BreakingClass::Breaking);
                    }
                    if plan.class != BreakingClass::Breaking {
                        assert_eq!(plan.spray, SprayEmission::NONE);
                    }
                    assert!(plan.spray.count <= PROFILE.max_spray_count);
                }
            }
        }
    }

    #[test]
    fn plan_is_deterministic() {
        let a = plan_surface_fx(PROFILE, inputs(2.5, -0.4, 6.0, 2.0));
        let b = plan_surface_fx(PROFILE, inputs(2.5, -0.4, 6.0, 2.0));
        assert_eq!(a, b);
    }
}
