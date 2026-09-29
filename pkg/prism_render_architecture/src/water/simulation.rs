//! Per-frame simulation stepping plan for the water solvers.
//!
//! [`plan_sim`] fuses the per-solver stability rules into one deterministic
//! schedule. Explicit height-field `SWE` stepping obeys a `CFL` bound and may
//! need several sub-steps per frame; spectral `IFFT`/`Tessendorf` and analytic
//! `Gerstner` displacement are unconditionally stable; and the `PBF`/`XPBD`
//! and `FLIP`/`APIC` particle solves run a fixed constraint schedule. Pure
//! scalar planning: no allocation, no transcendental calls, and only the
//! workspace-approved `sqrt` reaches the float unit (inside the upstream
//! dispersion helper).

use super::pbf::{self, PbfParams, PbfSolvePlan};
use super::spectrum::{self, SpectrumParams};
use super::swe;
use super::{SolverKind, EPS, TWO_PI};

/// Bit-exact equality for two `f32` values.
///
/// Compares the raw `u32` bit patterns rather than the values, so the float
/// equality operator is never used (satisfying the determinism policy) while
/// still giving structural equality for the plumbing types below.
#[must_use]
fn bits_eq(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits()
}

/// Ceiling of `frame_dt / stable_dt` as a sub-step count, per the `CFL` rule.
///
/// Returns at least `1`. The fractional part is detected with an epsilon guard
/// so a ratio that is an exact integer (up to rounding) is not rounded up.
/// Callers only invoke this when `stable_dt` is strictly positive and strictly
/// smaller than `frame_dt`, so the result is always `>= 1` before clamping.
#[must_use]
fn ceil_substeps(frame_dt: f32, stable_dt: f32) -> u32 {
    let ratio = frame_dt / stable_dt;
    let n = ratio as u32;
    if (n as f32) + EPS < ratio {
        n + 1
    } else {
        n.max(1)
    }
}

/// Static tuning for one water body's per-frame simulation schedule.
///
/// Aggregates the sea-state spectrum and the `PBF` density-solve tuning with a
/// small set of scheduler knobs (`CFL` safety factor, Jacobian fold threshold,
/// and an upper bound on sub-steps). Cheap to copy and fully value-typed.
#[derive(Clone, Copy, Debug)]
pub struct SimProfile {
    /// Wind-driven sea-state spectrum (feeds the largest-wave dispersion).
    pub spectrum: SpectrumParams,
    /// `PBF`/`XPBD` density-solve tuning for the particle solvers.
    pub pbf: PbfParams,
    /// `CFL` safety factor (`> 0`); clamped internally to `> EPS`.
    pub cfl_number: f32,
    /// Jacobian fold threshold forwarded to `is_wave_folding`.
    pub fold_threshold: f32,
    /// Upper bound on sub-steps per frame (`>= 1` is enforced).
    pub max_substeps: u32,
}

impl PartialEq for SimProfile {
    // Manual, bit-exact equality. The upstream `SpectrumParams` does not
    // implement `PartialEq`, so a derived impl cannot be used; comparing the
    // public scalar fields by bits also keeps the float equality operator out
    // of the source per the determinism policy.
    fn eq(&self, other: &Self) -> bool {
        self.spectrum.kind == other.spectrum.kind
            && bits_eq(self.spectrum.wind.x, other.spectrum.wind.x)
            && bits_eq(self.spectrum.wind.y, other.spectrum.wind.y)
            && bits_eq(self.spectrum.amplitude, other.spectrum.amplitude)
            && bits_eq(
                self.spectrum.peak_enhancement,
                other.spectrum.peak_enhancement,
            )
            && bits_eq(self.spectrum.min_wavelength, other.spectrum.min_wavelength)
            && self.spectrum.directional_exponent == other.spectrum.directional_exponent
            && self.pbf == other.pbf
            && bits_eq(self.cfl_number, other.cfl_number)
            && bits_eq(self.fold_threshold, other.fold_threshold)
            && self.max_substeps == other.max_substeps
    }
}

/// Per-frame dynamic inputs sampled from the live water state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimInputs {
    /// Which solver bucket this body runs.
    pub solver: SolverKind,
    /// Frame duration in seconds.
    pub frame_dt: f32,
    /// Grid cell size `dx` in metres.
    pub cell_size: f32,
    /// `SWE` local maximum signal speed `|u| + sqrt(g*h)` in m/s.
    pub max_signal_speed: f32,
    /// Frame minimum surface Jacobian (choppy-displacement fold measure).
    pub min_jacobian: f32,
}

/// The deterministic stepping schedule produced for one frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimStepPlan {
    /// Number of sub-steps to run this frame (`>= 1`).
    pub substep_count: u32,
    /// Stable timestep for a single sub-step (seconds).
    pub stable_dt: f32,
    /// Whether the full `frame_dt` alone is already stable.
    pub is_stable: bool,
    /// Whether the surface is folding (breaking crest / whitecap seed).
    pub wave_folding: bool,
    /// Angular frequency of the largest sustained wave (rad/s).
    pub peak_omega: f32,
    /// Particle-solver schedule, present only for `PBF`/`FLIP`/`APIC`.
    pub pbf: Option<PbfSolvePlan>,
    /// Total projection iterations across every sub-step.
    pub iteration_count: u32,
}

/// Builds the per-frame stepping schedule for one water body.
///
/// The spectral peak frequency and the folding flag are computed the same way
/// for every solver; the sub-step count, stable timestep, and iteration budget
/// branch on the solver bucket:
///
/// - `ShallowWater`: explicit stepping bounded by the `CFL` timestep, split
///   into as many sub-steps (up to `max_substeps`) as stability requires.
/// - `SpectralIfft`/`Gerstner`: unconditionally stable displacement, one step.
/// - `Pbf`/`FlipApic`: unconditionally stable `XPBD`/particle solve running a
///   fixed constraint-projection schedule in a single step.
///
/// Deterministic and pure: identical inputs always yield an identical plan.
#[must_use]
pub fn plan_sim(profile: SimProfile, inputs: SimInputs) -> SimStepPlan {
    let l = profile.spectrum.largest_wave();
    let k = if l > EPS { TWO_PI / l } else { 0.0 };
    let peak_omega = spectrum::dispersion(k);
    let wave_folding = spectrum::is_wave_folding(inputs.min_jacobian, profile.fold_threshold);

    let cfl = profile.cfl_number.max(EPS);
    let cap = profile.max_substeps.max(1);

    match inputs.solver {
        SolverKind::ShallowWater => {
            let stable_dt = swe::cfl_timestep(inputs.max_signal_speed, inputs.cell_size, cfl);
            let is_stable = swe::is_cfl_stable(
                inputs.frame_dt,
                inputs.cell_size,
                inputs.max_signal_speed,
                cfl,
            );
            let substep_count = if stable_dt > EPS && stable_dt < inputs.frame_dt {
                ceil_substeps(inputs.frame_dt, stable_dt).clamp(1, cap)
            } else {
                1
            };
            SimStepPlan {
                substep_count,
                stable_dt,
                is_stable,
                wave_folding,
                peak_omega,
                pbf: None,
                iteration_count: substep_count,
            }
        }
        SolverKind::SpectralIfft | SolverKind::Gerstner => SimStepPlan {
            substep_count: 1,
            stable_dt: inputs.frame_dt,
            is_stable: true,
            wave_folding,
            peak_omega,
            pbf: None,
            iteration_count: 1,
        },
        SolverKind::Pbf | SolverKind::FlipApic => {
            let p = pbf::plan_solve(profile.pbf);
            SimStepPlan {
                substep_count: 1,
                stable_dt: inputs.frame_dt,
                is_stable: true,
                wave_folding,
                peak_omega,
                pbf: Some(p),
                iteration_count: p.iterations,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::water::spectrum::SpectrumKind;
    use crate::water::Vec2;

    const PROFILE: SimProfile = SimProfile {
        spectrum: SpectrumParams {
            kind: SpectrumKind::Phillips,
            wind: Vec2::new(12.0, 0.0),
            amplitude: 4e-4,
            peak_enhancement: 1.0,
            min_wavelength: 0.2,
            directional_exponent: 2,
        },
        pbf: PbfParams {
            rest_density: 1000.0,
            particle_mass: 1.0,
            smoothing_radius: 0.1,
            relaxation_epsilon: 1e-3,
            artificial_pressure_k: 0.1,
            artificial_pressure_n: 4,
            artificial_pressure_delta_q: 0.2,
            solver_iterations: 4,
        },
        cfl_number: 0.5,
        fold_threshold: 0.0,
        max_substeps: 8,
    };

    fn inputs(solver: SolverKind, frame_dt: f32, dx: f32, speed: f32, jac: f32) -> SimInputs {
        SimInputs {
            solver,
            frame_dt,
            cell_size: dx,
            max_signal_speed: speed,
            min_jacobian: jac,
        }
    }

    #[test]
    fn shallow_water_fast_signal_needs_substeps() {
        let inp = inputs(SolverKind::ShallowWater, 0.1, 1.0, 20.0, 1.0);
        let plan = plan_sim(PROFILE, inp);
        assert!(plan.substep_count > 1, "fast signal should split the frame");
        assert!(plan.substep_count <= PROFILE.max_substeps);
        assert!(!plan.is_stable, "frame_dt alone must be unstable here");
        assert_eq!(plan.iteration_count, plan.substep_count);
        assert!(plan.pbf.is_none());
        // Per-substep dt must be strictly smaller than the frame.
        assert!(plan.stable_dt < inp.frame_dt);
    }

    #[test]
    fn shallow_water_slow_signal_single_substep() {
        let inp = inputs(SolverKind::ShallowWater, 0.016, 1.0, 1.0, 1.0);
        let plan = plan_sim(PROFILE, inp);
        assert_eq!(plan.substep_count, 1);
        assert!(plan.is_stable);
        assert_eq!(plan.iteration_count, 1);
        assert!(plan.pbf.is_none());
    }

    #[test]
    fn shallow_water_substeps_bounded_by_cap() {
        // A very fast signal would demand many sub-steps; the cap must hold.
        let inp = inputs(SolverKind::ShallowWater, 0.1, 1.0, 100_000.0, 1.0);
        let plan = plan_sim(PROFILE, inp);
        assert_eq!(plan.substep_count, PROFILE.max_substeps);
        assert!(!plan.is_stable);
    }

    #[test]
    fn shallow_water_substep_count_is_monotonic_in_speed() {
        let speeds = [0.5_f32, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 1000.0];
        let mut prev = 0_u32;
        for &s in &speeds {
            let plan = plan_sim(PROFILE, inputs(SolverKind::ShallowWater, 0.1, 1.0, s, 1.0));
            assert!(
                plan.substep_count >= prev,
                "substeps must not decrease as speed grows"
            );
            assert!(plan.substep_count >= 1 && plan.substep_count <= PROFILE.max_substeps);
            prev = plan.substep_count;
        }
    }

    #[test]
    fn spectral_and_gerstner_are_single_step() {
        for solver in [SolverKind::SpectralIfft, SolverKind::Gerstner] {
            let inp = inputs(solver, 0.033, 1.0, 50.0, 1.0);
            let plan = plan_sim(PROFILE, inp);
            assert_eq!(plan.substep_count, 1);
            assert!(plan.is_stable);
            assert!(plan.pbf.is_none());
            assert_eq!(plan.iteration_count, 1);
            assert!((plan.stable_dt - inp.frame_dt).abs() < EPS);
        }
    }

    #[test]
    fn particle_solvers_have_a_pbf_plan() {
        for solver in [SolverKind::Pbf, SolverKind::FlipApic] {
            let inp = inputs(solver, 0.033, 1.0, 50.0, 1.0);
            let plan = plan_sim(PROFILE, inp);
            assert_eq!(plan.substep_count, 1);
            assert!(plan.is_stable);
            let p = plan.pbf.expect("particle solver must carry a solve plan");
            assert!(p.iterations >= 1);
            assert_eq!(plan.iteration_count, p.iterations);
            assert!((plan.stable_dt - inp.frame_dt).abs() < EPS);
        }
    }

    #[test]
    fn peak_omega_is_positive_for_wind() {
        let plan = plan_sim(PROFILE, inputs(SolverKind::Gerstner, 0.016, 1.0, 1.0, 1.0));
        assert!(
            plan.peak_omega > 0.0,
            "positive wind sustains a longest wave"
        );
    }

    #[test]
    fn peak_omega_zero_without_wind() {
        let calm = SimProfile {
            spectrum: SpectrumParams {
                wind: Vec2::new(0.0, 0.0),
                ..PROFILE.spectrum
            },
            ..PROFILE
        };
        let plan = plan_sim(calm, inputs(SolverKind::Gerstner, 0.016, 1.0, 1.0, 1.0));
        assert!(
            plan.peak_omega.abs() < EPS,
            "no wind means no sustained wave"
        );
    }

    #[test]
    fn wave_folding_tracks_jacobian_threshold() {
        let folding = plan_sim(PROFILE, inputs(SolverKind::Gerstner, 0.016, 1.0, 1.0, -0.1));
        assert!(folding.wave_folding, "jacobian below threshold folds");
        let flat = plan_sim(PROFILE, inputs(SolverKind::Gerstner, 0.016, 1.0, 1.0, 1.0));
        assert!(!flat.wave_folding, "jacobian above threshold does not fold");
    }

    #[test]
    fn plan_is_deterministic() {
        for solver in [
            SolverKind::ShallowWater,
            SolverKind::SpectralIfft,
            SolverKind::Gerstner,
            SolverKind::Pbf,
            SolverKind::FlipApic,
        ] {
            let inp = inputs(solver, 0.1, 1.0, 20.0, -0.1);
            let a = plan_sim(PROFILE, inp);
            let b = plan_sim(PROFILE, inp);
            assert_eq!(a, b, "identical inputs must yield an identical plan");
        }
    }

    #[test]
    fn ceil_substeps_rounds_up_only_on_a_fraction() {
        assert_eq!(ceil_substeps(0.1, 0.025), 4); // exact multiple
        assert_eq!(ceil_substeps(0.1, 0.03), 4); // 3.33 -> 4
        assert_eq!(ceil_substeps(0.05, 0.05), 1); // ratio 1
    }

    #[test]
    fn profile_partial_eq_is_bit_exact() {
        assert_eq!(PROFILE, PROFILE);
        let bigger = SimProfile {
            max_substeps: PROFILE.max_substeps + 1,
            ..PROFILE
        };
        assert_ne!(PROFILE, bigger);
    }
}
