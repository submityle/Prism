//! Numerical integration stability policy for the particle subsystem
//! (design §25).
//!
//! Where [`super::simulation`] owns the per-particle integrators themselves,
//! this module is the *policy* layer that decides **how a frame is stepped** so
//! those integrators stay stable: how many fixed-`dt` substeps to take, when a
//! `CFL` limit forces more substeps, how far a single step is allowed to move a
//! particle, and which [`IntegratorKind`] fits a given stiffness / accuracy
//! budget. Everything here is a `CPU`-verifiable contract built from ordinary
//! arithmetic and `sqrt` only — no transcendental functions — so a future `GPU`
//! compute kernel can reproduce the same substep schedule bit for bit.
//!
//! # Fixed-`dt` substepping (physics-grade `VFX`)
//! A physics-oriented effect is advanced with a *fixed* per-substep `dt` and an
//! integer substep count, never a variable remainder step. [`plan_substeps`]
//! turns a frame `dt` plus a target substep `dt` into a [`SubstepPlan`] whose
//! `substeps * dt` reproduces the frame exactly (design §25), and
//! [`XpbdSubstepPlan`] extends that to positional (`XPBD`) constraints, which
//! are solved once per substep with the same fixed `dt` so the constraint
//! compliance scales consistently.
//!
//! # `CFL` condition
//! For fluid advection and fast particles the `Courant-Friedrichs-Lewy` (`CFL`)
//! number `max_speed * dt / cell_size` must not exceed a limit, otherwise a
//! particle jumps across more than one grid cell per step and the advection
//! becomes unstable. [`cfl_number`] and [`cfl_decision`] compute that ratio and
//! recommend a substep count (or report a clamp) using only multiply, divide,
//! and comparison.
//!
//! # Clamping (anti-explosion)
//! [`clamp_velocity`] and [`clamp_displacement`] cap the speed and the
//! per-step travel of a particle so a stiff force or a large `dt` cannot make a
//! particle "explode" to infinity. Both preserve direction (scale by
//! `limit / length`), so momentum *direction* is preserved while its magnitude
//! is reduced — clamping is intentionally dissipative and removes kinetic
//! energy rather than injecting it.
//!
//! # Stability tiers and selection
//! [`stable_dt_bound`] gives each integrator's approximate maximum stable `dt`
//! for a given stiffness, and [`select_integrator`] maps an
//! [`IntegrationNeeds`] request (stiffness, positional constraints, accuracy)
//! to an [`IntegratorKind`]: `Verlet` for constraint-driven motion, `RK2` when
//! high accuracy is required, and semi-implicit Euler as the cheap default.
//!
//! # Determinism (design §29)
//! Because every plan here is a *fixed* `dt` with a *fixed* integer substep
//! count and a fixed substep ordering, replaying the same frame inputs yields
//! the identical substep schedule. Combined with the stateless hash `RNG` in
//! [`super::simulation`], this is what makes recorded playback reproducible on
//! both the `CPU` and the `GPU`.

use super::{IntegratorKind, Vec3, EPS_LEN_SQ};

/// A fixed-`dt` substep schedule for one frame (design §25).
///
/// The contract is that `substeps` steps of exactly `dt` seconds each reproduce
/// the frame: `substeps as f32 * dt` equals the original frame `dt` (up to
/// floating-point rounding). Storing the count and the per-substep `dt`
/// together keeps the physics-grade "fixed `dt`, multiple substeps" invariant
/// explicit at every call site.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubstepPlan {
    /// Number of equal substeps to take this frame (always at least one).
    pub substeps: u32,
    /// The fixed per-substep timestep in seconds.
    pub dt: f32,
}

impl SubstepPlan {
    /// The total time covered by the plan, `substeps * dt`.
    ///
    /// Recomposing the frame `dt` this way (rather than trusting the caller's
    /// original value) is the check that a schedule is a faithful fixed-`dt`
    /// split of the frame.
    #[must_use]
    pub fn total_dt(self) -> f32 {
        self.substeps as f32 * self.dt
    }
}

/// A [`SubstepPlan`] paired with a positional-constraint (`XPBD`) solver
/// iteration count (design §25, §7).
///
/// `XPBD` constraints are solved on the *same* substep grid as the integrator:
/// each substep runs `solver_iterations` inner constraint iterations at the
/// plan's fixed `dt`. Sharing the substep `dt` is what keeps the constraint
/// compliance consistent across substeps — see [`XpbdSubstepPlan::compliance_over_dt_sq`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct XpbdSubstepPlan {
    /// The underlying fixed-`dt` substep schedule.
    pub base: SubstepPlan,
    /// Inner constraint-solver iterations per substep (always at least one).
    pub solver_iterations: u32,
}

impl XpbdSubstepPlan {
    /// Builds an `XPBD` plan, forcing at least one solver iteration per
    /// substep.
    #[must_use]
    pub fn new(base: SubstepPlan, solver_iterations: u32) -> Self {
        Self {
            base,
            solver_iterations: solver_iterations.max(1),
        }
    }

    /// The `XPBD` compliance term `alpha / dt^2` for the plan's substep `dt`.
    ///
    /// `XPBD` scales a constraint's raw compliance `alpha` by `1 / dt^2` so the
    /// same material stiffness behaves identically regardless of how many
    /// substeps a frame is split into. A non-positive substep `dt` yields `0.0`
    /// (a rigid constraint) rather than dividing by zero.
    #[must_use]
    pub fn compliance_over_dt_sq(self, compliance: f32) -> f32 {
        let dt = self.base.dt;
        if dt > 0.0 {
            compliance / (dt * dt)
        } else {
            0.0
        }
    }
}

/// The outcome of a `CFL`-driven substep decision (design §25).
///
/// Records the computed `CFL` number, the recommended substep count, and
/// whether that count was clamped to the caller's ceiling (meaning the `CFL`
/// limit could *not* be fully satisfied and the caller should also clamp
/// velocity or shrink the frame `dt`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CflDecision {
    /// The `CFL` number `max_speed * dt / cell_size` for the whole frame.
    pub cfl_number: f32,
    /// Recommended number of substeps to respect the `CFL` limit.
    pub substeps: u32,
    /// `true` when `substeps` hit the ceiling and the limit is still exceeded.
    pub clamped: bool,
}

/// Per-step clamp limits used to prevent a particle from "exploding"
/// (design §25).
///
/// A `max_speed` of `0.0` (or less) disables the speed clamp, and a `max_step`
/// of `0.0` (or less) disables the displacement clamp, so the limits are
/// opt-in per field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepLimits {
    /// Maximum allowed speed (velocity magnitude); non-positive disables it.
    pub max_speed: f32,
    /// Maximum allowed displacement magnitude per substep; non-positive
    /// disables it.
    pub max_step: f32,
}

impl StepLimits {
    /// Applies both clamps to a velocity and its `dt`-scaled displacement.
    ///
    /// The velocity is clamped first, then the displacement of the *clamped*
    /// velocity over `dt` is clamped independently, so neither the speed nor
    /// the single-step travel can exceed its limit. Both clamps preserve
    /// direction and are dissipative (see the module docs).
    #[must_use]
    pub fn apply(self, velocity: Vec3, dt: f32) -> ClampedStep {
        let clamped_velocity = clamp_velocity(velocity, self.max_speed);
        let displacement = clamp_displacement(clamped_velocity.scale(dt), self.max_step);
        ClampedStep {
            velocity: clamped_velocity,
            displacement,
        }
    }
}

/// The clamped velocity and displacement produced by [`StepLimits::apply`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClampedStep {
    /// The speed-clamped velocity.
    pub velocity: Vec3,
    /// The displacement-clamped position delta for this substep.
    pub displacement: Vec3,
}

/// A request describing what an effect needs from its integrator (design §25).
///
/// Fed to [`select_integrator`] to pick an [`IntegratorKind`]. `stiffness` is
/// the dominant spring/constraint stiffness (used by [`stable_dt_bound`], not
/// by the selection itself); `positional_constraints` marks `XPBD`-style
/// effects; `high_accuracy` requests a second-order integrator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntegrationNeeds {
    /// Dominant stiffness of the effect's forces or constraints.
    pub stiffness: f32,
    /// `true` when the effect is driven by positional (`XPBD`) constraints.
    pub positional_constraints: bool,
    /// `true` when second-order accuracy is required.
    pub high_accuracy: bool,
}

/// Splits a frame into fixed-`dt` substeps given a *target* substep `dt`
/// (design §25).
///
/// Computes `substeps = ceil(frame_dt / target_substep_dt)`, clamped to
/// `1..=max_substeps`, then returns the *actual* fixed `dt = frame_dt /
/// substeps` so the substeps recompose the frame exactly. Degenerate inputs
/// (a non-positive `frame_dt` or `target_substep_dt`) fall back to a single
/// substep of the whole frame.
#[must_use]
pub fn plan_substeps(frame_dt: f32, target_substep_dt: f32, max_substeps: u32) -> SubstepPlan {
    let ceiling = max_substeps.max(1);
    if frame_dt <= 0.0 || target_substep_dt <= 0.0 {
        return SubstepPlan {
            substeps: 1,
            dt: frame_dt.max(0.0),
        };
    }
    let needed = (frame_dt / target_substep_dt).ceil();
    let substeps = if needed >= ceiling as f32 {
        ceiling
    } else {
        // `needed` is finite and at least 1.0 here, so the cast is exact.
        (needed as u32).max(1)
    };
    SubstepPlan {
        substeps,
        dt: frame_dt / substeps as f32,
    }
}

/// Splits a frame into an explicit number of fixed-`dt` substeps (design §25).
///
/// Forces at least one substep and returns `dt = frame_dt / substeps`. A
/// non-positive `frame_dt` yields a zero `dt`, which lets callers pass an
/// idle frame through the same contract without special-casing it.
#[must_use]
pub fn plan_substeps_fixed(frame_dt: f32, substeps: u32) -> SubstepPlan {
    let substeps = substeps.max(1);
    let dt = if frame_dt > 0.0 {
        frame_dt / substeps as f32
    } else {
        0.0
    };
    SubstepPlan { substeps, dt }
}

/// The `CFL` number `max_speed * dt / cell_size` (design §25).
///
/// This is the fraction of a grid cell a particle at `max_speed` crosses in one
/// step of `dt`; advection is stable when it stays below the scheme's `CFL`
/// limit. A non-positive `cell_size` returns `0.0` rather than dividing by
/// zero.
#[must_use]
pub fn cfl_number(max_speed: f32, dt: f32, cell_size: f32) -> f32 {
    if cell_size > 0.0 {
        max_speed * dt / cell_size
    } else {
        0.0
    }
}

/// Recommends a substep count that keeps the `CFL` number under `cfl_limit`
/// (design §25).
///
/// Splitting a frame into `n` substeps divides the per-step `CFL` number by
/// `n`, so the required count is `ceil(cfl_number / cfl_limit)`, clamped to
/// `1..=max_substeps`. When that ceiling is not enough to satisfy the limit the
/// returned [`CflDecision`] has `clamped == true`, signalling the caller to
/// also clamp velocity (see [`clamp_velocity`]) or shrink the frame `dt`.
/// Degenerate inputs fall back to a single unclamped substep.
#[must_use]
pub fn cfl_decision(
    max_speed: f32,
    dt: f32,
    cell_size: f32,
    cfl_limit: f32,
    max_substeps: u32,
) -> CflDecision {
    let number = cfl_number(max_speed, dt, cell_size);
    let ceiling = max_substeps.max(1);
    if number <= cfl_limit || cfl_limit <= 0.0 {
        return CflDecision {
            cfl_number: number,
            substeps: 1,
            clamped: false,
        };
    }
    let needed = (number / cfl_limit).ceil();
    if needed >= ceiling as f32 {
        CflDecision {
            cfl_number: number,
            substeps: ceiling,
            clamped: true,
        }
    } else {
        CflDecision {
            cfl_number: number,
            substeps: (needed as u32).max(1),
            clamped: false,
        }
    }
}

/// Clamps a velocity to `max_speed`, preserving direction (design §25).
///
/// A non-positive `max_speed` disables the clamp. When the speed exceeds the
/// limit the vector is scaled by `max_speed / length`; this removes kinetic
/// energy (dissipative) and keeps the momentum *direction* unchanged.
#[must_use]
pub fn clamp_velocity(velocity: Vec3, max_speed: f32) -> Vec3 {
    if max_speed <= 0.0 {
        return velocity;
    }
    let speed_sq = velocity.length_squared();
    if speed_sq > max_speed * max_speed && speed_sq > EPS_LEN_SQ {
        velocity.scale(max_speed / speed_sq.sqrt())
    } else {
        velocity
    }
}

/// Clamps a per-step displacement to `max_step`, preserving direction
/// (design §25).
///
/// A non-positive `max_step` disables the clamp. This bounds how far a single
/// substep can move a particle regardless of its velocity, which is the last
/// line of defence against tunnelling and blow-ups when a force spikes.
#[must_use]
pub fn clamp_displacement(delta: Vec3, max_step: f32) -> Vec3 {
    if max_step <= 0.0 {
        return delta;
    }
    let len_sq = delta.length_squared();
    if len_sq > max_step * max_step && len_sq > EPS_LEN_SQ {
        delta.scale(max_step / len_sq.sqrt())
    } else {
        delta
    }
}

/// Approximate maximum stable `dt` for `kind` at a given `stiffness`
/// (design §25).
///
/// For an undamped spring the natural frequency is `omega = sqrt(stiffness)`
/// (unit mass), and each integrator is stable up to `dt <= coefficient /
/// omega`. The explicit first-order schemes (semi-implicit Euler, `Verlet`)
/// share a coefficient near `2`, while midpoint `RK2` tolerates a slightly
/// larger step near `2 * sqrt(2)`. A non-positive `stiffness` has no stability
/// bound and returns [`f32::INFINITY`].
#[must_use]
pub fn stable_dt_bound(kind: IntegratorKind, stiffness: f32) -> f32 {
    if stiffness <= 0.0 {
        return f32::INFINITY;
    }
    let coefficient = match kind {
        IntegratorKind::SemiImplicitEuler | IntegratorKind::Verlet => 2.0,
        IntegratorKind::Rk2 => 2.0 * 2.0_f32.sqrt(),
    };
    coefficient / stiffness.sqrt()
}

/// Chooses an [`IntegratorKind`] for the given [`IntegrationNeeds`]
/// (design §25).
///
/// Positional (`XPBD`) constraints take priority and select `Verlet`, whose
/// implied-velocity update pairs naturally with position projection. Otherwise
/// a `high_accuracy` request selects midpoint `RK2`, and everything else uses
/// semi-implicit Euler — the cheap, robust default. `stiffness` does not change
/// the choice here; use [`stable_dt_bound`] to size the step for it.
#[must_use]
pub fn select_integrator(needs: IntegrationNeeds) -> IntegratorKind {
    if needs.positional_constraints {
        IntegratorKind::Verlet
    } else if needs.high_accuracy {
        IntegratorKind::Rk2
    } else {
        IntegratorKind::SemiImplicitEuler
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_substeps_recomposes_the_frame() {
        // frame 0.1s, target 0.02s => ceil(5) = 5 substeps of 0.02s.
        let plan = plan_substeps(0.1, 0.02, 64);
        assert_eq!(plan.substeps, 5);
        assert!((plan.dt - 0.02).abs() < 1e-6);
        assert!((plan.total_dt() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn plan_substeps_rounds_up_and_shrinks_dt() {
        // 0.1 / 0.03 = 3.33.. => 4 substeps, dt shrinks to 0.025 to fit exactly.
        let plan = plan_substeps(0.1, 0.03, 64);
        assert_eq!(plan.substeps, 4);
        assert!((plan.dt - 0.025).abs() < 1e-6);
        assert!((plan.total_dt() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn plan_substeps_clamps_to_ceiling() {
        // Would need 100 substeps but is capped at 8.
        let plan = plan_substeps(1.0, 0.01, 8);
        assert_eq!(plan.substeps, 8);
        assert!((plan.total_dt() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn plan_substeps_handles_degenerate_inputs() {
        // dt = 0 boundary: single substep of zero duration.
        let zero = plan_substeps(0.0, 0.02, 8);
        assert_eq!(zero.substeps, 1);
        assert!(zero.dt.abs() < 1e-9);
        // Non-positive target falls back to one substep of the whole frame.
        let no_target = plan_substeps(0.1, 0.0, 8);
        assert_eq!(no_target.substeps, 1);
        assert!((no_target.dt - 0.1).abs() < 1e-6);
        // A zero ceiling is treated as one.
        let no_ceiling = plan_substeps(0.1, 0.01, 0);
        assert_eq!(no_ceiling.substeps, 1);
    }

    #[test]
    fn plan_substeps_fixed_divides_evenly() {
        let plan = plan_substeps_fixed(0.1, 4);
        assert_eq!(plan.substeps, 4);
        assert!((plan.dt - 0.025).abs() < 1e-6);
        // Zero substeps is forced to one; zero frame yields zero dt.
        let one = plan_substeps_fixed(0.1, 0);
        assert_eq!(one.substeps, 1);
        let idle = plan_substeps_fixed(0.0, 4);
        assert_eq!(idle.substeps, 4);
        assert!(idle.dt.abs() < 1e-9);
    }

    #[test]
    fn xpbd_plan_scales_compliance_by_inverse_dt_squared() {
        let base = plan_substeps_fixed(0.1, 5); // dt = 0.02
        let plan = XpbdSubstepPlan::new(base, 0);
        // Solver iterations forced to at least one.
        assert_eq!(plan.solver_iterations, 1);
        // alpha / dt^2 = 1e-4 / (0.02^2) = 1e-4 / 4e-4 = 0.25.
        assert!((plan.compliance_over_dt_sq(1.0e-4) - 0.25).abs() < 1e-6);
        // Zero dt gives a rigid (zero-compliance) term without dividing by zero.
        let rigid = XpbdSubstepPlan::new(plan_substeps_fixed(0.0, 4), 3);
        assert!(rigid.compliance_over_dt_sq(1.0e-4).abs() < 1e-9);
        assert_eq!(rigid.solver_iterations, 3);
    }

    #[test]
    fn cfl_number_is_travel_over_cell() {
        // 100 * 0.1 / 2 = 5.
        assert!((cfl_number(100.0, 0.1, 2.0) - 5.0).abs() < 1e-6);
        // Non-positive cell size avoids division by zero.
        assert!(cfl_number(100.0, 0.1, 0.0).abs() < 1e-9);
    }

    #[test]
    fn cfl_decision_recommends_substeps() {
        // number = 5, limit = 1 => needs 5 substeps, under the ceiling.
        let d = cfl_decision(100.0, 0.1, 2.0, 1.0, 64);
        assert!((d.cfl_number - 5.0).abs() < 1e-6);
        assert_eq!(d.substeps, 5);
        assert!(!d.clamped);
    }

    #[test]
    fn cfl_decision_clamps_when_ceiling_too_low() {
        // number = 5, limit = 1, ceiling = 4 => clamped at 4.
        let d = cfl_decision(100.0, 0.1, 2.0, 1.0, 4);
        assert_eq!(d.substeps, 4);
        assert!(d.clamped);
    }

    #[test]
    fn cfl_decision_single_step_when_within_limit() {
        // number = 0.5 <= limit 1 => one unclamped substep.
        let d = cfl_decision(10.0, 0.1, 2.0, 1.0, 8);
        assert_eq!(d.substeps, 1);
        assert!(!d.clamped);
        // A non-positive limit also degrades to a single substep.
        let no_limit = cfl_decision(100.0, 0.1, 2.0, 0.0, 8);
        assert_eq!(no_limit.substeps, 1);
        assert!(!no_limit.clamped);
    }

    #[test]
    fn clamp_velocity_limits_magnitude_only() {
        let clamped = clamp_velocity(Vec3::new(3.0, 4.0, 0.0), 2.5);
        assert!((clamped.length() - 2.5).abs() < 1e-6);
        // Direction is preserved: components stay proportional (3:4).
        assert!((clamped.y / clamped.x - 4.0 / 3.0).abs() < 1e-6);
        // Under the limit the velocity is untouched.
        assert_eq!(
            clamp_velocity(Vec3::new(1.0, 0.0, 0.0), 2.5),
            Vec3::new(1.0, 0.0, 0.0)
        );
        // A non-positive limit disables the clamp.
        assert_eq!(
            clamp_velocity(Vec3::new(9.0, 0.0, 0.0), 0.0),
            Vec3::new(9.0, 0.0, 0.0)
        );
    }

    #[test]
    fn clamp_displacement_limits_travel() {
        let clamped = clamp_displacement(Vec3::new(0.0, 10.0, 0.0), 1.0);
        assert!((clamped.length() - 1.0).abs() < 1e-6);
        // Within the limit, unchanged.
        assert_eq!(
            clamp_displacement(Vec3::new(0.0, 0.5, 0.0), 1.0),
            Vec3::new(0.0, 0.5, 0.0)
        );
    }

    #[test]
    fn step_limits_apply_both_clamps() {
        let limits = StepLimits {
            max_speed: 2.0,
            max_step: 0.1,
        };
        // Speed 10 clamped to 2, then displacement over dt=0.1 clamped to 0.1.
        let step = limits.apply(Vec3::new(10.0, 0.0, 0.0), 0.1);
        assert!((step.velocity.length() - 2.0).abs() < 1e-6);
        assert!((step.displacement.length() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn stable_dt_bound_shrinks_with_stiffness() {
        // Semi-implicit Euler at stiffness 4: omega = 2, bound = 2/2 = 1.
        let euler = stable_dt_bound(IntegratorKind::SemiImplicitEuler, 4.0);
        assert!((euler - 1.0).abs() < 1e-6);
        // Verlet shares the coefficient.
        let verlet = stable_dt_bound(IntegratorKind::Verlet, 4.0);
        assert!((verlet - euler).abs() < 1e-6);
        // RK2 tolerates a larger step at the same stiffness.
        let rk2 = stable_dt_bound(IntegratorKind::Rk2, 4.0);
        assert!(rk2 > euler);
        // No stiffness => unbounded.
        assert!(stable_dt_bound(IntegratorKind::SemiImplicitEuler, 0.0).is_infinite());
    }

    #[test]
    fn select_integrator_follows_priority() {
        // Constraints win, even when high accuracy is also requested.
        assert_eq!(
            select_integrator(IntegrationNeeds {
                stiffness: 100.0,
                positional_constraints: true,
                high_accuracy: true,
            }),
            IntegratorKind::Verlet
        );
        // High accuracy without constraints picks RK2.
        assert_eq!(
            select_integrator(IntegrationNeeds {
                stiffness: 1.0,
                positional_constraints: false,
                high_accuracy: true,
            }),
            IntegratorKind::Rk2
        );
        // The cheap default otherwise.
        assert_eq!(
            select_integrator(IntegrationNeeds {
                stiffness: 1.0,
                positional_constraints: false,
                high_accuracy: false,
            }),
            IntegratorKind::SemiImplicitEuler
        );
    }
}
