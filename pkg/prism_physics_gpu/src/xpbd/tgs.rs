//! Temporal Gauss-Seidel (`TGS`) substepping solver for distance constraints.
//!
//! `TGS` is the modern velocity-level alternative to the position-level
//! [`cpu_solve`](super::cpu_solve) `XPBD` sweep. Both split the frame into equal
//! substeps, but they resolve the constraint differently and that difference is
//! the whole point of this module:
//!
//! * `XPBD` corrects *positions* directly, then recovers velocity from the net
//!   motion. The constraint geometry (the direction between two particles) is
//!   sampled once per substep.
//! * `TGS` corrects *velocities* against a position-error *bias*, then
//!   integrates positions with the corrected velocity. Because the geometry is
//!   re-evaluated every substep — and, within a substep, between the biased and
//!   relaxation passes — stiff and fast-moving systems converge in far fewer
//!   iterations. This "re-integrate, then re-linearise" loop is the *temporal*
//!   in Temporal Gauss-Seidel, the scheme shipped by `PhysX` 5 and Chaos.
//!
//! # The substep loop
//!
//! For each of `substeps` equal substeps of size `h = dt / substeps`:
//!
//! 1. **Integrate velocities** — apply gravity and linear damping to every
//!    awake, dynamic particle.
//! 2. **Biased solve** — run `iterations` Gauss-Seidel sweeps over the
//!    colour-ordered constraints. Each sweep drives the relative velocity along
//!    the constraint to a target *bias* that removes a fraction of the position
//!    error `C` (the fraction is set by [`SoftParams`]). A soft constraint
//!    resists via `mass_scale` and relaxes via `impulse_scale`; the rigid limit
//!    removes the whole error over the substep.
//! 3. **Integrate positions** — advance positions with the corrected velocity.
//! 4. **Relax** — run `relax_iterations` *bias-free* sweeps to remove the bias
//!    velocity injected in step 2, so the error correction does not leave
//!    spurious kinetic energy behind.
//!
//! # Correctness model
//!
//! The CPU implementation here is the golden reference a future GPU port is
//! validated against (mirroring the [`cpu_solve`](super::cpu_solve) /
//! `shaders/xpbd.wgsl` pairing). It is cross-checked three ways in its tests:
//! free-fall against the closed-form symplectic-Euler trajectory, the rigid
//! distance limit against the exact rest length, and a settled hanging chain
//! against the independent `XPBD` golden — a rigid `TGS` and a rigid `XPBD`
//! solve the same constraint system and must reach the same static shape.
//!
//! Constraints are projected in *colour order* (no two constraints in a colour
//! share a particle) so the sequential Gauss-Seidel order here maps cleanly
//! onto a future one-colour-per-dispatch GPU kernel, exactly as the `XPBD`
//! solver does.
//!
//! # Provenance
//!
//! Temporal Gauss-Seidel substepping with soft constraints (`PhysX` 5 / Chaos
//! lineage; Catto, "Soft Constraints", GDC 2011 for the coefficient form). No
//! Unreal Engine source or derived code.

use glam::Vec3;

use super::coloring::Colouring;
use super::config::{XpbdConfig, XpbdError};
use super::constraint::DistanceConstraint;
use super::cpu::EPSILON;
use super::state::ParticleState;
use super::tgs_soft::SoftParams;

/// Tunables controlling the Temporal Gauss-Seidel solve.
///
/// The frame step is split into `substeps` equal substeps. Within each substep
/// the biased velocity constraint is swept `iterations` times, positions are
/// integrated, then the bias-free relaxation is swept `relax_iterations` times.
/// Constraint stiffness is specified as a soft spring (`hertz`, `damping_ratio`)
/// so it is independent of the substep size; `hertz = 0` requests the rigid
/// limit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TgsConfig {
    /// Uniform acceleration applied to every awake, dynamic particle each
    /// substep, typically gravity.
    pub gravity: Vec3,
    /// Number of equal substeps the frame step is divided into. Clamped to at
    /// least `1` when stepping.
    pub substeps: u32,
    /// Number of biased Gauss-Seidel sweeps per substep. Clamped to at least
    /// `1` when stepping.
    pub iterations: u32,
    /// Number of bias-free relaxation sweeps per substep. May be `0`.
    pub relax_iterations: u32,
    /// Constraint stiffness frequency, in hertz. `0` (or negative) requests the
    /// rigid limit.
    pub hertz: f32,
    /// Constraint damping ratio; `1` is critically damped.
    pub damping_ratio: f32,
    /// Linear velocity damping coefficient (per second). Each substep scales
    /// velocity by `(1 - damping * h).max(0)`.
    pub damping: f32,
}

impl TgsConfig {
    /// Default substep count.
    pub const DEFAULT_SUBSTEPS: u32 = 8;
    /// Default biased sweeps per substep.
    pub const DEFAULT_ITERATIONS: u32 = 1;
    /// Default relaxation sweeps per substep.
    pub const DEFAULT_RELAX_ITERATIONS: u32 = 1;
    /// Default constraint stiffness (rigid).
    pub const DEFAULT_HERTZ: f32 = 0.0;
    /// Default constraint damping ratio (critical).
    pub const DEFAULT_DAMPING_RATIO: f32 = 1.0;
    /// Default linear velocity damping (per second).
    pub const DEFAULT_DAMPING: f32 = 0.0;

    /// Creates a configuration.
    #[must_use]
    pub fn new(
        gravity: Vec3,
        substeps: u32,
        iterations: u32,
        relax_iterations: u32,
        hertz: f32,
        damping_ratio: f32,
        damping: f32,
    ) -> TgsConfig {
        TgsConfig {
            gravity,
            substeps,
            iterations,
            relax_iterations,
            hertz,
            damping_ratio,
            damping,
        }
    }

    /// Returns the effective substep count (at least `1`).
    #[must_use]
    pub fn effective_substeps(&self) -> u32 {
        self.substeps.max(1)
    }

    /// Returns the effective biased sweep count (at least `1`).
    #[must_use]
    pub fn effective_iterations(&self) -> u32 {
        self.iterations.max(1)
    }

    /// Returns the relaxation sweep count (may be `0`).
    #[must_use]
    pub fn effective_relax_iterations(&self) -> u32 {
        self.relax_iterations
    }

    /// Validates the configuration, returning the first violated invariant.
    ///
    /// # Errors
    ///
    /// Returns [`XpbdError::InvalidConfig`] when `gravity`, `damping`, `hertz`,
    /// or `damping_ratio` is not finite. Zero substep or iteration counts are
    /// permitted and clamped at step time.
    pub fn validate(&self) -> Result<(), XpbdError> {
        if !self.gravity.is_finite() {
            return Err(XpbdError::InvalidConfig("gravity must be finite"));
        }
        if !self.damping.is_finite() {
            return Err(XpbdError::InvalidConfig("damping must be finite"));
        }
        if !self.hertz.is_finite() {
            return Err(XpbdError::InvalidConfig("hertz must be finite"));
        }
        if !self.damping_ratio.is_finite() {
            return Err(XpbdError::InvalidConfig("damping_ratio must be finite"));
        }
        Ok(())
    }
}

impl Default for TgsConfig {
    fn default() -> Self {
        TgsConfig {
            gravity: XpbdConfig::DEFAULT_GRAVITY,
            substeps: Self::DEFAULT_SUBSTEPS,
            iterations: Self::DEFAULT_ITERATIONS,
            relax_iterations: Self::DEFAULT_RELAX_ITERATIONS,
            hertz: Self::DEFAULT_HERTZ,
            damping_ratio: Self::DEFAULT_DAMPING_RATIO,
            damping: Self::DEFAULT_DAMPING,
        }
    }
}

/// Advances `state` under `constraints` by `dt` seconds using Temporal
/// Gauss-Seidel substepping.
///
/// This is the golden reference a future GPU `TGS` kernel is validated against.
/// It mutates `state` in place; positions and velocities carry forward to the
/// next call. Every particle is integrated (the all-awake path); the island
/// stepper will supply a real awake mask through [`run_substeps_tgs`].
///
/// # Errors
///
/// Returns [`XpbdError`] when the config or state is invalid, a constraint
/// indexes a missing particle, or the constraint graph needs more colours than
/// supported. Does nothing (returns `Ok`) when there are no particles or `dt`
/// is non-positive.
pub fn tgs_solve(
    state: &mut ParticleState,
    constraints: &[DistanceConstraint],
    config: &TgsConfig,
    dt: f32,
) -> Result<(), XpbdError> {
    config.validate()?;
    if !state.is_consistent() {
        return Err(XpbdError::InvalidConfig(
            "particle state arrays must have equal length",
        ));
    }
    if state.is_empty() || dt <= 0.0 {
        return Ok(());
    }
    let awake = vec![true; state.len()];
    run_substeps_tgs(state, constraints, &awake, config, dt)
}

/// Runs the Temporal Gauss-Seidel substep loop, integrating only the particles
/// the `awake` mask selects.
///
/// Shared core of the all-awake [`tgs_solve`] and the island-aware stepper, so
/// both callers run the identical arithmetic and differ only in the mask they
/// supply. `awake` must have one entry per particle. Returns `Ok` without
/// touching `state` when the effective substep size is non-positive.
///
/// # Errors
///
/// Returns [`XpbdError`] when the constraint graph needs more colours than
/// supported or indexes a missing particle.
pub(crate) fn run_substeps_tgs(
    state: &mut ParticleState,
    constraints: &[DistanceConstraint],
    awake: &[bool],
    config: &TgsConfig,
    dt: f32,
) -> Result<(), XpbdError> {
    let particle_count = state.len() as u32;
    let colouring = Colouring::build(constraints, particle_count)?;
    let ordered = colouring.reorder(constraints);

    let substeps = config.effective_substeps();
    let iterations = config.effective_iterations();
    let relax_iterations = config.effective_relax_iterations();
    let h = dt / substeps as f32;
    if h <= 0.0 {
        return Ok(());
    }
    let damping_scale = (1.0 - config.damping * h).max(0.0);
    let biased = SoftParams::from_hertz(config.hertz, config.damping_ratio, h);
    let relaxed = SoftParams::relax();

    // Per-constraint accumulated impulse, reset before each sweep group so the
    // soft `impulse_scale` decay acts across the iterations of one group only.
    let mut impulse = vec![0.0f32; ordered.len()];

    for _ in 0..substeps {
        integrate_velocities(state, config.gravity, damping_scale, h, awake);
        sweep(
            &ordered,
            colouring.ranges(),
            &mut impulse,
            state,
            &biased,
            iterations,
        );
        integrate_positions(state, h, awake);
        sweep(
            &ordered,
            colouring.ranges(),
            &mut impulse,
            state,
            &relaxed,
            relax_iterations,
        );
    }
    Ok(())
}

/// Integrates velocity under gravity and linear damping for awake, dynamic
/// particles; pinned (inverse mass `0`) and sleeping particles are left alone.
fn integrate_velocities(
    state: &mut ParticleState,
    gravity: Vec3,
    damping_scale: f32,
    h: f32,
    awake: &[bool],
) {
    for (i, &alive) in awake.iter().enumerate() {
        if !alive || state.inverse_masses[i] <= 0.0 {
            continue;
        }
        let mut v = state.velocities[i];
        v += gravity * h;
        v *= damping_scale;
        state.velocities[i] = v;
    }
}

/// Advances positions with the current velocity for awake, dynamic particles.
fn integrate_positions(state: &mut ParticleState, h: f32, awake: &[bool]) {
    for (i, &alive) in awake.iter().enumerate() {
        if !alive || state.inverse_masses[i] <= 0.0 {
            continue;
        }
        state.positions[i] += state.velocities[i] * h;
    }
}

/// Runs `count` Gauss-Seidel sweeps of the soft velocity constraint over the
/// colour-ordered constraints, resetting the accumulated impulse first so the
/// `impulse_scale` decay acts across this group's iterations only.
fn sweep(
    ordered: &[DistanceConstraint],
    ranges: &[(u32, u32)],
    impulse: &mut [f32],
    state: &mut ParticleState,
    soft: &SoftParams,
    count: u32,
) {
    impulse.iter_mut().for_each(|p| *p = 0.0);
    for _ in 0..count {
        for &(start, end) in ranges {
            for gi in start..end {
                solve_constraint(
                    &ordered[gi as usize],
                    &mut impulse[gi as usize],
                    &state.positions,
                    &mut state.velocities,
                    &state.inverse_masses,
                    soft,
                );
            }
        }
    }
}

/// Applies one soft velocity correction for a single distance constraint.
///
/// Drives the relative velocity along the constraint toward the bias target
/// `-bias_rate * C`, scaled by the soft effective mass and decayed by the
/// accumulated impulse, then applies the resulting impulse to the two
/// velocities with the usual `+n` / `-n` unit gradients.
fn solve_constraint(
    con: &DistanceConstraint,
    impulse: &mut f32,
    positions: &[Vec3],
    velocities: &mut [Vec3],
    inverse_masses: &[f32],
    soft: &SoftParams,
) {
    let ia = con.a as usize;
    let ib = con.b as usize;
    let wa = inverse_masses[ia];
    let wb = inverse_masses[ib];
    let w_sum = wa + wb;
    if w_sum <= 0.0 {
        return;
    }
    let delta = positions[ia] - positions[ib];
    let length = delta.length();
    if length < EPSILON {
        return;
    }
    let normal = delta / length;
    let c = length - con.rest_length;
    let eff_mass = 1.0 / w_sum;
    let v_rel = (velocities[ia] - velocities[ib]).dot(normal);
    let bias = soft.bias_rate * c;
    let delta_impulse =
        -eff_mass * soft.mass_scale * (v_rel + bias) - soft.impulse_scale * *impulse;
    *impulse += delta_impulse;
    let correction = normal * delta_impulse;
    velocities[ia] += correction * wa;
    velocities[ib] -= correction * wb;
}

#[cfg(test)]
mod tests {
    use super::super::cpu::cpu_solve;
    use super::*;

    fn rigid_pair() -> (ParticleState, Vec<DistanceConstraint>) {
        let mut state = ParticleState::new();
        state.push(Vec3::ZERO, 1.0);
        state.push(Vec3::new(2.0, 0.0, 0.0), 1.0);
        let cons = vec![DistanceConstraint::new(0, 1, 1.0, 0.0)];
        (state, cons)
    }

    /// Builds a chain of `n` particles pinned at the first node, spaced one unit
    /// apart along `+x`, with unit rigid links.
    fn hanging_chain(n: u32) -> (ParticleState, Vec<DistanceConstraint>) {
        let mut state = ParticleState::new();
        state.push(Vec3::ZERO, 0.0);
        for i in 1..n {
            state.push(Vec3::new(i as f32, 0.0, 0.0), 1.0);
        }
        let cons = (0..n - 1)
            .map(|i| DistanceConstraint::new(i, i + 1, 1.0, 0.0))
            .collect();
        (state, cons)
    }

    #[test]
    fn no_gravity_rigid_pair_restores_rest_length() {
        let (mut state, cons) = rigid_pair();
        let config = TgsConfig::new(Vec3::ZERO, 1, 4, 1, 0.0, 1.0, 0.0);
        tgs_solve(&mut state, &cons, &config, 1.0 / 60.0).unwrap();
        let length = (state.positions[0] - state.positions[1]).length();
        assert!((length - 1.0).abs() < 1e-4, "length was {length}");
    }

    #[test]
    fn pinned_anchor_never_moves() {
        let mut state = ParticleState::new();
        state.push(Vec3::ZERO, 0.0);
        state.push(Vec3::new(0.0, -1.0, 0.0), 1.0);
        let cons = vec![DistanceConstraint::new(0, 1, 1.0, 0.0)];
        tgs_solve(&mut state, &cons, &TgsConfig::default(), 1.0 / 60.0).unwrap();
        assert_eq!(state.positions[0], Vec3::ZERO);
    }

    #[test]
    fn free_fall_matches_closed_form_without_constraints() {
        // No constraints: each substep is symplectic Euler v += g*h; x += v*h,
        // identical to the XPBD free-fall path.
        let mut state = ParticleState::new();
        state.push(Vec3::ZERO, 1.0);
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let config = TgsConfig::new(gravity, 4, 1, 1, 0.0, 1.0, 0.0);
        let dt = 1.0 / 60.0;
        tgs_solve(&mut state, &[], &config, dt).unwrap();

        let h = dt / 4.0;
        let mut v = Vec3::ZERO;
        let mut x = Vec3::ZERO;
        for _ in 0..4 {
            v += gravity * h;
            x += v * h;
        }
        assert!((state.positions[0] - x).length() < 1e-6);
        assert!((state.velocities[0] - v).length() < 1e-6);
    }

    #[test]
    fn rigid_tgs_settles_to_the_same_shape_as_xpbd() {
        // A rigid TGS and a rigid XPBD solve the same constraint system, so a
        // settled hanging chain must reach the same static catenary shape.
        let (mut tgs_state, cons) = hanging_chain(10);
        let (mut xpbd_state, _) = hanging_chain(10);
        let dt = 1.0 / 60.0;
        // Damping drives both solvers to the same static rest state (the chain
        // hanging straight down); without it a frictionless chain swings forever
        // and the two solvers merely sit at different phases of the same orbit.
        let tgs_config = TgsConfig::new(Vec3::new(0.0, -9.81, 0.0), 16, 4, 2, 0.0, 1.0, 2.0);
        // Match the solved effort: 16 substeps, many projection iterations.
        let xpbd_config = XpbdConfig::new(Vec3::new(0.0, -9.81, 0.0), 16, 40, 2.0);
        for _ in 0..1200 {
            tgs_solve(&mut tgs_state, &cons, &tgs_config, dt).unwrap();
            cpu_solve(&mut xpbd_state, &cons, &xpbd_config, dt).unwrap();
        }
        for i in 0..tgs_state.len() {
            let d = (tgs_state.positions[i] - xpbd_state.positions[i]).length();
            assert!(
                d < 5e-2,
                "node {i} differs by {d}: tgs {:?} xpbd {:?}",
                tgs_state.positions[i],
                xpbd_state.positions[i]
            );
        }
        // Every rigid link holds its rest length at equilibrium.
        for c in &cons {
            let len =
                (tgs_state.positions[c.a as usize] - tgs_state.positions[c.b as usize]).length();
            assert!(
                (len - 1.0).abs() < 2e-2,
                "link {}-{} length {len}",
                c.a,
                c.b
            );
        }
    }

    #[test]
    fn softer_constraints_stretch_more_under_load() {
        // Hang a single bob off a pinned anchor; a lower stiffness must sag
        // (stretch past the rest length) more than a higher stiffness.
        let make = || {
            let mut s = ParticleState::new();
            s.push(Vec3::ZERO, 0.0);
            s.push(Vec3::new(0.0, -1.0, 0.0), 1.0);
            s
        };
        let cons = vec![DistanceConstraint::new(0, 1, 1.0, 0.0)];
        let dt = 1.0 / 60.0;
        let mut soft = make();
        let mut stiff = make();
        let soft_cfg = TgsConfig::new(Vec3::new(0.0, -9.81, 0.0), 4, 1, 0, 2.0, 1.0, 0.5);
        let stiff_cfg = TgsConfig::new(Vec3::new(0.0, -9.81, 0.0), 4, 1, 0, 60.0, 1.0, 0.5);
        for _ in 0..180 {
            tgs_solve(&mut soft, &cons, &soft_cfg, dt).unwrap();
            tgs_solve(&mut stiff, &cons, &stiff_cfg, dt).unwrap();
        }
        let soft_len = (soft.positions[0] - soft.positions[1]).length();
        let stiff_len = (stiff.positions[0] - stiff.positions[1]).length();
        assert!(soft_len > 1.0, "soft link should sag, was {soft_len}");
        assert!(
            soft_len > stiff_len,
            "soft {soft_len} should stretch more than stiff {stiff_len}"
        );
    }

    #[test]
    fn damping_keeps_a_swinging_chain_finite_and_on_circle() {
        let mut state = ParticleState::new();
        state.push(Vec3::ZERO, 0.0);
        state.push(Vec3::new(1.0, 0.0, 0.0), 1.0);
        let cons = vec![DistanceConstraint::new(0, 1, 1.0, 0.0)];
        let config = TgsConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 2, 2, 0.0, 1.0, 2.0);
        for _ in 0..120 {
            tgs_solve(&mut state, &cons, &config, 1.0 / 60.0).unwrap();
        }
        let radius = (state.positions[1] - state.positions[0]).length();
        assert!((radius - 1.0).abs() < 1e-2, "radius drifted to {radius}");
        assert!(state.velocities[1].is_finite());
    }

    #[test]
    fn empty_or_nonpositive_dt_is_a_noop() {
        let (mut state, cons) = rigid_pair();
        let before = state.clone();
        tgs_solve(&mut state, &cons, &TgsConfig::default(), 0.0).unwrap();
        assert_eq!(state, before);
    }

    #[test]
    fn deterministic_across_identical_runs() {
        let (mut a, cons) = hanging_chain(8);
        let (mut b, _) = hanging_chain(8);
        let cfg = TgsConfig::default();
        for _ in 0..60 {
            tgs_solve(&mut a, &cons, &cfg, 1.0 / 60.0).unwrap();
            tgs_solve(&mut b, &cons, &cfg, 1.0 / 60.0).unwrap();
        }
        assert_eq!(a, b);
    }

    #[test]
    fn invalid_config_is_rejected() {
        let (mut state, cons) = rigid_pair();
        let bad = TgsConfig::new(Vec3::new(f32::NAN, 0.0, 0.0), 8, 1, 1, 0.0, 1.0, 0.0);
        assert!(tgs_solve(&mut state, &cons, &bad, 1.0 / 60.0).is_err());
    }
}
