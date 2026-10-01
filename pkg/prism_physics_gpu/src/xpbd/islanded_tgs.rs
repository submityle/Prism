//! Island-aware, sleep-skipping stepping of the `TGS` distance solver.
//!
//! [`tgs_solve`](super::tgs::tgs_solve) is the dense reference: it integrates
//! and projects every particle every frame with the Temporal Gauss-Seidel
//! sweep. That is the right contract for the `GPU` parity twin, but it wastes
//! work on a scene full of settled stacks, and the solver's residual jitter
//! keeps those stacks faintly alive forever.
//!
//! [`IslandedTgsSolver`] wraps the same numeric core with the island partition
//! and per-island sleeping from [`crate::island`]: each frame it partitions the
//! constraint graph, lets a persistent [`SleepState`] decide which islands have
//! come to rest, and then runs the substep loop over *only the awake islands'
//! constraints*, integrating *only awake particles*. A settled stack costs
//! nothing until something disturbs it, and the moment any member of an island
//! moves, the whole island wakes as a unit. Because the awake path funnels
//! through [`run_substeps_tgs`](super::tgs::run_substeps_tgs) — the identical
//! integrate / solve / relax primitives [`tgs_solve`](super::tgs::tgs_solve)
//! uses — an awake island's trajectory is bit-for-bit what the dense solver
//! would have produced.
//!
//! This is the `TGS` counterpart of
//! [`IslandedSolver`](super::islanded::IslandedSolver), which does the same for
//! the baseline `XPBD` core; the two share the per-frame [`IslandStep`] summary
//! and the identical sleep bookkeeping, differing only in the inner solve.
//!
//! Provenance: standard island partitioning + velocity-threshold island
//! sleeping layered over substep Temporal Gauss-Seidel (Müller et al.; Macklin
//! et al. soft-constraint TGS). No Unreal Engine source or derived code.

use glam::Vec3;

use crate::island::{build_islands, SleepConfig, SleepState};

use super::config::XpbdError;
use super::constraint::DistanceConstraint;
use super::islanded::IslandStep;
use super::state::ParticleState;
use super::tgs::{run_substeps_tgs, TgsConfig};

/// A stateful `TGS` stepper that partitions into islands and sleeps the ones
/// at rest.
///
/// Hold one of these across frames and drive it with [`step`](Self::step); it
/// owns the persistent [`SleepState`] that accumulates each particle's quiet
/// time. The tracker is keyed by particle, not by island id (which is not stable
/// across re-partitions), so rebuilding the island set from scratch every frame
/// — as [`step`](Self::step) does — never loses sleep progress.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IslandedTgsSolver {
    /// Per-particle sleep bookkeeping carried across frames.
    sleep: SleepState,
    /// Thresholds governing when a quiet island falls asleep.
    config: SleepConfig,
}

impl IslandedTgsSolver {
    /// Creates a solver with the given sleep thresholds and an empty tracker.
    ///
    /// The tracker sizes itself to the particle count on the first
    /// [`step`](Self::step), so there is no need to know the count up front.
    #[must_use]
    pub fn new(config: SleepConfig) -> IslandedTgsSolver {
        IslandedTgsSolver {
            sleep: SleepState::default(),
            config,
        }
    }

    /// The sleep thresholds this solver is using.
    #[must_use]
    pub fn sleep_config(&self) -> SleepConfig {
        self.config
    }

    /// Replaces the sleep thresholds; takes effect on the next step.
    pub fn set_sleep_config(&mut self, config: SleepConfig) {
        self.config = config;
    }

    /// Borrows the persistent sleep tracker for inspection.
    #[must_use]
    pub fn sleep_state(&self) -> &SleepState {
        &self.sleep
    }

    /// Forces `particle` awake, resetting its quiet timer.
    ///
    /// Use this to inject a disturbance the velocity test cannot see — a
    /// teleport, an externally applied impulse — before the next step.
    pub fn wake(&mut self, particle: u32) {
        self.sleep.wake(particle);
    }

    /// Forces every particle awake, resetting all quiet timers.
    pub fn wake_all(&mut self) {
        self.sleep.wake_all();
    }

    /// Advances `state` under `constraints` by `dt`, skipping asleep islands.
    ///
    /// The frame proceeds in four stages:
    /// 1. Partition the constraint graph into islands (pinned anchors are shared
    ///    but never bridge islands).
    /// 2. Update the sleep tracker from the carried-in velocities: islands whose
    ///    members have all been quiet for the dwell time sleep as a unit; any
    ///    island with a moving member is forced fully awake.
    /// 3. Run the substep `TGS` loop over the awake islands' constraints,
    ///    integrating only awake, dynamic particles.
    /// 4. Pin asleep particles' velocities to exactly zero so residual jitter
    ///    cannot creep them.
    ///
    /// # Errors
    ///
    /// Returns [`XpbdError`] when the config or particle state is invalid, a
    /// constraint indexes a missing particle, or the constraint graph needs more
    /// colours than supported. Returns an all-zero [`IslandStep`] (doing nothing)
    /// when there are no particles or `dt` is non-positive.
    pub fn step(
        &mut self,
        state: &mut ParticleState,
        constraints: &[DistanceConstraint],
        config: &TgsConfig,
        dt: f32,
    ) -> Result<IslandStep, XpbdError> {
        config.validate()?;
        if !state.is_consistent() {
            return Err(XpbdError::InvalidConfig(
                "particle state arrays must have equal length",
            ));
        }
        if state.is_empty() || dt <= 0.0 {
            // Keep the tracker sized to the state so later queries stay valid,
            // but take no step: a non-positive `dt` makes `update` resize and
            // return without accumulating any quiet time.
            let partition = build_islands(constraints, &state.inverse_masses, state.len() as u32)?;
            self.sleep.update(state, &partition, &self.config, 0.0);
            return Ok(IslandStep::default());
        }

        let particle_count = state.len() as u32;
        let islands = build_islands(constraints, &state.inverse_masses, particle_count)?;

        // Decide sleep from the velocities carried in from last frame. After
        // this, `is_asleep` reflects the current frame's decision.
        self.sleep.update(state, &islands, &self.config, dt);

        // A particle integrates this frame only when it is dynamic and awake;
        // pinned particles (inverse mass <= 0) are frozen by the primitives
        // regardless, so the mask only needs to gate dynamic ones.
        let awake: Vec<bool> = (0..state.len())
            .map(|p| state.inverse_masses[p] > 0.0 && !self.sleep.is_asleep(p as u32))
            .collect();

        // Solve only the awake islands' constraints. The indices come back
        // sorted, so gathering preserves a deterministic constraint order.
        let awake_indices = self.sleep.awake_constraints(&islands);
        let awake_constraints: Vec<DistanceConstraint> = awake_indices
            .iter()
            .map(|&i| constraints[i as usize])
            .collect();

        run_substeps_tgs(state, &awake_constraints, &awake, config, dt)?;

        // Residual jitter must not creep a sleeping particle: pin its velocity.
        let mut asleep_particles = 0usize;
        for p in 0..state.len() {
            if self.sleep.is_asleep(p as u32) {
                state.velocities[p] = Vec3::ZERO;
                asleep_particles += 1;
            }
        }

        let asleep_islands = (0..islands.len() as u32)
            .filter(|&id| self.sleep.is_island_asleep(&islands, id))
            .count();

        Ok(IslandStep {
            islands: islands.len(),
            asleep_islands,
            asleep_particles,
            solved_constraints: awake_constraints.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-box stack hanging from a pinned anchor: anchor 0 (pinned), then
    /// particles 1 and 2 below it on rigid links.
    fn hanging_chain() -> (ParticleState, Vec<DistanceConstraint>) {
        let mut state = ParticleState::new();
        state.push(Vec3::ZERO, 0.0); // pinned anchor
        state.push(Vec3::new(0.0, -1.0, 0.0), 1.0);
        state.push(Vec3::new(0.0, -2.0, 0.0), 1.0);
        let cons = vec![
            DistanceConstraint::new(0, 1, 1.0, 0.0),
            DistanceConstraint::new(1, 2, 1.0, 0.0),
        ];
        (state, cons)
    }

    /// A rigid `TGS` config strong enough to settle the hanging chain.
    fn rigid_config() -> TgsConfig {
        TgsConfig::new(Vec3::new(0.0, -9.81, 0.0), 4, 8, 1, 0.0, 1.0, 0.0)
    }

    /// Settles the chain by stepping until it sleeps, returning the solver and
    /// the final state.
    fn settle() -> (
        IslandedTgsSolver,
        ParticleState,
        Vec<DistanceConstraint>,
        TgsConfig,
    ) {
        let (mut state, cons) = hanging_chain();
        let config = rigid_config();
        let mut solver = IslandedTgsSolver::new(SleepConfig::new(0.05, 0.2));
        let dt = 1.0 / 60.0;
        for _ in 0..600 {
            solver.step(&mut state, &cons, &config, dt).unwrap();
        }
        (solver, state, cons, config)
    }

    /// A chain released from rest eventually sleeps as one island, after which
    /// the step solves no constraints.
    #[test]
    fn resting_chain_sleeps_and_skips_work() {
        let (mut solver, mut state, cons, config) = settle();
        assert_eq!(solver.sleep_state().asleep_count(), 2);

        let step = solver.step(&mut state, &cons, &config, 1.0 / 60.0).unwrap();
        assert_eq!(step.islands, 1);
        assert_eq!(step.asleep_islands, 1);
        assert_eq!(step.asleep_particles, 2);
        assert_eq!(step.solved_constraints, 0);
    }

    /// While asleep, the chain holds its rest length and its velocities stay
    /// pinned at zero (no residual creep).
    #[test]
    fn sleeping_chain_holds_rest_length_and_zero_velocity() {
        let (mut solver, mut state, cons, config) = settle();
        let before = state.positions.clone();

        for _ in 0..120 {
            solver.step(&mut state, &cons, &config, 1.0 / 60.0).unwrap();
        }

        for p in [1usize, 2] {
            assert!(state.velocities[p].length() < 1e-6, "velocity crept: {p}");
            assert!(
                (state.positions[p] - before[p]).length() < 1e-4,
                "position drifted: {p}"
            );
        }
        // Rest lengths preserved.
        let l01 = (state.positions[1] - state.positions[0]).length();
        let l12 = (state.positions[2] - state.positions[1]).length();
        assert!((l01 - 1.0).abs() < 1e-2, "link 0-1 was {l01}");
        assert!((l12 - 1.0).abs() < 1e-2, "link 1-2 was {l12}");
    }

    /// Waking the island re-engages the solver: the next step solves its
    /// constraints again.
    #[test]
    fn wake_all_re_engages_the_solver() {
        let (mut solver, mut state, cons, config) = settle();
        assert_eq!(solver.sleep_state().asleep_count(), 2);

        solver.wake_all();
        let step = solver.step(&mut state, &cons, &config, 1.0 / 60.0).unwrap();
        assert_eq!(step.asleep_islands, 0);
        assert_eq!(step.solved_constraints, 2);
    }

    /// An awake island's trajectory matches the dense [`tgs_solve`] exactly:
    /// before anything sleeps, the island-aware stepper is bit-for-bit the
    /// reference solver.
    #[test]
    fn awake_island_matches_dense_solver() {
        let (mut islanded_state, cons) = hanging_chain();
        let mut dense_state = islanded_state.clone();
        let config = rigid_config();
        // A dwell long enough that nothing sleeps during the compared window.
        let mut solver = IslandedTgsSolver::new(SleepConfig::new(0.01, 1_000.0));
        let dt = 1.0 / 60.0;

        for _ in 0..30 {
            solver
                .step(&mut islanded_state, &cons, &config, dt)
                .unwrap();
            super::super::tgs_solve(&mut dense_state, &cons, &config, dt).unwrap();
        }

        for p in 0..islanded_state.len() {
            assert!(
                (islanded_state.positions[p] - dense_state.positions[p]).length() < 1e-6,
                "particle {p} diverged from the dense solver"
            );
            assert!(
                (islanded_state.velocities[p] - dense_state.velocities[p]).length() < 1e-6,
                "particle {p} velocity diverged from the dense solver"
            );
        }
    }

    /// Two independent chains sleep and wake separately: nudging one leaves the
    /// other asleep.
    #[test]
    fn independent_islands_sleep_separately() {
        // Chain A: anchor 0, bob 1. Chain B: anchor 2, bob 3.
        let mut state = ParticleState::new();
        state.push(Vec3::ZERO, 0.0);
        state.push(Vec3::new(0.0, -1.0, 0.0), 1.0);
        state.push(Vec3::new(5.0, 0.0, 0.0), 0.0);
        state.push(Vec3::new(5.0, -1.0, 0.0), 1.0);
        let cons = vec![
            DistanceConstraint::new(0, 1, 1.0, 0.0),
            DistanceConstraint::new(2, 3, 1.0, 0.0),
        ];
        let config = rigid_config();
        let mut solver = IslandedTgsSolver::new(SleepConfig::new(0.05, 0.2));
        let dt = 1.0 / 60.0;

        for _ in 0..600 {
            solver.step(&mut state, &cons, &config, dt).unwrap();
        }
        let partition = build_islands(&cons, &state.inverse_masses, 4).unwrap();
        assert_eq!(partition.len(), 2);
        assert_eq!(solver.sleep_state().asleep_count(), 2);

        // Wake only bob 1 (chain A). Chain B must remain asleep.
        solver.wake(1);
        let step = solver.step(&mut state, &cons, &config, dt).unwrap();
        assert_eq!(step.asleep_islands, 1);
        assert_eq!(step.solved_constraints, 1);
    }

    /// An empty state or non-positive `dt` is a no-op returning a zero summary.
    #[test]
    fn empty_or_nonpositive_dt_is_a_noop() {
        let mut solver = IslandedTgsSolver::new(SleepConfig::default());
        let config = TgsConfig::default();

        let mut empty = ParticleState::new();
        let step = solver.step(&mut empty, &[], &config, 1.0 / 60.0).unwrap();
        assert_eq!(step, IslandStep::default());

        let (mut state, cons) = hanging_chain();
        let before = state.clone();
        let step = solver.step(&mut state, &cons, &config, 0.0).unwrap();
        assert_eq!(step, IslandStep::default());
        assert_eq!(state, before);
    }
}
