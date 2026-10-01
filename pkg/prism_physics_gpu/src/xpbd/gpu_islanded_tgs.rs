//! Island-aware, sleep-skipping stepping of the real-device `GPU` `TGS` solver.
//!
//! [`GpuIslandedTgsSolver`] is the `GPU` twin of
//! [`IslandedTgsSolver`](super::islanded_tgs::IslandedTgsSolver): it layers the
//! island partition and per-island sleeping of [`crate::island`] over
//! [`GpuTgsSolver`](super::gpu_tgs::GpuTgsSolver). Each frame it partitions the
//! constraint graph on the host, lets a persistent [`SleepState`] decide which
//! islands have come to rest, uploads an awake mask plus only the awake
//! islands' constraints, and dispatches the Temporal Gauss-Seidel substep loop
//! so the device skips settled stacks entirely.
//!
//! The sleep bookkeeping (partition, dwell accounting, mask construction,
//! velocity pinning) is the *same host code* the `CPU` stepper runs; only the
//! inner substep solve moves to the device. The device substep path is
//! [`solve_masked`](super::gpu_tgs::GpuTgsSolver::solve_masked), which funnels
//! through the identical kernels [`GpuTgsSolver::solve`](
//! super::gpu_tgs::GpuTgsSolver::solve) uses — so an awake island's trajectory
//! is, within the device's floating-point tolerance, exactly what the dense
//! `GPU` `TGS` solver (and therefore the `CPU` golden twin) would have produced.
//! The island-aware parity test checks the `GPU` stepper against the `CPU`
//! [`IslandedTgsSolver`] frame for frame.
//!
//! Provenance: standard island partitioning + velocity-threshold island
//! sleeping layered over substep Temporal Gauss-Seidel (Müller et al.; Macklin
//! et al. soft-constraint TGS); standard `wgpu` compute dispatch. No Unreal
//! Engine source or derived code.

use glam::Vec3;

use crate::context::GpuContext;
use crate::island::{build_islands, SleepConfig, SleepState};

use super::config::XpbdError;
use super::constraint::DistanceConstraint;
use super::gpu_tgs::GpuTgsSolver;
use super::islanded::IslandStep;
use super::state::ParticleState;
use super::tgs::TgsConfig;

/// A stateful, device-backed `TGS` stepper that partitions into islands and
/// sleeps the ones at rest.
///
/// Hold one across frames and drive it with [`step`](Self::step). It owns a
/// compiled [`GpuTgsSolver`] and the persistent [`SleepState`] that accumulates
/// each particle's quiet time. The tracker is keyed by particle, not by island
/// id (which is not stable across re-partitions), so rebuilding the island set
/// every frame never loses sleep progress.
pub struct GpuIslandedTgsSolver {
    /// The compiled device solver whose masked path runs each awake substep.
    solver: GpuTgsSolver,
    /// Per-particle sleep bookkeeping carried across frames.
    sleep: SleepState,
    /// Thresholds governing when a quiet island falls asleep.
    config: SleepConfig,
}

impl GpuIslandedTgsSolver {
    /// Compiles the device solver on `ctx` and starts with an empty tracker.
    ///
    /// The tracker sizes itself to the particle count on the first
    /// [`step`](Self::step), so there is no need to know the count up front.
    #[must_use]
    pub fn new(ctx: &GpuContext, config: SleepConfig) -> GpuIslandedTgsSolver {
        GpuIslandedTgsSolver {
            solver: GpuTgsSolver::new(ctx),
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
    pub fn wake(&mut self, particle: u32) {
        self.sleep.wake(particle);
    }

    /// Forces every particle awake, resetting all quiet timers.
    pub fn wake_all(&mut self) {
        self.sleep.wake_all();
    }

    /// Advances `state` under `constraints` by `dt` on the device, skipping
    /// asleep islands.
    ///
    /// The frame proceeds in four stages, matching
    /// [`IslandedTgsSolver::step`](super::islanded_tgs::IslandedTgsSolver::step):
    /// 1. Partition the constraint graph into islands (pinned anchors are shared
    ///    but never bridge islands).
    /// 2. Update the sleep tracker from the carried-in velocities: islands whose
    ///    members have all been quiet for the dwell time sleep as a unit; any
    ///    island with a moving member is forced fully awake.
    /// 3. Run the substep `TGS` loop *on the device* over the awake islands'
    ///    constraints, integrating only awake, dynamic particles.
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
        ctx: &GpuContext,
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
        // pinned particles (inverse mass <= 0) are frozen by the kernel
        // regardless, so the mask only needs to gate dynamic ones. The device
        // reads a `u32` per particle (1 = awake, 0 = frozen).
        let awake: Vec<u32> = (0..state.len())
            .map(|p| u32::from(state.inverse_masses[p] > 0.0 && !self.sleep.is_asleep(p as u32)))
            .collect();

        // Solve only the awake islands' constraints. The indices come back
        // sorted, so gathering preserves a deterministic constraint order.
        let awake_indices = self.sleep.awake_constraints(&islands);
        let awake_constraints: Vec<DistanceConstraint> = awake_indices
            .iter()
            .map(|&i| constraints[i as usize])
            .collect();

        self.solver
            .solve_masked(ctx, state, &awake_constraints, &awake, config, dt)?;

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
