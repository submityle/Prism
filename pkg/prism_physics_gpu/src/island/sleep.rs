//! Island-granular sleeping: skipping islands that have come to rest.
//!
//! Solving contacts and constraints for a stack that is already at rest is pure
//! wasted work, and — worse — the iterative solver's residual jitter keeps a
//! "resting" stack faintly alive forever. Every AAA solver therefore *sleeps*
//! bodies that have stayed slow for long enough and skips them until something
//! disturbs them. This module tracks that state per particle and decides sleep
//! **per island**, as a unit: an island sleeps only once *all* of its particles
//! have been quiet for [`SleepConfig::time_to_sleep`], and the moment any one
//! member moves, the whole island wakes. Deciding as a unit is what keeps a
//! resting stack from half-sleeping and what makes a nudge to the top box wake
//! the boxes beneath it.
//!
//! The tracker is a pure bookkeeping layer over an [`IslandSet`] and a
//! [`ParticleState`]; it never moves a particle. A solver consumes it through
//! [`SleepState::awake_constraints`], which returns just the constraints of the
//! awake islands, so the asleep islands cost nothing.
//!
//! Provenance: standard velocity-threshold island sleeping (as in `PhysX`,
//! `Box2D`, Chaos). No Unreal Engine source or derived code.

use crate::xpbd::ParticleState;

use super::set::IslandSet;

/// Thresholds governing when a quiet island falls asleep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SleepConfig {
    /// Linear speed (metres/second) at or below which a particle counts as
    /// quiet. A particle moving faster than this instantly wakes its island.
    linear_threshold: f32,
    /// Seconds an island must stay entirely quiet before it sleeps.
    time_to_sleep: f32,
}

impl SleepConfig {
    /// Creates a sleep config, clamping negative inputs to `0`.
    #[must_use]
    pub fn new(linear_threshold: f32, time_to_sleep: f32) -> SleepConfig {
        SleepConfig {
            linear_threshold: linear_threshold.max(0.0),
            time_to_sleep: time_to_sleep.max(0.0),
        }
    }

    /// The quiet-speed threshold in metres/second.
    #[must_use]
    pub fn linear_threshold(self) -> f32 {
        self.linear_threshold
    }

    /// The dwell time in seconds before a quiet island sleeps.
    #[must_use]
    pub fn time_to_sleep(self) -> f32 {
        self.time_to_sleep
    }
}

impl Default for SleepConfig {
    /// A conservative default: sleep at `1 cm/s` after half a second of quiet.
    fn default() -> SleepConfig {
        SleepConfig::new(0.01, 0.5)
    }
}

/// Per-particle sleep bookkeeping carried across frames.
///
/// One [`SleepState`] tracks the whole [`ParticleState`]. Feed it the current
/// [`IslandSet`] and velocities every frame through [`SleepState::update`]; it
/// accumulates each particle's quiet time and flips islands between awake and
/// asleep as a unit. Because the state is keyed by particle (not by island id,
/// which is not stable across re-partitions), it survives the island set being
/// rebuilt from scratch each frame.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SleepState {
    /// Accumulated quiet time per particle, capped at
    /// [`SleepConfig::time_to_sleep`].
    timers: Vec<f32>,
    /// Whether each particle is currently asleep.
    asleep: Vec<bool>,
}

impl SleepState {
    /// Creates a tracker for `particle_count` particles, all awake.
    #[must_use]
    pub fn new(particle_count: usize) -> SleepState {
        SleepState {
            timers: vec![0.0; particle_count],
            asleep: vec![false; particle_count],
        }
    }

    /// The number of particles tracked.
    #[must_use]
    pub fn len(&self) -> usize {
        self.timers.len()
    }

    /// Whether the tracker holds no particles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.timers.is_empty()
    }

    /// Whether `particle` is currently asleep (`false` for an out-of-range or
    /// pinned particle).
    #[must_use]
    pub fn is_asleep(&self, particle: u32) -> bool {
        self.asleep.get(particle as usize).copied().unwrap_or(false)
    }

    /// The accumulated quiet time of `particle`, or `0` when out of range.
    #[must_use]
    pub fn timer(&self, particle: u32) -> f32 {
        self.timers.get(particle as usize).copied().unwrap_or(0.0)
    }

    /// The number of particles currently asleep.
    #[must_use]
    pub fn asleep_count(&self) -> usize {
        self.asleep.iter().filter(|&&a| a).count()
    }

    /// Forces `particle` awake and resets its quiet timer.
    ///
    /// Use this to inject an external disturbance (a teleport, an applied
    /// impulse) the velocity test cannot see. Out-of-range indices are ignored.
    pub fn wake(&mut self, particle: u32) {
        if let Some(timer) = self.timers.get_mut(particle as usize) {
            *timer = 0.0;
        }
        if let Some(flag) = self.asleep.get_mut(particle as usize) {
            *flag = false;
        }
    }

    /// Forces every particle awake and resets all timers.
    pub fn wake_all(&mut self) {
        self.timers.iter_mut().for_each(|t| *t = 0.0);
        self.asleep.iter_mut().for_each(|a| *a = false);
    }

    /// Resizes the tracker to `particle_count`, leaving added particles awake.
    fn resize(&mut self, particle_count: usize) {
        self.timers.resize(particle_count, 0.0);
        self.asleep.resize(particle_count, false);
    }

    /// Advances the sleep state by `dt` seconds against the current velocities
    /// and island partition.
    ///
    /// A particle whose speed exceeds [`SleepConfig::linear_threshold`] has its
    /// timer reset and is woken immediately; a quiet particle accumulates `dt`
    /// (capped at [`SleepConfig::time_to_sleep`]). Then each island sleeps as a
    /// unit — asleep only when *every* member's timer has reached the dwell
    /// time — and any island with a still-moving member is forced fully awake.
    /// Dynamic particles that belong to no island (free bodies) are decided
    /// individually on the same dwell rule.
    ///
    /// The tracker resizes itself to match `state` if their lengths differ, so a
    /// caller that grows the particle set need not reallocate the tracker by
    /// hand; newly added particles start awake.
    pub fn update(
        &mut self,
        state: &ParticleState,
        islands: &IslandSet,
        config: &SleepConfig,
        dt: f32,
    ) {
        if self.len() != state.len() {
            self.resize(state.len());
        }
        if dt <= 0.0 {
            return;
        }

        // Per-particle quiet accounting. Pinned particles never move and are
        // never island members, so their bookkeeping is inert.
        for p in 0..state.len() {
            if state.inverse_masses[p] <= 0.0 {
                continue;
            }
            let speed = state.velocities[p].length();
            if speed > config.linear_threshold {
                self.timers[p] = 0.0;
                self.asleep[p] = false;
            } else {
                self.timers[p] = (self.timers[p] + dt).min(config.time_to_sleep);
            }
        }

        // Island-granular decision: an island sleeps only when all its
        // particles have reached the dwell time; otherwise it is forced awake.
        for island in islands.islands() {
            let all_quiet = island
                .particles()
                .iter()
                .all(|&p| self.timers[p as usize] >= config.time_to_sleep);
            for &p in island.particles() {
                self.asleep[p as usize] = all_quiet;
            }
        }

        // Free dynamic particles (in no island) sleep individually.
        for p in 0..state.len() {
            if state.inverse_masses[p] <= 0.0 {
                continue;
            }
            if islands.island_of(p as u32).is_none() {
                self.asleep[p] = self.timers[p] >= config.time_to_sleep;
            }
        }
    }

    /// Whether the island with id `island_id` is fully asleep.
    ///
    /// Because [`update`](Self::update) decides sleep per island as a unit, this
    /// is unambiguous: it reports the shared asleep flag of the island's
    /// particles (and `false` for an empty or unknown island).
    #[must_use]
    pub fn is_island_asleep(&self, islands: &IslandSet, island_id: u32) -> bool {
        match islands.island(island_id) {
            Some(island) => island
                .particles()
                .first()
                .is_some_and(|&p| self.is_asleep(p)),
            None => false,
        }
    }

    /// Collects the constraint indices of every *awake* island, in ascending
    /// order, so a solver can skip the asleep islands entirely.
    ///
    /// This is the payoff of sleeping: the returned list is exactly the work the
    /// solver still needs to do this frame.
    #[must_use]
    pub fn awake_constraints(&self, islands: &IslandSet) -> Vec<u32> {
        let mut active = Vec::new();
        for (id, island) in islands.islands().iter().enumerate() {
            if self.is_island_asleep(islands, id as u32) {
                continue;
            }
            active.extend_from_slice(island.constraints());
        }
        active.sort_unstable();
        active
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::island::build_islands;
    use crate::xpbd::ColouredEdge;
    use glam::Vec3;

    /// A minimal constraint edge used to build islands for the sleep tests.
    struct Edge(u32, u32);

    impl ColouredEdge for Edge {
        fn endpoints(&self) -> (u32, u32) {
            (self.0, self.1)
        }
    }

    /// Builds a particle state with `n` dynamic particles at the origin.
    fn dynamic_state(n: usize) -> ParticleState {
        let mut state = ParticleState::new();
        for _ in 0..n {
            state.push(Vec3::ZERO, 1.0);
        }
        state
    }

    /// The config exposes clamped thresholds and a sane default.
    #[test]
    fn config_clamps_and_defaults() {
        let cfg = SleepConfig::new(-1.0, -2.0);
        assert_eq!(cfg.linear_threshold(), 0.0);
        assert_eq!(cfg.time_to_sleep(), 0.0);

        let def = SleepConfig::default();
        assert_eq!(def.linear_threshold(), 0.01);
        assert_eq!(def.time_to_sleep(), 0.5);
    }

    /// After enough quiet dwell, an all-quiet island sleeps as a unit.
    #[test]
    fn quiet_island_sleeps_after_dwell() {
        let state = dynamic_state(2);
        let islands = build_islands(&[Edge(0, 1)], &state.inverse_masses, 2).expect("valid graph");
        let cfg = SleepConfig::new(0.01, 0.5);
        let mut sleep = SleepState::new(2);

        // 0.4 s of quiet: not yet asleep.
        sleep.update(&state, &islands, &cfg, 0.4);
        assert!(!sleep.is_asleep(0));
        assert!(!sleep.is_asleep(1));

        // Crossing the dwell time sleeps the whole island.
        sleep.update(&state, &islands, &cfg, 0.2);
        assert!(sleep.is_asleep(0));
        assert!(sleep.is_asleep(1));
        assert!(sleep.is_island_asleep(&islands, 0));
        assert_eq!(sleep.asleep_count(), 2);
    }

    /// One moving member forces the entire island awake, even after it slept.
    #[test]
    fn moving_member_wakes_whole_island() {
        let mut state = dynamic_state(2);
        let islands = build_islands(&[Edge(0, 1)], &state.inverse_masses, 2).expect("valid graph");
        let cfg = SleepConfig::new(0.01, 0.5);
        let mut sleep = SleepState::new(2);

        // Sleep the island first.
        sleep.update(&state, &islands, &cfg, 1.0);
        assert!(sleep.is_island_asleep(&islands, 0));

        // Nudge particle 1 above the threshold; the whole island must wake.
        state.velocities[1] = Vec3::new(1.0, 0.0, 0.0);
        sleep.update(&state, &islands, &cfg, 0.016);
        assert!(!sleep.is_asleep(0));
        assert!(!sleep.is_asleep(1));
        assert!(!sleep.is_island_asleep(&islands, 0));
    }

    /// A partially quiet island never sleeps: one member below dwell keeps all
    /// members awake.
    #[test]
    fn partial_quiet_island_stays_awake() {
        let mut state = dynamic_state(2);
        let islands = build_islands(&[Edge(0, 1)], &state.inverse_masses, 2).expect("valid graph");
        let cfg = SleepConfig::new(0.01, 0.5);
        let mut sleep = SleepState::new(2);

        // Particle 1 keeps moving every frame while 0 is quiet.
        state.velocities[1] = Vec3::new(1.0, 0.0, 0.0);
        for _ in 0..100 {
            sleep.update(&state, &islands, &cfg, 0.016);
        }
        assert!(!sleep.is_asleep(0));
        assert!(!sleep.is_asleep(1));
    }

    /// Free dynamic particles (in no island) sleep individually on the dwell
    /// rule.
    #[test]
    fn free_particle_sleeps_individually() {
        // No constraints: particle 0 belongs to no island.
        let state = dynamic_state(1);
        let islands = build_islands::<Edge>(&[], &state.inverse_masses, 1).expect("valid graph");
        assert_eq!(islands.island_of(0), None);
        let cfg = SleepConfig::new(0.01, 0.5);
        let mut sleep = SleepState::new(1);

        sleep.update(&state, &islands, &cfg, 1.0);
        assert!(sleep.is_asleep(0));
    }

    /// `wake` resets one particle; `wake_all` resets everything.
    #[test]
    fn wake_and_wake_all_reset_state() {
        let state = dynamic_state(2);
        let islands = build_islands(&[Edge(0, 1)], &state.inverse_masses, 2).expect("valid graph");
        let cfg = SleepConfig::new(0.01, 0.5);
        let mut sleep = SleepState::new(2);
        sleep.update(&state, &islands, &cfg, 1.0);
        assert!(sleep.is_asleep(0));

        sleep.wake(0);
        assert!(!sleep.is_asleep(0));
        assert_eq!(sleep.timer(0), 0.0);
        // Particle 1 is untouched by the single-particle wake.
        assert!(sleep.is_asleep(1));

        sleep.wake_all();
        assert!(!sleep.is_asleep(1));
        assert_eq!(sleep.timer(1), 0.0);
        assert_eq!(sleep.asleep_count(), 0);
    }

    /// `awake_constraints` returns only the constraints of awake islands, sorted.
    #[test]
    fn awake_constraints_skips_sleeping_islands() {
        // Two independent islands: {0,1} via edge 0, {2,3} via edge 1.
        let state = dynamic_state(4);
        let edges = [Edge(0, 1), Edge(2, 3)];
        let islands = build_islands(&edges, &state.inverse_masses, 4).expect("valid graph");
        let cfg = SleepConfig::new(0.01, 0.5);
        let mut sleep = SleepState::new(4);

        // Everything quiet: both islands sleep, no constraints remain.
        sleep.update(&state, &islands, &cfg, 1.0);
        assert!(sleep.awake_constraints(&islands).is_empty());

        // Wake island of particle 2 by force; only its constraint returns.
        let mut moving = dynamic_state(4);
        moving.velocities[2] = Vec3::new(1.0, 0.0, 0.0);
        sleep.update(&moving, &islands, &cfg, 0.016);
        assert_eq!(sleep.awake_constraints(&islands), vec![1]);
    }

    /// `update` resizes the tracker to match a grown particle state; added
    /// particles start awake.
    #[test]
    fn update_resizes_to_match_state() {
        let state = dynamic_state(3);
        let islands = build_islands::<Edge>(&[], &state.inverse_masses, 3).expect("valid graph");
        let cfg = SleepConfig::new(0.01, 0.5);
        // Tracker starts smaller than the state.
        let mut sleep = SleepState::new(1);
        sleep.update(&state, &islands, &cfg, 0.016);
        assert_eq!(sleep.len(), 3);
    }

    /// A non-positive `dt` leaves the state untouched.
    #[test]
    fn non_positive_dt_is_a_no_op() {
        let state = dynamic_state(1);
        let islands = build_islands::<Edge>(&[], &state.inverse_masses, 1).expect("valid graph");
        let cfg = SleepConfig::new(0.01, 0.5);
        let mut sleep = SleepState::new(1);
        sleep.update(&state, &islands, &cfg, 0.0);
        assert_eq!(sleep.timer(0), 0.0);
        assert!(!sleep.is_asleep(0));
    }

    /// Out-of-range queries degrade gracefully.
    #[test]
    fn out_of_range_queries_are_safe() {
        let sleep = SleepState::new(1);
        assert!(!sleep.is_asleep(99));
        assert_eq!(sleep.timer(99), 0.0);
        assert!(!sleep.is_empty());
    }
}
