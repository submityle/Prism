//! Per-frame groom simulation kernel: the deterministic pass ordering that
//! turns the standalone strand services into one guide-strand step.
//!
//! Each hair service in this module solves one concern in isolation — wind as a
//! force pre-pass ([`super::wind`]), the XPBD guide solve
//! ([`super::dynamics::simulate_guides`]), the analytic and signed-distance
//! body colliders ([`super::collision`], [`super::sdf_collision`]), the
//! approximate strand self-collision ([`super::self_collision`]), and the
//! hysteretic sleep gate ([`super::sleep`]). Something has to run them in the
//! right order once per frame, and that order is hair-domain knowledge, not
//! scheduler policy: external forces are injected first, the constraint solve
//! projects them (with body colliders applied inside each substep), then the
//! heavier SDF and self-collision passes clean up what the analytic proxies
//! missed, and finally the sleep state advances on the frame's motion energy.
//!
//! [`step_groom`] encodes exactly that order over the flat, offset-sliced groom
//! layout already used by [`super::dynamics::simulate_guides`]. It is pure,
//! deterministic array-in/array-out math (design §9): the same buffers, time,
//! and sleep state always produce bit-identical output. Crucially it does
//! **not** own the deformation budget (design §8) — a sleeping groom returns
//! immediately without touching a single particle so the scheduler is never
//! charged for still hair, and an awake groom advances exactly the work the
//! caller granted it. This module composes the existing kernels; it never
//! reimplements a solver.

use super::collision::Collider;
use super::dynamics::{simulate_guides, StrandParticle, Vec3, Vec3Len, XpbdParams};
use super::sdf_collision::{resolve_sdf_collisions, SdfPrimitive};
use super::self_collision::{resolve_self_collision, SelfCollisionParams};
use super::sleep::{groom_motion_energy, update_sleep, GroomSleepState, SleepThresholds};
use super::wind::{apply_wind, WindField};

/// Everything that stays constant across one groom for a single frame step.
///
/// The heavy geometry (particles, strand layout, colliders) is passed to
/// [`step_groom`] by reference; this bundles the small value parameters so the
/// call site stays readable and a caller can reuse one config across many
/// grooms that share tuning.
#[derive(Clone, Copy, Debug)]
pub struct GroomStepConfig {
    /// Ambient wind applied as a force pre-pass before the solve.
    pub wind: WindField,
    /// XPBD guide-solver parameters (gravity, dt, substeps, stiffnesses).
    pub xpbd: XpbdParams,
    /// Hysteretic sleep thresholds gating whether this frame simulates at all.
    pub sleep: SleepThresholds,
    /// Optional approximate strand self-collision; `None` disables that pass.
    pub self_collision: Option<SelfCollisionParams>,
    /// Push-out relaxation passes for the SDF body field; `0` disables it.
    pub sdf_iterations: u32,
}

/// Advances one groom by a single frame and returns its next sleep state.
///
/// `particles`, `rest_lengths`, and `goal_positions` are the flat, offset-
/// sliced buffers consumed by [`simulate_guides`]; `strand_lengths[k]` is the
/// particle count of strand `k` and the strands occupy consecutive ranges in
/// that order. `colliders` are the analytic body proxies applied inside every
/// substep; `sdf` is the tighter signed-distance body field applied afterward.
///
/// The frame runs in this order, matching the module docs:
/// 1. measure the groom's motion energy and advance the sleep gate;
/// 2. if the groom is now asleep, return immediately — no particle is touched
///    and the deformation budget is not charged;
/// 3. otherwise inject wind, run the XPBD guide solve (analytic colliders
///    inside each substep), then the SDF body pass and the self-collision pass;
/// 4. return the advanced sleep state.
///
/// A groom that wakes this frame (its incoming motion cleared `wake_above`)
/// simulates on the same frame, so a disturbed groom never skips a beat.
pub fn step_groom(
    particles: &mut [StrandParticle],
    strand_lengths: &[usize],
    rest_lengths: &[Vec3Len],
    goal_positions: &[Vec3],
    colliders: &[Collider],
    sdf: &[SdfPrimitive],
    time: f32,
    sleep_state: GroomSleepState,
    config: GroomStepConfig,
) -> GroomSleepState {
    // 1. Sleep gate on this frame's incoming motion energy (which reflects any
    //    external root/skinning disturbance since the last step).
    let energy = groom_motion_energy(particles);
    let next_state = update_sleep(sleep_state, energy, config.sleep);

    // 2. Asleep grooms cost nothing: no work, no budget.
    if !next_state.should_simulate() {
        return next_state;
    }

    // 3a. External forces first, so the solve reads them back as velocity.
    apply_wind(particles, config.wind, time, config.xpbd.dt);

    // 3b. The XPBD guide solve; analytic body proxies are projected inside each
    //     substep by `simulate_guides`.
    simulate_guides(
        particles,
        strand_lengths,
        rest_lengths,
        goal_positions,
        colliders,
        config.xpbd,
    );

    // 3c. Heavier body tier: push out of the signed-distance field.
    resolve_sdf_collisions(particles, sdf, config.sdf_iterations);

    // 3d. Approximate strand-vs-strand separation across the whole groom.
    if let Some(self_params) = config.self_collision {
        resolve_self_collision(particles, self_params);
    }

    next_state
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn base_config() -> GroomStepConfig {
        GroomStepConfig {
            wind: WindField::CALM,
            xpbd: XpbdParams {
                gravity: Vec3::new(0.0, -9.81, 0.0),
                dt: 1.0 / 60.0,
                substeps: 2,
                iterations: 4,
                edge_compliance: 0.0,
                local_stiffness: 0.0,
                global_stiffness: 0.0,
                lra_stiffness: 0.0,
                damping: 0.0,
            },
            sleep: SleepThresholds {
                sleep_below: 1.0e-8,
                wake_above: 1.0e-6,
                frames_to_sleep: 2,
            },
            self_collision: None,
            sdf_iterations: 0,
        }
    }

    // One strand: a pinned root plus two free particles hanging below it.
    fn one_strand() -> (Vec<StrandParticle>, Vec<usize>, Vec<Vec3Len>, Vec<Vec3>) {
        let particles = vec![
            StrandParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
            StrandParticle::free(Vec3::new(0.0, -1.0, 0.0)),
            StrandParticle::free(Vec3::new(0.0, -2.0, 0.0)),
        ];
        let strand_lengths = vec![3usize];
        // Per-particle rest lengths (last entry ignored).
        let rest = vec![1.0, 1.0, 0.0];
        let goals = particles.iter().map(|p| p.position).collect();
        (particles, strand_lengths, rest, goals)
    }

    #[test]
    fn awake_groom_moves_under_gravity() {
        let (mut ps, lens, rest, goals) = one_strand();
        let before = ps[2].position;
        let state = step_groom(
            &mut ps,
            &lens,
            &rest,
            &goals,
            &[],
            &[],
            0.0,
            GroomSleepState::AWAKE,
            base_config(),
        );
        // Root stays pinned; a free tip falls.
        assert_eq!(ps[0].position, Vec3::ZERO);
        assert!(ps[2].position.y < before.y);
        // Fresh motion means it is not asleep yet.
        assert!(state.should_simulate());
    }

    #[test]
    fn asleep_groom_is_untouched_and_costs_nothing() {
        let (mut ps, lens, rest, goals) = one_strand();
        // A groom already at rest (prev == current) has zero motion energy.
        for p in &mut ps {
            p.prev_position = p.position;
        }
        let asleep = GroomSleepState {
            asleep: true,
            quiet_frames: 5,
        };
        let snapshot: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
        let mut windy = base_config();
        windy.wind = WindField {
            direction: Vec3::new(1.0, 0.0, 0.0),
            speed: 100.0,
            gust_amplitude: 0.0,
            gust_frequency: 0.0,
            turbulence: 0.0,
        };
        let state = step_groom(&mut ps, &lens, &rest, &goals, &[], &[], 0.0, asleep, windy);
        // Still asleep, and not one particle moved despite a gale of wind.
        assert!(!state.should_simulate());
        for (p, s) in ps.iter().zip(snapshot.iter()) {
            assert_eq!(p.position, *s);
        }
    }

    #[test]
    fn disturbed_sleeper_wakes_and_simulates_same_frame() {
        let (mut ps, lens, rest, goals) = one_strand();
        // Inject a large implicit velocity (position moved far from prev): this
        // is what an external skinning kick looks like at frame start.
        ps[2].prev_position = ps[2].position.sub(Vec3::new(0.0, 0.0, 1.0));
        let asleep = GroomSleepState {
            asleep: true,
            quiet_frames: 9,
        };
        let before = ps[2].position;
        let state = step_groom(
            &mut ps,
            &lens,
            &rest,
            &goals,
            &[],
            &[],
            0.0,
            asleep,
            base_config(),
        );
        // The kick cleared `wake_above`, so it woke and simulated this frame.
        assert!(state.should_simulate());
        assert!(ps[2].position.sub(before).length() > 0.0);
    }

    #[test]
    fn quiet_groom_falls_asleep_after_threshold() {
        let (mut ps, lens, rest, goals) = one_strand();
        // At rest: zero motion energy every frame.
        for p in &mut ps {
            p.prev_position = p.position;
        }
        let cfg = GroomStepConfig {
            xpbd: XpbdParams {
                gravity: Vec3::ZERO,
                ..base_config().xpbd
            },
            ..base_config()
        };
        let mut state = GroomSleepState::AWAKE;
        // frames_to_sleep = 2: first quiet frame arms, second sleeps.
        state = step_groom(&mut ps, &lens, &rest, &goals, &[], &[], 0.0, state, cfg);
        assert!(state.should_simulate());
        state = step_groom(&mut ps, &lens, &rest, &goals, &[], &[], 0.0, state, cfg);
        assert!(!state.should_simulate());
    }

    #[test]
    fn step_is_deterministic() {
        let (mut a, lens, rest, goals) = one_strand();
        let mut b = a.clone();
        let cfg = base_config();
        let sa = step_groom(
            &mut a,
            &lens,
            &rest,
            &goals,
            &[],
            &[],
            0.25,
            GroomSleepState::AWAKE,
            cfg,
        );
        let sb = step_groom(
            &mut b,
            &lens,
            &rest,
            &goals,
            &[],
            &[],
            0.25,
            GroomSleepState::AWAKE,
            cfg,
        );
        assert_eq!(sa, sb);
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.position, pb.position);
        }
    }

    #[test]
    fn sdf_and_self_collision_passes_run_when_configured() {
        // Two single-particle strands overlapping at nearly the same spot; the
        // self-collision pass must separate them, and the SDF pass must eject a
        // particle from inside a sphere. Roots are free so motion is visible.
        let mut ps = vec![
            StrandParticle::free(Vec3::new(0.0, 3.0, 0.0)),
            StrandParticle::free(Vec3::new(0.05, 3.0, 0.0)),
        ];
        for p in &mut ps {
            p.prev_position = p.position; // start at rest so gravity is the only motion
        }
        let lens = vec![1usize, 1usize];
        let rest = vec![0.0, 0.0];
        let goals: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
        let sphere = [SdfPrimitive::Sphere {
            center: Vec3::new(0.0, 3.0, 0.0),
            radius: 1.0,
        }];
        let cfg = GroomStepConfig {
            xpbd: XpbdParams {
                gravity: Vec3::ZERO,
                ..base_config().xpbd
            },
            self_collision: Some(SelfCollisionParams {
                particle_radius: 0.5,
                stiffness: 1.0,
                cell_size: 1.0,
            }),
            sdf_iterations: super::super::sdf_collision::DEFAULT_SDF_ITERATIONS,
            sleep: SleepThresholds {
                sleep_below: 0.0,
                wake_above: 1.0e-12,
                frames_to_sleep: 100,
            },
            ..base_config()
        };
        let _ = step_groom(
            &mut ps,
            &lens,
            &rest,
            &goals,
            &[],
            &sphere,
            0.0,
            GroomSleepState::AWAKE,
            cfg,
        );
        // Self-collision separated the overlapping pair to >= 2*radius.
        let sep = ps[0].position.sub(ps[1].position).length();
        assert!(sep >= 1.0 - 1.0e-3, "self-collision separation {sep}");
        // SDF ejected both particles to the sphere surface (distance ~0).
        for p in &ps {
            let d = p.position.sub(Vec3::new(0.0, 3.0, 0.0)).length();
            assert!(
                d >= 1.0 - 1.0e-2,
                "particle still inside sphere at dist {d}"
            );
        }
    }
}
