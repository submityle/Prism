//! Sleep / activation gating for guide-strand grooms.
//!
//! A crowd scene has far more grooms than the per-frame deformation budget can
//! ever simulate. The cheapest strand is the one that never runs: a groom that
//! has come to rest (a standing NPC, an off-screen character) is put to *sleep*
//! and stops consuming the shared deformation budget until something disturbs
//! it (design §6.9, §8 "休眠 groom 不占预算"). This module owns that hysteresis
//! purely and deterministically (design §9): it reads the groom's kinetic
//! energy and returns the next sleep state, with no side effects and no hidden
//! clock.
//!
//! The gate is hysteretic on purpose. A groom only sleeps after staying quiet
//! for a run of frames (so a brief lull does not stall an active groom), but it
//! wakes the instant its energy crosses a higher threshold (so it never looks
//! frozen). The two thresholds must straddle a dead band to avoid chattering
//! on the edge.

use super::dynamics::{StrandParticle, Vec3};

/// Twice the kinetic energy per unit mass of one particle: the squared implicit
/// velocity stored as `position - prev_position`. The dynamics integrator keeps
/// velocity implicitly, so this reads it back without extra state.
#[must_use]
fn particle_speed_squared(particle: &StrandParticle) -> f32 {
    let v: Vec3 = particle.position.sub(particle.prev_position);
    v.length_squared()
}

/// Sums the squared implicit velocity over every particle of a strand (or a
/// whole groom's flat particle array): a cheap, allocation-free proxy for how
/// much the groom is moving this frame. Pinned particles contribute ~0 because
/// the integrator holds their `prev_position` equal to `position`.
#[must_use]
pub fn groom_motion_energy(particles: &[StrandParticle]) -> f32 {
    let mut sum = 0.0;
    for particle in particles {
        sum += particle_speed_squared(particle);
    }
    sum
}

/// Hysteresis thresholds for the sleep gate.
///
/// `sleep_below` and `wake_above` are compared against
/// [`groom_motion_energy`]; `wake_above` should be `>= sleep_below` to leave a
/// dead band. `frames_to_sleep` is how many consecutive quiet frames are
/// required before a groom sleeps; `0` or `1` sleeps on the first quiet frame.
#[derive(Clone, Copy, Debug)]
pub struct SleepThresholds {
    /// Motion energy at or below which a frame counts as "quiet".
    pub sleep_below: f32,
    /// Motion energy at or above which an asleep groom wakes immediately.
    pub wake_above: f32,
    /// Consecutive quiet frames required before sleeping.
    pub frames_to_sleep: u32,
}

/// The evolving sleep state of one groom, advanced once per frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroomSleepState {
    /// `true` while the groom is asleep and skipping simulation.
    pub asleep: bool,
    /// Consecutive quiet frames observed while awake; resets on any motion.
    pub quiet_frames: u32,
}

impl GroomSleepState {
    /// The initial state of a freshly spawned groom: awake and undisturbed.
    pub const AWAKE: Self = Self {
        asleep: false,
        quiet_frames: 0,
    };

    /// `true` when the groom should be simulated and charged to the deformation
    /// budget this frame (i.e. it is awake).
    #[must_use]
    pub fn should_simulate(self) -> bool {
        !self.asleep
    }
}

/// Advances one groom's sleep state given this frame's motion energy.
///
/// Wake takes priority: any energy at or above `wake_above` wakes an asleep
/// groom and clears the quiet counter. An awake groom accumulates quiet frames
/// while its energy stays at or below `sleep_below`, and sleeps once the run
/// reaches `frames_to_sleep`; any non-quiet frame resets the counter. Energy in
/// the dead band between the thresholds neither wakes a sleeper nor resets an
/// awake groom's progress toward sleep... it simply is not "quiet", so it holds
/// the counter without advancing it. Non-finite energy is treated as motion
/// (keeps the groom awake) so a NaN can never latch a groom asleep.
#[must_use]
pub fn update_sleep(
    state: GroomSleepState,
    motion_energy: f32,
    thresholds: SleepThresholds,
) -> GroomSleepState {
    let energetic = !motion_energy.is_finite() || motion_energy >= thresholds.wake_above;
    if energetic {
        return GroomSleepState::AWAKE;
    }
    if state.asleep {
        // Asleep and not energetic enough to wake: stay asleep.
        return state;
    }

    let quiet = motion_energy.is_finite() && motion_energy <= thresholds.sleep_below;
    if quiet {
        let quiet_frames = state.quiet_frames.saturating_add(1);
        if quiet_frames >= thresholds.frames_to_sleep.max(1) {
            GroomSleepState {
                asleep: true,
                quiet_frames,
            }
        } else {
            GroomSleepState {
                asleep: false,
                quiet_frames,
            }
        }
    } else {
        // In the dead band: awake, but making no progress toward sleep.
        GroomSleepState {
            asleep: false,
            quiet_frames: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thresholds() -> SleepThresholds {
        SleepThresholds {
            sleep_below: 1.0e-4,
            wake_above: 1.0e-2,
            frames_to_sleep: 3,
        }
    }

    fn moving(v: f32) -> StrandParticle {
        StrandParticle {
            position: Vec3::new(v, 0.0, 0.0),
            prev_position: Vec3::ZERO,
            inverse_mass: 1.0,
        }
    }

    #[test]
    fn motion_energy_sums_squared_velocity() {
        let particles = [moving(2.0), moving(0.0)];
        // 2^2 + 0 = 4.
        assert!((groom_motion_energy(&particles) - 4.0).abs() < 1.0e-6);
    }

    #[test]
    fn quiet_run_eventually_sleeps() {
        let mut state = GroomSleepState::AWAKE;
        for _ in 0..2 {
            state = update_sleep(state, 0.0, thresholds());
            assert!(!state.asleep);
        }
        // Third quiet frame reaches frames_to_sleep and sleeps.
        state = update_sleep(state, 0.0, thresholds());
        assert!(state.asleep);
    }

    #[test]
    fn motion_resets_progress() {
        let mut state = GroomSleepState::AWAKE;
        state = update_sleep(state, 0.0, thresholds());
        assert_eq!(state.quiet_frames, 1);
        // A frame in the dead band resets the counter without sleeping.
        state = update_sleep(state, 1.0e-3, thresholds());
        assert!(!state.asleep);
        assert_eq!(state.quiet_frames, 0);
    }

    #[test]
    fn energetic_frame_wakes_immediately() {
        let asleep = GroomSleepState {
            asleep: true,
            quiet_frames: 9,
        };
        let woken = update_sleep(asleep, 1.0, thresholds());
        assert!(!woken.asleep);
        assert_eq!(woken.quiet_frames, 0);
    }

    #[test]
    fn asleep_groom_stays_asleep_in_dead_band() {
        let asleep = GroomSleepState {
            asleep: true,
            quiet_frames: 5,
        };
        // Dead-band energy (above sleep_below, below wake_above) must not wake.
        let still = update_sleep(asleep, 1.0e-3, thresholds());
        assert!(still.asleep);
    }

    #[test]
    fn nan_energy_keeps_groom_awake() {
        let asleep = GroomSleepState {
            asleep: true,
            quiet_frames: 5,
        };
        let woken = update_sleep(asleep, f32::NAN, thresholds());
        assert!(!woken.asleep);
    }

    #[test]
    fn should_simulate_tracks_awake() {
        assert!(GroomSleepState::AWAKE.should_simulate());
        assert!(!GroomSleepState {
            asleep: true,
            quiet_frames: 3
        }
        .should_simulate());
    }

    #[test]
    fn zero_frames_to_sleep_still_needs_one_quiet_frame() {
        let t = SleepThresholds {
            frames_to_sleep: 0,
            ..thresholds()
        };
        let state = update_sleep(GroomSleepState::AWAKE, 0.0, t);
        assert!(state.asleep);
    }
}
