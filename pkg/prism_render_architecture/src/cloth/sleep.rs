//! Sleep / activation gating for mesh garments.
//!
//! A populated scene has far more garments than the per-frame deformation
//! budget can ever simulate. The cheapest garment is the one that never runs: a
//! dress that has come to rest (a standing NPC, an off-screen character) is put
//! to *sleep* and stops consuming the shared deformation budget until something
//! disturbs it (design §6 "休眠/激活", §8 "休眠服装不占预算"). This module owns
//! that hysteresis purely and deterministically (design §9): it reads a garment's
//! motion indicator and returns the next sleep state, with no side effects and
//! no hidden clock, mirroring the sleep gates shipped in production cloth
//! engines such as `Havok` Cloth and UE5 `Chaos` Cloth.
//!
//! The gate is hysteretic on purpose. A garment only sleeps after staying quiet
//! for a run of frames (so a brief lull does not stall an active garment), but
//! it wakes the instant its motion crosses a higher threshold (so it never
//! looks frozen). The wake threshold sits above the sleep threshold, leaving a
//! dead band that stops the state machine chattering on the edge.

use super::ClothParticle;

/// Whether a garment is being simulated this frame or resting.
///
/// A `Sleeping` garment emits no deformation request and is charged nothing
/// against the shared budget until it is woken again.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SleepState {
    /// The garment is simulated and charged to the deformation budget.
    Awake,
    /// The garment is at rest, skipping simulation and the budget.
    Sleeping,
}

impl SleepState {
    /// Returns `true` while the garment is awake and simulated.
    #[must_use]
    pub fn is_awake(self) -> bool {
        matches!(self, SleepState::Awake)
    }

    /// Returns `true` while the garment is asleep and skipping simulation.
    #[must_use]
    pub fn is_sleeping(self) -> bool {
        matches!(self, SleepState::Sleeping)
    }
}

/// Hysteresis thresholds and dwell time for the sleep gate.
///
/// `linear_threshold` and `wake_threshold` are compared against the motion
/// indicator from [`max_kinetic_indicator`] (a squared speed), so they carry
/// squared-speed units. `wake_threshold` should sit at or above
/// `linear_threshold` to leave a dead band; [`SleepParams::sanitized`] enforces
/// that. `frames_to_sleep` is how many consecutive quiet frames are required
/// before a garment sleeps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SleepParams {
    /// Motion indicator strictly below which a frame counts as "quiet".
    pub linear_threshold: f32,
    /// Consecutive quiet frames required before sleeping.
    pub frames_to_sleep: u32,
    /// Motion indicator strictly above which a sleeping garment wakes at once.
    pub wake_threshold: f32,
}

impl Default for SleepParams {
    /// Production-leaning defaults: a very small rest threshold, a ~30-frame
    /// dwell (about half a second at 60 Hz), and a wake threshold two orders of
    /// magnitude higher to form a wide, chatter-free dead band.
    fn default() -> Self {
        Self {
            linear_threshold: 1.0e-4,
            frames_to_sleep: 30,
            wake_threshold: 1.0e-2,
        }
    }
}

impl SleepParams {
    /// Returns a copy with the thresholds made safe: negatives and non-finite
    /// values are clamped to `0`, and `wake_threshold` is raised to at least
    /// `linear_threshold` so the wake band never falls below the sleep band.
    ///
    /// This keeps the state machine deterministic and free of `NaN`-driven
    /// surprises regardless of how the parameters were authored.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let linear_threshold = clamp_non_negative(self.linear_threshold);
        let wake = clamp_non_negative(self.wake_threshold);
        let wake_threshold = if wake > linear_threshold {
            wake
        } else {
            linear_threshold
        };
        Self {
            linear_threshold,
            frames_to_sleep: self.frames_to_sleep,
            wake_threshold,
        }
    }
}

/// Clamps a scalar to `[0, ∞)`, mapping negatives and `NaN` to `0`.
#[must_use]
fn clamp_non_negative(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

/// The evolving sleep state of one garment, advanced once per frame.
///
/// Construct with [`SleepTracker::new`] or [`Default`] (awake, zero quiet
/// frames), then call [`SleepTracker::update`] each frame with the garment's
/// motion indicator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SleepTracker {
    /// The current awake/sleeping state.
    pub state: SleepState,
    /// Consecutive quiet frames observed while awake; resets on any motion.
    pub quiet_frames: u32,
}

impl Default for SleepTracker {
    /// A freshly spawned garment: awake and undisturbed.
    fn default() -> Self {
        Self {
            state: SleepState::Awake,
            quiet_frames: 0,
        }
    }
}

impl SleepTracker {
    /// Builds a fresh tracker (awake, zero quiet frames).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Advances this garment's sleep state given this frame's motion indicator.
    ///
    /// Wake takes priority: a non-finite indicator or one strictly above
    /// `wake_threshold` wakes a sleeping garment and clears the quiet counter,
    /// so a `NaN` can never latch a garment asleep. An awake garment
    /// accumulates quiet frames while its indicator stays strictly below
    /// `linear_threshold`, and sleeps once the run reaches `frames_to_sleep`
    /// (treated as at least one frame). Any non-quiet frame resets the counter.
    /// The parameters are [`SleepParams::sanitized`] first, so the transition is
    /// fully deterministic. Returns the new state.
    pub fn update(&mut self, indicator: f32, params: SleepParams) -> SleepState {
        let params = params.sanitized();
        let energetic = !indicator.is_finite() || indicator > params.wake_threshold;
        match self.state {
            SleepState::Sleeping => {
                if energetic {
                    self.state = SleepState::Awake;
                    self.quiet_frames = 0;
                }
            }
            SleepState::Awake => {
                if energetic {
                    self.quiet_frames = 0;
                } else if indicator.is_finite() && indicator < params.linear_threshold {
                    self.quiet_frames = self.quiet_frames.saturating_add(1);
                    if self.quiet_frames >= params.frames_to_sleep.max(1) {
                        self.state = SleepState::Sleeping;
                    }
                } else {
                    // In the dead band: awake, but making no progress to sleep.
                    self.quiet_frames = 0;
                }
            }
        }
        self.state
    }

    /// Forces the garment awake in response to an external disturbance (a
    /// collision, a teleport, a wind gust) and resets the quiet counter, so the
    /// dwell timer restarts from scratch.
    pub fn wake_on_disturbance(&mut self) {
        self.state = SleepState::Awake;
        self.quiet_frames = 0;
    }
}

/// Returns the largest squared speed over a garment's particles: a cheap,
/// allocation-free proxy for how much the garment is moving this frame.
///
/// Pinned particles (waistbands, attachment points, anim-driven vertices) are
/// skipped because their motion is prescribed, not simulated, and should never
/// keep the garment awake. An empty (or fully pinned) slice returns `0`, which
/// reads as "at rest" and never panics.
#[must_use]
pub fn max_kinetic_indicator(particles: &[ClothParticle]) -> f32 {
    let mut max = 0.0;
    for particle in particles {
        if particle.is_pinned() {
            continue;
        }
        let speed_squared = particle.velocity.length_squared();
        if speed_squared > max {
            max = speed_squared;
        }
    }
    max
}

/// Returns `true` when a garment in the given state should be simulated and
/// charged to the deformation budget this frame.
///
/// A `Sleeping` garment returns `false`, so callers emit no deformation request
/// and reserve no budget for it (design §8 "休眠服装不占预算").
#[must_use]
pub fn should_simulate(state: SleepState) -> bool {
    state.is_awake()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloth::Vec3;

    const EPS: f32 = 1.0e-6;

    fn params() -> SleepParams {
        SleepParams {
            linear_threshold: 1.0e-4,
            frames_to_sleep: 3,
            wake_threshold: 1.0e-2,
        }
    }

    fn moving(speed: f32) -> ClothParticle {
        ClothParticle {
            position: Vec3::ZERO,
            velocity: Vec3::new(speed, 0.0, 0.0),
            inverse_mass: 1.0,
        }
    }

    #[test]
    fn state_predicates_are_exclusive() {
        assert!(SleepState::Awake.is_awake());
        assert!(!SleepState::Awake.is_sleeping());
        assert!(SleepState::Sleeping.is_sleeping());
        assert!(!SleepState::Sleeping.is_awake());
    }

    #[test]
    fn defaults_are_awake_and_hysteretic() {
        let tracker = SleepTracker::default();
        assert_eq!(tracker.state, SleepState::Awake);
        assert_eq!(tracker.quiet_frames, 0);
        assert_eq!(SleepTracker::new(), tracker);

        let p = SleepParams::default();
        assert!(p.wake_threshold >= p.linear_threshold);
        assert!(p.frames_to_sleep >= 1);
    }

    #[test]
    fn indicator_is_max_squared_speed_and_skips_pinned() {
        let particles = [moving(2.0), moving(1.0), ClothParticle::pinned(Vec3::ZERO)];
        // max(2^2, 1^2) = 4, pinned ignored.
        assert!((max_kinetic_indicator(&particles) - 4.0).abs() < EPS);
    }

    #[test]
    fn indicator_of_empty_slice_is_zero() {
        assert!(max_kinetic_indicator(&[]).abs() < EPS);
    }

    #[test]
    fn fully_pinned_garment_reads_as_at_rest() {
        let particles = [
            ClothParticle::pinned(Vec3::new(1.0, 0.0, 0.0)),
            ClothParticle::pinned(Vec3::new(0.0, 2.0, 0.0)),
        ];
        assert!(max_kinetic_indicator(&particles).abs() < EPS);
    }

    #[test]
    fn quiet_run_eventually_sleeps() {
        let mut tracker = SleepTracker::new();
        for _ in 0..2 {
            let state = tracker.update(0.0, params());
            assert!(state.is_awake());
        }
        // Third quiet frame reaches frames_to_sleep and sleeps.
        let state = tracker.update(0.0, params());
        assert!(state.is_sleeping());
        assert!(!should_simulate(state));
    }

    #[test]
    fn motion_resets_progress_in_dead_band() {
        let mut tracker = SleepTracker::new();
        tracker.update(0.0, params());
        assert_eq!(tracker.quiet_frames, 1);
        // Dead-band motion (above sleep, below wake) resets without sleeping.
        let state = tracker.update(1.0e-3, params());
        assert!(state.is_awake());
        assert_eq!(tracker.quiet_frames, 0);
    }

    #[test]
    fn energetic_frame_wakes_sleeper_immediately() {
        let mut tracker = SleepTracker {
            state: SleepState::Sleeping,
            quiet_frames: 9,
        };
        let state = tracker.update(1.0, params());
        assert!(state.is_awake());
        assert_eq!(tracker.quiet_frames, 0);
    }

    #[test]
    fn sleeper_stays_asleep_in_dead_band() {
        let mut tracker = SleepTracker {
            state: SleepState::Sleeping,
            quiet_frames: 5,
        };
        // Dead-band motion must not wake a sleeper.
        let state = tracker.update(1.0e-3, params());
        assert!(state.is_sleeping());
    }

    #[test]
    fn nan_indicator_keeps_garment_awake() {
        let mut tracker = SleepTracker {
            state: SleepState::Sleeping,
            quiet_frames: 5,
        };
        let state = tracker.update(f32::NAN, params());
        assert!(state.is_awake());
        assert_eq!(tracker.quiet_frames, 0);
    }

    #[test]
    fn wake_on_disturbance_resets_dwell() {
        let mut tracker = SleepTracker {
            state: SleepState::Sleeping,
            quiet_frames: 12,
        };
        tracker.wake_on_disturbance();
        assert_eq!(tracker.state, SleepState::Awake);
        assert_eq!(tracker.quiet_frames, 0);
    }

    #[test]
    fn zero_frames_to_sleep_still_needs_one_quiet_frame() {
        let p = SleepParams {
            frames_to_sleep: 0,
            ..params()
        };
        let mut tracker = SleepTracker::new();
        let state = tracker.update(0.0, p);
        assert!(state.is_sleeping());
    }

    #[test]
    fn sanitized_clamps_negatives_and_nan() {
        let raw = SleepParams {
            linear_threshold: -5.0,
            frames_to_sleep: 4,
            wake_threshold: f32::NAN,
        };
        let clean = raw.sanitized();
        assert!(clean.linear_threshold.abs() < EPS);
        assert!(clean.wake_threshold.abs() < EPS);
        assert_eq!(clean.frames_to_sleep, 4);
    }

    #[test]
    fn sanitized_lifts_wake_threshold_above_sleep() {
        let raw = SleepParams {
            linear_threshold: 1.0e-2,
            frames_to_sleep: 4,
            wake_threshold: 1.0e-4,
        };
        let clean = raw.sanitized();
        assert!(clean.wake_threshold >= clean.linear_threshold);
    }

    #[test]
    fn update_is_deterministic() {
        let mut a = SleepTracker::new();
        let mut b = SleepTracker::new();
        let inputs = [0.0f32, 0.0, 5.0e-4, 0.0, 0.0, 0.0, 1.0];
        for &i in &inputs {
            assert_eq!(a.update(i, params()), b.update(i, params()));
            assert_eq!(a, b);
        }
    }
}
