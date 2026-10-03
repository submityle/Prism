//! [`FrameStepper`]: single-step frame debugging.
//!
//! When paused, a stepper freezes variable time but can release exactly one (or
//! a few) fixed-size frames on request — the "advance one frame" button in a
//! debugger. It never touches the real clock; it only gates the *virtual* delta
//! a frame driver feeds forward. When running, it passes the real delta through
//! untouched.

use crate::Duration;

/// Whether a [`FrameStepper`] is advancing freely or frozen awaiting steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum StepState {
    /// Time advances with the real frame delta.
    #[default]
    Running,
    /// Time is frozen; only requested steps release a `step_delta`.
    Paused,
}

/// A gate that releases frames one at a time while paused.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameStepper {
    state: StepState,
    pending: u32,
    step_delta: Duration,
}

impl FrameStepper {
    /// Default per-step delta while paused: `1/60 s`.
    pub const DEFAULT_STEP_DELTA: Duration = Duration::from_micros(16_667);

    /// Create a running stepper with the default `1/60 s` step delta.
    #[inline]
    pub fn new() -> Self {
        Self {
            state: StepState::Running,
            pending: 0,
            step_delta: Self::DEFAULT_STEP_DELTA,
        }
    }

    /// Create a running stepper with an explicit per-step delta.
    #[inline]
    pub fn with_step_delta(step_delta: Duration) -> Self {
        Self {
            state: StepState::Running,
            pending: 0,
            step_delta,
        }
    }

    /// The current [`StepState`].
    #[inline]
    pub fn state(&self) -> StepState {
        self.state
    }

    /// Whether the stepper is paused.
    #[inline]
    pub fn is_paused(&self) -> bool {
        self.state == StepState::Paused
    }

    /// Whether the stepper is running.
    #[inline]
    pub fn is_running(&self) -> bool {
        self.state == StepState::Running
    }

    /// Pause: subsequent frames yield time only when steps are requested.
    #[inline]
    pub fn pause(&mut self) {
        self.state = StepState::Paused;
    }

    /// Resume free-running time. Any queued steps are discarded.
    #[inline]
    pub fn resume(&mut self) {
        self.state = StepState::Running;
        self.pending = 0;
    }

    /// Toggle between [`StepState::Running`] and [`StepState::Paused`].
    #[inline]
    pub fn toggle(&mut self) {
        match self.state {
            StepState::Running => self.pause(),
            StepState::Paused => self.resume(),
        }
    }

    /// Queue a single frame step (saturating). Only meaningful while paused.
    #[inline]
    pub fn request_step(&mut self) {
        self.pending = self.pending.saturating_add(1);
    }

    /// Queue `n` frame steps (saturating).
    #[inline]
    pub fn request_steps(&mut self, n: u32) {
        self.pending = self.pending.saturating_add(n);
    }

    /// How many steps are queued.
    #[inline]
    pub fn pending_steps(&self) -> u32 {
        self.pending
    }

    /// The per-step delta released while paused.
    #[inline]
    pub fn step_delta(&self) -> Duration {
        self.step_delta
    }

    /// Set the per-step delta released while paused.
    #[inline]
    pub fn set_step_delta(&mut self, step_delta: Duration) {
        self.step_delta = step_delta;
    }

    /// Compute the virtual delta to feed this frame and consume one queued step
    /// if paused.
    ///
    /// - Running: returns `real_delta` unchanged.
    /// - Paused with a queued step: consumes one step and returns `step_delta`.
    /// - Paused with no queued step: returns [`Duration::ZERO`] (frozen).
    #[inline]
    pub fn next_delta(&mut self, real_delta: Duration) -> Duration {
        match self.state {
            StepState::Running => real_delta,
            StepState::Paused => {
                if self.pending > 0 {
                    self.pending -= 1;
                    self.step_delta
                } else {
                    Duration::ZERO
                }
            }
        }
    }

    /// Whether the last [`next_delta`](Self::next_delta) for `real_delta` would
    /// release time (running, or paused with a queued step). Pure query; does
    /// not consume a step.
    #[inline]
    pub fn would_advance(&self) -> bool {
        self.state == StepState::Running || self.pending > 0
    }
}

impl Default for FrameStepper {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
