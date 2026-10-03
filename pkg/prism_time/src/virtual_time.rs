//! [`Time<Virtual>`]: controllable game time with time dilation and pause.
//!
//! Virtual time scales the real delta by a relative speed, clamps it to a
//! configurable maximum (the spiral-of-death guard), and freezes advancement
//! while paused. Gameplay, animation, particles, and cameras read this clock so
//! slow-motion, bullet-time, and pause affect them uniformly.

use crate::{Duration, Time, TimeKind};

/// Context for [`Time<Virtual>`]: scalable, pausable game time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Virtual {
    /// Upper bound on a single frame's *real* delta before scaling, so a hitch
    /// cannot inject an unbounded virtual step.
    max_delta: Duration,
    /// Requested relative speed (`1.0` = real time, `2.0` = double speed).
    relative_speed: f64,
    /// Speed actually applied last advance (`0.0` while paused).
    effective_speed: f64,
    /// Whether the clock is paused.
    paused: bool,
}

impl Virtual {
    /// Default clamp on a single frame's real delta (`0.25 s`), matching the
    /// death-spiral guard described in the design doc.
    pub const DEFAULT_MAX_DELTA: Duration = Duration::from_millis(250);
}

impl Default for Virtual {
    #[inline]
    fn default() -> Self {
        Self {
            max_delta: Self::DEFAULT_MAX_DELTA,
            relative_speed: 1.0,
            effective_speed: 1.0,
            paused: false,
        }
    }
}

impl TimeKind for Virtual {}

impl Time<Virtual> {
    /// Create a fresh virtual clock (speed `1.0`, not paused, default clamp).
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance the virtual clock from a real frame delta.
    ///
    /// The real delta is first clamped to [`Virtual::max_delta`], then scaled by
    /// the effective speed (`0.0` while paused, otherwise the relative speed).
    /// A paused clock therefore reports a zero delta while `elapsed` holds.
    #[inline]
    pub fn advance_by(&mut self, real_delta: Duration) {
        let max_delta = self.context.max_delta;
        let clamped = if real_delta > max_delta {
            max_delta
        } else {
            real_delta
        };
        let effective = if self.context.paused {
            0.0
        } else {
            self.context.relative_speed.max(0.0)
        };
        self.context.effective_speed = effective;
        let scaled = if effective == 1.0 {
            clamped
        } else {
            clamped.mul_f64(effective)
        };
        self.advance_generic(scaled);
    }

    /// Requested relative speed (`f32`).
    #[inline]
    pub fn relative_speed(&self) -> f32 {
        self.context.relative_speed as f32
    }

    /// Requested relative speed (`f64`).
    #[inline]
    pub fn relative_speed_f64(&self) -> f64 {
        self.context.relative_speed
    }

    /// Speed actually applied on the last [`advance_by`](Self::advance_by)
    /// (`0.0` while paused).
    #[inline]
    pub fn effective_speed(&self) -> f32 {
        self.context.effective_speed as f32
    }

    /// Speed actually applied on the last advance (`f64`).
    #[inline]
    pub fn effective_speed_f64(&self) -> f64 {
        self.context.effective_speed
    }

    /// Set the relative speed from an `f32`. Negative values are clamped to
    /// `0.0`; non-finite values are ignored.
    #[inline]
    pub fn set_relative_speed(&mut self, speed: f32) {
        self.set_relative_speed_f64(speed as f64);
    }

    /// Set the relative speed from an `f64`. Negative values are clamped to
    /// `0.0`; non-finite values are ignored.
    #[inline]
    pub fn set_relative_speed_f64(&mut self, speed: f64) {
        if speed.is_finite() {
            self.context.relative_speed = speed.max(0.0);
        }
    }

    /// The current max-delta clamp.
    #[inline]
    pub fn max_delta(&self) -> Duration {
        self.context.max_delta
    }

    /// Set the max-delta clamp applied to the real delta before scaling.
    #[inline]
    pub fn set_max_delta(&mut self, max_delta: Duration) {
        self.context.max_delta = max_delta;
    }

    /// Whether the clock is paused.
    #[inline]
    pub fn is_paused(&self) -> bool {
        self.context.paused
    }

    /// Pause the clock: subsequent advances report a zero delta while `elapsed`
    /// is held. The relative speed is preserved for when it unpauses.
    #[inline]
    pub fn pause(&mut self) {
        self.context.paused = true;
    }

    /// Unpause the clock, restoring the previously set relative speed.
    #[inline]
    pub fn unpause(&mut self) {
        self.context.paused = false;
    }
}
