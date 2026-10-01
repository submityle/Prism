//! Time-driven animation drivers: [`Tween`] and the enter/exit [`Transition`]
//! helper.

use crate::easing::Easing;
use crate::lerp::Lerp;
use crate::math::clampf;

/// A tween interpolates `from -> to` over `duration` using an [`Easing`],
/// advanced by feeding elapsed time into [`Tween::step`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tween<T: Lerp> {
    /// Starting value (progress `0.0`).
    pub from: T,
    /// Target value (progress `1.0`).
    pub to: T,
    /// Total duration in seconds. Non-positive durations complete instantly.
    pub duration: f32,
    /// Easing applied to the normalized progress before interpolation.
    pub easing: Easing,
    /// Elapsed time so far, in seconds.
    pub elapsed: f32,
}

impl<T: Lerp + Clone> Tween<T> {
    /// Create a tween from `from` to `to` over `duration` seconds.
    #[inline]
    pub fn new(from: T, to: T, duration: f32, easing: Easing) -> Self {
        Self {
            from,
            to,
            duration,
            easing,
            elapsed: 0.0,
        }
    }

    /// Normalized, un-eased progress in `[0, 1]`.
    #[inline]
    pub fn progress(&self) -> f32 {
        if self.duration <= 0.0 {
            1.0
        } else {
            clampf(self.elapsed / self.duration, 0.0, 1.0)
        }
    }

    /// The current interpolated value for the present `elapsed` time.
    #[inline]
    pub fn value(&self) -> T {
        let eased = self.easing.sample(self.progress());
        self.from.lerp(&self.to, eased)
    }

    /// Advance by `dt` seconds and return the new interpolated value.
    #[inline]
    pub fn step(&mut self, dt: f32) -> T {
        self.elapsed += dt;
        // Keep elapsed bounded so it cannot drift unboundedly once finished.
        if self.duration > 0.0 && self.elapsed > self.duration {
            self.elapsed = self.duration;
        }
        self.value()
    }

    /// Whether the tween has reached its end.
    #[inline]
    pub fn finished(&self) -> bool {
        self.duration <= 0.0 || self.elapsed >= self.duration
    }

    /// Restart the tween from the beginning.
    #[inline]
    pub fn reset(&mut self) {
        self.elapsed = 0.0;
    }
}

/// The phase of a [`Transition`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransitionPhase {
    /// Fully hidden and idle (appearance `0.0`).
    Exited,
    /// Animating towards fully shown.
    Entering,
    /// Fully shown and idle (appearance `1.0`).
    Entered,
    /// Animating towards fully hidden.
    Exiting,
}

/// An enter/exit transition driving a scalar "appearance" value in `[0, 1]`,
/// where `0.0` is fully hidden and `1.0` is fully shown.
///
/// Call [`Transition::enter`] or [`Transition::exit`] to retarget, then feed
/// elapsed time into [`Transition::step`]. Retargeting mid-flight interpolates
/// from the current appearance, so interrupted transitions stay continuous.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transition {
    duration: f32,
    easing: Easing,
    from: f32,
    to: f32,
    elapsed: f32,
}

impl Transition {
    /// Create a transition that starts fully hidden (appearance `0.0`).
    #[inline]
    pub fn new(duration: f32, easing: Easing) -> Self {
        Self {
            duration,
            easing,
            from: 0.0,
            to: 0.0,
            // Start settled so `value()` reports 0.0 before any animation.
            elapsed: if duration > 0.0 { duration } else { 0.0 },
        }
    }

    /// Begin animating towards fully shown, from the current appearance.
    #[inline]
    pub fn enter(&mut self) {
        self.retarget(1.0);
    }

    /// Begin animating towards fully hidden, from the current appearance.
    #[inline]
    pub fn exit(&mut self) {
        self.retarget(0.0);
    }

    fn retarget(&mut self, to: f32) {
        self.from = self.value();
        self.to = to;
        self.elapsed = 0.0;
    }

    /// Normalized progress of the current enter/exit animation in `[0, 1]`.
    #[inline]
    fn progress(&self) -> f32 {
        if self.duration <= 0.0 {
            1.0
        } else {
            clampf(self.elapsed / self.duration, 0.0, 1.0)
        }
    }

    /// The current appearance value in `[0, 1]`.
    #[inline]
    pub fn value(&self) -> f32 {
        let eased = self.easing.sample(self.progress());
        self.from + (self.to - self.from) * eased
    }

    /// Advance by `dt` seconds and return the new appearance value.
    #[inline]
    pub fn step(&mut self, dt: f32) -> f32 {
        self.elapsed += dt;
        if self.duration > 0.0 && self.elapsed > self.duration {
            self.elapsed = self.duration;
        }
        self.value()
    }

    /// Whether the current enter/exit animation has finished.
    #[inline]
    pub fn finished(&self) -> bool {
        self.duration <= 0.0 || self.elapsed >= self.duration
    }

    /// The current [`TransitionPhase`].
    #[inline]
    pub fn phase(&self) -> TransitionPhase {
        if self.finished() {
            if self.to >= 0.5 {
                TransitionPhase::Entered
            } else {
                TransitionPhase::Exited
            }
        } else if self.to >= self.from {
            TransitionPhase::Entering
        } else {
            TransitionPhase::Exiting
        }
    }
}
