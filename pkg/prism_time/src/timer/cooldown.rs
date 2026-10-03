//! [`Cooldown`] and [`Throttle`]: gameplay rate-limiting primitives.
//!
//! A [`Cooldown`] models an ability on cooldown: it starts *ready*, is put on
//! cooldown with [`Cooldown::trigger`], and becomes ready again once the
//! duration has been ticked off. A [`Throttle`] rate-limits an action so it
//! fires at most once per interval, firing immediately the first time.

use crate::Duration;

/// An ability-style cooldown that counts down to ready.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cooldown {
    duration: Duration,
    remaining: Duration,
}

impl Cooldown {
    /// Create a cooldown of `duration`, starting ready (remaining zero).
    #[inline]
    pub const fn new(duration: Duration) -> Self {
        Self {
            duration,
            remaining: Duration::ZERO,
        }
    }

    /// Create a cooldown from a duration in seconds (`f32`).
    #[inline]
    pub fn from_seconds(seconds: f32) -> Self {
        Self::new(Duration::from_secs_f32(seconds.max(0.0)))
    }

    /// Whether the cooldown has elapsed and an action is allowed.
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.remaining.is_zero()
    }

    /// Put the cooldown on its full duration regardless of current state.
    #[inline]
    pub fn trigger(&mut self) {
        self.remaining = self.duration;
    }

    /// If ready, consume readiness and start the cooldown, returning `true`.
    /// Otherwise returns `false` and leaves the cooldown untouched.
    #[inline]
    pub fn try_trigger(&mut self) -> bool {
        if self.is_ready() {
            self.trigger();
            true
        } else {
            false
        }
    }

    /// Advance the cooldown by `delta`, saturating at ready. Returns `self`.
    #[inline]
    pub fn tick(&mut self, delta: Duration) -> &Self {
        self.remaining = self.remaining.saturating_sub(delta);
        self
    }

    /// Time until ready.
    #[inline]
    pub fn remaining(&self) -> Duration {
        self.remaining
    }

    /// Seconds until ready (`f32`).
    #[inline]
    pub fn remaining_secs(&self) -> f32 {
        self.remaining.as_secs_f32()
    }

    /// Fraction of the cooldown remaining, clamped to `0.0..=1.0`.
    #[inline]
    pub fn fraction_remaining(&self) -> f32 {
        if self.duration.is_zero() {
            0.0
        } else {
            (self.remaining.as_secs_f32() / self.duration.as_secs_f32()).clamp(0.0, 1.0)
        }
    }

    /// The configured cooldown duration.
    #[inline]
    pub fn duration(&self) -> Duration {
        self.duration
    }

    /// Change the cooldown duration. Does not change the current remaining.
    #[inline]
    pub fn set_duration(&mut self, duration: Duration) {
        self.duration = duration;
    }

    /// Force the cooldown back to ready.
    #[inline]
    pub fn reset(&mut self) {
        self.remaining = Duration::ZERO;
    }
}

/// Rate-limits an action to at most once per `interval`, firing immediately the
/// first time it is ready.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Throttle {
    interval: Duration,
    elapsed: Duration,
    ready: bool,
}

impl Throttle {
    /// Create a throttle with the given `interval`, starting ready.
    #[inline]
    pub const fn new(interval: Duration) -> Self {
        Self {
            interval,
            elapsed: Duration::ZERO,
            ready: true,
        }
    }

    /// Create a throttle from an interval in seconds (`f32`).
    #[inline]
    pub fn from_seconds(seconds: f32) -> Self {
        Self::new(Duration::from_secs_f32(seconds.max(0.0)))
    }

    /// Whether the throttle will allow an action right now.
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Advance the throttle by `delta`. Once the interval elapses while blocked,
    /// the throttle becomes ready again. Returns `self`.
    #[inline]
    pub fn tick(&mut self, delta: Duration) -> &Self {
        if !self.ready {
            self.elapsed = self.elapsed.saturating_add(delta);
            if self.elapsed >= self.interval {
                self.ready = true;
                self.elapsed = Duration::ZERO;
            }
        }
        self
    }

    /// If ready, fire and block for the interval, returning `true`. Otherwise
    /// returns `false`.
    #[inline]
    pub fn try_trigger(&mut self) -> bool {
        if self.ready {
            self.ready = false;
            self.elapsed = Duration::ZERO;
            true
        } else {
            false
        }
    }

    /// The configured interval.
    #[inline]
    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// Change the interval. Does not change the current blocked progress.
    #[inline]
    pub fn set_interval(&mut self, interval: Duration) {
        self.interval = interval;
    }

    /// Force the throttle back to ready.
    #[inline]
    pub fn reset(&mut self) {
        self.ready = true;
        self.elapsed = Duration::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cooldown_starts_ready() {
        let mut cd = Cooldown::from_seconds(2.0);
        assert!(cd.is_ready());
        assert!(cd.try_trigger());
        assert!(!cd.is_ready());
        // A second trigger is denied while on cooldown.
        assert!(!cd.try_trigger());
    }

    #[test]
    fn cooldown_ticks_to_ready() {
        let mut cd = Cooldown::from_seconds(1.0);
        cd.trigger();
        cd.tick(Duration::from_millis(400));
        assert!(!cd.is_ready());
        assert!((cd.fraction_remaining() - 0.6).abs() < 1e-6);
        cd.tick(Duration::from_millis(600));
        assert!(cd.is_ready());
        assert_eq!(cd.remaining(), Duration::ZERO);
    }

    #[test]
    fn throttle_fires_once_per_interval() {
        let mut th = Throttle::from_seconds(1.0);
        assert!(th.try_trigger()); // immediate first fire
        assert!(!th.try_trigger()); // blocked
        th.tick(Duration::from_millis(500));
        assert!(!th.is_ready());
        th.tick(Duration::from_millis(500));
        assert!(th.is_ready());
        assert!(th.try_trigger());
    }
}
