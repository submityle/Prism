//! [`Timer`]: a countdown/periodic timer built on [`Stopwatch`].
//!
//! A timer tracks elapsed time against a target `duration`. In [`TimerMode::Once`]
//! it latches finished once the duration is reached; in [`TimerMode::Repeating`]
//! it wraps the leftover time and reports how many whole periods elapsed in the
//! last tick ([`Timer::times_finished_this_tick`]), so no periods are dropped on
//! a large delta.

use super::Stopwatch;
use crate::Duration;

/// How a [`Timer`] behaves when it reaches its duration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TimerMode {
    /// Latch finished once the duration is reached and stop advancing.
    #[default]
    Once,
    /// Wrap around and keep running, counting whole periods per tick.
    Repeating,
}

/// A timer counting elapsed time toward a `duration`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Timer {
    stopwatch: Stopwatch,
    duration: Duration,
    mode: TimerMode,
    finished: bool,
    times_finished_this_tick: u32,
}

impl Timer {
    /// Create a timer with the given `duration` and `mode`.
    #[inline]
    pub fn new(duration: Duration, mode: TimerMode) -> Self {
        Self {
            stopwatch: Stopwatch::new(),
            duration,
            mode,
            finished: false,
            times_finished_this_tick: 0,
        }
    }

    /// Create a timer from a duration in seconds (`f32`).
    #[inline]
    pub fn from_seconds(seconds: f32, mode: TimerMode) -> Self {
        Self::new(Duration::from_secs_f32(seconds.max(0.0)), mode)
    }

    /// Whether the timer has reached its duration.
    ///
    /// For [`TimerMode::Repeating`] this is true only on the ticks where a
    /// period boundary was crossed (it mirrors [`just_finished`](Self::just_finished)
    /// semantics for repeating timers, which never permanently latch).
    #[inline]
    pub fn finished(&self) -> bool {
        self.finished
    }

    /// Whether the timer finished during the last [`tick`](Self::tick).
    #[inline]
    pub fn just_finished(&self) -> bool {
        self.times_finished_this_tick > 0
    }

    /// Elapsed time toward the duration.
    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.stopwatch.elapsed()
    }

    /// Elapsed seconds (`f32`).
    #[inline]
    pub fn elapsed_secs(&self) -> f32 {
        self.stopwatch.elapsed_secs()
    }

    /// Elapsed seconds (`f64`).
    #[inline]
    pub fn elapsed_secs_f64(&self) -> f64 {
        self.stopwatch.elapsed_secs_f64()
    }

    /// Overwrite the elapsed time, recomputing the finished flag.
    #[inline]
    pub fn set_elapsed(&mut self, elapsed: Duration) {
        self.stopwatch.set_elapsed(elapsed);
        self.finished = self.stopwatch.elapsed() >= self.duration;
    }

    /// The target duration.
    #[inline]
    pub fn duration(&self) -> Duration {
        self.duration
    }

    /// Change the target duration. Does not change elapsed time.
    #[inline]
    pub fn set_duration(&mut self, duration: Duration) {
        self.duration = duration;
    }

    /// The timer mode.
    #[inline]
    pub fn mode(&self) -> TimerMode {
        self.mode
    }

    /// Change the timer mode. Switching to [`TimerMode::Repeating`] clears a
    /// latched finished flag so the timer keeps running.
    #[inline]
    pub fn set_mode(&mut self, mode: TimerMode) {
        if self.mode == TimerMode::Once && mode == TimerMode::Repeating && self.finished {
            self.finished = false;
        }
        self.mode = mode;
    }

    /// Whether the timer is paused.
    #[inline]
    pub fn is_paused(&self) -> bool {
        self.stopwatch.is_paused()
    }

    /// Pause the timer.
    #[inline]
    pub fn pause(&mut self) {
        self.stopwatch.pause();
    }

    /// Resume the timer.
    #[inline]
    pub fn unpause(&mut self) {
        self.stopwatch.unpause();
    }

    /// Reset the timer to zero elapsed and clear the finished flag.
    #[inline]
    pub fn reset(&mut self) {
        self.stopwatch.reset();
        self.finished = false;
        self.times_finished_this_tick = 0;
    }

    /// Fraction of the duration elapsed, clamped to `0.0..=1.0`.
    #[inline]
    pub fn fraction(&self) -> f32 {
        if self.duration.is_zero() {
            1.0
        } else {
            (self.elapsed_secs() / self.duration.as_secs_f32()).clamp(0.0, 1.0)
        }
    }

    /// Fraction of the duration remaining, clamped to `0.0..=1.0`.
    #[inline]
    pub fn fraction_remaining(&self) -> f32 {
        1.0 - self.fraction()
    }

    /// Time remaining until the duration (saturating at zero).
    #[inline]
    pub fn remaining(&self) -> Duration {
        self.duration.saturating_sub(self.elapsed())
    }

    /// Remaining seconds (`f32`).
    #[inline]
    pub fn remaining_secs(&self) -> f32 {
        self.remaining().as_secs_f32()
    }

    /// Number of whole periods completed during the last [`tick`](Self::tick).
    ///
    /// For [`TimerMode::Once`] this is `0` or `1`. For [`TimerMode::Repeating`]
    /// a large delta can complete several periods at once.
    #[inline]
    pub fn times_finished_this_tick(&self) -> u32 {
        self.times_finished_this_tick
    }

    /// Advance the timer by `delta`. Returns `self` for chaining.
    pub fn tick(&mut self, delta: Duration) -> &Self {
        if self.is_paused() {
            self.times_finished_this_tick = 0;
            return self;
        }

        // A one-shot timer that already finished does not advance further.
        if self.mode == TimerMode::Once && self.finished {
            self.times_finished_this_tick = 0;
            return self;
        }

        self.stopwatch.tick(delta);
        let elapsed = self.stopwatch.elapsed();
        self.finished = elapsed >= self.duration;

        if !self.finished {
            self.times_finished_this_tick = 0;
            return self;
        }

        match self.mode {
            TimerMode::Once => {
                self.times_finished_this_tick = 1;
                // Clamp elapsed to the duration so `fraction()` reads exactly 1.0.
                self.stopwatch.set_elapsed(self.duration);
            }
            TimerMode::Repeating => {
                if self.duration.is_zero() {
                    // Degenerate zero-period timer: it "completes" every tick.
                    self.times_finished_this_tick = u32::MAX;
                    self.stopwatch.set_elapsed(Duration::ZERO);
                } else {
                    let period = self.duration.as_nanos();
                    let total = elapsed.as_nanos();
                    let periods = total / period;
                    self.times_finished_this_tick =
                        u32::try_from(periods).unwrap_or(u32::MAX);
                    let remainder_nanos = total % period;
                    self.stopwatch.set_elapsed(nanos_to_duration(remainder_nanos));
                }
            }
        }
        self
    }
}

/// Convert a `u128` nanosecond count (always `< duration < u64::MAX` secs in
/// practice here, since it is a remainder) back into a [`Duration`].
#[inline]
fn nanos_to_duration(nanos: u128) -> Duration {
    const NANOS_PER_SEC: u128 = 1_000_000_000;
    let secs = (nanos / NANOS_PER_SEC) as u64;
    let sub = (nanos % NANOS_PER_SEC) as u32;
    Duration::new(secs, sub)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn once_latches_and_clamps() {
        let mut t = Timer::from_seconds(1.0, TimerMode::Once);
        t.tick(Duration::from_millis(600));
        assert!(!t.finished());
        assert!(!t.just_finished());
        assert!((t.fraction() - 0.6).abs() < 1e-6);

        t.tick(Duration::from_millis(600));
        assert!(t.finished());
        assert!(t.just_finished());
        assert_eq!(t.times_finished_this_tick(), 1);
        assert_eq!(t.elapsed(), Duration::from_secs(1));
        assert!((t.fraction() - 1.0).abs() < 1e-6);

        // Further ticks do nothing and clear just_finished.
        t.tick(Duration::from_secs(10));
        assert!(t.finished());
        assert!(!t.just_finished());
        assert_eq!(t.elapsed(), Duration::from_secs(1));
    }

    #[test]
    fn repeating_wraps_and_counts_periods() {
        let mut t = Timer::from_seconds(1.0, TimerMode::Repeating);
        // A 2.5s delta completes two whole 1s periods, leaving 0.5s.
        t.tick(Duration::from_millis(2500));
        assert!(t.finished());
        assert!(t.just_finished());
        assert_eq!(t.times_finished_this_tick(), 2);
        assert_eq!(t.elapsed(), Duration::from_millis(500));

        // Next small tick does not complete a period.
        t.tick(Duration::from_millis(100));
        assert!(!t.just_finished());
        assert_eq!(t.times_finished_this_tick(), 0);
        assert_eq!(t.elapsed(), Duration::from_millis(600));
    }

    #[test]
    fn remaining_and_fraction_remaining() {
        let mut t = Timer::from_seconds(4.0, TimerMode::Once);
        t.tick(Duration::from_secs(1));
        assert_eq!(t.remaining(), Duration::from_secs(3));
        assert!((t.fraction_remaining() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn paused_timer_ignores_ticks() {
        let mut t = Timer::from_seconds(1.0, TimerMode::Once);
        t.pause();
        t.tick(Duration::from_secs(5));
        assert_eq!(t.elapsed(), Duration::ZERO);
        assert!(!t.finished());
        t.unpause();
        t.tick(Duration::from_secs(5));
        assert!(t.finished());
    }

    #[test]
    fn zero_duration_repeating_is_bounded() {
        let mut t = Timer::new(Duration::ZERO, TimerMode::Repeating);
        t.tick(Duration::from_secs(1));
        assert_eq!(t.times_finished_this_tick(), u32::MAX);
        assert_eq!(t.elapsed(), Duration::ZERO);
        assert!((t.fraction() - 1.0).abs() < 1e-6);
    }
}
