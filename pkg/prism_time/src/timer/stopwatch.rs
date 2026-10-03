//! [`Stopwatch`]: a pausable accumulator of elapsed time.
//!
//! A stopwatch is the simplest timing primitive: it tracks how much time has
//! been fed into it via [`Stopwatch::tick`], can be paused/resumed, and reset.
//! It is the building block for [`Timer`](super::Timer).

use crate::Duration;

/// A pausable stopwatch that accumulates ticked time.
///
/// Unlike [`Time`](crate::Time) it is not tied to any clock context; the caller
/// decides which delta to feed it (usually the active clock's `delta()`), which
/// keeps it deterministic and `no_std` friendly.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stopwatch {
    elapsed: Duration,
    paused: bool,
}

impl Stopwatch {
    /// Create a stopwatch at zero, running (not paused).
    #[inline]
    pub const fn new() -> Self {
        Self {
            elapsed: Duration::ZERO,
            paused: false,
        }
    }

    /// Total accumulated time.
    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// Total accumulated time in seconds (`f32`).
    #[inline]
    pub fn elapsed_secs(&self) -> f32 {
        self.elapsed.as_secs_f32()
    }

    /// Total accumulated time in seconds (`f64`).
    #[inline]
    pub fn elapsed_secs_f64(&self) -> f64 {
        self.elapsed.as_secs_f64()
    }

    /// Overwrite the accumulated time (used by [`Timer`](super::Timer) when it
    /// wraps a repeating period).
    #[inline]
    pub fn set_elapsed(&mut self, elapsed: Duration) {
        self.elapsed = elapsed;
    }

    /// Advance by `delta` unless paused. Returns `self` for chaining.
    #[inline]
    pub fn tick(&mut self, delta: Duration) -> &Self {
        if !self.paused {
            self.elapsed = self.elapsed.saturating_add(delta);
        }
        self
    }

    /// Pause the stopwatch; subsequent [`tick`](Self::tick)s are ignored.
    #[inline]
    pub fn pause(&mut self) {
        self.paused = true;
    }

    /// Resume a paused stopwatch.
    #[inline]
    pub fn unpause(&mut self) {
        self.paused = false;
    }

    /// Whether the stopwatch is paused.
    #[inline]
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Reset accumulated time to zero. Does not change the paused state.
    #[inline]
    pub fn reset(&mut self) {
        self.elapsed = Duration::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_accumulate() {
        let mut sw = Stopwatch::new();
        sw.tick(Duration::from_millis(100));
        sw.tick(Duration::from_millis(150));
        assert_eq!(sw.elapsed(), Duration::from_millis(250));
        assert!((sw.elapsed_secs() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn pause_freezes_accumulation() {
        let mut sw = Stopwatch::new();
        sw.tick(Duration::from_millis(100));
        sw.pause();
        sw.tick(Duration::from_millis(100));
        assert_eq!(sw.elapsed(), Duration::from_millis(100));
        sw.unpause();
        sw.tick(Duration::from_millis(50));
        assert_eq!(sw.elapsed(), Duration::from_millis(150));
    }

    #[test]
    fn reset_zeroes_without_touching_pause() {
        let mut sw = Stopwatch::new();
        sw.pause();
        sw.set_elapsed(Duration::from_secs(5));
        sw.reset();
        assert_eq!(sw.elapsed(), Duration::ZERO);
        assert!(sw.is_paused());
    }
}
