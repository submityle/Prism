//! A monotonic [`Instant`] source.
//!
//! With the `std` feature, [`Instant`] wraps the platform monotonic clock and
//! [`Instant::now`] is available. Without `std`, [`Instant`] is a monotonic
//! nanosecond tick that callers advance explicitly (e.g. from a platform HAL),
//! keeping the clock usable in headless/`no_std` builds.

use core::ops::Add;
use core::time::Duration;

#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
/// A monotonic point in time.
pub struct Instant(std::time::Instant);

#[cfg(feature = "std")]
impl Instant {
    /// The current monotonic instant.
    #[inline]
    pub fn now() -> Self {
        Self(std::time::Instant::now())
    }
    /// Duration since `earlier`, clamped to zero if `earlier` is later
    /// (monotonic clocks never go backwards, but this stays defensive).
    #[inline]
    pub fn saturating_duration_since(self, earlier: Self) -> Duration {
        self.0.saturating_duration_since(earlier.0)
    }
}

#[cfg(feature = "std")]
impl Add<Duration> for Instant {
    type Output = Instant;
    #[inline]
    fn add(self, rhs: Duration) -> Instant {
        Instant(self.0 + rhs)
    }
}

#[cfg(not(feature = "std"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
/// A monotonic point in time, measured in nanoseconds from an arbitrary epoch.
pub struct Instant(u64);

#[cfg(not(feature = "std"))]
impl Instant {
    /// Construct from a raw monotonic nanosecond tick (platform-supplied).
    #[inline]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }
    /// The raw nanosecond tick.
    #[inline]
    pub const fn as_nanos(self) -> u64 {
        self.0
    }
    /// Duration since `earlier`, clamped to zero if `earlier` is later.
    #[inline]
    pub fn saturating_duration_since(self, earlier: Self) -> Duration {
        Duration::from_nanos(self.0.saturating_sub(earlier.0))
    }
}

#[cfg(not(feature = "std"))]
impl Add<Duration> for Instant {
    type Output = Instant;
    #[inline]
    fn add(self, rhs: Duration) -> Instant {
        Instant(self.0.saturating_add(rhs.as_nanos() as u64))
    }
}
