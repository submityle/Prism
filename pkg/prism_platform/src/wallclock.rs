//! Real / system wall-clock time source.
//!
//! This is the wall-clock half of the M5 milestone (design doc §7 高精度时钟
//! "墙钟 `SystemTime`", §22 M5). It provides UTC / Unix-epoch time suitable for
//! **log timestamps** and correlating engine events with the outside world — it
//! is explicitly **not** for game logic or frame stepping, which must use the
//! monotonic [`crate::clock`] source instead.
//!
//! ## Wall clock vs. monotonic clock
//! A wall clock answers "what time is it in the world?" and can jump backwards
//! or forwards when the system clock is adjusted (NTP, manual changes, daylight
//! saving, suspend/resume). A monotonic clock ([`crate::clock::now`]) answers
//! "how much time has elapsed?" and never goes backwards, but has no relation
//! to calendar time. The two are different tools; this module keeps them
//! distinct by type so the distinction is visible at every call site. Use
//! [`WallClock::sample`] to capture both at once when a log line needs a
//! human-readable timestamp *and* a monotonic ordering key.
//!
//! Requires the `std` feature: wall-clock time comes from
//! [`std::time::SystemTime`].

use core::fmt;
use core::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::clock::{self, MonotonicNanos};

/// A wall-clock instant, i.e. a point on the system (calendar) clock.
///
/// Wraps [`std::time::SystemTime`]. Unlike [`crate::clock::MonotonicNanos`],
/// successive reads are **not** guaranteed to be ordered: the system clock can
/// be stepped backwards. Use this for timestamps, not for measuring elapsed
/// time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WallTime(SystemTime);

impl WallTime {
    /// Read the current wall-clock time.
    #[must_use]
    pub fn now() -> Self {
        Self(SystemTime::now())
    }

    /// The Unix epoch (`1970-01-01T00:00:00Z`) as a [`WallTime`].
    #[must_use]
    pub const fn unix_epoch() -> Self {
        Self(UNIX_EPOCH)
    }

    /// Borrow the underlying [`std::time::SystemTime`].
    #[must_use]
    pub const fn as_system_time(self) -> SystemTime {
        self.0
    }

    /// Build a [`WallTime`] from a [`std::time::SystemTime`].
    #[must_use]
    pub const fn from_system_time(time: SystemTime) -> Self {
        Self(time)
    }

    /// Whole seconds since the Unix epoch.
    ///
    /// Negative for instants before `1970-01-01T00:00:00Z`. Truncates toward
    /// zero at sub-second resolution; use [`WallTime::unix_nanos`] for the full
    /// resolution.
    #[must_use]
    pub fn unix_seconds(self) -> i64 {
        match self.0.duration_since(UNIX_EPOCH) {
            Ok(dur) => dur.as_secs() as i64,
            Err(err) => -(err.duration().as_secs() as i64),
        }
    }

    /// Milliseconds since the Unix epoch (negative before the epoch).
    #[must_use]
    pub fn unix_millis(self) -> i128 {
        match self.0.duration_since(UNIX_EPOCH) {
            Ok(dur) => dur.as_millis() as i128,
            Err(err) => -(err.duration().as_millis() as i128),
        }
    }

    /// Nanoseconds since the Unix epoch (negative before the epoch).
    ///
    /// `i128` is wide enough to hold the full nanosecond range for any
    /// representable [`SystemTime`].
    #[must_use]
    pub fn unix_nanos(self) -> i128 {
        match self.0.duration_since(UNIX_EPOCH) {
            Ok(dur) => dur.as_nanos() as i128,
            Err(err) => -(err.duration().as_nanos() as i128),
        }
    }

    /// Signed duration from `earlier` to `self`, in nanoseconds.
    ///
    /// Positive when `self` is later than `earlier`, negative otherwise. This
    /// tolerates the non-monotonic nature of the wall clock, unlike
    /// [`WallTime::duration_since`], which fails on backward steps.
    #[must_use]
    pub fn signed_nanos_since(self, earlier: WallTime) -> i128 {
        self.unix_nanos() - earlier.unix_nanos()
    }

    /// Positive [`Duration`] elapsed from `earlier` to `self`.
    ///
    /// Returns [`None`] when `self` is before `earlier` (the wall clock stepped
    /// backwards between the two reads). Callers that must tolerate backward
    /// steps should use [`WallTime::signed_nanos_since`].
    #[must_use]
    pub fn duration_since(self, earlier: WallTime) -> Option<Duration> {
        self.0.duration_since(earlier.0).ok()
    }
}

impl From<SystemTime> for WallTime {
    fn from(time: SystemTime) -> Self {
        Self(time)
    }
}

impl From<WallTime> for SystemTime {
    fn from(time: WallTime) -> Self {
        time.0
    }
}

impl fmt::Display for WallTime {
    /// Formats as a signed Unix-nanosecond count (a stable, locale-free form
    /// suitable for machine-readable logs). Human-calendar formatting lives in
    /// higher layers (`prism_time`), which own locale and timezone policy.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}ns@unix", self.unix_nanos())
    }
}

/// A paired wall-clock and monotonic reading captured together.
///
/// Logging and diagnostics typically want a human-readable timestamp
/// ([`WallClockSample::wall`]) for display *and* a monotonic key
/// ([`WallClockSample::monotonic`]) for correct ordering even when the wall
/// clock is stepped. [`WallClock::sample`] captures both back-to-back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WallClockSample {
    /// The wall-clock (calendar) reading.
    pub wall: WallTime,
    /// The monotonic reading captured alongside it.
    pub monotonic: MonotonicNanos,
}

/// The real system wall clock.
///
/// A zero-sized handle; all reads go straight to the OS. Exists to give the
/// facade a nameable type parallel to the other platform subsystems and to host
/// the [`WallClock::sample`] correlation helper.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WallClock;

impl WallClock {
    /// Construct a handle to the system wall clock.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Read the current wall-clock time.
    #[must_use]
    pub fn now(self) -> WallTime {
        WallTime::now()
    }

    /// Capture the wall clock and the monotonic clock together.
    ///
    /// The two reads are taken back-to-back (wall first, monotonic second);
    /// they are close but not atomic. This is the recommended way to timestamp
    /// a log event that must also be orderable.
    #[must_use]
    pub fn sample(self) -> WallClockSample {
        let wall = WallTime::now();
        let monotonic = clock::now();
        WallClockSample { wall, monotonic }
    }
}

/// Read the current wall-clock time (convenience for [`WallTime::now`]).
#[must_use]
pub fn now() -> WallTime {
    WallTime::now()
}

/// Capture a paired wall-clock + monotonic sample (convenience for
/// [`WallClock::sample`]).
#[must_use]
pub fn sample() -> WallClockSample {
    WallClock::new().sample()
}
