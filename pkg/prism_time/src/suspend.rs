//! **§24.7 — suspend / resume (CPU-deterministic portion).** The time-source
//! semantics a process needs when the OS suspends it (mobile background, console
//! sleep, window minimise) and later resumes.
//!
//! The hazard §24.7 names: on resume, the wall-clock gap accumulated while
//! suspended must **not** be charged to the simulation as one giant frame delta
//! (that teleports physics through walls and snaps animation). [`SuspendableClock`]
//! fixes this deterministically: it advances a simulated elapsed measure only
//! across *running* intervals, so the suspended span contributes zero to
//! `elapsed` / ticks and resume produces no jump.
//!
//! It consumes a monotonically-increasing wall timestamp (as a [`Duration`]
//! since some epoch) and exposes:
//!
//! - [`advance_to`](SuspendableClock::advance_to) — fold the running wall delta
//!   since the last timestamp into simulated time, with an optional per-step
//!   **max catch-up clamp** ([`with_max_delta`](SuspendableClock::with_max_delta),
//!   the §12 max-delta guard) that discards the excess of an over-long step.
//! - [`suspend`](SuspendableClock::suspend) — count the running portion up to
//!   the suspend instant, then stop advancing.
//! - [`resume`](SuspendableClock::resume) — restart from the resume instant so
//!   the suspended gap is excluded; returns the skipped [`Duration`] for logging.
//!
//! All math is integer nanoseconds (`u128`); no wall clock is read internally
//! and no floating point enters the path, so a given timestamp sequence yields
//! a bit-identical simulated timeline across runs. `no_std + alloc`, no `unsafe`.
//!
//! ## Honest boundary
//! Delivering the real OS suspend / resume signals (and deciding the background
//! policy — freeze `Virtual`, keep `Real` running for the network heartbeat, or
//! run a low-frequency background tick) lives in `prism_platform` / `prism_app`,
//! which forward those events to this clock (design doc §24.7). Wiring the
//! real platform suspend/resume event source remains **PLANNED**; this layer is
//! the deterministic correction applied to caller-fed timestamps only, and it
//! never touches an OS API or a clock itself.

use crate::Duration;

/// Nanoseconds in one second.
const NANOS_PER_SEC: u128 = 1_000_000_000;

/// Convert a `u128` nanosecond count into a [`Duration`], saturating the
/// seconds field rather than overflowing.
#[inline]
fn duration_from_nanos_u128(nanos: u128) -> Duration {
    let secs = (nanos / NANOS_PER_SEC).min(u64::MAX as u128) as u64;
    let sub = (nanos % NANOS_PER_SEC) as u32;
    Duration::new(secs, sub)
}

/// A time source with deterministic suspend / resume semantics.
///
/// Drive it with a monotonically-increasing wall timestamp (nanoseconds since
/// an arbitrary epoch, passed as a [`Duration`]). It accumulates *simulated*
/// elapsed time across running intervals only; the span between a
/// [`suspend`](Self::suspend) and the matching [`resume`](Self::resume) is
/// excluded, so resume introduces no delta spike. An optional per-step max
/// catch-up clamp bounds any single running delta.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SuspendableClock {
    /// Simulated elapsed time in nanoseconds (advances only while running).
    simulated: u128,
    /// Last observed wall timestamp in nanoseconds, or `None` before the first
    /// observation. While suspended this holds the suspend instant.
    last_wall: Option<u128>,
    /// Whether the clock is currently suspended.
    suspended: bool,
    /// Optional per-step catch-up clamp in nanoseconds. A single running delta
    /// larger than this is folded in only up to the cap; the excess is dropped.
    max_delta: Option<u128>,
    /// Total wall time (ns) excluded while suspended, for reporting.
    total_suspended: u128,
    /// Total wall time (ns) discarded by the max-delta clamp, for reporting.
    total_clamped: u128,
}

impl SuspendableClock {
    /// A fresh, running clock at simulated time zero with no catch-up clamp.
    /// The first [`advance_to`](Self::advance_to) establishes the baseline and
    /// applies a zero delta.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            simulated: 0,
            last_wall: None,
            suspended: false,
            max_delta: None,
            total_suspended: 0,
            total_clamped: 0,
        }
    }

    /// Set the per-step max catch-up clamp: any single running delta is folded
    /// in only up to `max_delta` (the design doc §12 guard against a hitch
    /// becoming one huge step). A zero clamp means every running step
    /// contributes nothing.
    #[inline]
    #[must_use]
    pub const fn with_max_delta(mut self, max_delta: Duration) -> Self {
        self.max_delta = Some(max_delta.as_nanos());
        self
    }

    /// The current per-step catch-up clamp, if any.
    #[inline]
    #[must_use]
    pub fn max_delta(&self) -> Option<Duration> {
        self.max_delta.map(duration_from_nanos_u128)
    }

    /// Set (or, with `None`, clear) the per-step catch-up clamp.
    #[inline]
    pub fn set_max_delta(&mut self, max_delta: Option<Duration>) {
        self.max_delta = max_delta.map(|d| d.as_nanos());
    }

    /// Whether the clock is currently suspended.
    #[inline]
    #[must_use]
    pub const fn is_suspended(&self) -> bool {
        self.suspended
    }

    /// Simulated elapsed time as a [`Duration`].
    #[inline]
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        duration_from_nanos_u128(self.simulated)
    }

    /// Simulated elapsed time in exact nanoseconds.
    #[inline]
    #[must_use]
    pub const fn elapsed_nanos(&self) -> u128 {
        self.simulated
    }

    /// Simulated elapsed time as an integer tick count at a fixed `tick`
    /// duration (`floor(elapsed / tick)`). A zero `tick` yields `0`. This is
    /// the deterministic `tick` view the design doc §24.7 asks the suspend
    /// correction to keep consistent with `elapsed`.
    #[inline]
    #[must_use]
    pub fn ticks(&self, tick: Duration) -> u128 {
        self.simulated.checked_div(tick.as_nanos()).unwrap_or(0)
    }

    /// Total wall time excluded while suspended, as a [`Duration`].
    #[inline]
    #[must_use]
    pub fn total_suspended(&self) -> Duration {
        duration_from_nanos_u128(self.total_suspended)
    }

    /// Total wall time discarded by the max-delta clamp, as a [`Duration`].
    #[inline]
    #[must_use]
    pub fn total_clamped(&self) -> Duration {
        duration_from_nanos_u128(self.total_clamped)
    }

    /// Fold the running wall delta since the previous timestamp into simulated
    /// time and return the simulated delta actually applied.
    ///
    /// - While **suspended**, this is a no-op and returns [`Duration::ZERO`]:
    ///   timestamps observed during suspension do not advance simulated time
    ///   (the suspended span is excluded by [`resume`](Self::resume)).
    /// - On the **first** observation it only establishes the baseline and
    ///   returns zero.
    /// - Otherwise the delta is `wall - last_wall` (saturating; a
    ///   non-monotonic backward timestamp yields zero), clamped to the optional
    ///   [`max_delta`](Self::max_delta) with the excess discarded.
    pub fn advance_to(&mut self, wall: Duration) -> Duration {
        let w = wall.as_nanos();
        if self.suspended {
            // Ignore timestamps while suspended; `resume` excludes the gap.
            return Duration::ZERO;
        }
        let Some(last) = self.last_wall else {
            self.last_wall = Some(w);
            return Duration::ZERO;
        };
        let raw = w.saturating_sub(last);
        self.last_wall = Some(w);
        let applied = match self.max_delta {
            Some(cap) if raw > cap => {
                self.total_clamped = self.total_clamped.saturating_add(raw - cap);
                cap
            }
            _ => raw,
        };
        self.simulated = self.simulated.saturating_add(applied);
        duration_from_nanos_u128(applied)
    }

    /// Suspend at wall instant `at`: first fold the running portion up to `at`
    /// into simulated time (honouring the catch-up clamp), then stop advancing.
    /// Returns the simulated delta applied for that final running portion.
    ///
    /// A second `suspend` while already suspended is a no-op (returns zero).
    pub fn suspend(&mut self, at: Duration) -> Duration {
        if self.suspended {
            return Duration::ZERO;
        }
        // `advance_to` folds the running portion up to `at` and sets
        // `last_wall = at`, which then serves as the suspend instant so
        // `resume` can measure the excluded span.
        let applied = self.advance_to(at);
        self.suspended = true;
        applied
    }

    /// Resume at wall instant `at`: exclude the `[suspend_instant, at]` span
    /// from simulated time by rebasing the reference to `at`, so the next
    /// [`advance_to`](Self::advance_to) measures from here. Returns the skipped
    /// (suspended) [`Duration`] for logging.
    ///
    /// A `resume` while not suspended is a no-op (returns zero).
    pub fn resume(&mut self, at: Duration) -> Duration {
        if !self.suspended {
            return Duration::ZERO;
        }
        let a = at.as_nanos();
        let skipped = match self.last_wall {
            Some(suspend_at) => a.saturating_sub(suspend_at),
            None => 0,
        };
        self.total_suspended = self.total_suspended.saturating_add(skipped);
        self.last_wall = Some(a);
        self.suspended = false;
        duration_from_nanos_u128(skipped)
    }

    /// Reset simulated time, the baseline, suspension state, and the reporting
    /// counters to their initial values, keeping the catch-up clamp.
    #[inline]
    pub fn reset(&mut self) {
        self.simulated = 0;
        self.last_wall = None;
        self.suspended = false;
        self.total_suspended = 0;
        self.total_clamped = 0;
    }
}

impl Default for SuspendableClock {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
