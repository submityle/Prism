//! Frame budgets, deadlines and the clock abstraction that drives yielding.
//!
//! Time-slicing needs a notion of "how much of this frame is left", but a
//! `no_std` UI crate owns no clock. So the host supplies one through the
//! [`Clock`] trait — typically wrapping a monotonic timer — and the scheduler
//! only ever *reads* elapsed microseconds through it. Tests and deterministic
//! replays use [`ManualClock`], whose cursor the caller advances by hand.

use core::cell::Cell;

/// A monotonic source of elapsed microseconds.
///
/// Implementations must be non-decreasing: a later call to [`now_micros`] must
/// never observe a smaller value than an earlier one, otherwise a deadline
/// could appear to move backwards.
///
/// [`now_micros`]: Clock::now_micros
pub trait Clock {
    /// The current reading in microseconds since an arbitrary fixed origin.
    fn now_micros(&self) -> u64;
}

/// The per-frame time-slice target.
///
/// A 60 Hz frame is ~`16_666` µs; leaving headroom for paint and present, Loom
/// defaults the reconciliation slice to `8_000` µs so a long update yields with
/// time to spare rather than overrunning the frame.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FrameBudget {
    micros: u64,
}

impl FrameBudget {
    /// A conservative half-frame slice for 60 Hz displays.
    pub const DEFAULT: FrameBudget = FrameBudget { micros: 8_000 };

    /// Builds a budget of `micros` microseconds, clamped to at least 1 µs so a
    /// deadline always permits forward progress on at least one unit of work.
    #[must_use]
    pub const fn from_micros(micros: u64) -> FrameBudget {
        FrameBudget {
            micros: if micros == 0 { 1 } else { micros },
        }
    }

    /// Builds a budget from a refresh rate in hertz, reserving `reserve_frac`
    /// (0.0–1.0) of the frame for paint/present.
    ///
    /// A non-positive or non-finite `hz` falls back to [`DEFAULT`]; the reserve
    /// fraction is clamped into `0.0..=0.9` so some slice always remains.
    ///
    /// [`DEFAULT`]: FrameBudget::DEFAULT
    #[must_use]
    pub fn for_refresh_hz(hz: f32, reserve_frac: f32) -> FrameBudget {
        if !(hz.is_finite()) || hz <= 0.0 {
            return FrameBudget::DEFAULT;
        }
        let frame_us = 1_000_000.0 / hz;
        let keep = 1.0 - reserve_frac.clamp(0.0, 0.9);
        let slice = frame_us * keep;
        // `slice` is strictly positive and well under u64::MAX microseconds.
        FrameBudget::from_micros(slice as u64)
    }

    /// The slice length in microseconds.
    #[must_use]
    pub const fn micros(self) -> u64 {
        self.micros
    }

    /// Opens a [`Deadline`] starting at `start_micros`.
    #[must_use]
    pub const fn deadline_from(self, start_micros: u64) -> Deadline {
        Deadline {
            end: start_micros.saturating_add(self.micros),
        }
    }

    /// Opens a [`Deadline`] starting at the clock's current reading.
    #[must_use]
    pub fn deadline(self, clock: &impl Clock) -> Deadline {
        self.deadline_from(clock.now_micros())
    }
}

impl Default for FrameBudget {
    fn default() -> FrameBudget {
        FrameBudget::DEFAULT
    }
}

/// An absolute instant past which the scheduler should yield the frame.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Deadline {
    end: u64,
}

impl Deadline {
    /// Builds a deadline that expires at `end_micros` on the clock timeline.
    #[must_use]
    pub const fn at_micros(end_micros: u64) -> Deadline {
        Deadline { end: end_micros }
    }

    /// A deadline that is already expired, forcing an immediate yield.
    pub const EXPIRED: Deadline = Deadline { end: 0 };

    /// A deadline that never expires, letting work drain fully.
    pub const NEVER: Deadline = Deadline { end: u64::MAX };

    /// The absolute expiry reading in microseconds.
    #[must_use]
    pub const fn end_micros(self) -> u64 {
        self.end
    }

    /// Microseconds remaining at `now`, saturating at zero once expired.
    #[must_use]
    pub const fn remaining_at(self, now: u64) -> u64 {
        self.end.saturating_sub(now)
    }

    /// Whether `now` has reached or passed the deadline.
    #[must_use]
    pub const fn is_expired_at(self, now: u64) -> bool {
        now >= self.end
    }

    /// Whether the clock has reached or passed the deadline.
    #[must_use]
    pub fn is_expired(self, clock: &impl Clock) -> bool {
        self.is_expired_at(clock.now_micros())
    }
}

/// A caller-advanced [`Clock`] for deterministic tests and replays.
///
/// The cursor only ever moves forward: [`advance`](ManualClock::advance) adds
/// to it and [`set`](ManualClock::set) refuses to rewind, upholding the
/// monotonicity [`Clock`] requires.
#[derive(Debug, Default)]
pub struct ManualClock {
    now: Cell<u64>,
}

impl ManualClock {
    /// Creates a clock reading `start` microseconds.
    #[must_use]
    pub const fn new(start: u64) -> ManualClock {
        ManualClock {
            now: Cell::new(start),
        }
    }

    /// Advances the cursor by `delta` microseconds, saturating at [`u64::MAX`].
    pub fn advance(&self, delta: u64) {
        self.now.set(self.now.get().saturating_add(delta));
    }

    /// Moves the cursor to `value` when that does not rewind it.
    ///
    /// A `value` below the current reading is ignored so the clock stays
    /// monotonic.
    pub fn set(&self, value: u64) {
        if value > self.now.get() {
            self.now.set(value);
        }
    }
}

impl Clock for ManualClock {
    fn now_micros(&self) -> u64 {
        self.now.get()
    }
}
