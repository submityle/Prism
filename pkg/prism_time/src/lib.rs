//! # `prism_time`
//!
//! Prism's time kernel. It provides the monotonic time source and the generic
//! [`Time<T>`] clock with `delta`/`elapsed` accessors, plus the three clock
//! contexts [`Real`], [`Virtual`], and [`Fixed`] and a default-context switch
//! ([`Clocks`]).
//!
//! ## Design
//! A single generic [`Time<T>`] reuses one set of `delta`/`elapsed` accessors;
//! the context `T` decides advancement semantics. Both `f32` (fast, enough for
//! a frame) and `f64` (long-session precision) deltas are exposed so long runs
//! do not accumulate `f32` error.
//!
//! The three clocks mirror the design doc's separation:
//! - [`Time<Real>`]: monotonic wall-clock time, unaffected by pause/scale.
//! - [`Time<Virtual>`]: game time with time dilation (scale) and pause, plus a
//!   max-delta clamp that guards against the fixed-step spiral of death.
//! - [`Time<Fixed>`]: a deterministic fixed-timestep accumulator driven by the
//!   virtual delta; it exposes how many fixed steps to run and the leftover
//!   `overstep` (interpolation alpha).
//!
//! [`Clocks`] bundles all three and exposes a context-less default [`Time<()>`]
//! that the app points at `Virtual` during variable update and at `Fixed`
//! around the fixed-update schedule, so a system can call `delta_secs()` and
//! get the correct value without naming a context.
//!
//! ## Milestone status (per the design-doc roadmap)
//! - **M0 (done):** monotonic [`Instant`]/[`Duration`], unit conversions, and
//!   [`Time<Real>`] with monotonic/advancement tests.
//! - **M1 (this crate, done):** [`Time<Virtual>`] (scale/pause/clamp),
//!   [`Time<Fixed>`] (accumulator + overstep), and the default-context switch.
//! - **M2+ (planned):** timers/stopwatch, rational/fixed-point deterministic
//!   stepping, smoothing, and network clocks.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

mod clock;
mod fixed;
mod instant;
mod virtual_time;

pub use clock::{Clocks, DefaultSource};
pub use core::time::Duration;
pub use fixed::Fixed;
pub use instant::Instant;
pub use virtual_time::Virtual;

/// Marker for a time context, selecting advancement semantics.
pub trait TimeKind: Default {}

/// Real wall-clock time, advanced by the platform monotonic clock.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Real {
    startup: Option<Instant>,
    last_update: Option<Instant>,
}

impl TimeKind for Real {}

/// The context-less default clock. Its readings are copied from whichever
/// context [`Clocks`] has made active (see [`Clocks::set_source`]).
impl TimeKind for () {}

/// A clock. The context `T` decides how it advances; the accessors below are
/// shared across all clock kinds.
#[derive(Clone, Copy, Debug)]
pub struct Time<T: TimeKind> {
    context: T,
    delta: Duration,
    elapsed: Duration,
    delta_secs: f32,
    delta_secs_f64: f64,
    elapsed_secs: f32,
    elapsed_secs_f64: f64,
}

impl<T: TimeKind> Default for Time<T> {
    #[inline]
    fn default() -> Self {
        Self {
            context: T::default(),
            delta: Duration::ZERO,
            elapsed: Duration::ZERO,
            delta_secs: 0.0,
            delta_secs_f64: 0.0,
            elapsed_secs: 0.0,
            elapsed_secs_f64: 0.0,
        }
    }
}

impl<T: TimeKind> Time<T> {
    /// Time advanced on the last update.
    #[inline]
    pub fn delta(&self) -> Duration {
        self.delta
    }
    /// Last delta in seconds (`f32`).
    #[inline]
    pub fn delta_secs(&self) -> f32 {
        self.delta_secs
    }
    /// Last delta in seconds (`f64`).
    #[inline]
    pub fn delta_secs_f64(&self) -> f64 {
        self.delta_secs_f64
    }
    /// Total time elapsed since the clock started.
    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }
    /// Total elapsed seconds (`f32`).
    #[inline]
    pub fn elapsed_secs(&self) -> f32 {
        self.elapsed_secs
    }
    /// Total elapsed seconds (`f64`).
    #[inline]
    pub fn elapsed_secs_f64(&self) -> f64 {
        self.elapsed_secs_f64
    }
    /// Read-only access to the context.
    #[inline]
    pub fn context(&self) -> &T {
        &self.context
    }

    /// Advance the shared accessors by `delta`. Used by every context after it
    /// has computed its own step. Private so each context exposes its own
    /// semantic entry point (e.g. [`Time::<Virtual>::advance_by`]).
    #[inline]
    fn advance_generic(&mut self, delta: Duration) {
        self.delta = delta;
        self.elapsed = self.elapsed.saturating_add(delta);
        self.delta_secs = delta.as_secs_f32();
        self.delta_secs_f64 = delta.as_secs_f64();
        self.elapsed_secs = self.elapsed.as_secs_f32();
        self.elapsed_secs_f64 = self.elapsed.as_secs_f64();
    }
}

impl Time<Real> {
    /// Create a fresh real-time clock. The first [`Time::update`] establishes
    /// the baseline and reports a zero delta.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance using the platform monotonic clock.
    #[cfg(feature = "std")]
    #[inline]
    pub fn update(&mut self) {
        self.update_with_instant(Instant::now());
    }

    /// Advance using a caller-supplied instant. Deterministic and `no_std`
    /// friendly; the backbone of the `update` convenience and of tests.
    #[inline]
    pub fn update_with_instant(&mut self, instant: Instant) {
        let delta = match self.context.last_update {
            // Monotonic clock: never report a negative delta.
            Some(last) => instant.saturating_duration_since(last),
            None => {
                self.context.startup = Some(instant);
                Duration::ZERO
            }
        };
        self.context.last_update = Some(instant);
        self.advance_generic(delta);
    }

    /// Advance by an explicit delta (e.g. for a headless/fixed feeder).
    #[inline]
    pub fn update_with_delta(&mut self, delta: Duration) {
        self.advance_generic(delta);
    }

    /// The instant of the first update, if any.
    #[inline]
    pub fn startup(&self) -> Option<Instant> {
        self.context.startup
    }
}

/// Common imports.
pub mod prelude {
    pub use crate::{Clocks, DefaultSource, Duration, Fixed, Instant, Real, Time, Virtual};
}

#[cfg(test)]
mod tests;
