//! [`ServerTick`]: an authoritative fixed-rate server tick clock.
//!
//! A networked simulation is anchored to the server's tick number: the server
//! advances a fixed-rate clock (e.g. 60 Hz) and stamps authoritative state with
//! the current tick; clients align their simulation and inputs to that tick.
//!
//! [`ServerTick`] wraps the deterministic [`TickClock`](crate::TickClock) so the
//! tick count advances by exact integer arithmetic (no floating drift over a
//! 24/7 session) and is bit-identical across runs — the determinism contract
//! that lock-step and rollback networking rely on. On top of the raw clock it
//! adds the server-authoritative operations: reading the current tick, mapping
//! a tick to its exact time, and *reconciling* the local estimate to an
//! authoritative tick received from the server.

use crate::{Duration, TickClock, TickSnapshot};

/// An authoritative fixed-rate tick clock for networked simulation.
///
/// Advance it with wall/real deltas via [`advance`](Self::advance); it reports
/// how many whole server ticks elapsed. The leftover is exposed as an
/// interpolation [`overstep_fraction`](Self::overstep_fraction). The clock is
/// deterministic: identical delta sequences yield identical tick counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServerTick {
    clock: TickClock,
}

impl ServerTick {
    /// Create an authoritative clock at `hz` ticks per second (e.g. `60`),
    /// starting at tick 0.
    ///
    /// # Panics
    /// Panics if `hz` is zero.
    #[inline]
    #[must_use]
    pub const fn from_hz(hz: u64) -> Self {
        Self {
            clock: TickClock::from_hz(hz),
        }
    }

    /// Build from an existing deterministic [`TickClock`].
    #[inline]
    #[must_use]
    pub const fn from_clock(clock: TickClock) -> Self {
        Self { clock }
    }

    /// The current authoritative tick number.
    #[inline]
    #[must_use]
    pub const fn tick(&self) -> u64 {
        self.clock.tick()
    }

    /// Read-only access to the underlying deterministic clock.
    #[inline]
    #[must_use]
    pub const fn clock(&self) -> &TickClock {
        &self.clock
    }

    /// The interpolation alpha into the current (incomplete) tick, in `[0, 1)`.
    #[inline]
    #[must_use]
    pub fn overstep_fraction(&self) -> f32 {
        self.clock.overstep_fraction()
    }

    /// The interpolation alpha as `f64`.
    #[inline]
    #[must_use]
    pub fn overstep_fraction_f64(&self) -> f64 {
        self.clock.overstep_fraction_f64()
    }

    /// Elapsed simulated time (`tick * step`), as a [`Duration`].
    #[inline]
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.clock.elapsed()
    }

    /// The exact duration of one server tick.
    #[inline]
    #[must_use]
    pub fn tick_duration(&self) -> Duration {
        self.clock.step().as_duration()
    }

    /// The exact time of the *start* of tick `tick` (`tick * step`),
    /// as a [`Duration`] from the clock epoch.
    #[inline]
    #[must_use]
    pub fn time_of_tick(&self, tick: u64) -> Duration {
        let step = self.clock.step();
        let nanos =
            (tick as u128).saturating_mul(step.nanos_num() as u128) / (step.nanos_den() as u128);
        let secs = (nanos / 1_000_000_000).min(u64::MAX as u128) as u64;
        let sub = (nanos % 1_000_000_000) as u32;
        Duration::new(secs, sub)
    }

    /// Feed a real delta and advance the authoritative tick, returning the
    /// number of whole ticks that elapsed this call.
    ///
    /// Bounded by the underlying clock's max-substeps guard so a hitch cannot
    /// run an unbounded number of ticks in one call.
    #[inline]
    pub fn advance(&mut self, delta: Duration) -> u64 {
        self.clock.accumulate(delta);
        self.clock.expend_all()
    }

    /// Advance exactly one tick, ignoring the accumulator. For lock-step
    /// drivers stepped by an authoritative external tick cadence.
    #[inline]
    pub fn step_once(&mut self) {
        self.clock.step_once();
    }

    /// Reconcile to an authoritative tick number received from the server.
    ///
    /// This snaps the integer tick to `authoritative_tick` while preserving the
    /// sub-tick accumulator (so the interpolation alpha stays continuous).
    /// Returns the signed tick correction applied (`authoritative - previous`),
    /// which callers can use to decide whether a visible rollback/resim is
    /// needed. Tick numbering is authoritative, so this is a hard set — unlike
    /// the *wall-clock* offset, which is smoothed via [`ClockSync`].
    ///
    /// [`ClockSync`]: crate::ClockSync
    #[inline]
    pub fn reconcile_to(&mut self, authoritative_tick: u64) -> i64 {
        let previous = self.clock.tick();
        let correction = authoritative_tick as i64 - previous as i64;
        let snapshot = TickSnapshot {
            tick: authoritative_tick,
            accum: self.clock.overstep_subunits(),
            step_num: self.clock.step().nanos_num(),
            step_den: self.clock.step().nanos_den(),
        };
        self.clock.restore(&snapshot);
        correction
    }

    /// Reset to tick 0 with an empty accumulator, keeping the rate.
    #[inline]
    pub fn reset(&mut self) {
        self.clock.reset();
    }
}

impl Default for ServerTick {
    /// A 64 Hz authoritative clock, matching the crate's default fixed rate.
    #[inline]
    fn default() -> Self {
        Self::from_clock(TickClock::default())
    }
}

#[cfg(test)]
mod tests {
    use super::ServerTick;
    use crate::Duration;

    #[test]
    fn advance_counts_whole_ticks() {
        let mut s = ServerTick::from_hz(60);
        // 100 ms at 60 Hz = exactly 6 ticks (6 * 1/60 s = 0.1 s), no leftover.
        let ran = s.advance(Duration::from_millis(100));
        assert_eq!(ran, 6);
        assert_eq!(s.tick(), 6);
        assert_eq!(s.overstep_fraction_f64(), 0.0);
        // A further 10 ms leaves a partial tick (interpolation alpha > 0).
        s.advance(Duration::from_millis(10));
        assert!(s.overstep_fraction_f64() > 0.0);
    }

    #[test]
    fn tick_rate_is_drift_free_over_a_long_run() {
        let mut s = ServerTick::from_hz(60);
        // Feed exactly one second, 1 ms at a time, for an hour of sim.
        for _ in 0..(1000 * 60 * 60) {
            s.advance(Duration::from_millis(1));
        }
        // 3600 s * 60 Hz = 216_000 ticks, exactly.
        assert_eq!(s.tick(), 216_000);
    }

    #[test]
    fn time_of_tick_matches_elapsed() {
        let mut s = ServerTick::from_hz(50); // 20 ms exact
        for _ in 0..10 {
            s.advance(Duration::from_millis(20)); // one tick each (within cap)
        }
        assert_eq!(s.tick(), 10);
        assert_eq!(s.time_of_tick(10), Duration::from_millis(200));
        assert_eq!(s.tick_duration(), Duration::from_millis(20));
    }

    #[test]
    fn reconcile_snaps_tick_and_reports_correction() {
        let mut s = ServerTick::from_hz(60);
        s.advance(Duration::from_millis(100)); // tick 6
        let alpha_before = s.overstep_fraction_f64();
        // Server says we should actually be at tick 10.
        let correction = s.reconcile_to(10);
        assert_eq!(correction, 4);
        assert_eq!(s.tick(), 10);
        // Sub-tick alpha preserved across the reconcile.
        assert!((s.overstep_fraction_f64() - alpha_before).abs() < 1e-12);
        // Reconciling backwards reports a negative correction.
        assert_eq!(s.reconcile_to(8), -2);
        assert_eq!(s.tick(), 8);
    }

    #[test]
    fn determinism_double_run_is_bit_equivalent() {
        let feed = [17u64, 16, 16, 17, 33, 8, 16, 16];
        let run = || {
            let mut s = ServerTick::from_hz(60);
            let mut ticks = 0u64;
            for _ in 0..5000 {
                for &ms in &feed {
                    ticks += s.advance(Duration::from_millis(ms));
                }
            }
            (s, ticks)
        };
        let (a, ta) = run();
        let (b, tb) = run();
        assert_eq!(a, b);
        assert_eq!(ta, tb);
        assert_eq!(a.tick(), ta);
    }

    #[test]
    fn default_is_64hz() {
        let s = ServerTick::default();
        assert_eq!(s.tick_duration(), Duration::from_micros(15_625));
    }
}
