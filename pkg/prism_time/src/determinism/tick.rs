//! [`TickClock`]: a deterministic integer-tick fixed-timestep clock, plus a
//! serializable [`TickSnapshot`] for replay/restore.
//!
//! Where [`Time<Fixed>`](crate::Time) accumulates [`Duration`] (whole
//! nanoseconds) and so drifts for steps like `1/60 s`, [`TickClock`] keeps time
//! as an **integer tick counter** advanced by an exact [`RationalStep`]. All
//! advancement is integer arithmetic in sub-units of `1/den` nanoseconds, so
//! the tick count is bit-identical across runs with no floating-point drift.
//!
//! The API mirrors [`Time<Fixed>`](crate::Time) so it drops into the same
//! "fixed timestep with accumulator" loop:
//!
//! ```
//! use prism_time::{Duration, RationalStep, TickClock};
//!
//! let mut clock = TickClock::from_hz(60);
//! clock.accumulate(Duration::from_millis(50));
//! while clock.expend() {
//!     // run one deterministic fixed step
//! }
//! let _alpha = clock.overstep_fraction(); // interpolation factor in [0, 1)
//! # let _ = RationalStep::from_hz(60);
//! ```

use crate::{Duration, RationalStep};

/// Nanoseconds in one second.
const NANOS_PER_SEC: u128 = 1_000_000_000;

/// A deterministic integer-tick fixed-timestep clock.
///
/// Time is `tick * step` where `step` is an exact [`RationalStep`]. The
/// accumulator holds leftover time in sub-units of `1/den` nanoseconds; one
/// tick costs `num` sub-units, so [`expend`](Self::expend) is an exact integer
/// comparison and subtraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TickClock {
    /// Exact rational timestep.
    step: RationalStep,
    /// Integer tick counter (the authoritative, drift-free elapsed measure).
    tick: u64,
    /// Leftover time in sub-units of `1/den` nanoseconds; always `< num` after
    /// draining with [`expend`](Self::expend).
    accum: u128,
    /// Cap on accumulated ticks (death-spiral guard), mirroring
    /// [`Fixed`](crate::Fixed).
    max_substeps: u32,
}

impl TickClock {
    /// Default cap on accumulated ticks per accumulate (death-spiral guard),
    /// matching [`Fixed::DEFAULT_MAX_SUBSTEPS`](crate::Fixed::DEFAULT_MAX_SUBSTEPS).
    pub const DEFAULT_MAX_SUBSTEPS: u32 = 8;

    /// Create a clock with the given exact [`RationalStep`], starting at tick 0.
    #[inline]
    pub const fn new(step: RationalStep) -> Self {
        Self {
            step,
            tick: 0,
            accum: 0,
            max_substeps: Self::DEFAULT_MAX_SUBSTEPS,
        }
    }

    /// Create a clock from a tick rate in hertz (e.g. `60`), stored exactly.
    ///
    /// # Panics
    /// Panics if `hz` is zero.
    #[inline]
    pub const fn from_hz(hz: u64) -> Self {
        Self::new(RationalStep::from_hz(hz))
    }

    /// Builder: set the accumulated-tick cap. A value of `0` drops all
    /// accumulated time on [`accumulate`](Self::accumulate).
    #[inline]
    pub const fn with_max_substeps(mut self, max_substeps: u32) -> Self {
        self.max_substeps = max_substeps;
        self
    }

    /// The exact rational timestep.
    #[inline]
    pub const fn step(self) -> RationalStep {
        self.step
    }

    /// The current integer tick count (the authoritative elapsed measure).
    #[inline]
    pub const fn tick(self) -> u64 {
        self.tick
    }

    /// The accumulated-tick cap (death-spiral guard).
    #[inline]
    pub const fn max_substeps(self) -> u32 {
        self.max_substeps
    }

    /// Set the accumulated-tick cap. A value of `0` drops accumulated time.
    #[inline]
    pub fn set_max_substeps(&mut self, max_substeps: u32) {
        self.max_substeps = max_substeps;
    }

    /// Feed a delta into the accumulator using exact integer arithmetic.
    ///
    /// The delta's whole nanoseconds are converted into `1/den` sub-units
    /// (`nanos * den`) and added. The accumulator is then capped at
    /// `max_substeps` ticks so a frame hitch cannot schedule an unbounded run of
    /// fixed steps.
    #[inline]
    pub fn accumulate(&mut self, delta: Duration) {
        let den = self.step.nanos_den() as u128;
        let sub = delta.as_nanos().saturating_mul(den);
        self.accum = self.accum.saturating_add(sub);
        let cap = (self.step.nanos_num() as u128).saturating_mul(self.max_substeps as u128);
        if self.accum > cap {
            self.accum = cap;
        }
    }

    /// Consume one tick if the accumulator holds at least one timestep.
    ///
    /// Returns `true` and increments the tick counter when a step was
    /// available, otherwise `false`. Drive the schedule with
    /// `while clock.expend() { .. }`.
    #[inline]
    pub fn expend(&mut self) -> bool {
        let num = self.step.nanos_num() as u128;
        if self.accum >= num {
            self.accum -= num;
            self.tick += 1;
            true
        } else {
            false
        }
    }

    /// Drain every available tick, returning how many ran. Bounded by
    /// `max_substeps` because [`accumulate`](Self::accumulate) caps the
    /// accumulator.
    #[inline]
    pub fn expend_all(&mut self) -> u64 {
        let mut steps = 0;
        while self.expend() {
            steps += 1;
        }
        steps
    }

    /// Advance the tick counter by one unconditionally, ignoring the
    /// accumulator. Useful for lock-step/replay drivers that step by an
    /// authoritative externally supplied tick count.
    #[inline]
    pub fn step_once(&mut self) {
        self.tick += 1;
    }

    /// Leftover accumulated time in sub-units of `1/den` nanoseconds (`< num`
    /// after draining). This is the exact overstep; see
    /// [`overstep`](Self::overstep) for a (lossy) [`Duration`] view.
    #[inline]
    pub const fn overstep_subunits(self) -> u128 {
        self.accum
    }

    /// Leftover accumulated time as a [`Duration`], flooring to whole
    /// nanoseconds (`accum / den`). Lossy when the step is not exact
    /// nanoseconds; the exact value is [`overstep_subunits`](Self::overstep_subunits).
    #[inline]
    pub fn overstep(self) -> Duration {
        let den = self.step.nanos_den() as u128;
        let nanos = self.accum / den;
        duration_from_nanos_u128(nanos)
    }

    /// Overstep as a fraction of one timestep in `[0, 1)` after draining: the
    /// interpolation alpha for the presentation layer (`f32`).
    #[inline]
    pub fn overstep_fraction(self) -> f32 {
        self.overstep_fraction_f64() as f32
    }

    /// Overstep fraction as `f64`.
    #[inline]
    pub fn overstep_fraction_f64(self) -> f64 {
        self.accum as f64 / self.step.nanos_num() as f64
    }

    /// Elapsed time as a [`Duration`] (`tick * step`), flooring to whole
    /// nanoseconds. The authoritative drift-free measure is
    /// [`tick`](Self::tick); this derived view is for display/consumers that
    /// want wall-clock-shaped output.
    #[inline]
    pub fn elapsed(self) -> Duration {
        let nanos = (self.tick as u128).saturating_mul(self.step.nanos_num() as u128)
            / (self.step.nanos_den() as u128);
        duration_from_nanos_u128(nanos)
    }

    /// Elapsed seconds (`f64`), computed from the integer tick count.
    #[inline]
    pub fn elapsed_secs_f64(self) -> f64 {
        self.tick as f64 * self.step.as_secs_f64()
    }

    /// Reset to tick 0 with an empty accumulator, keeping the step and cap.
    #[inline]
    pub fn reset(&mut self) {
        self.tick = 0;
        self.accum = 0;
    }

    /// Capture the full deterministic state (tick, accumulator, and step) into
    /// a serializable [`TickSnapshot`] for replay/restore.
    #[inline]
    pub const fn snapshot(self) -> TickSnapshot {
        TickSnapshot {
            tick: self.tick,
            accum: self.accum,
            step_num: self.step.nanos_num(),
            step_den: self.step.nanos_den(),
        }
    }

    /// Restore the full deterministic state from a [`TickSnapshot`]. After
    /// restore, feeding the identical subsequent deltas reproduces an identical
    /// tick sequence (replay determinism). The `max_substeps` cap is retained
    /// from the live clock (it is a policy knob, not part of the time state).
    #[inline]
    pub fn restore(&mut self, snapshot: &TickSnapshot) {
        self.step = RationalStep::new(snapshot.step_num, snapshot.step_den);
        self.tick = snapshot.tick;
        self.accum = snapshot.accum;
    }

    /// Build a clock directly from a snapshot (with the default cap).
    #[inline]
    pub fn from_snapshot(snapshot: &TickSnapshot) -> Self {
        let mut clock = Self::new(RationalStep::new(snapshot.step_num, snapshot.step_den));
        clock.tick = snapshot.tick;
        clock.accum = snapshot.accum;
        clock
    }
}

impl Default for TickClock {
    /// A `1/64 s` clock, matching [`Fixed`](crate::Fixed)'s default.
    #[inline]
    fn default() -> Self {
        Self::new(RationalStep::default())
    }
}

/// Convert a `u128` nanosecond count to a [`Duration`], saturating the seconds
/// field rather than overflowing.
#[inline]
fn duration_from_nanos_u128(nanos: u128) -> Duration {
    let secs = (nanos / NANOS_PER_SEC).min(u64::MAX as u128) as u64;
    let sub = (nanos % NANOS_PER_SEC) as u32;
    Duration::new(secs, sub)
}

/// A serializable snapshot of a [`TickClock`]'s deterministic state.
///
/// It is plain integer data (`Copy`, with a fixed-size little-endian byte
/// encoding via [`to_bytes`](Self::to_bytes)/[`from_bytes`](Self::from_bytes)),
/// so it can be persisted or sent over the wire with any serializer and later
/// fed back to [`TickClock::restore`] to reproduce identical stepping.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TickSnapshot {
    /// Integer tick counter at snapshot time.
    pub tick: u64,
    /// Accumulator in sub-units of `1/step_den` nanoseconds.
    pub accum: u128,
    /// Step numerator (nanoseconds), reduced.
    pub step_num: u64,
    /// Step denominator, reduced.
    pub step_den: u64,
}

/// Byte length of the little-endian [`TickSnapshot`] encoding.
const SNAPSHOT_BYTES: usize = 8 + 16 + 8 + 8;

impl TickSnapshot {
    /// Encode as a fixed-size little-endian byte array (`tick`, `accum`,
    /// `step_num`, `step_den`).
    #[inline]
    pub fn to_bytes(self) -> [u8; SNAPSHOT_BYTES] {
        let mut out = [0u8; SNAPSHOT_BYTES];
        out[0..8].copy_from_slice(&self.tick.to_le_bytes());
        out[8..24].copy_from_slice(&self.accum.to_le_bytes());
        out[24..32].copy_from_slice(&self.step_num.to_le_bytes());
        out[32..40].copy_from_slice(&self.step_den.to_le_bytes());
        out
    }

    /// Decode from the little-endian byte array produced by
    /// [`to_bytes`](Self::to_bytes).
    #[inline]
    pub fn from_bytes(bytes: [u8; SNAPSHOT_BYTES]) -> Self {
        let mut tick = [0u8; 8];
        tick.copy_from_slice(&bytes[0..8]);
        let mut accum = [0u8; 16];
        accum.copy_from_slice(&bytes[8..24]);
        let mut step_num = [0u8; 8];
        step_num.copy_from_slice(&bytes[24..32]);
        let mut step_den = [0u8; 8];
        step_den.copy_from_slice(&bytes[32..40]);
        Self {
            tick: u64::from_le_bytes(tick),
            accum: u128::from_le_bytes(accum),
            step_num: u64::from_le_bytes(step_num),
            step_den: u64::from_le_bytes(step_den),
        }
    }

    /// Length in bytes of the [`to_bytes`](Self::to_bytes) encoding.
    pub const BYTES: usize = SNAPSHOT_BYTES;
}

#[cfg(test)]
mod tests {
    use super::{TickClock, TickSnapshot};
    use crate::{Duration, RationalStep};

    #[test]
    fn exact_nanos_step_has_no_drift_over_millions_of_ticks() {
        // 50 Hz => 20 ms exact. Feed 20 ms per frame for 2,000,000 frames.
        let mut clock = TickClock::from_hz(50);
        let frame = Duration::from_millis(20);
        let frames: u64 = 2_000_000;
        let mut total_steps = 0u64;
        for _ in 0..frames {
            clock.accumulate(frame);
            total_steps += clock.expend_all();
        }
        assert_eq!(total_steps, frames);
        assert_eq!(clock.tick(), frames);
        assert_eq!(clock.overstep_subunits(), 0);
        // elapsed = 2e6 * 20 ms = 40_000 s, exactly.
        assert_eq!(clock.elapsed(), Duration::from_secs(40_000));
    }

    #[test]
    fn rational_step_is_exact() {
        // 1/60 s = 50_000_000/3 ns. 0.05 s is exactly 3 ticks with no leftover.
        let mut clock = TickClock::from_hz(60);
        assert_eq!(clock.step(), RationalStep::from_hz(60));
        clock.accumulate(Duration::from_millis(50));
        assert_eq!(clock.expend_all(), 3);
        assert_eq!(clock.tick(), 3);
        assert_eq!(clock.overstep_subunits(), 0);
    }

    #[test]
    fn rational_step_accumulates_exact_leftover() {
        // One 1/60 s step consumed from a 20 ms frame; leftover is exact.
        let mut clock = TickClock::from_hz(60);
        clock.accumulate(Duration::from_millis(20)); // 20_000_000 ns
        // sub-units = 20_000_000 * 3 = 60_000_000; one tick costs 50_000_000.
        assert!(clock.expend());
        assert!(!clock.expend());
        assert_eq!(clock.tick(), 1);
        // leftover 10_000_000 sub-units = 10/3 ms as a fraction of the step.
        assert_eq!(clock.overstep_subunits(), 10_000_000);
        // alpha = 10_000_000 / 50_000_000 = 0.2.
        assert!((clock.overstep_fraction_f64() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn no_drift_across_awkward_frame_pattern() {
        // Feed a jittery 16/17 ms pattern at 60 Hz for a long run; the integer
        // tick count must equal floor(total_ns / step) exactly, with the
        // leftover accounted for to the sub-unit (zero drift).
        let mut clock = TickClock::from_hz(60);
        let pattern = [
            Duration::from_millis(16),
            Duration::from_millis(17),
            Duration::from_millis(16),
            Duration::from_micros(16_666),
        ];
        let mut total_nanos: u128 = 0;
        let mut steps = 0u64;
        for i in 0..400_000u64 {
            let d = pattern[(i % 4) as usize];
            total_nanos += d.as_nanos();
            clock.accumulate(d);
            steps += clock.expend_all();
        }
        assert_eq!(steps, clock.tick());
        // Reconstruct exact leftover: total sub-units - ticks * num.
        let num = clock.step().nanos_num() as u128;
        let den = clock.step().nanos_den() as u128;
        let expected_ticks = (total_nanos * den) / num;
        let expected_leftover = (total_nanos * den) % num;
        assert_eq!(clock.tick() as u128, expected_ticks);
        assert_eq!(clock.overstep_subunits(), expected_leftover);
    }

    #[test]
    fn alpha_is_monotonic_within_a_step() {
        let mut clock = TickClock::from_hz(100); // 10 ms exact
        let mut last = -1.0f64;
        // Feed 1 ms at a time; alpha climbs 0 -> ~0.9 then resets after a tick.
        for _ in 0..9 {
            clock.accumulate(Duration::from_millis(1));
            clock.expend_all();
            let a = clock.overstep_fraction_f64();
            assert!(a > last, "alpha should increase: {a} !> {last}");
            last = a;
        }
        assert!((clock.overstep_fraction_f64() - 0.9).abs() < 1e-12);
        // The tenth millisecond completes a step and resets the fraction.
        clock.accumulate(Duration::from_millis(1));
        assert_eq!(clock.expend_all(), 1);
        assert_eq!(clock.overstep_fraction_f64(), 0.0);
    }

    #[test]
    fn snapshot_restore_reproduces_identical_stepping() {
        // Run to an arbitrary mid-step state, then snapshot.
        let mut clock = TickClock::from_hz(60);
        let warmup = [
            Duration::from_millis(7),
            Duration::from_millis(13),
            Duration::from_micros(4_321),
            Duration::from_millis(21),
        ];
        for d in warmup {
            clock.accumulate(d);
            clock.expend_all();
        }
        let snap = clock.snapshot();

        // Replay feed applied to a copy taken at snapshot time ("run A").
        let replay_feed = [
            Duration::from_millis(16),
            Duration::from_micros(16_666),
            Duration::from_millis(50),
            Duration::from_micros(123),
        ];
        let mut run_a = TickClock::from_snapshot(&snap);
        let mut ticks_a = [0u64; 4];
        for (slot, &d) in ticks_a.iter_mut().zip(replay_feed.iter()) {
            run_a.accumulate(d);
            *slot = run_a.expend_all();
        }

        // Mutate the live clock to prove restore wipes the divergence.
        clock.accumulate(Duration::from_secs(1));
        clock.expend_all();
        assert_ne!(clock.tick(), snap.tick);

        // Restore and replay the identical feed; must match run A bit-for-bit.
        clock.restore(&snap);
        assert_eq!(clock, TickClock::from_snapshot(&snap));
        let mut ticks_b = [0u64; 4];
        for (slot, &d) in ticks_b.iter_mut().zip(replay_feed.iter()) {
            clock.accumulate(d);
            *slot = clock.expend_all();
        }
        assert_eq!(ticks_a, ticks_b);
        assert_eq!(clock.tick(), run_a.tick());
        assert_eq!(clock.overstep_subunits(), run_a.overstep_subunits());
    }

    #[test]
    fn snapshot_bytes_round_trip() {
        let mut clock = TickClock::from_hz(60);
        clock.accumulate(Duration::from_micros(9_999));
        clock.expend_all();
        let snap = clock.snapshot();
        let bytes = snap.to_bytes();
        assert_eq!(bytes.len(), TickSnapshot::BYTES);
        let decoded = TickSnapshot::from_bytes(bytes);
        assert_eq!(decoded, snap);

        let mut restored = TickClock::default();
        restored.restore(&decoded);
        assert_eq!(restored, clock);
    }

    #[test]
    fn step_once_advances_without_accumulator() {
        let mut clock = TickClock::from_hz(60);
        clock.step_once();
        clock.step_once();
        assert_eq!(clock.tick(), 2);
        assert_eq!(clock.overstep_subunits(), 0);
    }

    #[test]
    fn max_substeps_caps_a_hitch() {
        let mut clock = TickClock::from_hz(100).with_max_substeps(4); // 10 ms step
        clock.accumulate(Duration::from_secs(10)); // 1000 steps worth
        assert_eq!(clock.expend_all(), 4);
        assert_eq!(clock.tick(), 4);
    }

    #[test]
    fn reset_zeroes_state_but_keeps_step() {
        let mut clock = TickClock::from_hz(60);
        clock.accumulate(Duration::from_millis(100));
        clock.expend_all();
        clock.reset();
        assert_eq!(clock.tick(), 0);
        assert_eq!(clock.overstep_subunits(), 0);
        assert_eq!(clock.step(), RationalStep::from_hz(60));
    }
}
