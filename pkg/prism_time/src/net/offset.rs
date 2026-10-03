//! Clock-offset estimation and smooth convergence.
//!
//! A client cannot read the server's clock directly; it *estimates* the offset
//! between them from round-trip timing, NTP-style, then applies it. Two hazards
//! must be handled, both called out by the design doc:
//!
//! 1. **Jitter / outliers.** Network queueing inflates round-trip time
//!    unpredictably, biasing naive offset samples. [`ClockOffsetEstimator`]
//!    filters a window of samples and trusts the lowest-RTT one (the least
//!    queued, à la NTP), which rejects transient spikes.
//! 2. **Snap / jump.** Applying a freshly estimated offset instantly would
//!    teleport remote entities. [`ClockSync`] converges the *applied* offset
//!    toward the target smoothly and at a bounded rate, so corrections are
//!    gradual and never overshoot.

/// One NTP-style offset measurement: the estimated clock offset and the
/// round-trip time it was derived from.
///
/// Offset is `server_clock - client_clock` in nanoseconds (positive when the
/// server clock is ahead). RTT is the measured round trip minus the server's
/// own processing interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OffsetSample {
    /// Estimated `server - client` offset, in nanoseconds (signed).
    pub offset_nanos: i64,
    /// Round-trip time this sample was measured over, in nanoseconds.
    pub rtt_nanos: u64,
}

impl OffsetSample {
    /// Build a sample directly from an offset and RTT.
    #[inline]
    #[must_use]
    pub const fn new(offset_nanos: i64, rtt_nanos: u64) -> Self {
        Self {
            offset_nanos,
            rtt_nanos,
        }
    }

    /// Derive a sample from the four NTP timestamps (all monotonic nanoseconds
    /// in their own clock's epoch):
    ///
    /// - `client_send` (`t0`): client transmit time (client clock).
    /// - `server_recv` (`t1`): server receive time (server clock).
    /// - `server_send` (`t2`): server transmit time (server clock).
    /// - `client_recv` (`t3`): client receive time (client clock).
    ///
    /// Offset `= ((t1 - t0) + (t2 - t3)) / 2`, RTT `= (t3 - t0) - (t2 - t1)`.
    /// The server-interval subtraction and total are saturated so malformed
    /// (non-monotonic) timestamps cannot overflow or produce a negative RTT.
    #[inline]
    #[must_use]
    pub fn from_round_trip(
        client_send: u64,
        server_recv: u64,
        server_send: u64,
        client_recv: u64,
    ) -> Self {
        let t0 = client_send as i128;
        let t1 = server_recv as i128;
        let t2 = server_send as i128;
        let t3 = client_recv as i128;
        let offset = ((t1 - t0) + (t2 - t3)) / 2;
        let total = client_recv.saturating_sub(client_send);
        let server_interval = server_send.saturating_sub(server_recv);
        let rtt = total.saturating_sub(server_interval);
        Self {
            offset_nanos: offset as i64,
            rtt_nanos: rtt,
        }
    }
}

/// Size of the sliding sample window used for outlier-resistant filtering.
const WINDOW: usize = 16;

/// A sliding-window filter over [`OffsetSample`]s that yields a jitter-robust
/// offset estimate.
///
/// It keeps the most recent [`WINDOW`] samples and reports the offset of the
/// lowest-RTT sample — the measurement least distorted by network queueing.
/// Feed the result into [`ClockSync::set_target`] to converge toward it
/// smoothly.
#[derive(Clone, Copy, Debug)]
pub struct ClockOffsetEstimator {
    samples: [OffsetSample; WINDOW],
    len: usize,
    /// Next write slot (ring buffer).
    head: usize,
}

impl ClockOffsetEstimator {
    /// Create an empty estimator.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            samples: [OffsetSample::new(0, 0); WINDOW],
            len: 0,
            head: 0,
        }
    }

    /// The window capacity (number of samples retained).
    pub const CAPACITY: usize = WINDOW;

    /// Add a sample, evicting the oldest once the window is full.
    #[inline]
    pub fn add(&mut self, sample: OffsetSample) {
        self.samples[self.head] = sample;
        self.head = (self.head + 1) % WINDOW;
        if self.len < WINDOW {
            self.len += 1;
        }
    }

    /// Convenience: build a sample from NTP timestamps and add it.
    #[inline]
    pub fn add_round_trip(
        &mut self,
        client_send: u64,
        server_recv: u64,
        server_send: u64,
        client_recv: u64,
    ) {
        self.add(OffsetSample::from_round_trip(
            client_send,
            server_recv,
            server_send,
            client_recv,
        ));
    }

    /// Number of samples currently retained.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no samples have been added.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Drop all samples.
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
        self.head = 0;
    }

    /// The lowest-RTT sample in the window (the filtered best estimate), or
    /// `None` if empty.
    #[inline]
    #[must_use]
    pub fn best(&self) -> Option<OffsetSample> {
        self.samples[..self.len]
            .iter()
            .copied()
            .min_by_key(|s| s.rtt_nanos)
    }

    /// The filtered offset estimate (offset of the lowest-RTT sample), or
    /// `None` if empty.
    #[inline]
    #[must_use]
    pub fn best_offset(&self) -> Option<i64> {
        self.best().map(|s| s.offset_nanos)
    }

    /// The smallest RTT seen in the window, or `None` if empty.
    #[inline]
    #[must_use]
    pub fn min_rtt(&self) -> Option<u64> {
        self.samples[..self.len].iter().map(|s| s.rtt_nanos).min()
    }
}

impl Default for ClockOffsetEstimator {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// A smooth, rate-limited corrector that eases the *applied* clock offset
/// toward a target estimate without snapping or overshooting.
///
/// Each [`advance`](Self::advance) moves the applied offset toward the target
/// at a rate proportional to the remaining error (exponential-style ease)
/// clamped to a maximum slew rate (nanoseconds of correction per real second),
/// then clamped so it never passes the target. This directly addresses the
/// design-doc hazard that offset jumps teleport remote entities.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClockSync {
    applied_nanos: f64,
    target_nanos: f64,
    /// Proportional convergence gain, per second.
    convergence_hz: f64,
    /// Maximum correction rate, in nanoseconds per real second.
    max_slew_ns_per_sec: f64,
}

impl ClockSync {
    /// Default proportional gain (`2.0`/s): closes ~86% of the error per second
    /// absent the slew clamp.
    pub const DEFAULT_CONVERGENCE_HZ: f64 = 2.0;
    /// Default slew clamp (`50 ms` of correction per second): fast enough to
    /// track real drift, slow enough that a correction is imperceptible.
    pub const DEFAULT_MAX_SLEW_NS_PER_SEC: f64 = 50_000_000.0;

    /// Create a corrector starting at offset `0` with default tuning.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            applied_nanos: 0.0,
            target_nanos: 0.0,
            convergence_hz: Self::DEFAULT_CONVERGENCE_HZ,
            max_slew_ns_per_sec: Self::DEFAULT_MAX_SLEW_NS_PER_SEC,
        }
    }

    /// Builder: set the proportional convergence gain (per second, clamped to
    /// `>= 0`).
    #[inline]
    #[must_use]
    pub fn with_convergence_hz(mut self, hz: f64) -> Self {
        if hz.is_finite() {
            self.convergence_hz = hz.max(0.0);
        }
        self
    }

    /// Builder: set the maximum slew rate (nanoseconds per second, clamped to
    /// `>= 0`).
    #[inline]
    #[must_use]
    pub fn with_max_slew(mut self, ns_per_sec: f64) -> Self {
        if ns_per_sec.is_finite() {
            self.max_slew_ns_per_sec = ns_per_sec.max(0.0);
        }
        self
    }

    /// Set the target offset to converge toward (typically
    /// [`ClockOffsetEstimator::best_offset`]).
    #[inline]
    pub fn set_target(&mut self, target_nanos: i64) {
        self.target_nanos = target_nanos as f64;
    }

    /// Immediately set both the applied and target offset (no ramp). Use once
    /// at session start to seed the initial estimate.
    #[inline]
    pub fn snap_to(&mut self, offset_nanos: i64) {
        self.applied_nanos = offset_nanos as f64;
        self.target_nanos = offset_nanos as f64;
    }

    /// The target offset.
    #[inline]
    #[must_use]
    pub fn target(&self) -> i64 {
        round_to_i64(self.target_nanos)
    }

    /// The currently applied offset in nanoseconds (rounded to the nearest ns).
    #[inline]
    #[must_use]
    pub fn offset(&self) -> i64 {
        round_to_i64(self.applied_nanos)
    }

    /// The currently applied offset in seconds.
    #[inline]
    #[must_use]
    pub fn offset_secs_f64(&self) -> f64 {
        self.applied_nanos / 1_000_000_000.0
    }

    /// Remaining error (`target - applied`) in nanoseconds.
    #[inline]
    #[must_use]
    pub fn error_nanos(&self) -> f64 {
        self.target_nanos - self.applied_nanos
    }

    /// Whether the applied offset is within `tolerance_nanos` of the target.
    #[inline]
    #[must_use]
    pub fn is_converged(&self, tolerance_nanos: f64) -> bool {
        self.error_nanos().abs() <= tolerance_nanos.max(0.0)
    }

    /// Advance the smooth correction by `delta` real seconds.
    ///
    /// The applied offset moves toward the target at `error * convergence_hz`,
    /// clamped to `±max_slew`, then clamped so it cannot overshoot. Returns the
    /// new applied offset in nanoseconds.
    #[inline]
    pub fn advance(&mut self, delta: core::time::Duration) -> i64 {
        let dt = delta.as_secs_f64();
        if dt > 0.0 {
            let error = self.target_nanos - self.applied_nanos;
            let desired_rate = error * self.convergence_hz;
            let rate = desired_rate.clamp(-self.max_slew_ns_per_sec, self.max_slew_ns_per_sec);
            let step = rate * dt;
            self.applied_nanos += step;
            // Clamp against overshoot in either direction.
            if (error > 0.0 && self.applied_nanos > self.target_nanos)
                || (error < 0.0 && self.applied_nanos < self.target_nanos)
            {
                self.applied_nanos = self.target_nanos;
            }
        }
        self.offset()
    }

    /// Apply the current offset to a client-clock reading, yielding the
    /// estimated server-clock reading in nanoseconds (saturating).
    #[inline]
    #[must_use]
    pub fn to_server_nanos(&self, client_nanos: u64) -> u64 {
        let v = client_nanos as i128 + round_to_i64(self.applied_nanos) as i128;
        v.clamp(0, u64::MAX as i128) as u64
    }
}

impl Default for ClockSync {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Round a finite `f64` nanosecond value to the nearest `i64`, saturating.
#[inline]
fn round_to_i64(v: f64) -> i64 {
    let r = v.round();
    if r >= i64::MAX as f64 {
        i64::MAX
    } else if r <= i64::MIN as f64 {
        i64::MIN
    } else {
        r as i64
    }
}

#[cfg(test)]
mod tests {
    use super::{ClockOffsetEstimator, ClockSync, OffsetSample};
    use crate::Duration;

    #[test]
    fn round_trip_recovers_known_offset() {
        // Server clock is +1000 ns ahead; symmetric 200 ns one-way latency.
        // client_send=0, server_recv=1000+200=1200, server_send=1300 (100 ns
        // processing), client_recv = (1300 - 1000) + 200 = 500.
        let s = OffsetSample::from_round_trip(0, 1200, 1300, 500);
        assert_eq!(s.offset_nanos, 1000);
        // RTT = total(500) - server_interval(100) = 400.
        assert_eq!(s.rtt_nanos, 400);
    }

    #[test]
    fn estimator_trusts_lowest_rtt_sample() {
        let mut e = ClockOffsetEstimator::new();
        assert!(e.is_empty());
        // Jittered samples: true offset 1000, spikes bias the offset upward
        // but carry large RTT, so the filter rejects them.
        e.add(OffsetSample::new(1500, 9000)); // spike
        e.add(OffsetSample::new(1000, 400)); // clean, low RTT
        e.add(OffsetSample::new(1300, 5000)); // spike
        e.add(OffsetSample::new(1020, 500)); // clean-ish
        assert_eq!(e.best_offset(), Some(1000));
        assert_eq!(e.min_rtt(), Some(400));
        assert_eq!(e.len(), 4);
    }

    #[test]
    fn estimator_window_evicts_oldest() {
        let mut e = ClockOffsetEstimator::new();
        // Fill beyond capacity; the earliest low-RTT sample should age out.
        e.add(OffsetSample::new(999, 1)); // best, but will be evicted
        for i in 0..ClockOffsetEstimator::CAPACITY {
            e.add(OffsetSample::new(2000 + i as i64, 100));
        }
        assert_eq!(e.len(), ClockOffsetEstimator::CAPACITY);
        // The RTT-1 sample is gone; best RTT is now 100.
        assert_eq!(e.min_rtt(), Some(100));
        assert_ne!(e.best_offset(), Some(999));
    }

    #[test]
    fn sync_converges_smoothly_without_snapping() {
        let mut sync = ClockSync::new();
        sync.set_target(30_000_000); // 30 ms target
        let mut last = 0i64;
        let mut max_step = 0i64;
        // 5 s at 60 fps.
        for _ in 0..300 {
            let now = sync.advance(Duration::from_micros(16_666));
            max_step = max_step.max((now - last).abs());
            last = now;
        }
        // Converged close to target after the smooth ramp.
        assert!(sync.is_converged(100_000.0), "err {}", sync.error_nanos());
        // No single frame jumped more than the slew cap allows
        // (50 ms/s * 16.666 ms ~= 833_300 ns), i.e. never a snap.
        assert!(max_step <= 900_000, "max per-frame step {max_step}");
    }

    #[test]
    fn sync_never_overshoots() {
        let mut sync = ClockSync::new().with_convergence_hz(1000.0); // very aggressive
        sync.set_target(-5_000_000);
        for _ in 0..1000 {
            sync.advance(Duration::from_millis(16));
        }
        assert_eq!(sync.offset(), -5_000_000);
        assert!(sync.is_converged(1.0));
    }

    #[test]
    fn slew_rate_bounds_large_corrections() {
        let mut sync = ClockSync::new().with_max_slew(10_000_000.0); // 10 ms/s
        sync.set_target(1_000_000_000); // 1 s off — huge
        let first = sync.advance(Duration::from_secs(1));
        // One second of correction is clamped to the 10 ms slew budget.
        assert_eq!(first, 10_000_000);
    }

    #[test]
    fn snap_to_seeds_without_ramp() {
        let mut sync = ClockSync::new();
        sync.snap_to(12_345);
        assert_eq!(sync.offset(), 12_345);
        assert_eq!(sync.target(), 12_345);
        assert!(sync.is_converged(0.0));
    }

    #[test]
    fn to_server_nanos_applies_offset() {
        let mut sync = ClockSync::new();
        sync.snap_to(1000);
        assert_eq!(sync.to_server_nanos(5000), 6000);
    }

    #[test]
    fn determinism_double_run_is_bit_equivalent() {
        let targets = [30_000_000i64, 10_000_000, -5_000_000, 42_000_000];
        let run = || {
            let mut sync = ClockSync::new();
            let mut trace = [0i64; 400];
            for (i, slot) in trace.iter_mut().enumerate() {
                sync.set_target(targets[(i / 100) % targets.len()]);
                *slot = sync.advance(Duration::from_micros(16_666));
            }
            trace
        };
        assert_eq!(run(), run());
    }
}
