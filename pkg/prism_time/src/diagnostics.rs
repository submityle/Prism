//! Frame-timing diagnostics: [`FrameBudget`] and [`FrameStats`].
//!
//! [`FrameBudget`] holds a target frame time (e.g. `1/60 s`) and answers
//! whether a frame fit, by how much it overran, and the headroom or
//! utilisation. [`FrameStats`] is a fixed-capacity ring of recent frame times
//! exposing min/max/mean and jitter, so a profiler overlay can be driven
//! without heap allocation.

use crate::Duration;

/// A per-frame time budget derived from a target frame rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameBudget {
    target: Duration,
}

impl FrameBudget {
    /// Create a budget from an explicit target frame time.
    #[inline]
    pub fn new(target: Duration) -> Self {
        Self { target }
    }

    /// Create a budget from a target rate in hertz (frames per second).
    ///
    /// # Panics
    /// Panics if `hz` is not strictly positive and finite.
    #[inline]
    pub fn from_hz(hz: f64) -> Self {
        assert!(hz > 0.0 && hz.is_finite(), "frame rate must be positive");
        Self::new(Duration::from_secs_f64(1.0 / hz))
    }

    /// The target frame time.
    #[inline]
    pub fn target(&self) -> Duration {
        self.target
    }

    /// Set the target frame time.
    #[inline]
    pub fn set_target(&mut self, target: Duration) {
        self.target = target;
    }

    /// Whether `frame` exceeded the budget.
    #[inline]
    pub fn is_over_budget(&self, frame: Duration) -> bool {
        frame > self.target
    }

    /// How far `frame` overran the budget, or `None` if it fit.
    #[inline]
    pub fn overrun(&self, frame: Duration) -> Option<Duration> {
        frame.checked_sub(self.target).filter(|d| !d.is_zero())
    }

    /// Remaining headroom within the budget, or `None` if it overran.
    #[inline]
    pub fn headroom(&self, frame: Duration) -> Option<Duration> {
        self.target.checked_sub(frame)
    }

    /// Fraction of the budget consumed (`1.0` = exactly on budget, `>1.0` over).
    /// A zero target reports `0.0` (no meaningful budget).
    #[inline]
    pub fn utilization(&self, frame: Duration) -> f64 {
        if self.target.is_zero() {
            0.0
        } else {
            frame.as_secs_f64() / self.target.as_secs_f64()
        }
    }
}

/// A fixed-capacity ring buffer of recent frame times.
///
/// Holds up to `N` samples. Statistics are computed over the samples currently
/// buffered; the buffer allocates nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameStats<const N: usize> {
    samples: [Duration; N],
    head: usize,
    len: usize,
}

impl<const N: usize> FrameStats<N> {
    /// Create an empty stats ring.
    #[inline]
    pub fn new() -> Self {
        Self {
            samples: [Duration::ZERO; N],
            head: 0,
            len: 0,
        }
    }

    /// Record one frame time, evicting the oldest sample once full.
    #[inline]
    pub fn record(&mut self, frame: Duration) {
        if N == 0 {
            return;
        }
        self.samples[self.head] = frame;
        self.head = (self.head + 1) % N;
        if self.len < N {
            self.len += 1;
        }
    }

    /// Number of buffered samples.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no samples are buffered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the buffer is at capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len == N
    }

    /// The most recently recorded frame time, if any.
    #[inline]
    pub fn latest(&self) -> Option<Duration> {
        if self.len == 0 {
            None
        } else {
            // `head` points at the next write slot; the latest is one before it.
            let idx = (self.head + N - 1) % N;
            Some(self.samples[idx])
        }
    }

    /// The smallest buffered frame time, if any.
    #[inline]
    pub fn min(&self) -> Option<Duration> {
        self.iter().min()
    }

    /// The largest buffered frame time, if any.
    #[inline]
    pub fn max(&self) -> Option<Duration> {
        self.iter().max()
    }

    /// The arithmetic mean of the buffered frame times, if any. Computed in
    /// `u128` nanoseconds to avoid `f32`/`f64` drift over long windows.
    #[inline]
    pub fn mean(&self) -> Option<Duration> {
        if self.len == 0 {
            return None;
        }
        let sum: u128 = self.iter().map(|d| d.as_nanos()).sum();
        Some(duration_from_nanos_u128(sum / self.len as u128))
    }

    /// Peak-to-peak jitter (`max - min`) over the buffered frames, if any.
    #[inline]
    pub fn jitter(&self) -> Option<Duration> {
        match (self.min(), self.max()) {
            (Some(lo), Some(hi)) => Some(hi.saturating_sub(lo)),
            _ => None,
        }
    }

    /// Population standard deviation of the buffered frame times, if any.
    ///
    /// Variance is accumulated in `u128` nanosecond-squared space and the
    /// square root is taken with an exact integer method, so the result is
    /// deterministic and needs no floating-point `sqrt` (keeping the crate pure
    /// `no_std`). The returned [`Duration`] is the standard deviation in
    /// nanoseconds.
    #[inline]
    pub fn stddev(&self) -> Option<Duration> {
        if self.len == 0 {
            return None;
        }
        let n = self.len as u128;
        let sum: u128 = self.iter().map(|d| d.as_nanos()).sum();
        let mean = sum / n;
        let variance: u128 = self
            .iter()
            .map(|d| {
                let x = d.as_nanos();
                let diff = x.abs_diff(mean);
                diff * diff
            })
            .sum::<u128>()
            / n;
        Some(duration_from_nanos_u128(isqrt_u128(variance)))
    }

    /// Iterate buffered samples from oldest to newest.
    #[inline]
    fn iter(&self) -> impl Iterator<Item = Duration> + '_ {
        let start = if self.len == N { self.head } else { 0 };
        (0..self.len).map(move |i| self.samples[(start + i) % N])
    }
}

impl<const N: usize> Default for FrameStats<N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Integer square root of a `u128` via a bit-by-bit (digit-by-digit) method.
/// Returns `floor(sqrt(value))` exactly, with no floating-point.
#[inline]
fn isqrt_u128(value: u128) -> u128 {
    if value < 2 {
        return value;
    }
    let mut result: u128 = 0;
    // Highest power-of-four `bit` not exceeding `value`.
    let mut bit: u128 = 1 << ((127 - value.leading_zeros()) & !1);
    let mut remainder = value;
    while bit != 0 {
        if remainder >= result + bit {
            remainder -= result + bit;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }
    result
}

/// Convert a `u128` nanosecond count to a [`Duration`], saturating the seconds
/// field.
#[inline]
fn duration_from_nanos_u128(nanos: u128) -> Duration {
    let secs = (nanos / 1_000_000_000).min(u64::MAX as u128) as u64;
    let sub = (nanos % 1_000_000_000) as u32;
    Duration::new(secs, sub)
}
