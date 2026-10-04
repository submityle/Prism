//! [`MonotonicBaseline`]: wrap-safe accumulation of a fixed-width hardware
//! counter into a monotonic, drift-free elapsed measure.

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

/// Accumulates a bounded-width monotonic counter into a drift-free elapsed
/// measure, absorbing register wrap-around.
///
/// Platform monotonic counters are read as an integer that counts at a fixed
/// frequency (`ticks_per_sec`) in a register of a fixed width (`width_bits`).
/// When the register overflows its width it wraps back to zero. As long as the
/// counter is polled at least once per wrap period, this type reconstructs the
/// true forward progress: each [`update`](Self::update) adds the wrap-corrected
/// tick delta to an unbounded `u128` tick total.
///
/// The authoritative elapsed measure is the **integer tick count**
/// ([`elapsed_ticks`](Self::elapsed_ticks)); the [`Duration`] and `f64` views
/// are derived from it, so no `f32`/`f64` accumulation error builds up over a
/// long session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MonotonicBaseline {
    /// Counter frequency in ticks per second (`> 0`).
    ticks_per_sec: u64,
    /// Low-bit mask for the counter width (`2^width_bits - 1`). Raw readings
    /// are reduced modulo `width_bits` before use.
    mask: u64,
    /// Last raw reading (masked to the counter width), or `None` before the
    /// first update establishes the baseline.
    last_raw: Option<u64>,
    /// Wrap-corrected total ticks since the baseline reading.
    elapsed_ticks: u128,
}

impl MonotonicBaseline {
    /// Create a baseline for a counter running at `ticks_per_sec` in a
    /// `width_bits`-wide register.
    ///
    /// # Panics
    /// Panics if `ticks_per_sec` is zero or `width_bits` is not in `1..=64`.
    #[inline]
    #[must_use]
    pub const fn new(ticks_per_sec: u64, width_bits: u32) -> Self {
        assert!(ticks_per_sec != 0, "counter frequency must be non-zero");
        assert!(
            width_bits >= 1 && width_bits <= 64,
            "counter width must be in 1..=64 bits"
        );
        let mask = if width_bits == 64 {
            u64::MAX
        } else {
            (1u64 << width_bits) - 1
        };
        Self {
            ticks_per_sec,
            mask,
            last_raw: None,
            elapsed_ticks: 0,
        }
    }

    /// A full-width (`64`-bit) counter at `ticks_per_sec`.
    #[inline]
    #[must_use]
    pub const fn full_width(ticks_per_sec: u64) -> Self {
        Self::new(ticks_per_sec, 64)
    }

    /// Feed the latest raw counter reading. Returns the wrap-corrected elapsed
    /// time *this update* as a [`Duration`]. The first call establishes the
    /// baseline and returns [`Duration::ZERO`].
    ///
    /// Correctness assumes fewer than one full wrap elapsed since the previous
    /// reading (true whenever the counter is polled at least once per wrap
    /// period, which for any realistic frequency/width is far longer than a
    /// frame).
    pub fn update(&mut self, raw: u64) -> Duration {
        let masked = raw & self.mask;
        let delta_ticks: u128 = match self.last_raw {
            None => 0,
            Some(last) => u128::from(masked.wrapping_sub(last) & self.mask),
        };
        self.last_raw = Some(masked);
        self.elapsed_ticks = self.elapsed_ticks.saturating_add(delta_ticks);
        self.ticks_to_duration(delta_ticks)
    }

    /// Authoritative wrap-corrected elapsed tick count since the baseline.
    #[inline]
    #[must_use]
    pub const fn elapsed_ticks(&self) -> u128 {
        self.elapsed_ticks
    }

    /// Elapsed time as a [`Duration`], derived from the integer tick count by
    /// flooring to whole nanoseconds.
    #[inline]
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.ticks_to_duration(self.elapsed_ticks)
    }

    /// Elapsed seconds as `f64`, derived from the integer tick count. The
    /// authoritative measure is [`elapsed_ticks`](Self::elapsed_ticks); this is
    /// for display / ratio math.
    #[inline]
    #[must_use]
    pub fn elapsed_secs_f64(&self) -> f64 {
        self.elapsed_ticks as f64 / self.ticks_per_sec as f64
    }

    /// The configured counter frequency in ticks per second.
    #[inline]
    #[must_use]
    pub const fn ticks_per_sec(&self) -> u64 {
        self.ticks_per_sec
    }

    /// Whether a baseline reading has been taken yet.
    #[inline]
    #[must_use]
    pub const fn is_started(&self) -> bool {
        self.last_raw.is_some()
    }

    /// Drop the baseline and zero the elapsed total, keeping the frequency and
    /// width. The next [`update`](Self::update) re-establishes the baseline.
    #[inline]
    pub fn reset(&mut self) {
        self.last_raw = None;
        self.elapsed_ticks = 0;
    }

    /// Convert a tick count to a [`Duration`] (`ticks * 1e9 / freq`, floored).
    #[inline]
    #[must_use]
    fn ticks_to_duration(&self, ticks: u128) -> Duration {
        let nanos = ticks.saturating_mul(NANOS_PER_SEC) / u128::from(self.ticks_per_sec);
        duration_from_nanos_u128(nanos)
    }
}
