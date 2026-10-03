//! Monotonic clock source.

/// A monotonic timestamp in nanoseconds since an unspecified epoch. Suitable
/// for measuring durations; not wall-clock time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MonotonicNanos(pub u64);

impl MonotonicNanos {
    /// Nanoseconds elapsed since `earlier` (saturating).
    pub fn saturating_since(self, earlier: MonotonicNanos) -> u64 {
        self.0.saturating_sub(earlier.0)
    }
}

/// Read the platform monotonic clock.
///
/// With `std` this wraps `std::time::Instant` relative to a process-start
/// anchor. Without `std` it returns zero (a real no_std backend lands with the
/// platform clock milestone).
#[cfg(feature = "std")]
pub fn now() -> MonotonicNanos {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    let start = *START.get_or_init(Instant::now);
    MonotonicNanos(start.elapsed().as_nanos() as u64)
}

/// Read the platform monotonic clock (no_std stub: always zero).
#[cfg(not(feature = "std"))]
pub fn now() -> MonotonicNanos {
    MonotonicNanos(0)
}
