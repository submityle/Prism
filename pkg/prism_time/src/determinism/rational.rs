//! [`RationalStep`]: an exact rational timestep measured in nanoseconds.
//!
//! A fixed timestep such as `1/60 s` cannot be represented exactly as an
//! integer number of nanoseconds (`16_666_666.666… ns`), and representing it as
//! an `f32`/`f64` second count accumulates binary rounding error over a long
//! run. [`RationalStep`] stores the step as an exact fraction `num/den`
//! nanoseconds, so the deterministic clock can advance with pure integer
//! arithmetic and never drift.

use crate::Duration;

/// Nanoseconds in one second.
const NANOS_PER_SEC: u64 = 1_000_000_000;

/// Greatest common divisor (Euclid), used to keep the fraction reduced so the
/// integer accumulator in [`TickClock`](crate::TickClock) stays small.
const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// An exact rational timestep: `num/den` nanoseconds per tick.
///
/// The fraction is always stored in lowest terms. Construct it from a rate
/// ([`RationalStep::from_hz`]), from an exact nanosecond count
/// ([`RationalStep::from_nanos`]), or from an arbitrary fraction
/// ([`RationalStep::new`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RationalStep {
    /// Numerator of the step, in nanoseconds.
    num: u64,
    /// Denominator of the step.
    den: u64,
}

impl RationalStep {
    /// Create a step of `num/den` nanoseconds, reduced to lowest terms.
    ///
    /// # Panics
    /// Panics if `num` or `den` is zero (a zero step would never advance and a
    /// zero denominator is undefined).
    #[inline]
    pub const fn new(num: u64, den: u64) -> Self {
        assert!(num != 0, "rational step numerator must be non-zero");
        assert!(den != 0, "rational step denominator must be non-zero");
        let g = gcd(num, den);
        Self {
            num: num / g,
            den: den / g,
        }
    }

    /// Create a step of exactly `nanos` nanoseconds (`den == 1`).
    ///
    /// # Panics
    /// Panics if `nanos` is zero.
    #[inline]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self::new(nanos, 1)
    }

    /// Create a step from a tick rate in hertz: one tick is `1e9 / hz` ns,
    /// stored exactly. `60` yields `1/60 s`, `50_000_000/3 ns` reduced.
    ///
    /// # Panics
    /// Panics if `hz` is zero.
    #[inline]
    pub const fn from_hz(hz: u64) -> Self {
        assert!(hz != 0, "rational step rate must be non-zero");
        Self::new(NANOS_PER_SEC, hz)
    }

    /// The step numerator, in nanoseconds (reduced).
    #[inline]
    pub const fn nanos_num(self) -> u64 {
        self.num
    }

    /// The step denominator (reduced).
    #[inline]
    pub const fn nanos_den(self) -> u64 {
        self.den
    }

    /// `true` when the step is an exact whole number of nanoseconds.
    #[inline]
    pub const fn is_exact_nanos(self) -> bool {
        self.den == 1
    }

    /// The step length in seconds (`f64`). Lossy; the exact state lives in the
    /// integer `num`/`den` pair, this is for display and ratio math only.
    #[inline]
    pub fn as_secs_f64(self) -> f64 {
        self.num as f64 / (self.den as f64 * NANOS_PER_SEC as f64)
    }

    /// The step as a [`Duration`], flooring to whole nanoseconds.
    ///
    /// This is lossy whenever [`is_exact_nanos`](Self::is_exact_nanos) is
    /// `false` (e.g. `1/60 s` floors from `16_666_666.666… ns`). Use it to seed
    /// drift-tolerant consumers such as [`Timer`](crate::Timer); the
    /// deterministic clock itself never routes through this conversion.
    #[inline]
    pub fn as_duration(self) -> Duration {
        Duration::from_nanos(self.num / self.den)
    }
}

impl Default for RationalStep {
    /// `1/64 s`, matching [`Fixed`](crate::Fixed)'s default timestep.
    #[inline]
    fn default() -> Self {
        Self::from_hz(64)
    }
}

#[cfg(test)]
mod tests {
    use super::{RationalStep, NANOS_PER_SEC};

    #[test]
    fn from_hz_is_reduced_and_exact() {
        // 1/60 s = 1e9/60 ns = 50_000_000/3 ns after reducing by gcd 20.
        let s = RationalStep::from_hz(60);
        assert_eq!(s.nanos_num(), 50_000_000);
        assert_eq!(s.nanos_den(), 3);
        assert!(!s.is_exact_nanos());
        assert!((s.as_secs_f64() - 1.0 / 60.0).abs() < 1e-12);
    }

    #[test]
    fn from_hz_divisor_is_exact_nanos() {
        // 50 Hz divides a second evenly: 20_000_000 ns exactly.
        let s = RationalStep::from_hz(50);
        assert_eq!(s.nanos_num(), 20_000_000);
        assert_eq!(s.nanos_den(), 1);
        assert!(s.is_exact_nanos());
        assert_eq!(s.as_duration().as_nanos(), 20_000_000);
    }

    #[test]
    fn new_reduces_to_lowest_terms() {
        let s = RationalStep::new(1_000_000_000, 1000);
        // gcd(1e9, 1000) = 1000 => 1_000_000 / 1.
        assert_eq!(s.nanos_num(), 1_000_000);
        assert_eq!(s.nanos_den(), 1);
    }

    #[test]
    fn from_nanos_round_trips() {
        let s = RationalStep::from_nanos(16_666_667);
        assert_eq!(s.nanos_num(), 16_666_667);
        assert_eq!(s.nanos_den(), 1);
    }

    #[test]
    fn default_is_one_sixtyfourth() {
        let s = RationalStep::default();
        assert_eq!(s, RationalStep::from_hz(64));
        assert_eq!(s.nanos_num(), NANOS_PER_SEC / 64);
    }

    #[test]
    #[should_panic(expected = "non-zero")]
    fn zero_numerator_panics() {
        let _ = RationalStep::new(0, 1);
    }

    #[test]
    #[should_panic(expected = "non-zero")]
    fn zero_denominator_panics() {
        let _ = RationalStep::new(1, 0);
    }
}
