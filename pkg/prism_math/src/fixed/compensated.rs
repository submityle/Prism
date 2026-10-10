//! Compensated (error-free) summation for `f32`/`f64` slices (design doc
//! §24.4).
//!
//! Long accumulation chains (big-world coordinate accumulation, deterministic
//! reduction, massive transform chains, physics integration) let `f32`/`f64`
//! rounding error pile up into visible drift. [`KahanSum`] and [`NeumaierSum`]
//! carry a running compensation term so the sum stays close to the
//! infinitely-precise result.
//!
//! ## Relationship to the fixed-point path
//! These helpers are the **single-platform, low-drift** companion to the
//! [`Fixed`](super::Fixed) path's **cross-platform bit-exact** guarantee
//! (design doc §24.4 "与 §13 定点档并列"). Floating-point addition is *not*
//! associative, so a compensated sum is only reproducible for a fixed
//! evaluation order and on a platform with the same float rounding; it is a
//! precision aid for deterministic reduction, not a substitute for
//! [`Fixed`](super::Fixed) when cross-platform bit-exactness is required.
//!
//! [`NeumaierSum`] is the improved Kahan–Babuška variant that also handles the
//! case where the next term is larger in magnitude than the running total.

use core::ops::AddAssign;

mod sealed {
    /// Sealed marker for the float types the compensated sums support.
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Floating-point element usable with [`KahanSum`]/[`NeumaierSum`].
///
/// Sealed: implemented only for `f32` and `f64`.
pub trait CompensableFloat: sealed::Sealed + Copy {
    /// The additive identity (`0.0`).
    const ZERO: Self;
    /// `self + rhs`.
    fn add(self, rhs: Self) -> Self;
    /// `self - rhs`.
    fn sub(self, rhs: Self) -> Self;
    /// Absolute value.
    fn abs(self) -> Self;
    /// `self >= rhs`.
    fn ge(self, rhs: Self) -> bool;
}

impl CompensableFloat for f32 {
    const ZERO: Self = 0.0;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
    #[inline]
    fn abs(self) -> Self {
        libm::fabsf(self)
    }
    #[inline]
    fn ge(self, rhs: Self) -> bool {
        self >= rhs
    }
}

impl CompensableFloat for f64 {
    const ZERO: Self = 0.0;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
    #[inline]
    fn abs(self) -> Self {
        libm::fabs(self)
    }
    #[inline]
    fn ge(self, rhs: Self) -> bool {
        self >= rhs
    }
}

/// Kahan compensated summation accumulator.
///
/// Maintains a running `sum` and a `compensation` term capturing the low-order
/// bits lost in each addition.
#[derive(Clone, Copy, Debug)]
pub struct KahanSum<T: CompensableFloat> {
    sum: T,
    compensation: T,
}

impl<T: CompensableFloat> Default for KahanSum<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: CompensableFloat> KahanSum<T> {
    /// A fresh accumulator at zero.
    #[inline]
    pub const fn new() -> Self {
        Self {
            sum: T::ZERO,
            compensation: T::ZERO,
        }
    }
    /// Add one term.
    #[inline]
    pub fn add(&mut self, value: T) {
        let y = value.sub(self.compensation);
        let t = self.sum.add(y);
        // `(t - sum)` recovers the high part actually added; the difference
        // from `y` is the rounding error, carried into the next step.
        self.compensation = t.sub(self.sum).sub(y);
        self.sum = t;
    }
    /// The current compensated total.
    #[inline]
    pub fn sum(&self) -> T {
        self.sum
    }
}

impl<T: CompensableFloat> AddAssign<T> for KahanSum<T> {
    #[inline]
    fn add_assign(&mut self, value: T) {
        self.add(value);
    }
}

/// Neumaier (improved Kahan–Babuška) compensated summation accumulator.
///
/// More robust than plain Kahan when an incoming term is larger in magnitude
/// than the running total.
#[derive(Clone, Copy, Debug)]
pub struct NeumaierSum<T: CompensableFloat> {
    sum: T,
    compensation: T,
}

impl<T: CompensableFloat> Default for NeumaierSum<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: CompensableFloat> NeumaierSum<T> {
    /// A fresh accumulator at zero.
    #[inline]
    pub const fn new() -> Self {
        Self {
            sum: T::ZERO,
            compensation: T::ZERO,
        }
    }
    /// Add one term.
    #[inline]
    pub fn add(&mut self, value: T) {
        let t = self.sum.add(value);
        let correction = if self.sum.abs().ge(value.abs()) {
            // Running total is larger: low bits of `value` are lost.
            self.sum.sub(t).add(value)
        } else {
            // Incoming term is larger: low bits of `sum` are lost.
            value.sub(t).add(self.sum)
        };
        self.compensation = self.compensation.add(correction);
        self.sum = t;
    }
    /// The current compensated total (`sum + compensation`).
    #[inline]
    pub fn sum(&self) -> T {
        self.sum.add(self.compensation)
    }
}

impl<T: CompensableFloat> AddAssign<T> for NeumaierSum<T> {
    #[inline]
    fn add_assign(&mut self, value: T) {
        self.add(value);
    }
}

/// Kahan-compensated sum of a slice.
#[inline]
pub fn kahan_sum<T: CompensableFloat>(values: &[T]) -> T {
    let mut acc = KahanSum::new();
    for &v in values {
        acc.add(v);
    }
    acc.sum()
}

/// Neumaier-compensated sum of a slice.
#[inline]
pub fn neumaier_sum<T: CompensableFloat>(values: &[T]) -> T {
    let mut acc = NeumaierSum::new();
    for &v in values {
        acc.add(v);
    }
    acc.sum()
}
