//! Conservative interval arithmetic with outward rounding.
//!
//! An [`Interval`] stores an inclusive lower and upper `f32` bound and
//! guarantees that the mathematically exact result of every operation lies
//! inside the returned interval. This is the "never miss" counterpart to the
//! exact predicates in [`crate::geom`]: exact predicates give an absolutely
//! correct sign, interval arithmetic gives an absolutely correct enclosure.
//!
//! ## Why this exists
//! Conservative culling and continuous collision detection (`CCD`) must never
//! wrongly discard something: a conservative frustum/occlusion test may keep an
//! object that is actually hidden, but must never cull one that is actually
//! visible, and a `CCD` sweep must never step through a thin wall. Running the
//! bounding computation in interval arithmetic turns rounding error from a
//! silent correctness hazard into a tracked, enclosed quantity.
//!
//! ## Rounding model
//! Rust/`no_std` cannot portably switch the hardware floating-point rounding
//! mode, so each operation rounds to nearest (the IEEE 754 default, within half
//! a unit in the last place, `ULP`) and then inflates the result outward by one
//! `ULP` via [`next_down`]/[`next_up`]. Because the rounding error is at most
//! half a `ULP` and the inflation is a full `ULP`, the exact value is always
//! enclosed. The inflation is deterministic and bit-exact across platforms.

use crate::float::f32 as mf;
use crate::vec::Vec3;

/// Smallest representable `f32` strictly greater than `x`.
///
/// `NaN` and `+inf` are returned unchanged. Both signed zeros step to the
/// smallest positive subnormal.
#[inline]
#[must_use]
pub const fn next_up(x: f32) -> f32 {
    if x.is_nan() || x == f32::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f32::from_bits(1);
    }
    let bits = x.to_bits();
    // Positive values increase magnitude; negative values decrease it.
    let stepped = if bits >> 31 == 0 { bits + 1 } else { bits - 1 };
    f32::from_bits(stepped)
}

/// Smallest representable `f32` strictly less than `x`.
///
/// `NaN` and `-inf` are returned unchanged. Both signed zeros step to the
/// smallest negative subnormal.
#[inline]
#[must_use]
pub const fn next_down(x: f32) -> f32 {
    if x.is_nan() || x == f32::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return f32::from_bits(0x8000_0001);
    }
    let bits = x.to_bits();
    let stepped = if bits >> 31 == 0 { bits - 1 } else { bits + 1 };
    f32::from_bits(stepped)
}

#[inline]
fn min4(a: f32, b: f32, c: f32, d: f32) -> f32 {
    a.min(b).min(c).min(d)
}

#[inline]
fn max4(a: f32, b: f32, c: f32, d: f32) -> f32 {
    a.max(b).max(c).max(d)
}

/// A closed, conservatively-rounded `f32` interval `[lo, hi]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    lo: f32,
    hi: f32,
}

impl Interval {
    /// The whole real line, `[-inf, +inf]`. Any value is contained.
    pub const UNBOUNDED: Self = Self {
        lo: f32::NEG_INFINITY,
        hi: f32::INFINITY,
    };

    /// Build an interval from two bounds, ordering them so `lo <= hi`.
    #[inline]
    #[must_use]
    pub fn new(a: f32, b: f32) -> Self {
        Self {
            lo: a.min(b),
            hi: a.max(b),
        }
    }

    /// A degenerate interval containing the single value `v`.
    #[inline]
    #[must_use]
    pub const fn point(v: f32) -> Self {
        Self { lo: v, hi: v }
    }

    /// Lower bound.
    #[inline]
    #[must_use]
    pub const fn lo(self) -> f32 {
        self.lo
    }

    /// Upper bound.
    #[inline]
    #[must_use]
    pub const fn hi(self) -> f32 {
        self.hi
    }

    /// Width `hi - lo`, rounded outward.
    #[inline]
    #[must_use]
    pub fn width(self) -> f32 {
        next_up(self.hi - self.lo)
    }

    /// Approximate midpoint. The exact midpoint is enclosed by `self`.
    #[inline]
    #[must_use]
    pub fn midpoint(self) -> f32 {
        self.lo + (self.hi - self.lo) * 0.5
    }

    /// `true` when `v` lies within the closed interval.
    #[inline]
    #[must_use]
    pub fn contains(self, v: f32) -> bool {
        self.lo <= v && v <= self.hi
    }

    /// `true` when every value of `other` is also in `self`.
    #[inline]
    #[must_use]
    pub fn contains_interval(self, other: Self) -> bool {
        self.lo <= other.lo && other.hi <= self.hi
    }

    /// `true` when the two intervals share at least one value.
    #[inline]
    #[must_use]
    pub fn overlaps(self, other: Self) -> bool {
        self.lo <= other.hi && other.lo <= self.hi
    }

    /// Smallest interval containing both operands (interval union/hull).
    #[inline]
    #[must_use]
    pub fn hull(self, other: Self) -> Self {
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    /// Intersection, or `None` when the intervals are disjoint.
    #[inline]
    #[must_use]
    pub fn intersect(self, other: Self) -> Option<Self> {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo <= hi {
            Some(Self { lo, hi })
        } else {
            None
        }
    }

    /// Conservative absolute value: the enclosure of `|x|` for all `x` in
    /// `self`.
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        if self.lo >= 0.0 {
            self
        } else if self.hi <= 0.0 {
            Self {
                lo: -self.hi,
                hi: -self.lo,
            }
        } else {
            // Straddles zero: lower bound is 0, upper is the larger magnitude.
            Self {
                lo: 0.0,
                hi: mf::abs(self.lo).max(mf::abs(self.hi)),
            }
        }
    }

    /// Conservative square root. Negative parts of the interval are clamped to
    /// zero (the real square root is undefined there).
    #[inline]
    #[must_use]
    pub fn sqrt(self) -> Self {
        let lo = mf::sqrt(self.lo.max(0.0));
        let hi = mf::sqrt(self.hi.max(0.0));
        Self {
            lo: next_down(lo),
            hi: next_up(hi),
        }
    }
}

impl From<f32> for Interval {
    #[inline]
    fn from(v: f32) -> Self {
        Self::point(v)
    }
}

impl core::ops::Neg for Interval {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
}

impl core::ops::Add for Interval {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            lo: next_down(self.lo + rhs.lo),
            hi: next_up(self.hi + rhs.hi),
        }
    }
}

impl core::ops::Sub for Interval {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            lo: next_down(self.lo - rhs.hi),
            hi: next_up(self.hi - rhs.lo),
        }
    }
}

impl core::ops::Mul for Interval {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let p0 = self.lo * rhs.lo;
        let p1 = self.lo * rhs.hi;
        let p2 = self.hi * rhs.lo;
        let p3 = self.hi * rhs.hi;
        Self {
            lo: next_down(min4(p0, p1, p2, p3)),
            hi: next_up(max4(p0, p1, p2, p3)),
        }
    }
}

impl core::ops::Div for Interval {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        // A divisor straddling zero yields an unbounded enclosure.
        if rhs.lo <= 0.0 && rhs.hi >= 0.0 {
            return Self::UNBOUNDED;
        }
        let q0 = self.lo / rhs.lo;
        let q1 = self.lo / rhs.hi;
        let q2 = self.hi / rhs.lo;
        let q3 = self.hi / rhs.hi;
        Self {
            lo: next_down(min4(q0, q1, q2, q3)),
            hi: next_up(max4(q0, q1, q2, q3)),
        }
    }
}

/// A conservatively-rounded axis-aligned box of three [`Interval`]s, usable as
/// a "never miss" bounding volume for culling and `CCD` broad-phase tests.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntervalVec3 {
    /// Interval along the x axis.
    pub x: Interval,
    /// Interval along the y axis.
    pub y: Interval,
    /// Interval along the z axis.
    pub z: Interval,
}

impl IntervalVec3 {
    /// Build from three per-axis intervals.
    #[inline]
    #[must_use]
    pub const fn new(x: Interval, y: Interval, z: Interval) -> Self {
        Self { x, y, z }
    }

    /// A degenerate box containing a single point.
    #[inline]
    #[must_use]
    pub const fn point(p: Vec3) -> Self {
        Self {
            x: Interval::point(p.x),
            y: Interval::point(p.y),
            z: Interval::point(p.z),
        }
    }

    /// `true` when `p` lies inside the box on every axis.
    #[inline]
    #[must_use]
    pub fn contains(self, p: Vec3) -> bool {
        self.x.contains(p.x) && self.y.contains(p.y) && self.z.contains(p.z)
    }

    /// `true` when the two boxes overlap on every axis (separating-axis test).
    #[inline]
    #[must_use]
    pub fn overlaps(self, other: Self) -> bool {
        self.x.overlaps(other.x) && self.y.overlaps(other.y) && self.z.overlaps(other.z)
    }

    /// Smallest box containing both operands.
    #[inline]
    #[must_use]
    pub fn hull(self, other: Self) -> Self {
        Self {
            x: self.x.hull(other.x),
            y: self.y.hull(other.y),
            z: self.z.hull(other.z),
        }
    }
}

impl core::ops::Add for IntervalVec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl core::ops::Sub for IntervalVec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{Rng, SplitMix64};

    fn rand_f32(rng: &mut SplitMix64) -> f32 {
        // Spread across a few orders of magnitude, both signs.
        let u = (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32; // [0,1)
        let scale = 1.0 + (rng.next_u32() % 1000) as f32;
        (u - 0.5) * 2.0 * scale
    }

    #[test]
    fn ulp_steps_bracket_a_value() {
        for &v in &[0.0f32, 1.0, -1.0, 1234.5, -0.001, f32::MIN_POSITIVE] {
            assert!(next_down(v) < v, "next_down failed for {v}");
            assert!(next_up(v) > v, "next_up failed for {v}");
        }
        assert_eq!(next_up(f32::INFINITY), f32::INFINITY);
        assert_eq!(next_down(f32::NEG_INFINITY), f32::NEG_INFINITY);
    }

    #[test]
    fn arithmetic_encloses_exact_result() {
        let mut rng = SplitMix64::new(0xc0ff_ee99);
        for _ in 0..50_000 {
            let a = rand_f32(&mut rng);
            let b = rand_f32(&mut rng);
            let ia = Interval::point(a);
            let ib = Interval::point(b);

            let (af, bf) = (f64::from(a), f64::from(b));

            let add = ia + ib;
            assert!(
                f64::from(add.lo()) <= af + bf && af + bf <= f64::from(add.hi()),
                "add not enclosed: {a}+{b}"
            );

            let sub = ia - ib;
            assert!(
                f64::from(sub.lo()) <= af - bf && af - bf <= f64::from(sub.hi()),
                "sub not enclosed: {a}-{b}"
            );

            let mul = ia * ib;
            assert!(
                f64::from(mul.lo()) <= af * bf && af * bf <= f64::from(mul.hi()),
                "mul not enclosed: {a}*{b}"
            );

            if b.abs() > 1e-3 {
                let div = ia / ib;
                let q = af / bf;
                assert!(
                    f64::from(div.lo()) <= q && q <= f64::from(div.hi()),
                    "div not enclosed: {a}/{b}"
                );
            }
        }
    }

    #[test]
    fn wide_interval_arithmetic_encloses_corners() {
        let mut rng = SplitMix64::new(0x1234_5678);
        for _ in 0..20_000 {
            let a = Interval::new(rand_f32(&mut rng), rand_f32(&mut rng));
            let b = Interval::new(rand_f32(&mut rng), rand_f32(&mut rng));
            let prod = a * b;
            // Every corner product must be enclosed.
            for &x in &[a.lo(), a.hi()] {
                for &y in &[b.lo(), b.hi()] {
                    let p = f64::from(x) * f64::from(y);
                    assert!(
                        f64::from(prod.lo()) <= p && p <= f64::from(prod.hi()),
                        "corner {x}*{y} not enclosed"
                    );
                }
            }
        }
    }

    #[test]
    fn divisor_straddling_zero_is_unbounded() {
        let num = Interval::new(1.0, 2.0);
        let den = Interval::new(-1.0, 1.0);
        let q = num / den;
        assert_eq!(q, Interval::UNBOUNDED);
    }

    #[test]
    fn sqrt_encloses_and_handles_negatives() {
        let i = Interval::new(-4.0, 9.0);
        let s = i.sqrt();
        assert!(s.lo() <= 0.0 && s.hi() >= 3.0);
        assert!(s.contains(3.0));
    }

    #[test]
    fn hull_and_intersect() {
        let a = Interval::new(0.0, 2.0);
        let b = Interval::new(1.0, 3.0);
        assert_eq!(a.hull(b), Interval::new(0.0, 3.0));
        assert_eq!(a.intersect(b), Some(Interval::new(1.0, 2.0)));
        assert_eq!(a.intersect(Interval::new(5.0, 6.0)), None);
        assert!(a.overlaps(b));
    }

    #[test]
    fn abs_straddling_zero() {
        let a = Interval::new(-3.0, 2.0).abs();
        assert_eq!(a.lo(), 0.0);
        assert_eq!(a.hi(), 3.0);
        let b = Interval::new(-5.0, -2.0).abs();
        assert_eq!(b, Interval::new(2.0, 5.0));
    }

    #[test]
    fn vec3_box_contains_and_overlaps() {
        let unit = IntervalVec3::new(
            Interval::new(0.0, 1.0),
            Interval::new(0.0, 1.0),
            Interval::new(0.0, 1.0),
        );
        assert!(unit.contains(Vec3::new(0.5, 0.5, 0.5)));
        assert!(!unit.contains(Vec3::new(0.5, 1.5, 0.5)));
        let shifted = IntervalVec3::new(
            Interval::new(0.5, 1.5),
            Interval::new(0.5, 1.5),
            Interval::new(0.5, 1.5),
        );
        assert!(unit.overlaps(shifted));
        let far = IntervalVec3::new(
            Interval::new(2.0, 3.0),
            Interval::new(2.0, 3.0),
            Interval::new(2.0, 3.0),
        );
        assert!(!unit.overlaps(far));
    }
}
