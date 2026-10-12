//! Deterministic trigonometry on [`Fixed`] (Q32.32): [`Fixed::sin`],
//! [`Fixed::cos`], [`Fixed::sin_cos`], [`Fixed::tan`], and [`Fixed::atan2`].
//!
//! All routines are pure integer math (no hardware float), so results are
//! bit-identical across platforms. `sin`/`cos` range-reduce the angle modulo
//! `τ`, fold it into `[-π/2, π/2]`, and evaluate a 9th-order odd Taylor series.
//! `atan2` reduces to a first-octant ratio and evaluates a well-known degree-9
//! odd minimax polynomial for `atan` on `[0, 1]`.
//!
//! Accuracy (verified by the M4 tests): `sin`/`cos` are within `2e-5`
//! absolute of a high-precision reference across a full period; `atan2` is
//! within `3e-4` radians.

use super::Fixed;

// Odd Taylor coefficients for sin on [-π/2, π/2], as raw Q32.32 integers.
const INV_6: Fixed = Fixed::from_bits(715_827_883); // 1/3!
const INV_120: Fixed = Fixed::from_bits(35_791_394); // 1/5!
const INV_5040: Fixed = Fixed::from_bits(852_176); // 1/7!
const INV_362880: Fixed = Fixed::from_bits(11_836); // 1/9!

// Minimax atan coefficients on [0, 1] (odd polynomial in z).
const ATAN_C0: Fixed = Fixed::from_bits(4_294_391_770); // 0.9998660
const ATAN_C1: Fixed = Fixed::from_bits(-1_418_625_550); // -0.3302995
const ATAN_C2: Fixed = Fixed::from_bits(773_699_704); // 0.1801410
const ATAN_C3: Fixed = Fixed::from_bits(-365_643_451); // -0.0851330
const ATAN_C4: Fixed = Fixed::from_bits(89_486_073); // 0.0208351

/// Evaluate `sin(r)` for `r` already folded into `[-π/2, π/2]`.
#[inline]
fn sin_core(r: Fixed) -> Fixed {
    let r2 = r.saturating_mul(r);
    let r3 = r2.saturating_mul(r);
    let r5 = r3.saturating_mul(r2);
    let r7 = r5.saturating_mul(r2);
    let r9 = r7.saturating_mul(r2);
    r.saturating_sub(r3.saturating_mul(INV_6))
        .saturating_add(r5.saturating_mul(INV_120))
        .saturating_sub(r7.saturating_mul(INV_5040))
        .saturating_add(r9.saturating_mul(INV_362880))
}

/// Reduce an angle to `[-π/2, π/2]` returning the folded angle; the caller
/// gets a value whose sine equals the sine of the original angle.
#[inline]
fn fold_to_half_pi(x: Fixed) -> Fixed {
    let tau = Fixed::TAU.to_bits();
    let pi = Fixed::PI.to_bits();
    let half_pi = Fixed::FRAC_PI_2.to_bits();
    // Reduce modulo τ into (-τ, τ), then into [-π, π].
    let mut r = x.to_bits() % tau;
    if r > pi {
        r -= tau;
    } else if r < -pi {
        r += tau;
    }
    // Fold [-π, π] into [-π/2, π/2] using sin(π - r) = sin(r).
    if r > half_pi {
        r = pi - r;
    } else if r < -half_pi {
        r = -pi - r;
    }
    Fixed::from_bits(r)
}

/// Reduce first, then shift by a bounded amount. Shifting an unreduced `x`
/// with saturating addition would change the phase for inputs near `MIN/MAX`.
#[inline]
fn fold_cos_to_half_pi(x: Fixed) -> Fixed {
    let reduced = x.to_bits() % Fixed::TAU.to_bits();
    fold_to_half_pi(Fixed::from_bits(reduced + Fixed::FRAC_PI_2.to_bits()))
}

impl Fixed {
    /// Sine of an angle in radians.
    #[inline]
    pub fn sin(self) -> Self {
        sin_core(fold_to_half_pi(self))
    }

    /// Cosine of an angle in radians.
    #[inline]
    pub fn cos(self) -> Self {
        // cos(x) = sin(x + π/2). Reduce before adding so extreme raw values
        // cannot saturate and lose their periodic phase.
        sin_core(fold_cos_to_half_pi(self))
    }

    /// Sine and cosine together.
    #[inline]
    pub fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    /// Tangent of an angle in radians (`sin/cos`; saturates near the poles).
    #[inline]
    pub fn tan(self) -> Self {
        let (s, c) = self.sin_cos();
        s.saturating_div(c)
    }

    /// Four-quadrant arctangent of `self / x`, in radians over `(-π, π]`.
    ///
    /// `atan2(0, 0)` returns [`Fixed::ZERO`].
    #[inline]
    pub fn atan2(self, x: Self) -> Self {
        let y = self;
        if x.is_zero() && y.is_zero() {
            return Self::ZERO;
        }
        // Unsigned magnitudes represent |i64::MIN| exactly (it is one raw
        // unit larger than Fixed::MAX), unlike Fixed::abs(), which saturates.
        let ax = x.to_bits().unsigned_abs();
        let ay = y.to_bits().unsigned_abs();
        // First-octant angle in [0, π/2].
        let a = if ax >= ay {
            atan_unit(Fixed::from_bits(ratio_q32(ay, ax)))
        } else {
            Self::FRAC_PI_2.saturating_sub(atan_unit(Fixed::from_bits(ratio_q32(ax, ay))))
        };
        // Place into the correct quadrant from the signs of x and y.
        let a = if x.is_negative() {
            Self::PI.saturating_sub(a)
        } else {
            a
        };
        if y.is_negative() {
            a.saturating_neg()
        } else {
            a
        }
    }
}

/// Return `(numerator / denominator)` as a Q32.32 raw ratio, rounded down.
/// Preconditions: denominator > 0 and numerator <= denominator.
#[inline]
fn ratio_q32(numerator: u64, denominator: u64) -> i64 {
    (((numerator as u128) << 32) / denominator as u128) as i64
}

/// `atan(z)` for `z ∈ [0, 1]` via an odd minimax polynomial.
#[inline]
fn atan_unit(z: Fixed) -> Fixed {
    let z2 = z.saturating_mul(z);
    // Horner on z²: C0 + z²(C1 + z²(C2 + z²(C3 + z²·C4)))
    let mut acc = ATAN_C4;
    acc = ATAN_C3.saturating_add(z2.saturating_mul(acc));
    acc = ATAN_C2.saturating_add(z2.saturating_mul(acc));
    acc = ATAN_C1.saturating_add(z2.saturating_mul(acc));
    acc = ATAN_C0.saturating_add(z2.saturating_mul(acc));
    z.saturating_mul(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_of_extreme_raw_angles_preserves_phase_shift() {
        for raw in [i64::MIN, i64::MIN + 1, i64::MAX - 1, i64::MAX] {
            let angle = Fixed::from_bits(raw);
            let reduced = raw % Fixed::TAU.to_bits();
            let expected = sin_core(fold_to_half_pi(Fixed::from_bits(
                reduced + Fixed::FRAC_PI_2.to_bits(),
            )));
            assert_eq!(angle.cos(), expected, "raw angle {raw}");
        }
        assert_ne!(Fixed::MAX.cos(), Fixed::MAX.sin());
    }

    #[test]
    fn atan2_uses_exact_minimum_magnitude() {
        let diagonal = Fixed::from_bits(i64::MIN).atan2(Fixed::from_bits(i64::MIN));
        assert_eq!(diagonal, Fixed::NEG_ONE.atan2(Fixed::NEG_ONE));

        let near_diagonal = Fixed::from_bits(i64::MIN + 1).atan2(Fixed::from_bits(i64::MIN));
        assert_ne!(near_diagonal, diagonal);
        assert!(near_diagonal.to_bits().abs_diff(diagonal.to_bits()) <= 2);
        assert_ne!(
            Fixed::MAX.atan2(Fixed::from_bits(i64::MIN)),
            diagonal,
            "MIN and MAX magnitudes must not collapse to the same ratio"
        );
        assert_eq!(
            Fixed::from_bits(i64::MIN).atan2(Fixed::ZERO),
            -Fixed::FRAC_PI_2
        );
        assert_eq!(Fixed::ZERO.atan2(Fixed::from_bits(i64::MIN)), Fixed::PI);
    }
}
