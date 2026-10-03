//! Deterministic `sqrt`, `exp`, and `ln` on [`Fixed`] (Q32.32).
//!
//! Every routine here is pure integer math (plus `i128` intermediates): no
//! hardware float is touched, so results are bit-identical on every platform.
//! The algorithms are the classic range-reduction + polynomial / digit-by-digit
//! approaches:
//!
//! - [`Fixed::sqrt`] uses an exact `u128` digit-by-digit integer square root.
//! - [`Fixed::exp`] reduces `x = k·ln2 + r` and evaluates a 5th-order Taylor
//!   series for `exp(r)`, then scales by `2^k` with a bit shift.
//! - [`Fixed::ln`] normalizes `x = m·2^e` with `m ∈ [1, 2)` and evaluates the
//!   `atanh` series `ln(m) = 2·(u + u³/3 + u⁵/5 + u⁷/7)` with
//!   `u = (m-1)/(m+1)`.
//!
//! Accuracy bounds (empirically verified by the M4 tests over the documented
//! domains) are stated on each method.

use super::Fixed;

// Reciprocal-factorial and series constants as raw Q32.32 integers.
const INV_2: Fixed = Fixed::from_bits(2_147_483_648); // 1/2
const INV_6: Fixed = Fixed::from_bits(715_827_883); // 1/6
const INV_24: Fixed = Fixed::from_bits(178_956_971); // 1/24
const INV_120: Fixed = Fixed::from_bits(35_791_394); // 1/120
const INV_3: Fixed = Fixed::from_bits(1_431_655_765); // 1/3
const INV_5: Fixed = Fixed::from_bits(858_993_459); // 1/5
const INV_7: Fixed = Fixed::from_bits(613_566_757); // 1/7
const INV_LN2: Fixed = Fixed::from_bits(6_196_328_019); // 1/ln(2) = log2(e)

/// Exact floor integer square root of a `u128`, digit-by-digit.
#[inline]
const fn isqrt_u128(n: u128) -> u128 {
    let mut x = n;
    let mut c: u128 = 0;
    // Largest power of four not exceeding `n`.
    let mut d: u128 = 1u128 << 126;
    while d > n {
        d >>= 2;
    }
    while d != 0 {
        if x >= c + d {
            x -= c + d;
            c = (c >> 1) + d;
        } else {
            c >>= 1;
        }
        d >>= 2;
    }
    c
}

impl Fixed {
    /// Non-negative square root.
    ///
    /// Returns [`Fixed::ZERO`] for negative inputs (the deterministic choice —
    /// there is no NaN in fixed point). Computed via an exact integer square
    /// root, so the result is the correctly-rounded-down Q32.32 value: the
    /// error is in `[0, 2^-32)` plus at most one ulp, i.e. `|sqrt(x)² - x|` is
    /// tiny. The M4 tests bound the absolute error against a rational
    /// reference at `< 1e-6`.
    #[inline]
    pub const fn sqrt(self) -> Self {
        if self.to_bits() <= 0 {
            return Self::ZERO;
        }
        // value = raw / 2^32; sqrt(value)·2^32 = sqrt(raw · 2^32).
        let radicand = (self.to_bits() as u128) << 32;
        Self::from_bits(isqrt_u128(radicand) as i64)
    }

    /// Natural exponential `e^x`.
    ///
    /// Range-reduces `x = k·ln2 + r` and evaluates a 5th-order Taylor series on
    /// the small remainder `r`, then scales by `2^k`. Saturates to
    /// [`Fixed::MAX`] on overflow (large `x`) and to [`Fixed::ZERO`] on deep
    /// underflow (very negative `x`). Over `x ∈ [-10, 10]` the M4 tests bound
    /// the relative error at `< 1e-4`.
    #[inline]
    pub fn exp(self) -> Self {
        // k = round(x / ln2)
        let k = self.saturating_mul(INV_LN2).round().to_int();
        if k > 63 {
            return Self::MAX;
        }
        if k < -63 {
            return Self::ZERO;
        }
        let r = self.saturating_sub(Fixed::from_int(k).saturating_mul(Self::LN_2));
        // exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120
        let r2 = r.saturating_mul(r);
        let r3 = r2.saturating_mul(r);
        let r4 = r3.saturating_mul(r);
        let r5 = r4.saturating_mul(r);
        let exp_r = Self::ONE
            .saturating_add(r)
            .saturating_add(r2.saturating_mul(INV_2))
            .saturating_add(r3.saturating_mul(INV_6))
            .saturating_add(r4.saturating_mul(INV_24))
            .saturating_add(r5.saturating_mul(INV_120));
        // Scale by 2^k via a shift on the raw integer.
        let scaled: i128 = if k >= 0 {
            (exp_r.to_bits() as i128) << (k as u32)
        } else {
            (exp_r.to_bits() as i128) >> ((-k) as u32)
        };
        if scaled > i64::MAX as i128 {
            Self::MAX
        } else if scaled < i64::MIN as i128 {
            Self::MIN
        } else {
            Self::from_bits(scaled as i64)
        }
    }

    /// Natural logarithm `ln(x)`.
    ///
    /// Returns [`Fixed::MIN`] as a sentinel for `x ≤ 0` (undefined). Normalizes
    /// `x = m·2^e` with `m ∈ [1, 2)` and sums `e·ln2` with the fast-converging
    /// `atanh` series for `ln(m)`. Over `x ∈ [1e-3, 1e3]` the M4 tests bound
    /// the absolute error at `< 1e-4`.
    #[inline]
    pub fn ln(self) -> Self {
        let raw = self.to_bits();
        if raw <= 0 {
            return Self::MIN;
        }
        // floor(log2(value)) = floor(log2(raw)) - 32.
        let msb = 63 - (raw as u64).leading_zeros() as i64;
        let e = msb - 32;
        // m_raw = raw / 2^e, landing m in [1, 2) ⇔ m_raw in [2^32, 2^33).
        let m_raw: i64 = if e >= 0 { raw >> (e as u32) } else { raw << ((-e) as u32) };
        let m = Self::from_bits(m_raw);
        // u = (m - 1) / (m + 1), ln(m) = 2·(u + u³/3 + u⁵/5 + u⁷/7).
        let u = m.saturating_sub(Self::ONE).saturating_div(m.saturating_add(Self::ONE));
        let u2 = u.saturating_mul(u);
        let u3 = u2.saturating_mul(u);
        let u5 = u3.saturating_mul(u2);
        let u7 = u5.saturating_mul(u2);
        let series = u
            .saturating_add(u3.saturating_mul(INV_3))
            .saturating_add(u5.saturating_mul(INV_5))
            .saturating_add(u7.saturating_mul(INV_7));
        let ln_m = series.saturating_mul(Self::TWO);
        Fixed::from_int(e).saturating_mul(Self::LN_2).saturating_add(ln_m)
    }
}
