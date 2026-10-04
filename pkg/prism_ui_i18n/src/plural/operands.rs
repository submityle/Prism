//! `CLDR` plural operands.
//!
//! The `CLDR` plural rules do not operate on a bare number; they operate on a
//! fixed set of *operands* derived from the number's **decimal
//! representation** (the exact digits the user sees), per
//! [UTS#35 Part 4](https://unicode.org/reports/tr35/tr35-numbers.html#Operands).
//! All operands are integers, so selection stays in pure integer arithmetic —
//! no floating-point is used anywhere, preserving determinism.
//!
//! | Operand | Meaning                                                        |
//! |---------|----------------------------------------------------------------|
//! | `n`     | absolute value of the source number                            |
//! | `i`     | integer digits of `n`                                          |
//! | `v`     | count of visible fraction digits, **with** trailing zeros      |
//! | `w`     | count of visible fraction digits, **without** trailing zeros   |
//! | `f`     | visible fraction digits **with** trailing zeros, as an integer |
//! | `t`     | visible fraction digits **without** trailing zeros, as integer |
//! | `e`     | compact decimal exponent (`c` is a synonym), `0` when absent    |
//!
//! Because `n` can be fractional, rules that compare `n` (e.g. `n = 1` or
//! `n % 100 = 11..99`) only match when the value is an exact integer. The
//! helpers [`PluralOperands::n_eq`], [`PluralOperands::n_in`], and
//! [`PluralOperands::n_mod`] encode that semantics directly so rule code reads
//! like the `CLDR` source.

/// The decimal operands backing a `CLDR` plural selection.
///
/// Construct from an integer with [`PluralOperands::from_u64`] /
/// [`PluralOperands::from_i64`], or from an exact decimal string with
/// [`PluralOperands::parse`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PluralOperands {
    /// Integer digits of the absolute value (`i`).
    pub i: u64,
    /// Visible fraction-digit count including trailing zeros (`v`).
    pub v: u32,
    /// Visible fraction-digit count excluding trailing zeros (`w`).
    pub w: u32,
    /// Visible fraction digits including trailing zeros, as integer (`f`).
    pub f: u64,
    /// Visible fraction digits excluding trailing zeros, as integer (`t`).
    pub t: u64,
    /// Compact decimal exponent (`e` / `c`); `0` when the number is not compact.
    pub e: u32,
}

impl PluralOperands {
    /// Operands for a non-negative integer: `i = n`, every fraction operand `0`.
    pub const fn from_u64(n: u64) -> Self {
        PluralOperands {
            i: n,
            v: 0,
            w: 0,
            f: 0,
            t: 0,
            e: 0,
        }
    }

    /// Operands for a signed integer, using its absolute value per `CLDR`.
    pub const fn from_i64(n: i64) -> Self {
        Self::from_u64(n.unsigned_abs())
    }

    /// Whether the number is an exact integer (`v == 0 && e == 0`).
    ///
    /// `CLDR` comparisons against `n` (equality, ranges, modulo) can only
    /// succeed for exact integers; this gates those helpers.
    const fn is_integer(&self) -> bool {
        self.v == 0
    }

    /// `n = value`: true only when the value is exactly `value`.
    pub const fn n_eq(&self, value: u64) -> bool {
        self.is_integer() && self.i == value
    }

    /// `n = lo..hi` (inclusive integer range membership).
    pub const fn n_in(&self, lo: u64, hi: u64) -> bool {
        self.is_integer() && self.i >= lo && self.i <= hi
    }

    /// `n % modulus`, defined only for exact integers (returns `None` for
    /// fractional values so range/equality tests against it correctly fail).
    pub const fn n_mod(&self, modulus: u64) -> Option<u64> {
        if self.is_integer() {
            Some(self.i % modulus)
        } else {
            None
        }
    }

    /// `n % modulus = lo..hi` with the fractional-value semantics of [`n_mod`].
    ///
    /// [`n_mod`]: PluralOperands::n_mod
    pub const fn n_mod_in(&self, modulus: u64, lo: u64, hi: u64) -> bool {
        match self.n_mod(modulus) {
            Some(r) => r >= lo && r <= hi,
            None => false,
        }
    }

    /// Parse an exact decimal string such as `"1234.560"` or `"-12"`.
    ///
    /// A compact exponent may be supplied with a trailing `c<k>` or `e<k>`
    /// (e.g. `"1.2c6"` meaning `1.2 x 10^6`); this sets the `e`/`c` operand but
    /// does not expand the digits, matching `CLDR` compact-number operands.
    ///
    /// Returns `None` if the string is not a well-formed decimal. A leading `+`
    /// or `-` sign is accepted and ignored (operands use the absolute value).
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.strip_prefix(['+', '-']).unwrap_or(s);
        if s.is_empty() {
            return None;
        }

        // Split off an optional compact exponent suffix (`c<k>` / `e<k>`).
        let (mantissa, exp) = match s.find(['c', 'e', 'C', 'E']) {
            Some(idx) => {
                let (m, rest) = s.split_at(idx);
                let digits = &rest[1..];
                if m.is_empty() || digits.is_empty() {
                    return None;
                }
                let e: u32 = digits.parse().ok()?;
                (m, e)
            }
            None => (s, 0),
        };

        let (int_part, frac_part) = match mantissa.split_once('.') {
            Some((int_part, frac_part)) => (int_part, frac_part),
            None => (mantissa, ""),
        };

        // The integer part may be empty (".5") meaning zero integer digits.
        if int_part.is_empty() && frac_part.is_empty() {
            return None;
        }
        if !int_part.bytes().all(|b| b.is_ascii_digit())
            || !frac_part.bytes().all(|b| b.is_ascii_digit())
        {
            return None;
        }

        let i: u64 = if int_part.is_empty() {
            0
        } else {
            int_part.parse().ok()?
        };

        let v = frac_part.len() as u32;
        let trimmed = frac_part.trim_end_matches('0');
        let w = trimmed.len() as u32;
        let f: u64 = if frac_part.is_empty() {
            0
        } else {
            frac_part.parse().ok()?
        };
        let t: u64 = if trimmed.is_empty() {
            0
        } else {
            trimmed.parse().ok()?
        };

        Some(PluralOperands {
            i,
            v,
            w,
            f,
            t,
            e: exp,
        })
    }
}

impl From<u64> for PluralOperands {
    fn from(value: u64) -> Self {
        Self::from_u64(value)
    }
}

impl From<i64> for PluralOperands {
    fn from(value: i64) -> Self {
        Self::from_i64(value)
    }
}
