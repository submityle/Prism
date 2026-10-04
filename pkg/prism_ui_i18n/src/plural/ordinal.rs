//! `CLDR` ordinal plural rule families.
//!
//! Ordinal rules select the category used to format a *position* ("1st", "2nd",
//! "3rd"), and are distinct from the cardinal rules. Ordinals always apply to
//! exact non-negative integers, so selection reads the integer value directly.
//! Locales outside the mapped set fall back to [`OrdinalFamily::OtherOnly`].

use super::{PluralCategory, PluralOperands};

/// A `CLDR` ordinal plural rule family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OrdinalFamily {
    /// Root/most languages: every position is `other` (de, es, nl, ...).
    OtherOnly,
    /// English-style one/two/few/other ("1st", "2nd", "3rd", "4th") (en).
    English,
    /// `one: n = 1`, otherwise `other` (fr).
    OneIsOne,
}

impl OrdinalFamily {
    /// Select the ordinal plural category for `ops` under this family.
    pub fn select(self, ops: &PluralOperands) -> PluralCategory {
        use PluralCategory::{Few, One, Other, Two};
        match self {
            OrdinalFamily::OtherOnly => Other,

            // one: n % 10 = 1 and n % 100 != 11
            // two: n % 10 = 2 and n % 100 != 12
            // few: n % 10 = 3 and n % 100 != 13
            OrdinalFamily::English => {
                if ops.n_mod(10) == Some(1) && ops.n_mod(100) != Some(11) {
                    One
                } else if ops.n_mod(10) == Some(2) && ops.n_mod(100) != Some(12) {
                    Two
                } else if ops.n_mod(10) == Some(3) && ops.n_mod(100) != Some(13) {
                    Few
                } else {
                    Other
                }
            }

            // one: n = 1
            OrdinalFamily::OneIsOne => {
                if ops.n_eq(1) {
                    One
                } else {
                    Other
                }
            }
        }
    }
}
