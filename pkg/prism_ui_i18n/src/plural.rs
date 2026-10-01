//! `CLDR`-style plural category selection.
//!
//! Selection is driven entirely by integer arithmetic over a `u64` count; no
//! floating-point or transcendental math is used. Two concrete rule families
//! are implemented plus two trivial ones, and the fallback is always
//! [`PluralCategory::Other`].

/// The six `CLDR` plural categories.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PluralCategory {
    /// The `zero` category (used by some Semitic/Baltic languages).
    Zero,
    /// The `one` (singular) category.
    One,
    /// The `two` (dual) category.
    Two,
    /// The `few` (paucal) category.
    Few,
    /// The `many` category.
    Many,
    /// The `other` (general plural / fallback) category.
    Other,
}

/// A plural rule family selecting a [`PluralCategory`] from an integer count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PluralRules {
    /// English-style: `one` when `n == 1`, otherwise `other`.
    English,
    /// Slavic-style (e.g. Russian) cardinal rule using `one`/`few`/`many`.
    Slavic,
    /// East-Asian-style (e.g. Chinese/Japanese): always `other`.
    Asian,
    /// Degenerate rule that always returns `other`.
    OtherOnly,
}

impl PluralRules {
    /// Select the plural category for integer count `n`.
    pub fn select(&self, n: u64) -> PluralCategory {
        match self {
            PluralRules::English => {
                if n == 1 {
                    PluralCategory::One
                } else {
                    PluralCategory::Other
                }
            }
            PluralRules::Slavic => {
                let rem10 = n % 10;
                let rem100 = n % 100;
                if rem10 == 1 && rem100 != 11 {
                    PluralCategory::One
                } else if (2..=4).contains(&rem10) && !(12..=14).contains(&rem100) {
                    PluralCategory::Few
                } else {
                    PluralCategory::Many
                }
            }
            PluralRules::Asian | PluralRules::OtherOnly => PluralCategory::Other,
        }
    }
}
