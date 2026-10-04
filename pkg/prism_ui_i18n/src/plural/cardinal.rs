//! `CLDR` cardinal plural rule families.
//!
//! Each [`CardinalFamily`] variant is one of the distinct rule expressions in
//! the Unicode `CLDR` cardinal plural data. The variants are named after a
//! representative language, but each applies to every locale that shares the
//! same rule (see [`crate::plural::registry`]). Selection is driven entirely by
//! the integer [`PluralOperands`]; no floating-point is used.
//!
//! The implemented families cover the overwhelming majority of major UI
//! locales (English/Germanic, Romance, East-Slavic, West-Slavic, Arabic,
//! Celtic, Baltic, and all East/South-East-Asian "other-only" languages). Each
//! family is validated against the `CLDR` published `@integer`/`@decimal`
//! sample sets in `plural/tests.rs`; locales outside the mapped set fall back
//! to [`CardinalFamily::OtherOnly`] (the `CLDR` root rule).

use super::{PluralCategory, PluralOperands};

/// A `CLDR` cardinal plural rule family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CardinalFamily {
    /// Root rule: everything is `other` (zh, ja, ko, th, vi, id, ms, ...).
    OtherOnly,
    /// `one: i = 1 and v = 0` (en, de, nl, sv, da, no, fi, it, es, ...).
    Germanic,
    /// `one: i = 0,1`; `many` for exact millions (fr, pt).
    French,
    /// East-Slavic one/few/many (ru, uk, be).
    EastSlavic,
    /// Polish one/few/many (pl).
    Polish,
    /// Czech/Slovak one/few/many (cs, sk).
    CzechSlovak,
    /// Arabic zero/one/two/few/many/other (ar).
    Arabic,
    /// Welsh zero/one/two/few/many/other (cy).
    Welsh,
    /// Lithuanian one/few/many (lt).
    Lithuanian,
    /// Romanian one/few/other (ro).
    Romanian,
    /// Slovenian one/two/few/other (sl).
    Slovenian,
    /// Irish one/two/few/many/other (ga).
    Irish,
}

impl CardinalFamily {
    /// Select the plural category for `ops` under this family.
    pub fn select(self, ops: &PluralOperands) -> PluralCategory {
        use PluralCategory::{Few, Many, One, Other, Two, Zero};
        match self {
            CardinalFamily::OtherOnly => Other,

            // one: i = 1 and v = 0
            CardinalFamily::Germanic => {
                if ops.i == 1 && ops.v == 0 {
                    One
                } else {
                    Other
                }
            }

            // one: i = 0,1
            // many: i != 0 and i % 1000000 = 0 and v = 0
            CardinalFamily::French => {
                if ops.i == 0 || ops.i == 1 {
                    One
                } else if ops.i != 0 && ops.i.is_multiple_of(1_000_000) && ops.v == 0 {
                    Many
                } else {
                    Other
                }
            }

            // one:  v = 0 and i % 10 = 1 and i % 100 != 11
            // few:  v = 0 and i % 10 = 2..4 and i % 100 != 12..14
            // many: v = 0 and i % 10 = 0
            //       or v = 0 and i % 10 = 5..9
            //       or v = 0 and i % 100 = 11..14
            CardinalFamily::EastSlavic => {
                if ops.v != 0 {
                    return Other;
                }
                let i10 = ops.i % 10;
                let i100 = ops.i % 100;
                if i10 == 1 && i100 != 11 {
                    One
                } else if (2..=4).contains(&i10) && !(12..=14).contains(&i100) {
                    Few
                } else {
                    Many
                }
            }

            // one:  i = 1 and v = 0
            // few:  v = 0 and i % 10 = 2..4 and i % 100 != 12..14
            // many: v = 0 and i != 1 and i % 10 = 0..1
            //       or v = 0 and i % 10 = 5..9
            //       or v = 0 and i % 100 = 12..14
            CardinalFamily::Polish => {
                if ops.v != 0 {
                    return Other;
                }
                let i10 = ops.i % 10;
                let i100 = ops.i % 100;
                if ops.i == 1 {
                    One
                } else if (2..=4).contains(&i10) && !(12..=14).contains(&i100) {
                    Few
                } else if (ops.i != 1 && i10 <= 1)
                    || (5..=9).contains(&i10)
                    || (12..=14).contains(&i100)
                {
                    Many
                } else {
                    Other
                }
            }

            // one:  i = 1 and v = 0
            // few:  i = 2..4 and v = 0
            // many: v != 0
            CardinalFamily::CzechSlovak => {
                if ops.v != 0 {
                    Many
                } else if ops.i == 1 {
                    One
                } else if (2..=4).contains(&ops.i) {
                    Few
                } else {
                    Other
                }
            }

            // zero: n = 0        one: n = 1        two: n = 2
            // few:  n % 100 = 3..10     many: n % 100 = 11..99
            CardinalFamily::Arabic => {
                if ops.n_eq(0) {
                    Zero
                } else if ops.n_eq(1) {
                    One
                } else if ops.n_eq(2) {
                    Two
                } else if ops.n_mod_in(100, 3, 10) {
                    Few
                } else if ops.n_mod_in(100, 11, 99) {
                    Many
                } else {
                    Other
                }
            }

            // zero: n = 0  one: n = 1  two: n = 2  few: n = 3  many: n = 6
            CardinalFamily::Welsh => {
                if ops.n_eq(0) {
                    Zero
                } else if ops.n_eq(1) {
                    One
                } else if ops.n_eq(2) {
                    Two
                } else if ops.n_eq(3) {
                    Few
                } else if ops.n_eq(6) {
                    Many
                } else {
                    Other
                }
            }

            // one:  n % 10 = 1 and n % 100 != 11..19
            // few:  n % 10 = 2..9 and n % 100 != 11..19
            // many: f != 0
            CardinalFamily::Lithuanian => {
                if ops.f != 0 {
                    Many
                } else if ops.n_mod(10) == Some(1) && !ops.n_mod_in(100, 11, 19) {
                    One
                } else if ops.n_mod_in(10, 2, 9) && !ops.n_mod_in(100, 11, 19) {
                    Few
                } else {
                    Other
                }
            }

            // one: i = 1 and v = 0
            // few: v != 0 or n = 0 or n != 1 and n % 100 = 1..19
            CardinalFamily::Romanian => {
                if ops.i == 1 && ops.v == 0 {
                    One
                } else if ops.v != 0
                    || ops.n_eq(0)
                    || (!ops.n_eq(1) && ops.n_mod_in(100, 1, 19))
                {
                    Few
                } else {
                    Other
                }
            }

            // one: v = 0 and i % 100 = 1
            // two: v = 0 and i % 100 = 2
            // few: v = 0 and i % 100 = 3..4 or v != 0
            CardinalFamily::Slovenian => {
                let i100 = ops.i % 100;
                if ops.v == 0 && i100 == 1 {
                    One
                } else if ops.v == 0 && i100 == 2 {
                    Two
                } else if ops.v != 0 || (3..=4).contains(&i100) {
                    Few
                } else {
                    Other
                }
            }

            // one: n = 1  two: n = 2  few: n = 3..6  many: n = 7..10
            CardinalFamily::Irish => {
                if ops.n_eq(1) {
                    One
                } else if ops.n_eq(2) {
                    Two
                } else if ops.n_in(3, 6) {
                    Few
                } else if ops.n_in(7, 10) {
                    Many
                } else {
                    Other
                }
            }
        }
    }
}
