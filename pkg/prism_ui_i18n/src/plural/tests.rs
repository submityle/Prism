//! Conformance tests for the `CLDR` plural rule families.
//!
//! Each family is checked against representative members of the Unicode `CLDR`
//! published `@integer`/`@decimal` sample sets, which are designed to cover the
//! boundaries of every rule clause. Integer samples use [`select`], decimal
//! samples use [`select_decimal`].
//!
//! [`select`]: super::PluralRules::select
//! [`select_decimal`]: super::PluralRules::select_decimal

use super::{
    CardinalFamily, OrdinalFamily, PluralCategory, PluralOperands, PluralRules, PluralType,
};
use PluralCategory::{Few, Many, One, Other, Two, Zero};

/// Assert that every integer in `ns` maps to `expected` under `family`.
fn check_ints(family: CardinalFamily, expected: PluralCategory, ns: &[u64]) {
    for &n in ns {
        let got = family.select(&PluralOperands::from_u64(n));
        assert_eq!(got, expected, "{family:?} cardinal n={n}");
    }
}

/// Assert that every decimal string in `ds` maps to `expected` under `family`.
fn check_decimals(family: CardinalFamily, expected: PluralCategory, ds: &[&str]) {
    for &d in ds {
        let ops = PluralOperands::parse(d).unwrap_or_else(|| panic!("parse {d}"));
        let got = family.select(&ops);
        assert_eq!(got, expected, "{family:?} cardinal decimal={d}");
    }
}

// -------------------------------------------------------------------------
// Operand parsing
// -------------------------------------------------------------------------

#[test]
fn operands_from_integer() {
    let ops = PluralOperands::from_u64(1234);
    assert_eq!(ops, PluralOperands { i: 1234, v: 0, w: 0, f: 0, t: 0, e: 0 });
}

#[test]
fn operands_parse_decimal_with_trailing_zeros() {
    // "1.230": i=1, v=3, w=2, f=230, t=23.
    let ops = PluralOperands::parse("1.230").expect("parse");
    assert_eq!(ops, PluralOperands { i: 1, v: 3, w: 2, f: 230, t: 23, e: 0 });
}

#[test]
fn operands_parse_integer_only_and_leading_dot() {
    assert_eq!(
        PluralOperands::parse("12").expect("parse"),
        PluralOperands { i: 12, v: 0, w: 0, f: 0, t: 0, e: 0 }
    );
    // ".5" => i=0, v=1, f=5.
    assert_eq!(
        PluralOperands::parse(".5").expect("parse"),
        PluralOperands { i: 0, v: 1, w: 1, f: 5, t: 5, e: 0 }
    );
}

#[test]
fn operands_parse_sign_and_compact_exponent() {
    assert_eq!(PluralOperands::parse("-12").expect("parse").i, 12);
    let ops = PluralOperands::parse("1.2c6").expect("parse");
    assert_eq!(ops.e, 6);
    assert_eq!(ops.i, 1);
    assert_eq!(ops.f, 2);
}

#[test]
fn operands_parse_rejects_garbage() {
    assert!(PluralOperands::parse("").is_none());
    assert!(PluralOperands::parse("abc").is_none());
    assert!(PluralOperands::parse("1.2.3").is_none());
    assert!(PluralOperands::parse("1.2x3").is_none());
}

// -------------------------------------------------------------------------
// Cardinal families (CLDR sample-set conformance)
// -------------------------------------------------------------------------

#[test]
fn cardinal_germanic() {
    let f = CardinalFamily::Germanic;
    check_ints(f, One, &[1]);
    check_ints(f, Other, &[0, 2, 3, 10, 11, 20, 100, 1000, 10000, 1_000_000]);
    // one requires v = 0, so every fractional value is other.
    check_decimals(f, Other, &["0.0", "0.5", "1.0", "1.5", "2.0", "100.0"]);
}

#[test]
fn cardinal_french() {
    let f = CardinalFamily::French;
    check_ints(f, One, &[0, 1]);
    check_ints(f, Many, &[1_000_000, 2_000_000, 3_000_000]);
    check_ints(f, Other, &[2, 3, 17, 100, 1000, 10000, 100_000, 1_000_001]);
    // one: i = 0,1 regardless of fraction digits.
    check_decimals(f, One, &["0.0", "0.5", "1.0", "1.5", "0.9", "1.9"]);
    check_decimals(f, Other, &["2.0", "2.5", "10.5"]);
}

#[test]
fn cardinal_east_slavic() {
    let f = CardinalFamily::EastSlavic;
    check_ints(f, One, &[1, 21, 31, 41, 51, 61, 71, 81, 101, 1001]);
    check_ints(f, Few, &[2, 3, 4, 22, 23, 24, 32, 33, 34, 102, 103, 104, 1002]);
    check_ints(
        f,
        Many,
        &[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 100, 1000],
    );
    check_decimals(f, Other, &["0.0", "0.5", "1.0", "1.5", "2.0", "5.5"]);
}

#[test]
fn cardinal_polish() {
    let f = CardinalFamily::Polish;
    check_ints(f, One, &[1]);
    check_ints(f, Few, &[2, 3, 4, 22, 23, 24, 32, 33, 34, 102, 103, 104]);
    check_ints(
        f,
        Many,
        &[0, 5, 6, 9, 10, 11, 12, 13, 14, 15, 19, 20, 21, 25, 100, 1000, 1_000_000],
    );
    check_decimals(f, Other, &["0.0", "0.5", "1.0", "1.5", "2.0"]);
}

#[test]
fn cardinal_czech_slovak() {
    let f = CardinalFamily::CzechSlovak;
    check_ints(f, One, &[1]);
    check_ints(f, Few, &[2, 3, 4]);
    check_ints(f, Other, &[0, 5, 6, 9, 10, 11, 100, 1000, 1_000_000]);
    // many: v != 0 (any fraction, including whole-looking "1.0").
    check_decimals(f, Many, &["0.0", "0.5", "1.0", "1.5", "2.0", "10.0"]);
}

#[test]
fn cardinal_arabic() {
    let f = CardinalFamily::Arabic;
    check_ints(f, Zero, &[0]);
    check_ints(f, One, &[1]);
    check_ints(f, Two, &[2]);
    check_ints(f, Few, &[3, 4, 5, 6, 7, 8, 9, 10, 103, 104, 110, 1003]);
    check_ints(f, Many, &[11, 12, 25, 26, 99, 111, 1011, 1099]);
    check_ints(f, Other, &[100, 101, 102, 200, 300, 1000, 10000, 1_000_000]);
    check_decimals(f, Other, &["0.1", "0.5", "1.5", "2.5", "3.5"]);
}

#[test]
fn cardinal_welsh() {
    let f = CardinalFamily::Welsh;
    check_ints(f, Zero, &[0]);
    check_ints(f, One, &[1]);
    check_ints(f, Two, &[2]);
    check_ints(f, Few, &[3]);
    check_ints(f, Many, &[6]);
    check_ints(f, Other, &[4, 5, 7, 8, 9, 10, 11, 100, 1000]);
    check_decimals(f, Other, &["0.5", "1.5", "3.5", "6.5"]);
}

#[test]
fn cardinal_lithuanian() {
    let f = CardinalFamily::Lithuanian;
    check_ints(f, One, &[1, 21, 31, 41, 51, 61, 71, 81, 91, 101, 1001]);
    check_ints(f, Few, &[2, 3, 4, 5, 6, 7, 8, 9, 22, 102, 109]);
    check_ints(f, Other, &[0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 100, 1000]);
    // many: f != 0 (any value with a non-zero fraction).
    check_decimals(f, Many, &["0.1", "0.5", "1.1", "10.9", "11.1"]);
}

#[test]
fn cardinal_romanian() {
    let f = CardinalFamily::Romanian;
    check_ints(f, One, &[1]);
    check_ints(f, Few, &[0, 2, 3, 12, 19, 101, 112, 119, 1012]);
    check_ints(f, Other, &[20, 21, 35, 80, 100, 120, 1000, 1_000_000]);
    // few also covers every fractional value (v != 0).
    check_decimals(f, Few, &["0.5", "1.5", "2.5", "20.5"]);
}

#[test]
fn cardinal_slovenian() {
    let f = CardinalFamily::Slovenian;
    check_ints(f, One, &[1, 101, 201, 301, 1001]);
    check_ints(f, Two, &[2, 102, 202, 1002]);
    check_ints(f, Few, &[3, 4, 103, 104, 1003, 1004]);
    check_ints(f, Other, &[0, 5, 6, 7, 10, 11, 100, 200, 1000]);
    // few also covers every fractional value (v != 0).
    check_decimals(f, Few, &["0.5", "1.5", "2.5", "5.5"]);
}

#[test]
fn cardinal_irish() {
    let f = CardinalFamily::Irish;
    check_ints(f, One, &[1]);
    check_ints(f, Two, &[2]);
    check_ints(f, Few, &[3, 4, 5, 6]);
    check_ints(f, Many, &[7, 8, 9, 10]);
    check_ints(f, Other, &[0, 11, 12, 20, 100, 1000]);
    check_decimals(f, Other, &["0.5", "1.5", "3.5", "7.5"]);
}

// -------------------------------------------------------------------------
// Ordinal families
// -------------------------------------------------------------------------

#[test]
fn ordinal_english() {
    let f = OrdinalFamily::English;
    let sel = |n: u64| f.select(&PluralOperands::from_u64(n));
    for n in [1, 21, 31, 41, 101, 1001] {
        assert_eq!(sel(n), One, "en ordinal n={n}");
    }
    for n in [2, 22, 32, 102] {
        assert_eq!(sel(n), Two, "en ordinal n={n}");
    }
    for n in [3, 23, 33, 103] {
        assert_eq!(sel(n), Few, "en ordinal n={n}");
    }
    for n in [0, 4, 5, 11, 12, 13, 14, 20, 100, 111, 112, 113] {
        assert_eq!(sel(n), Other, "en ordinal n={n}");
    }
}

#[test]
fn ordinal_one_is_one_and_other_only() {
    let fr = OrdinalFamily::OneIsOne;
    assert_eq!(fr.select(&PluralOperands::from_u64(1)), One);
    for n in [0, 2, 3, 11, 21, 100] {
        assert_eq!(fr.select(&PluralOperands::from_u64(n)), Other, "fr ordinal n={n}");
    }
    let de = OrdinalFamily::OtherOnly;
    for n in [0, 1, 2, 3, 11, 21, 100] {
        assert_eq!(de.select(&PluralOperands::from_u64(n)), Other, "de ordinal n={n}");
    }
}

// -------------------------------------------------------------------------
// Locale resolution + PluralRules facade
// -------------------------------------------------------------------------

#[test]
fn locale_resolution_cardinal() {
    assert_eq!(PluralRules::cardinal("en"), PluralRules::Cardinal(CardinalFamily::Germanic));
    assert_eq!(PluralRules::cardinal("en-US"), PluralRules::Cardinal(CardinalFamily::Germanic));
    assert_eq!(PluralRules::cardinal("pt-BR"), PluralRules::Cardinal(CardinalFamily::French));
    assert_eq!(PluralRules::cardinal("ru"), PluralRules::Cardinal(CardinalFamily::EastSlavic));
    assert_eq!(PluralRules::cardinal("pl"), PluralRules::Cardinal(CardinalFamily::Polish));
    assert_eq!(PluralRules::cardinal("cs"), PluralRules::Cardinal(CardinalFamily::CzechSlovak));
    assert_eq!(PluralRules::cardinal("ar-EG"), PluralRules::Cardinal(CardinalFamily::Arabic));
    assert_eq!(PluralRules::cardinal("zh_Hant"), PluralRules::Cardinal(CardinalFamily::OtherOnly));
    // Unknown language -> CLDR root (other-only).
    assert_eq!(PluralRules::cardinal("xx"), PluralRules::Cardinal(CardinalFamily::OtherOnly));
}

#[test]
fn locale_resolution_ordinal() {
    assert_eq!(PluralRules::ordinal("en"), PluralRules::Ordinal(OrdinalFamily::English));
    assert_eq!(PluralRules::ordinal("fr"), PluralRules::Ordinal(OrdinalFamily::OneIsOne));
    assert_eq!(PluralRules::ordinal("de"), PluralRules::Ordinal(OrdinalFamily::OtherOnly));
    assert_eq!(PluralRules::ordinal("en").plural_type(), PluralType::Ordinal);
    assert_eq!(PluralRules::cardinal("en").plural_type(), PluralType::Cardinal);
}

#[test]
fn facade_select_paths() {
    let ru = PluralRules::cardinal("ru");
    assert_eq!(ru.select(1), One);
    assert_eq!(ru.select(2), Few);
    assert_eq!(ru.select(5), Many);
    assert_eq!(ru.select_decimal("1.5"), Other);
    assert_eq!(ru.select_i64(-21), One);
    // Malformed decimal degrades to other.
    assert_eq!(ru.select_decimal("not-a-number"), Other);
}

// -------------------------------------------------------------------------
// Legacy alias back-compatibility
// -------------------------------------------------------------------------

#[test]
fn legacy_aliases_match_families() {
    for n in [0, 1, 2, 11, 21, 100, 1000] {
        assert_eq!(
            PluralRules::English.select(n),
            CardinalFamily::Germanic.select(&PluralOperands::from_u64(n)),
            "English alias n={n}"
        );
        assert_eq!(
            PluralRules::Slavic.select(n),
            CardinalFamily::EastSlavic.select(&PluralOperands::from_u64(n)),
            "Slavic alias n={n}"
        );
        assert_eq!(PluralRules::Asian.select(n), Other, "Asian alias n={n}");
        assert_eq!(PluralRules::OtherOnly.select(n), Other, "OtherOnly alias n={n}");
    }
}

#[test]
fn legacy_english_one_other() {
    assert_eq!(PluralRules::English.select(1), One);
    assert_eq!(PluralRules::English.select(0), Other);
    assert_eq!(PluralRules::English.select(2), Other);
}

#[test]
fn legacy_slavic_one_few_many() {
    assert_eq!(PluralRules::Slavic.select(1), One);
    assert_eq!(PluralRules::Slavic.select(2), Few);
    assert_eq!(PluralRules::Slavic.select(5), Many);
    assert_eq!(PluralRules::Slavic.select(11), Many);
    assert_eq!(PluralRules::Slavic.select(21), One);
}
