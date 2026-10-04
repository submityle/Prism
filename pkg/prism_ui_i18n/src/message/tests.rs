//! Conformance tests for the `ICU`-style message format subset.

use alloc::string::String;

use crate::format::Args;
use crate::plural::PluralRules;

use super::parser::ParseErrorKind;
use super::MessagePattern;

/// Render `src` against `args` using English cardinal rules in the `en` locale.
fn render(src: &str, args: &Args) -> String {
    MessagePattern::parse(src)
        .expect("pattern should parse")
        .format(args, PluralRules::English, "en")
}

#[test]
fn plain_text_passthrough() {
    assert_eq!(render("just text", &Args::new()), "just text");
}

#[test]
fn simple_argument() {
    let args = Args::new().with("name", "Ada");
    assert_eq!(render("Hello, {name}!", &args), "Hello, Ada!");
}

#[test]
fn missing_argument_left_verbatim() {
    assert_eq!(render("Hi {name}!", &Args::new()), "Hi {name}!");
}

#[test]
fn escaped_braces() {
    let args = Args::new().with("sym", "x");
    assert_eq!(render("Use {{ and }} like {sym}.", &args), "Use { and } like x.");
}

#[test]
fn select_gender() {
    let pattern = "{gender, select, male {he} female {she} other {they}}";
    assert_eq!(render(pattern, &Args::new().with("gender", "male")), "he");
    assert_eq!(render(pattern, &Args::new().with("gender", "female")), "she");
    assert_eq!(render(pattern, &Args::new().with("gender", "nb")), "they");
    // A missing select argument falls back to the `other` arm.
    assert_eq!(render(pattern, &Args::new()), "they");
}

#[test]
fn plural_one_other_with_pound() {
    let pattern = "{count, plural, one {# file} other {# files}}";
    assert_eq!(render(pattern, &Args::new().with("count", 1)), "1 file");
    assert_eq!(render(pattern, &Args::new().with("count", 0)), "0 files");
    assert_eq!(render(pattern, &Args::new().with("count", 7)), "7 files");
}

#[test]
fn plural_exact_selector_matches_original_value() {
    let pattern = "{count, plural, =0 {no files} one {# file} other {# files}}";
    assert_eq!(render(pattern, &Args::new().with("count", 0)), "no files");
    assert_eq!(render(pattern, &Args::new().with("count", 1)), "1 file");
    assert_eq!(render(pattern, &Args::new().with("count", 3)), "3 files");
}

#[test]
fn plural_offset_adjusts_pound_and_keyword() {
    // "You and # others": offset:1 → one remaining person, keyword uses value-1.
    let pattern =
        "{count, plural, offset:1 =1 {You} one {You and # other} other {You and # others}}";
    // =1 matches the original value before offset.
    assert_eq!(render(pattern, &Args::new().with("count", 1)), "You");
    // value 2 → adjusted 1 → `one`, `#` renders 1.
    assert_eq!(render(pattern, &Args::new().with("count", 2)), "You and 1 other");
    // value 5 → adjusted 4 → `other`, `#` renders 4.
    assert_eq!(render(pattern, &Args::new().with("count", 5)), "You and 4 others");
}

#[test]
fn selectordinal_english() {
    let pattern =
        "{place, selectordinal, one {#st} two {#nd} few {#rd} other {#th}}";
    let ord = |n: i64| {
        MessagePattern::parse(pattern)
            .expect("parse")
            .format(&Args::new().with("place", n), PluralRules::English, "en")
    };
    assert_eq!(ord(1), "1st");
    assert_eq!(ord(2), "2nd");
    assert_eq!(ord(3), "3rd");
    assert_eq!(ord(4), "4th");
    assert_eq!(ord(11), "11th");
    assert_eq!(ord(21), "21st");
    assert_eq!(ord(22), "22nd");
    assert_eq!(ord(23), "23rd");
}

#[test]
fn nested_select_in_plural_inherits_pound() {
    // `#` inside the nested select arm refers to the enclosing plural value.
    let pattern = "{count, plural, one {{gender, select, male {# brother} other {# sibling}}} \
         other {{gender, select, male {# brothers} other {# siblings}}}}";
    let args = Args::new().with("count", 1).with("gender", "male");
    assert_eq!(render(pattern, &args), "1 brother");
    let args = Args::new().with("count", 3).with("gender", "female");
    assert_eq!(render(pattern, &args), "3 siblings");
}

#[test]
fn pound_literal_at_top_level() {
    assert_eq!(render("issue #", &Args::new()), "issue #");
}

#[test]
fn pound_literal_inside_select() {
    // Outside any plural, `#` stays literal even within a select arm.
    let pattern = "{kind, select, bug {bug #} other {item #}}";
    assert_eq!(render(pattern, &Args::new().with("kind", "bug")), "bug #");
    assert_eq!(render(pattern, &Args::new().with("kind", "task")), "item #");
}

#[test]
fn non_numeric_plural_argument_falls_back_to_zero() {
    let pattern = "{count, plural, one {# file} other {# files}}";
    // A string value is not a number → treated as 0 → `other`.
    assert_eq!(render(pattern, &Args::new().with("count", "oops")), "0 files");
}

#[test]
fn error_missing_other_arm() {
    let err = MessagePattern::parse("{count, plural, one {# file}}").unwrap_err();
    assert_eq!(err.reason, ParseErrorKind::MissingOther);
}

#[test]
fn error_missing_other_in_select() {
    let err = MessagePattern::parse("{g, select, male {he}}").unwrap_err();
    assert_eq!(err.reason, ParseErrorKind::MissingOther);
}

#[test]
fn error_unclosed_brace() {
    let err = MessagePattern::parse("Hello, {name").unwrap_err();
    assert_eq!(err.reason, ParseErrorKind::UnclosedBrace);
}

#[test]
fn error_unknown_type() {
    let err = MessagePattern::parse("{x, frobnicate, other {y}}").unwrap_err();
    assert_eq!(
        err.reason,
        ParseErrorKind::UnknownType(String::from("frobnicate"))
    );
}

#[test]
fn error_empty_name() {
    let err = MessagePattern::parse("{}").unwrap_err();
    assert_eq!(err.reason, ParseErrorKind::EmptyName);
}

#[test]
fn error_bad_keyword() {
    let err = MessagePattern::parse("{n, plural, lots {x} other {y}}").unwrap_err();
    assert_eq!(err.reason, ParseErrorKind::BadKeyword(String::from("lots")));
}
