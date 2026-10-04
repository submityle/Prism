//! Locale -> plural rule-family resolution.
//!
//! Maps a BCP-47-style locale identifier (only its primary language subtag is
//! significant for plural rules) to its `CLDR` cardinal and ordinal families.
//! Matching is case-insensitive on the language subtag; the region, script, and
//! variant subtags are ignored because `CLDR` plural rules are keyed by
//! language. Unmapped languages fall back to the `CLDR` root rule
//! ([`CardinalFamily::OtherOnly`] / [`OrdinalFamily::OtherOnly`]).

use super::{CardinalFamily, OrdinalFamily};

/// Extract the lowercase primary language subtag from a locale id.
///
/// Accepts `-` or `_` as subtag separators (e.g. `"pt-BR"`, `"zh_Hant"`).
fn language_subtag(locale: &str) -> &str {
    let end = locale
        .find(['-', '_'])
        .unwrap_or(locale.len());
    &locale[..end]
}

/// Compare a locale's language subtag against a lowercase candidate.
fn lang_eq(locale_lang: &str, candidate: &str) -> bool {
    locale_lang.eq_ignore_ascii_case(candidate)
}

/// Resolve the cardinal rule family for `locale`.
///
/// Returns [`CardinalFamily::OtherOnly`] for any language not in the mapped
/// set, matching the `CLDR` root behaviour.
pub fn cardinal_family(locale: &str) -> CardinalFamily {
    let lang = language_subtag(locale);
    const GERMANIC: &[&str] = &[
        "en", "de", "nl", "sv", "da", "no", "nb", "nn", "fo", "af", "et", "fi",
        "it", "es", "ca", "gl", "eu",
    ];
    const FRENCH: &[&str] = &["fr", "pt"];
    const EAST_SLAVIC: &[&str] = &["ru", "uk", "be"];
    const CZECH_SLOVAK: &[&str] = &["cs", "sk"];

    if GERMANIC.iter().any(|c| lang_eq(lang, c)) {
        CardinalFamily::Germanic
    } else if FRENCH.iter().any(|c| lang_eq(lang, c)) {
        CardinalFamily::French
    } else if EAST_SLAVIC.iter().any(|c| lang_eq(lang, c)) {
        CardinalFamily::EastSlavic
    } else if CZECH_SLOVAK.iter().any(|c| lang_eq(lang, c)) {
        CardinalFamily::CzechSlovak
    } else if lang_eq(lang, "pl") {
        CardinalFamily::Polish
    } else if lang_eq(lang, "ar") {
        CardinalFamily::Arabic
    } else if lang_eq(lang, "cy") {
        CardinalFamily::Welsh
    } else if lang_eq(lang, "lt") {
        CardinalFamily::Lithuanian
    } else if lang_eq(lang, "ro") || lang_eq(lang, "mo") {
        CardinalFamily::Romanian
    } else if lang_eq(lang, "sl") {
        CardinalFamily::Slovenian
    } else if lang_eq(lang, "ga") {
        CardinalFamily::Irish
    } else {
        // CLDR root rule: no plural distinctions. This also covers the
        // "other-only" languages (zh, ja, ko, th, vi, id, ms, lo, km, my, ...).
        CardinalFamily::OtherOnly
    }
}

/// Resolve the ordinal rule family for `locale`.
///
/// Returns [`OrdinalFamily::OtherOnly`] for any language not in the mapped set.
pub fn ordinal_family(locale: &str) -> OrdinalFamily {
    let lang = language_subtag(locale);
    if lang_eq(lang, "en") {
        OrdinalFamily::English
    } else if lang_eq(lang, "fr") {
        OrdinalFamily::OneIsOne
    } else {
        OrdinalFamily::OtherOnly
    }
}
