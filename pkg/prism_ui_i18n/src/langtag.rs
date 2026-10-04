//! BCP 47 language-tag canonicalization and RFC 4647 language matching.
//!
//! The rest of the crate keys catalogs and plural rules off opaque locale
//! strings (see [`LocaleId`](crate::LocaleId)); the only structural handling so
//! far is primary-subtag extraction inside the plural registry. This module is
//! the complementary negotiation layer: it canonicalizes a tag's casing per
//! BCP 47 and resolves a user's requested locales against the set an
//! application actually ships, using the two standard RFC 4647 algorithms.
//!
//! * [`canonicalize`] applies BCP 47 §2.1.1 case folding: the primary language
//!   subtag is lowercase, a script subtag is `Titlecase`, a region subtag is
//!   `UPPERCASE`, and everything else (variants, singletons, extension and
//!   private-use subtags) is lowercase. Both `-` and `_` are accepted as
//!   separators and normalized to `-`.
//! * [`truncation_chain`] produces the progressively truncated lookup sequence
//!   from RFC 4647 §3.4, dropping a trailing single-character singleton subtag
//!   alongside the subtag it introduces.
//! * [`lookup`] implements RFC 4647 §3.4 "Lookup": the single best available
//!   tag for an ordered list of requested ranges, or a caller-supplied default.
//! * [`basic_filter`] implements RFC 4647 §3.3 "Basic Filtering": every
//!   available tag that equals a range or extends it at a subtag boundary.
//!
//! Everything is pure string/integer manipulation and `no_std`-friendly (it
//! only needs `alloc`).

use alloc::string::String;
use alloc::vec::Vec;

/// Canonicalizes the casing of a BCP 47 language tag.
///
/// Subtags are classified by position and shape per BCP 47 §2.1.1:
///
/// * subtag 0 (the primary language) is lowercased;
/// * the first 4-letter subtag after the language, before any region or
///   singleton, is a script and is `Titlecased`;
/// * the first 2-letter subtag, before any later region or singleton, is a
///   region and is `UPPERCASED` (a 3-digit numeric region is left as digits);
/// * once a single-character singleton subtag (such as `x` or `u`) is seen,
///   that subtag and everything after it is lowercased;
/// * all remaining subtags (variants, extensions, private use) are lowercased.
///
/// Both `-` and `_` are accepted as separators and emitted as `-`. Empty
/// subtags produced by doubled, leading, or trailing separators are dropped.
/// Non-ASCII subtags are passed through [`str::to_lowercase`] without being
/// treated as a script or region.
///
/// ```
/// use prism_ui_i18n::canonicalize;
///
/// assert_eq!(canonicalize("EN-latn-gb-boont"), "en-Latn-GB-boont");
/// assert_eq!(canonicalize("zh_hant_cn"), "zh-Hant-CN");
/// assert_eq!(canonicalize("de-DE-x-FOO"), "de-DE-x-foo");
/// ```
#[must_use]
pub fn canonicalize(tag: &str) -> String {
    let mut out = String::with_capacity(tag.len());
    let mut index = 0usize;
    let mut seen_script = false;
    let mut seen_region = false;
    let mut in_singleton = false;

    for subtag in tag.split(['-', '_']) {
        if subtag.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push('-');
        }

        let ascii = subtag.is_ascii();
        let len = subtag.len();
        let all_alpha = subtag.bytes().all(|b| b.is_ascii_alphabetic());
        let all_digit = subtag.bytes().all(|b| b.is_ascii_digit());

        if index == 0 || in_singleton || (ascii && len == 1) {
            push_lowercase(&mut out, subtag);
        } else if ascii && len == 4 && all_alpha && !seen_script && !seen_region {
            push_titlecase(&mut out, subtag);
            seen_script = true;
        } else if ascii && len == 2 && all_alpha && !seen_region {
            push_uppercase(&mut out, subtag);
            seen_region = true;
        } else if ascii && len == 3 && all_digit && !seen_region {
            out.push_str(subtag);
            seen_region = true;
        } else {
            push_lowercase(&mut out, subtag);
        }

        if ascii && len == 1 {
            in_singleton = true;
        }
        index += 1;
    }
    out
}

fn push_lowercase(out: &mut String, subtag: &str) {
    out.push_str(&subtag.to_lowercase());
}

fn push_uppercase(out: &mut String, subtag: &str) {
    out.push_str(&subtag.to_uppercase());
}

fn push_titlecase(out: &mut String, subtag: &str) {
    let mut chars = subtag.chars();
    let Some(first) = chars.next() else {
        return;
    };
    for upper in first.to_uppercase() {
        out.push(upper);
    }
    for lower in chars.flat_map(char::to_lowercase) {
        out.push(lower);
    }
}

/// Builds the RFC 4647 §3.4 lookup truncation sequence for a language range.
///
/// The range is canonicalized, then progressively truncated one subtag at a
/// time. When a truncation leaves a single-character singleton as the trailing
/// subtag, that singleton is removed together with the subtag it introduced, so
/// the chain never ends on a bare singleton. The first element is the full
/// canonicalized range and each later element is a strict subtag-prefix of the
/// previous one.
///
/// A `*` range (which conveys no tag to match in the lookup scheme) and the
/// empty string both yield an empty chain.
///
/// ```
/// use prism_ui_i18n::truncation_chain;
///
/// assert_eq!(
///     truncation_chain("zh-hant-cn-x-private1-private2"),
///     vec![
///         "zh-Hant-CN-x-private1-private2".to_string(),
///         "zh-Hant-CN-x-private1".to_string(),
///         "zh-Hant-CN".to_string(),
///         "zh-Hant".to_string(),
///         "zh".to_string(),
///     ],
/// );
/// ```
#[must_use]
pub fn truncation_chain(range: &str) -> Vec<String> {
    let canon = canonicalize(range);
    let mut chain = Vec::new();
    if canon.is_empty() || canon == "*" {
        return chain;
    }

    let mut current = canon;
    loop {
        chain.push(current.clone());
        let Some(cut) = current.rfind('-') else {
            break;
        };
        current.truncate(cut);
        // Drop any run of trailing single-character singleton subtags along with
        // the subtags they introduce, so the chain never matches on a dangling
        // extension/private-use singleton.
        while let Some(prev) = current.rfind('-')
            && current.len() - prev - 1 == 1
        {
            current.truncate(prev);
        }
        if current.is_empty() || (!current.contains('-') && current.chars().count() == 1) {
            break;
        }
    }
    chain
}

/// Resolves the single best available tag for an ordered list of requested
/// language ranges, per RFC 4647 §3.4 "Lookup".
///
/// Each range is tried in priority order; for a range, its
/// [`truncation_chain`] is walked from longest to shortest and the first
/// `available` tag whose canonical form equals the current truncation wins.
/// Comparison is case-insensitive because both sides are canonicalized. If no
/// range matches, `default` is returned.
///
/// ```
/// use prism_ui_i18n::lookup;
///
/// let available = ["en", "zh-Hant", "fr"];
/// // zh-Hant-TW truncates to zh-Hant before zh, so the script match wins.
/// assert_eq!(lookup(&["zh-Hant-TW"], &available, Some("en")), Some("zh-Hant"));
/// // The first range misses entirely; the second resolves.
/// assert_eq!(lookup(&["sr-Latn", "fr-CA"], &available, Some("en")), Some("fr"));
/// // Nothing matches, so the default is used.
/// assert_eq!(lookup(&["de"], &available, Some("en")), Some("en"));
/// ```
#[must_use]
pub fn lookup<'a>(
    ranges: &[&str],
    available: &'a [&'a str],
    default: Option<&'a str>,
) -> Option<&'a str> {
    let canon_available: Vec<String> = available.iter().map(|&tag| canonicalize(tag)).collect();
    for range in ranges {
        for prefix in truncation_chain(range) {
            for (position, candidate) in canon_available.iter().enumerate() {
                if *candidate == prefix {
                    return Some(available[position]);
                }
            }
        }
    }
    default
}

/// Returns every available tag that matches a single language range, per
/// RFC 4647 §3.3 "Basic Filtering".
///
/// A tag matches when its canonical form equals the canonical range or extends
/// it at a subtag boundary (`de` matches `de` and `de-DE` but not `den`). The
/// wildcard range `*` matches every available tag. Results preserve the order
/// of `available`.
///
/// ```
/// use prism_ui_i18n::basic_filter;
///
/// let available = ["de", "de-DE", "de-Latn-DE", "den", "en-de"];
/// assert_eq!(basic_filter("de", &available), vec!["de", "de-DE", "de-Latn-DE"]);
/// ```
#[must_use]
pub fn basic_filter<'a>(range: &str, available: &'a [&'a str]) -> Vec<&'a str> {
    let canon_range = canonicalize(range);
    let mut out = Vec::new();
    if canon_range == "*" {
        out.extend_from_slice(available);
        return out;
    }

    let mut prefix = canon_range.clone();
    prefix.push('-');
    for &tag in available {
        let canon = canonicalize(tag);
        if canon == canon_range || canon.starts_with(&prefix) {
            out.push(tag);
        }
    }
    out
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;

    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, bound: u32) -> u32 {
            (self.next_u64() % u64::from(bound)) as u32
        }
    }

    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    fn random_subtag(rng: &mut SplitMix64) -> String {
        let len = rng.below(8) + 1;
        let mut s = String::new();
        for _ in 0..len {
            let pick = rng.below(ALPHABET.len() as u32) as usize;
            s.push(ALPHABET[pick] as char);
        }
        s
    }

    fn random_tag(rng: &mut SplitMix64) -> String {
        let count = rng.below(5) + 1;
        let mut parts = Vec::new();
        for _ in 0..count {
            parts.push(random_subtag(rng));
        }
        parts.join("-")
    }

    #[test]
    fn canonicalize_golden_cases() {
        assert_eq!(canonicalize("en"), "en");
        assert_eq!(canonicalize("EN"), "en");
        assert_eq!(canonicalize("en-us"), "en-US");
        assert_eq!(canonicalize("zh-hant"), "zh-Hant");
        assert_eq!(canonicalize("ZH-HANT-CN"), "zh-Hant-CN");
        assert_eq!(canonicalize("mn-cyrl-mn"), "mn-Cyrl-MN");
        assert_eq!(canonicalize("en-latn-gb-boont"), "en-Latn-GB-boont");
        assert_eq!(canonicalize("es-419"), "es-419");
        assert_eq!(canonicalize("sl-rozaj-biske"), "sl-rozaj-biske");
        assert_eq!(canonicalize("pt_BR"), "pt-BR");
        assert_eq!(canonicalize("de-DE-x-FOO"), "de-DE-x-foo");
        // Doubled / leading / trailing separators are dropped.
        assert_eq!(canonicalize("-en--US-"), "en-US");
    }

    #[test]
    fn canonicalize_is_idempotent() {
        let mut rng = SplitMix64(0x1234_5678_9ABC_DEF0);
        for _ in 0..4000 {
            let tag = random_tag(&mut rng);
            let once = canonicalize(&tag);
            let twice = canonicalize(&once);
            assert_eq!(once, twice, "canonicalize not idempotent for {tag:?}");
        }
    }

    #[test]
    fn truncation_chain_golden_and_invariants() {
        assert_eq!(
            truncation_chain("zh-Hant-TW"),
            vec!["zh-Hant-TW".to_string(), "zh-Hant".to_string(), "zh".to_string()],
        );
        assert_eq!(truncation_chain("en-US"), vec!["en-US".to_string(), "en".to_string()]);
        assert!(truncation_chain("*").is_empty());
        assert!(truncation_chain("").is_empty());

        let mut rng = SplitMix64(0x0F0F_0F0F_1234_5678);
        for _ in 0..4000 {
            let tag = random_tag(&mut rng);
            let chain = truncation_chain(&tag);
            if chain.is_empty() {
                continue;
            }
            assert_eq!(chain[0], canonicalize(&tag), "chain head must be canonical range");
            for window in chain.windows(2) {
                let longer = &window[0];
                let shorter = &window[1];
                assert!(longer.len() > shorter.len(), "chain must strictly shrink: {chain:?}");
                assert!(
                    longer.as_bytes()[shorter.len()] == b'-'
                        && longer.starts_with(shorter.as_str()),
                    "each element must be a subtag-prefix of the previous: {chain:?}",
                );
            }
            // Past the head (which echoes the raw range verbatim), no element
            // may end on a dangling single-character singleton tail (e.g. a
            // trailing "-x"); such a subtag is always dropped together with the
            // subtag it introduces.
            for element in &chain[1..] {
                if let Some(dash) = element.rfind('-') {
                    assert!(
                        element[dash + 1..].chars().count() != 1,
                        "element {element:?} ends on a singleton tail: {chain:?}",
                    );
                }
            }
        }
    }

    #[test]
    fn lookup_golden_cases() {
        let available = ["en", "zh-Hant", "fr"];
        assert_eq!(lookup(&["zh-Hant-TW"], &available, Some("en")), Some("zh-Hant"));
        assert_eq!(lookup(&["fr-CA"], &available, Some("en")), Some("fr"));
        assert_eq!(lookup(&["de"], &available, Some("en")), Some("en"));
        assert_eq!(lookup(&["de"], &available, None), None);
        assert_eq!(lookup(&["*"], &available, Some("en")), Some("en"));
        // Priority order: first range misses, second resolves.
        assert_eq!(lookup(&["sr-Latn", "en"], &available, None), Some("en"));
    }

    #[test]
    fn lookup_matches_first_chain_prefix() {
        // Independent oracle: for a single range, the resolved tag must be the
        // available tag whose canonical form equals the earliest-available
        // truncation prefix of the range, and nothing earlier may match.
        let pool = ["en", "en-US", "zh", "zh-Hant", "zh-Hant-TW", "fr-FR", "sr-Latn-RS"];
        let mut rng = SplitMix64(0xDEAD_BEEF_0BAD_F00D);
        for _ in 0..4000 {
            let range = random_tag(&mut rng);
            // Random available subset.
            let mut available: Vec<&str> = Vec::new();
            for &tag in &pool {
                if rng.below(2) == 0 {
                    available.push(tag);
                }
            }
            let got = lookup(&[range.as_str()], &available, None);
            let chain = truncation_chain(&range);
            let canon_available: Vec<String> =
                available.iter().map(|&t| canonicalize(t)).collect();

            // Compute the expected match independently via chain indexing.
            let mut expected: Option<&str> = None;
            'outer: for prefix in &chain {
                for (i, cand) in canon_available.iter().enumerate() {
                    if cand == prefix {
                        expected = Some(available[i]);
                        break 'outer;
                    }
                }
            }
            assert_eq!(got, expected, "range {range:?} available {available:?}");

            if let Some(matched) = got {
                let canon_matched = canonicalize(matched);
                assert!(chain.contains(&canon_matched), "match must lie on the chain");
            }
        }
    }

    #[test]
    fn basic_filter_golden_and_membership() {
        let available = ["de", "de-DE", "de-Latn-DE", "den", "en-de"];
        assert_eq!(basic_filter("de", &available), vec!["de", "de-DE", "de-Latn-DE"]);
        assert_eq!(basic_filter("*", &available), available.to_vec());
        assert!(basic_filter("fr", &available).is_empty());

        let pool = ["de", "de-DE", "de-Latn-DE", "den", "en", "en-GB", "zh-Hant"];
        let mut rng = SplitMix64(0x5151_A1A1_B2B2_C3C3);
        for _ in 0..4000 {
            let range = random_tag(&mut rng);
            let result = basic_filter(&range, &pool);
            let canon_range = canonicalize(&range);
            let mut prefix = canon_range.clone();
            prefix.push('-');
            for &tag in &pool {
                let canon = canonicalize(tag);
                let want = canon_range == "*"
                    || canon == canon_range
                    || canon.starts_with(&prefix);
                assert_eq!(
                    result.contains(&tag),
                    want,
                    "membership mismatch for range {range:?} tag {tag:?}",
                );
            }
        }
    }
}
