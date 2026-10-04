//! Query-string building and order-preserving parsing with
//! `application/x-www-form-urlencoded` semantics.
//!
//! [`Location`](crate::Location) already exposes query parameters as a
//! deduplicated map, which is what route guards and typed extraction want. This
//! module is the complementary half used to *construct* links and to read
//! queries that carry repeated keys:
//!
//! * [`QueryString`] is an insertion-order builder modelled on the web
//!   `URLSearchParams` API — `append` keeps duplicate keys in order and
//!   [`QueryString::build`] renders the encoded string.
//! * [`serialize_query_pairs`] renders an iterator of borrowed key/value pairs
//!   directly, for callers that already hold their parameters.
//! * [`parse_query_pairs`] decodes a query string into an ordered vector that
//!   preserves duplicates (the `getAll` use case), unlike the map on
//!   [`Location`](crate::Location).
//!
//! Encoding delegates to [`form_encode`](crate::encoding) /
//! [`form_decode`](crate::encoding), so a space is `+`, a literal `+` is `%2B`,
//! and all other reserved bytes are `%XX`. Because those two are exact inverses
//! for any UTF-8 text, parsing a freshly serialized set of pairs reproduces the
//! original pairs exactly. All work is pure string/integer manipulation and is
//! `no_std`-friendly (requires `alloc`).

use alloc::string::String;
use alloc::vec::Vec;

use crate::encoding::{form_decode, form_encode};

/// Serializes an iterator of borrowed `(key, value)` pairs into an
/// `application/x-www-form-urlencoded` query string, preserving the iteration
/// order and any duplicate keys.
///
/// Each key and value is encoded with [`form_encode`](crate::encoding) and
/// joined as `key=value` with `&` separators. The result has no leading `?`.
/// An empty iterator yields an empty string.
///
/// ```
/// use prism_ui_router::serialize_query_pairs;
///
/// let qs = serialize_query_pairs([("q", "rust lang"), ("tag", "a&b")]);
/// assert_eq!(qs, "q=rust+lang&tag=a%26b");
/// ```
pub fn serialize_query_pairs<'a, I>(pairs: I) -> String
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    let mut out = String::new();
    for (key, value) in pairs {
        if !out.is_empty() {
            out.push('&');
        }
        out.push_str(&form_encode(key));
        out.push('=');
        out.push_str(&form_encode(value));
    }
    out
}

/// Parses a query string into an ordered vector of decoded `(key, value)`
/// pairs, preserving duplicate keys.
///
/// A single leading `?` is tolerated and stripped. Pairs are split on `&`;
/// empty pairs (from a leading, trailing, or doubled `&`) are skipped. Within a
/// pair the first `=` separates key from value, and a pair with no `=` is read
/// as a key with an empty value. Keys and values are decoded with
/// [`form_decode`](crate::encoding).
///
/// Unlike [`Location::query`](crate::Location::query), repeated keys are all
/// retained in order, which is the `getAll` behaviour callers need for
/// multi-valued filters.
///
/// ```
/// use prism_ui_router::parse_query_pairs;
///
/// let pairs = parse_query_pairs("tag=a&tag=b&q=rust+lang");
/// assert_eq!(
///     pairs,
///     vec![
///         ("tag".to_string(), "a".to_string()),
///         ("tag".to_string(), "b".to_string()),
///         ("q".to_string(), "rust lang".to_string()),
///     ],
/// );
/// ```
#[must_use]
pub fn parse_query_pairs(query: &str) -> Vec<(String, String)> {
    let trimmed = query.strip_prefix('?').unwrap_or(query);
    let mut out = Vec::new();
    for pair in trimmed.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (key, value) = match pair.split_once('=') {
            Some((key, value)) => (key, value),
            None => (pair, ""),
        };
        out.push((form_decode(key), form_decode(value)));
    }
    out
}

/// An insertion-order query-string builder modelled on the web
/// `URLSearchParams` API.
///
/// Keys are kept in append order and duplicates are allowed, so building a
/// query for `?tag=a&tag=b` is just two [`QueryString::append`] calls. Call
/// [`QueryString::build`] to render the encoded string.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QueryString {
    pairs: Vec<(String, String)>,
}

impl QueryString {
    /// Creates an empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a `(key, value)` pair, keeping insertion order and allowing
    /// duplicate keys. Returns `&mut self` for chaining.
    pub fn append(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.pairs.push((key.into(), value.into()));
        self
    }

    /// Returns whether no pairs have been appended.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Returns the number of appended pairs (counting duplicates).
    #[must_use]
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Renders the encoded `application/x-www-form-urlencoded` string, with no
    /// leading `?`.
    #[must_use]
    pub fn build(&self) -> String {
        serialize_query_pairs(
            self.pairs
                .iter()
                .map(|(key, value)| (key.as_str(), value.as_str())),
        )
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::std_instead_of_alloc, reason = "tests run under std")]

    use super::{parse_query_pairs, serialize_query_pairs, QueryString};
    use crate::Location;
    use alloc::collections::BTreeMap;
    use alloc::string::{String, ToString};
    use alloc::vec::Vec;

    #[test]
    fn serializes_golden_pairs() {
        let qs = serialize_query_pairs([("a", "1"), ("b", "x y"), ("c", "a&b"), ("d", "1+2")]);
        assert_eq!(qs, "a=1&b=x+y&c=a%26b&d=1%2B2");
        assert_eq!(serialize_query_pairs(core::iter::empty::<(&str, &str)>()), "");
    }

    #[test]
    fn parses_duplicates_empties_and_leading_question_mark() {
        assert_eq!(
            parse_query_pairs("tag=a&tag=b&q=x+y"),
            vec![
                ("tag".to_string(), "a".to_string()),
                ("tag".to_string(), "b".to_string()),
                ("q".to_string(), "x y".to_string()),
            ]
        );
        // Leading `?` tolerated; empty pairs skipped; missing `=` is empty value.
        assert_eq!(
            parse_query_pairs("?&a=1&&flag&"),
            vec![
                ("a".to_string(), "1".to_string()),
                ("flag".to_string(), String::new()),
            ]
        );
        assert_eq!(parse_query_pairs(""), Vec::<(String, String)>::new());
        assert_eq!(parse_query_pairs("&&&"), Vec::<(String, String)>::new());
    }

    #[test]
    fn builder_preserves_order_and_duplicates() {
        let mut qs = QueryString::new();
        assert!(qs.is_empty());
        qs.append("tag", "a").append("tag", "b").append("q", "x y");
        assert_eq!(qs.len(), 3);
        assert!(!qs.is_empty());
        assert_eq!(qs.build(), "tag=a&tag=b&q=x+y");
    }

    #[test]
    fn builder_default_matches_new() {
        assert_eq!(QueryString::default(), QueryString::new());
        assert_eq!(QueryString::new().build(), "");
    }

    // --- Property tests over a deterministic generator ---------------------

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

    /// Characters chosen to exercise every encoding branch: unreserved,
    /// space (`+`), the reserved query delimiters, a literal `+`/`%`, and a
    /// multibyte code point.
    const ALPHABET: &[char] = &[
        'a', 'Z', '0', '-', '_', '.', '~', ' ', '&', '=', '?', '#', '+', '%', '/', 'é', '世',
    ];

    fn gen_text(rng: &mut SplitMix64, max_len: u32) -> String {
        let len = rng.below(max_len + 1);
        let mut out = String::new();
        for _ in 0..len {
            out.push(ALPHABET[rng.below(ALPHABET.len() as u32) as usize]);
        }
        out
    }

    #[test]
    fn parse_is_left_inverse_of_serialize() {
        // For any set of pairs, decoding a freshly encoded query reproduces the
        // originals exactly (form_encode/form_decode are inverses and every
        // pair serializes to a non-empty `key=value` token).
        let mut rng = SplitMix64(0x0123_4567_89AB_CDEF);
        for _ in 0..3_000 {
            let count = rng.below(6);
            let mut pairs: Vec<(String, String)> = Vec::new();
            for _ in 0..count {
                pairs.push((gen_text(&mut rng, 6), gen_text(&mut rng, 6)));
            }
            let encoded = serialize_query_pairs(
                pairs.iter().map(|(k, v)| (k.as_str(), v.as_str())),
            );
            assert_eq!(parse_query_pairs(&encoded), pairs);
        }
    }

    #[test]
    fn builder_output_agrees_with_location_for_unique_keys() {
        // When keys are unique, the builder's output parses (via Location) to
        // the same deduplicated map, cross-checking against the existing
        // parser rather than our own.
        let mut rng = SplitMix64(0x7777_1111_3333_9999);
        for _ in 0..1_500 {
            let count = rng.below(6);
            let mut builder = QueryString::new();
            let mut expected: BTreeMap<String, String> = BTreeMap::new();
            for index in 0..count {
                // Force unique keys with an index prefix.
                let key = alloc::format!("k{index}{}", gen_text(&mut rng, 3));
                let value = gen_text(&mut rng, 5);
                builder.append(key.clone(), value.clone());
                expected.insert(key, value);
            }
            let encoded = builder.build();
            let location = Location::new(&alloc::format!("/path?{encoded}"));
            let mut got: BTreeMap<String, String> = BTreeMap::new();
            for key in expected.keys() {
                got.insert(
                    key.clone(),
                    location.query(key).map(ToString::to_string).unwrap_or_default(),
                );
            }
            assert_eq!(got, expected);
        }
    }
}
