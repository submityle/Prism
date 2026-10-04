//! URL-ish path parsing into a reactive-friendly [`Location`] value.
//!
//! A [`Location`] splits an input string such as `/users/42/posts?page=2#top`
//! into its path, decoded path segments, query parameters, and fragment. The
//! parsing is pure string work with no allocation beyond the owned pieces it
//! stores, and it is `no_std`-friendly (requires `alloc`).

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::encoding::{form_decode, percent_decode};

/// A parsed location: a path, its segments, a query map, and an optional
/// fragment.
///
/// Construct one with [`Location::new`]. The original path portion is preserved
/// verbatim (including any trailing slash) and can be read with
/// [`Location::path`], while [`Location::segments`] exposes the non-empty path
/// segments used for route matching, percent-decoded. Query components are
/// decoded with form-urlencoded semantics (`+` is a space) and the fragment is
/// percent-decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Location {
    path: String,
    segments: Vec<String>,
    query: BTreeMap<String, String>,
    fragment: Option<String>,
}

impl Location {
    /// Parses `input` into a [`Location`].
    ///
    /// The input is split into `path?query#fragment` order: the fragment is
    /// everything after the first `#`, the query is everything after the first
    /// `?` in the remainder, and the path is what is left. An empty path is
    /// normalised to `/`.
    #[must_use]
    pub fn new(input: &str) -> Self {
        // Split off the fragment first, as it is the trailing component.
        let (before_fragment, fragment) = match input.split_once('#') {
            Some((head, frag)) => (head, Some(percent_decode(frag))),
            None => (input, None),
        };

        // Then split the query from the path.
        let (raw_path, raw_query) = match before_fragment.split_once('?') {
            Some((head, query)) => (head, Some(query)),
            None => (before_fragment, None),
        };

        let path = if raw_path.is_empty() {
            "/".to_string()
        } else {
            raw_path.to_string()
        };

        let segments = raw_path
            .split('/')
            .filter(|segment| !segment.is_empty())
            .map(percent_decode)
            .collect();

        let mut query = BTreeMap::new();
        if let Some(raw_query) = raw_query {
            for pair in raw_query.split('&') {
                if pair.is_empty() {
                    continue;
                }
                let (key, value) = match pair.split_once('=') {
                    Some((key, value)) => (key, value),
                    None => (pair, ""),
                };
                query.insert(form_decode(key), form_decode(value));
            }
        }

        Self {
            path,
            segments,
            query,
            fragment,
        }
    }

    /// Returns the path portion of the location, verbatim (e.g. `/users/42`).
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns the non-empty path segments, in order.
    #[must_use]
    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    /// Returns the value of query parameter `key`, if present.
    #[must_use]
    pub fn query(&self, key: &str) -> Option<&str> {
        self.query.get(key).map(String::as_str)
    }

    /// Returns the fragment (the part after `#`), if present.
    #[must_use]
    pub fn fragment(&self) -> Option<&str> {
        self.fragment.as_deref()
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::std_instead_of_alloc, reason = "tests run under std")]

    use super::Location;

    #[test]
    fn parses_segments_query_and_fragment() {
        let loc = Location::new("/users/42/posts?page=2&sort=asc#top");
        assert_eq!(loc.path(), "/users/42/posts");
        assert_eq!(loc.segments(), ["users", "42", "posts"]);
        assert_eq!(loc.query("page"), Some("2"));
        assert_eq!(loc.query("sort"), Some("asc"));
        assert_eq!(loc.query("missing"), None);
        assert_eq!(loc.fragment(), Some("top"));
    }

    #[test]
    fn root_path_has_no_segments() {
        let loc = Location::new("/");
        assert_eq!(loc.path(), "/");
        assert!(loc.segments().is_empty());
        assert_eq!(loc.fragment(), None);
    }

    #[test]
    fn empty_input_normalises_to_root() {
        let loc = Location::new("");
        assert_eq!(loc.path(), "/");
        assert!(loc.segments().is_empty());
    }

    #[test]
    fn trailing_slash_is_ignored_for_segments() {
        let loc = Location::new("/users/42/");
        assert_eq!(loc.path(), "/users/42/");
        assert_eq!(loc.segments(), ["users", "42"]);
    }

    #[test]
    fn query_without_value_is_empty_string() {
        let loc = Location::new("/search?q");
        assert_eq!(loc.query("q"), Some(""));
    }

    #[test]
    fn segments_are_percent_decoded() {
        let loc = Location::new("/users/john%20doe/caf%C3%A9");
        assert_eq!(loc.segments(), ["users", "john doe", "café"]);
        // The raw path is preserved verbatim for round-tripping / history.
        assert_eq!(loc.path(), "/users/john%20doe/caf%C3%A9");
    }

    #[test]
    fn encoded_slash_stays_within_one_segment() {
        let loc = Location::new("/files/a%2Fb/c");
        assert_eq!(loc.segments(), ["files", "a/b", "c"]);
    }

    #[test]
    fn query_uses_form_decoding_with_plus_as_space() {
        let loc = Location::new("/search?q=hello+world&tag=a%26b");
        assert_eq!(loc.query("q"), Some("hello world"));
        assert_eq!(loc.query("tag"), Some("a&b"));
    }

    #[test]
    fn fragment_is_percent_decoded() {
        let loc = Location::new("/doc#section%201");
        assert_eq!(loc.fragment(), Some("section 1"));
    }
}
