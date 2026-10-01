//! Compiled route patterns and their matches.
//!
//! A [`RoutePattern`] is compiled from a pattern string containing static
//! segments, named parameters (`:name`), and an optional trailing wildcard
//! (`*name`). Matching a [`Location`] against a pattern yields a [`RouteMatch`]
//! holding the captured parameters and wildcard tail.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::path::Location;

/// A single compiled pattern segment.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Segment {
    /// A literal segment that must match exactly.
    Static(String),
    /// A `:name` segment that captures exactly one path segment.
    Param(String),
    /// A `*name` segment that captures all remaining path segments.
    Wildcard(String),
}

/// A compiled route pattern.
///
/// Compile one with [`RoutePattern::new`]. Static segments must match exactly,
/// a `:name` segment captures a single path segment, and a trailing `*name`
/// segment captures the rest of the path (joined by `/`). A wildcard is only
/// honoured as the final segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutePattern {
    segments: Vec<Segment>,
    has_wildcard: bool,
}

/// The result of matching a [`Location`] against a [`RoutePattern`].
///
/// Captured `:name` parameters are read with [`RouteMatch::param`], and the
/// trailing `*name` capture (if the pattern had one) with
/// [`RouteMatch::wildcard`].
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct RouteMatch {
    params: BTreeMap<String, String>,
    wildcard: Option<String>,
}

impl RoutePattern {
    /// Compiles `pattern` into a [`RoutePattern`].
    ///
    /// The pattern is split on `/`; empty segments are ignored. A segment that
    /// starts with `:` becomes a named parameter, one that starts with `*`
    /// becomes a wildcard, and anything else is a static segment.
    #[must_use]
    pub fn new(pattern: &str) -> Self {
        let mut segments = Vec::new();
        let mut has_wildcard = false;
        for raw in pattern.split('/').filter(|segment| !segment.is_empty()) {
            if let Some(name) = raw.strip_prefix(':') {
                segments.push(Segment::Param(name.to_string()));
            } else if let Some(name) = raw.strip_prefix('*') {
                segments.push(Segment::Wildcard(name.to_string()));
                has_wildcard = true;
            } else {
                segments.push(Segment::Static(raw.to_string()));
            }
        }
        Self {
            segments,
            has_wildcard,
        }
    }

    /// Attempts to match `location` against this pattern.
    ///
    /// Returns `Some(RouteMatch)` on success, or `None` if a static segment
    /// differs or the segment counts do not line up. Without a wildcard the
    /// pattern and location must have the same number of segments; with a
    /// trailing wildcard the location must have at least as many leading
    /// segments as the pattern's fixed part.
    #[must_use]
    pub fn match_path(&self, location: &Location) -> Option<RouteMatch> {
        let path_segments = location.segments();
        let mut result = RouteMatch::default();

        if self.has_wildcard {
            // The wildcard is always the final pattern segment; the preceding
            // segments must each line up with a location segment.
            let fixed_len = self.segments.len() - 1;
            if path_segments.len() < fixed_len {
                return None;
            }
            for (segment, value) in self.segments.iter().zip(path_segments.iter()) {
                match segment {
                    Segment::Static(expected) => {
                        if expected != value {
                            return None;
                        }
                    }
                    Segment::Param(name) => {
                        result.params.insert(name.clone(), value.clone());
                    }
                    Segment::Wildcard(_) => break,
                }
            }
            let rest = path_segments[fixed_len..].join("/");
            result.wildcard = Some(rest);
            return Some(result);
        }

        if self.segments.len() != path_segments.len() {
            return None;
        }
        for (segment, value) in self.segments.iter().zip(path_segments.iter()) {
            match segment {
                Segment::Static(expected) => {
                    if expected != value {
                        return None;
                    }
                }
                Segment::Param(name) => {
                    result.params.insert(name.clone(), value.clone());
                }
                Segment::Wildcard(_) => return None,
            }
        }
        Some(result)
    }
}

impl RouteMatch {
    /// Returns the captured value of parameter `name`, if present.
    #[must_use]
    pub fn param(&self, name: &str) -> Option<&str> {
        self.params.get(name).map(String::as_str)
    }

    /// Returns the captured wildcard tail, if the pattern had a wildcard.
    #[must_use]
    pub fn wildcard(&self) -> Option<&str> {
        self.wildcard.as_deref()
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::std_instead_of_alloc, reason = "tests run under std")]

    use super::RoutePattern;
    use crate::path::Location;

    #[test]
    fn static_pattern_matches_exactly() {
        let pattern = RoutePattern::new("/users/me");
        assert!(pattern.match_path(&Location::new("/users/me")).is_some());
        assert!(pattern.match_path(&Location::new("/users/you")).is_none());
    }

    #[test]
    fn param_is_captured() {
        let pattern = RoutePattern::new("/users/:id/posts");
        let matched = pattern
            .match_path(&Location::new("/users/42/posts"))
            .expect("should match");
        assert_eq!(matched.param("id"), Some("42"));
        assert_eq!(matched.wildcard(), None);
    }

    #[test]
    fn wildcard_captures_remaining_path() {
        let pattern = RoutePattern::new("/files/*rest");
        let matched = pattern
            .match_path(&Location::new("/files/a/b/c.txt"))
            .expect("should match");
        assert_eq!(matched.wildcard(), Some("a/b/c.txt"));
    }

    #[test]
    fn segment_count_mismatch_rejects() {
        let pattern = RoutePattern::new("/users/:id");
        assert!(pattern
            .match_path(&Location::new("/users/42/posts"))
            .is_none());
        assert!(pattern.match_path(&Location::new("/users")).is_none());
    }
}
